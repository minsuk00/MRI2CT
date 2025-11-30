import os
import glob
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from fused_ssim import fused_ssim

def set_seed(seed=42):
    """Sets the seed for reproducibility."""
    print(f"🌱 Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

def minmax(arr, minclip=None, maxclip=None):
    if not (minclip is None and maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    return arr

def pad_to_multiple_np(arr, multiple=16):
    D, H, W = arr.shape
    pad_D = (multiple - D % multiple) % multiple
    pad_H = (multiple - H % multiple) % multiple
    pad_W = (multiple - W % multiple) % multiple
    return np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant'), (pad_D, pad_H, pad_W)

def load_image_pair(root, subj_id):
    ct_path = glob.glob(os.path.join(root, subj_id, "ct_resampled.nii*"))[0]
    mr_path = glob.glob(os.path.join(root, subj_id, "registration_output", "moved_*.nii*"))[0]
    
    mr_img, ct_img = tio.ScalarImage(mr_path), tio.ScalarImage(ct_path)
    mri, ct = mr_img.data[0].numpy(), ct_img.data[0].numpy()
    
    mri = minmax(mri)
    ct = minmax(ct, minclip=-450, maxclip=450)
    
    mri, pad_vals = pad_to_multiple_np(mri, multiple=16)
    ct, _ = pad_to_multiple_np(ct, multiple=16)
    
    print(f"Data Loaded | MRI: {mri.shape}, CT: {ct.shape}")
    return mri, ct, pad_vals

def load_segmentation(root, subj_id, seg_filename="labels_moved*.nii.gz", pad_vals=None):
    seg_path = glob.glob(os.path.join(root, subj_id, "registration_output", seg_filename))[0]
    
    if not os.path.exists(seg_path):
        raise FileNotFoundError(f"Segmentation not found for {subj_id} in {paths_to_check}")

    seg_img = tio.LabelMap(seg_path)
    seg = seg_img.data[0].numpy().astype(np.int16)
    
    if pad_vals is not None:
        pad_D, pad_H, pad_W = pad_vals
        seg = np.pad(seg, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
        
    print(f"Seg Loaded  | Shape: {seg.shape}, Max Label: {seg.max()}")
    return seg

def one_hot_encode(seg, num_classes=None):
    if num_classes is None:
        num_classes = int(seg.max()) + 1
        
    seg_flat = seg.flatten()
    one_hot = np.eye(num_classes, dtype=np.float32)[seg_flat]
    
    D, H, W = seg.shape
    one_hot = one_hot.reshape(D, H, W, num_classes)
    one_hot = np.transpose(one_hot, (3, 0, 1, 2))
    return one_hot

# --- Dataset & Loader ---
class RandomPatchDataset_single(Dataset):
    def __init__(self, feats, target, patch_size=(96, 96, 96), samples_per_epoch=100):
        self.feats = feats
        self.target = target
        self.patch_size = np.array(patch_size)
        self.samples = samples_per_epoch
        self.dims = np.array(target.shape) # D, H, W
        
    def __len__(self):
        return self.samples
    
    def __getitem__(self, idx):
        # Fix: Sample coordinates individually to avoid array errors in np.random.randint
        max_start = np.maximum(self.dims - self.patch_size, 0)
        
        start_d = np.random.randint(0, max_start[0] + 1)
        start_h = np.random.randint(0, max_start[1] + 1)
        start_w = np.random.randint(0, max_start[2] + 1)
        
        start = np.array([start_d, start_h, start_w])
        end = start + self.patch_size
        
        f_patch = self.feats[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        t_patch = self.target[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        return torch.from_numpy(f_patch).float(), torch.from_numpy(t_patch).unsqueeze(0).float()

# ds = RandomPatchDataset_single(
#     feats_mri, 
#     ct_target, 
#     patch_size=(args.patch_size, args.patch_size, args.patch_size), 
#     samples_per_epoch=args.patches_per_epoch
# )
# return DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)

# --- TorchIO Adapter ---
class TioAdapter:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = self 
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            yield batch['input'][tio.DATA], batch['target'][tio.DATA]

# --- Simple Stochastic Datasets ---
class StochasticVoxelDataset(Dataset):
    """
    Optimized MLP Dataset.
    Instead of 1 voxel per call, returns a CHUNK of voxels per call.
    This reduces Python overhead by factor of `chunk_size`.
    """
    def __init__(self, data_list, dataset_len, chunk_size=4096):
        self.subjects = []
        self.weights = []
        self.chunk_size = chunk_size
        self.dataset_len = dataset_len # Virtual length (number of chunks to yield per epoch)
        
        for feats, target in data_list:
            # feats: [C, D, H, W] -> Transpose to [D, H, W, C] for fast channel slicing
            feats_t = np.transpose(feats, (1, 2, 3, 0))
            self.subjects.append((feats_t, target))
            self.weights.append(target.size)
            
        total_voxels = sum(self.weights)
        self.probs = [w / total_voxels for w in self.weights]
        
    def __len__(self):
        return self.dataset_len
        
    def __getitem__(self, idx):
        # 1. Pick Subject
        subj_idx = np.random.choice(len(self.subjects), p=self.probs)
        feats, target = self.subjects[subj_idx]
        
        # 2. Pick 'chunk_size' Random Voxels at once (Vectorized)
        D, H, W, C = feats.shape
        
        # Random coordinates
        z = np.random.randint(0, D, size=self.chunk_size)
        y = np.random.randint(0, H, size=self.chunk_size)
        x = np.random.randint(0, W, size=self.chunk_size)
        
        # 3. Extract (NumPy advanced indexing is fast)
        # feat_chunk: [chunk_size, C]
        feat_chunk = feats[z, y, x] 
        # targ_chunk: [chunk_size]
        targ_chunk = target[z, y, x]
        
        # 4. Coords [0,1]
        # Normalize and stack
        z_n = z / (D - 1)
        y_n = y / (H - 1)
        x_n = x / (W - 1)
        
        # [chunk_size, 3]
        coords_chunk = np.stack([x_n, y_n, z_n], axis=1).astype(np.float32)
        
        return (
            torch.from_numpy(feat_chunk).float(),
            torch.from_numpy(coords_chunk).float(),
            torch.from_numpy(targ_chunk).unsqueeze(1).float() # [chunk, 1]
        )

def collate_flatten(batch):
    """
    Flattens a list of chunks into a single batch.
    Input: List of (Chunk, C) tensors
    Output: (Batch*Chunk, C) tensor
    """
    feats, coords, targs = zip(*batch)
    
    # Stack -> [Batch, Chunk, C] -> View -> [Batch*Chunk, C]
    feats = torch.stack(feats).view(-1, feats[0].shape[-1])
    coords = torch.stack(coords).view(-1, coords[0].shape[-1])
    targs = torch.stack(targs).view(-1, targs[0].shape[-1])
    
    return feats, coords, targs
        
# --- Unified Dataloaders ---
def get_dataloader_single(feats_mri, ct_target, args):
    """Compatibility wrapper for single subject script"""
    return get_dataloader_multi([(feats_mri, ct_target)], args)

def get_dataloader_multi(data_list, args):
    """
    Args:
        data_list: List of tuples [(feats_mri, ct_target), ...]
    """
    # 1. MLP LOADING (Stochastic)
    if args.model_type == "mlp":
        print(f"⚡ Creating MLP Dataloader (Vectorized Stochastic) for {len(data_list)} subjects...")
        
        # Optimization: Fetch 4096 points per __getitem__ call
        chunk_size = 4096
        # Adjust loader batch size to match target total batch size
        # e.g., Target 131072 // 4096 = 32 batches per step
        loader_batch_size = args.batch_size // chunk_size
        # We want 500 update steps per epoch
        dataset_len = args.steps_per_epoch * loader_batch_size
        
        ds = StochasticVoxelDataset(data_list, dataset_len=dataset_len, chunk_size=chunk_size)
        
        return DataLoader(
            ds, 
            batch_size=loader_batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_flatten # Flattens chunks into one big batch
        )
    # 2. CNN LOADING (TorchIO SubjectsDataset)
    elif args.model_type == "cnn":
        print(f"⚡ Creating CNN Dataloader for {len(data_list)} subjects...")
        subjects = []
        for feats, target in data_list:
            # Auto-pad subjects smaller than patch_size
            c, d, h, w = feats.shape
            
            p_d = max(0, args.patch_size - d)
            p_h = max(0, args.patch_size - h)
            p_w = max(0, args.patch_size - w)
            
            if p_d > 0 or p_h > 0 or p_w > 0:
                pad_d1, pad_d2 = p_d // 2, p_d - p_d // 2
                pad_h1, pad_h2 = p_h // 2, p_h - p_h // 2
                pad_w1, pad_w2 = p_w // 2, p_w - p_w // 2
                
                # Pad (C, D, H, W) -> pad last 3 dims
                feats = np.pad(feats, ((0,0), (pad_d1, pad_d2), (pad_h1, pad_h2), (pad_w1, pad_w2)), mode='constant') # pad with 0
                target = np.pad(target, ((pad_d1, pad_d2), (pad_h1, pad_h2), (pad_w1, pad_w2)), mode='constant')
                # print(f"   ⚠️ Auto-padded subject from {(d,h,w)} to {feats.shape[1:]}")

            t_feats = torch.from_numpy(feats).float()
            t_ct = torch.from_numpy(target).unsqueeze(0).float()
            
            # Recalculate prob_map on PADDED tensor so padding (0) counts as air
            prob_map = (t_ct > 0.01).float()
            
            subjects.append(tio.Subject(
                input=tio.ScalarImage(tensor=t_feats),
                target=tio.ScalarImage(tensor=t_ct),
                prob_map=tio.Image(tensor=prob_map)
            ))
            
        dataset = tio.SubjectsDataset(subjects)
        
        # Queue parameters
        patch_size = (args.patch_size, args.patch_size, args.patch_size)
        sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
        
        queue = tio.Queue(
            subjects_dataset=dataset,
            max_length=max(args.patches_per_epoch, 50),
            samples_per_volume=args.patches_per_epoch,
            sampler=sampler,
            num_workers=0, # Faster for in-memory
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        
        loader = DataLoader(queue, batch_size=1, num_workers=0, pin_memory=False)
        return TioAdapter(loader)

    
# --- Loss & Metrics ---
def unpad_np(arr, pad_vals):
    pad_D, pad_H, pad_W = pad_vals
    s_d = slice(None, -pad_D) if pad_D > 0 else slice(None)
    s_h = slice(None, -pad_H) if pad_H > 0 else slice(None)
    s_w = slice(None, -pad_W) if pad_W > 0 else slice(None)
    return arr[s_d, s_h, s_w]
    
def unpad_torch(data, pad_vals):
    """
    data: (B, C, D, H, W) 5D Tensor
    pad_vals: [d_pad, h_pad, w_pad]
    """
    if pad_vals is None:
        return data

    d_pad, h_pad, w_pad = pad_vals
    b, c, d, h, w = data.shape
    
    d_end = d - d_pad if d_pad > 0 else d
    h_end = h - h_pad if h_pad > 0 else h
    w_end = w - w_pad if w_pad > 0 else w
    
    return data[..., :d_end, :h_end, :w_end]
    
def compute_metrics_deprecated(pred, target, pad_vals):
    pred = unpad_np(pred, pad_vals)
    target = unpad_np(target, pad_vals)
    
    mae = np.mean(np.abs(pred - target))
    psnrs, ssims = [], []
    
    for z in range(pred.shape[2]):
        p_slice = pred[..., z]
        t_slice = target[..., z]
        
        # Skip empty background slices
        if t_slice.max() < 1e-6:
            continue

        mse = np.mean((p_slice - t_slice) ** 2)
        if mse < 1e-10:
            psnrs.append(100.0)
        else:
            val = psnr2d(t_slice, p_slice, data_range=1.0)
            psnrs.append(val)
            
        ssims.append(fused_ssim(p_slice, t_slice , train=False))
        
    return mae, np.mean(psnrs) if psnrs else 0.0, np.mean(ssims) if ssims else 0.0

def compute_metrics(pred, target, data_range=1.0):
    b, c, d, h, w = pred.shape
    pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)

    # SSIM
    ssim_val = fused_ssim(pred_2d, targ_2d, train=False).item()

    # PSNR
    mse = torch.mean((pred_2d - targ_2d) ** 2, dim=[1, 2, 3])
    mse = torch.clamp(mse, min=1e-10) 
    psnr_2d = 10.0 * torch.log10((data_range ** 2) / mse)
    psnr_val = torch.mean(psnr_2d).item()

    # MAE
    mae_val = torch.mean(torch.abs(pred - target)).item()
    
    return mae_val, psnr_val, ssim_val

class CompositeLoss(nn.Module):
    def __init__(self, weights={"l1": 1.0, "l2": 0.0, "ssim": 0.0}):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        total_loss = 0.0
        loss_components = {}
        
        if self.weights.get("l1", 0) > 0:
            val_l1 = self.l1(pred, target)
            total_loss += self.weights["l1"] * val_l1
            loss_components["loss_l1"] = val_l1.item()
            
        if self.weights.get("l2", 0) > 0:
            val_l2 = self.l2(pred, target)
            total_loss += self.weights["l2"] * val_l2
            loss_components["loss_l2"] = val_l2.item()
            
        if self.weights.get("ssim", 0) > 0:
            if pred.ndim == 5: 
                b, c, d, h, w = pred.shape
                pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
                
                pred32 = pred_2d.float()
                targ32 = targ_2d.float()
                ssim_score = fused_ssim(pred32, targ32, train=True)
                
                val_ssim_loss = 1.0 - ssim_score
                total_loss += self.weights["ssim"] * val_ssim_loss
                loss_components["loss_ssim"] = val_ssim_loss.item()
        
        return total_loss, loss_components