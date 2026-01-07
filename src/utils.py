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
    """Handles both Numpy and Torch inputs."""
    if torch.is_tensor(arr):
        if minclip is not None and maxclip is not None:
            arr = torch.clamp(arr, minclip, maxclip)
        denom = arr.max() - arr.min()
        if denom == 0: return torch.zeros_like(arr)
        return (arr - arr.min()) / denom
    else:
        if not (minclip is None) & (maxclip is None):
            arr = np.clip(arr, minclip, maxclip)
        denom = arr.max() - arr.min()
        if denom == 0: return np.zeros_like(arr)
        return (arr - arr.min()) / denom

def one_hot_encode(seg, num_classes=None):
    """Handles both Numpy and Torch inputs. Returns (C, D, H, W)."""
    if torch.is_tensor(seg):
        # Torch implementation
        if num_classes is None:
            num_classes = int(seg.max()) + 1
        
        # F.one_hot needs Long type and expects indices in last dim usually, 
        # but here input is (C=1, D, H, W) or (D, H, W)
        if seg.ndim == 4: seg = seg.squeeze(0) # Remove channel dim
        
        seg = seg.long()
        seg_flat = seg.view(-1)
        # Mask out values out of range
        mask = (seg_flat >= 0) & (seg_flat < num_classes)
        seg_flat = seg_flat * mask # Zero out invalid
        
        # Encode
        one_hot = F.one_hot(seg_flat, num_classes=num_classes).float() # [N, C]
        
        # Reshape back: [D, H, W, C] -> Permute to [C, D, H, W]
        D, H, W = seg.shape
        one_hot = one_hot.view(D, H, W, num_classes).permute(3, 0, 1, 2)
        return one_hot
        
    else:
        # Numpy implementation
        if num_classes is None:
            num_classes = int(seg.max()) + 1
        seg_flat = seg.flatten()
        seg_flat[seg_flat >= num_classes] = 0
        
        one_hot = np.eye(num_classes, dtype=np.float32)[seg_flat]
        D, H, W = seg.shape
        one_hot = one_hot.reshape(D, H, W, num_classes)
        one_hot = np.transpose(one_hot, (3, 0, 1, 2))
        return one_hot

def pad_to_multiple_np(arr, multiple=16):
    D, H, W = arr.shape
    pad_D = (multiple - D % multiple) % multiple
    pad_H = (multiple - H % multiple) % multiple
    pad_W = (multiple - W % multiple) % multiple
    return np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant'), (pad_D, pad_H, pad_W)

def get_subject_paths(root, subj_id, seg_name="labels_moved.nii.gz"):
    paths = {}
    
    # 1. CT
    ct = glob.glob(os.path.join(root, subj_id, "ct_resampled.nii*"))
    if not ct:
        raise FileNotFoundError(f"CT missing for {subj_id}")
    paths['ct'] = ct[0]
    
    # 2. MRI
    mr = glob.glob(os.path.join(root, subj_id, "registration_output", "moved_*.nii*"))
    if not mr:
        raise FileNotFoundError(f"MRI missing for {subj_id}")
    paths['mri'] = mr[0]
    
    # 3. Seg (Optional check, but we return path if found)
    seg = glob.glob(os.path.join(root, subj_id, "registration_output", seg_name))
    if not seg:
        raise FileNotFoundError(f"Segmentation missing for {subj_id}")
    paths['seg'] = seg[0]
            
    return paths

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
        raise FileNotFoundError(f"Segmentation not found for {subj_id} in {seg_path}")

    seg_img = tio.LabelMap(seg_path)
    seg = seg_img.data[0].numpy().astype(np.int16)
    
    if pad_vals is not None:
        pad_D, pad_H, pad_W = pad_vals
        seg = np.pad(seg, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant', constant_values=0)
        
    print(f"Seg Loaded  | Shape: {seg.shape}, Max Label: {seg.max()}")
    return seg



class ProjectPreprocessing(tio.Transform):
    """
    Applies MinMax Norm, One-Hot Encoding, and Prob Map creation
    AFTER loading from disk but BEFORE patching.
    """
    def __init__(self, use_seg=False, seg_classes=51, patch_size=96, **kwargs):
        super().__init__(**kwargs)
        self.use_seg = use_seg
        self.seg_classes = seg_classes
        self.patch_size = patch_size

    def apply_transform(self, subject):
        # 0. Safety Padding (Prevent Crash on Small Volumes)
        # We check one image (CT) to determine if padding is needed
        shape = subject['ct'].spatial_shape # (W, H, D) order in TorchIO
        
        # Calc padding for multiple of 16
        mult = 16
        pad_w = (mult - shape[0] % mult) % mult
        pad_h = (mult - shape[1] % mult) % mult
        pad_d = (mult - shape[2] % mult) % mult
        
        # Also ensure minimum patch size
        pad_w = max(pad_w, max(0, self.patch_size - shape[0]))
        pad_h = max(pad_h, max(0, self.patch_size - shape[1]))
        pad_d = max(pad_d, max(0, self.patch_size - shape[2]))

        if pad_w > 0 or pad_h > 0 or pad_d > 0:
            # Symmetric padding is safer, but pad_to_multiple usually does one-side.
            # Let's stick to 'end' padding to match numpy behavior if possible, 
            # or split. TorchIO Pad splits by default.
            # Let's padding_mode='constant' (0)
            # Pad args: (w_ini, w_fin, h_ini, h_fin, ...)
            pad = (0, pad_w, 0, pad_h, 0, pad_d) # Pad at end to match pad_to_multiple_np logic
            pad_transform = tio.Pad(pad, padding_mode=0)
            subject = pad_transform(subject)
            
            # Update vol_shape metadata to match PADDED size
            if 'vol_shape' in subject:
                new_shape = subject['ct'].spatial_shape 
                subject['vol_shape'] = torch.tensor(new_shape).float().view(1, 1, 1, 3)

        # 1. Normalize CT (-450 to 450)
        ct_data = subject['ct'].data
        # subject['ct'].set_data(minmax(ct_data, -450, 450))
        subject['ct'].set_data(minmax(ct_data, -450, 450).to(torch.float32))

        # 2. Normalize MRI
        mr_data = subject['mri'].data
        # subject['mri'].set_data(minmax(mr_data))
        subject['mri'].set_data(minmax(mr_data).to(torch.float32))

        # 3. Probability Map
        # Re-access data in case it was padded
        prob = (subject['ct'].data > 0.01).to(torch.float32)
        # prob = (prob > 0.5).long()  # re-threshold to avoid interpolation artifacts

        subject.add_image(tio.LabelMap(tensor=prob, affine=subject['mri'].affine), 'prob_map')
        # prob_map = tio.LabelMap.from_image(subject['ct']) 
        # prob_map.set_data(prob)
        # subject.add_image(prob_map, 'prob_map')

        # 4. Segmentation One-Hot
        if self.use_seg and 'seg' in subject:
            seg_hot = one_hot_encode(subject['seg'].data, num_classes=self.seg_classes)
            subject['seg'].set_data(seg_hot)
            
        return subject
    
# --- Dataset & Loader ---
def get_augmentations():
    """
    Returns the augmentation pipeline.
    Includes Physics-based (BiasField) and Spatial (Elastic/Affine) transforms.
    """
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomBiasField(p=0.5), # Bias Field: Simulates MRI intensity inhomogeneity
        tio.RandomNoise(std=0.02, p=0.25), # Noise: Simulates sensor noise
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5), # Rigid Transformation
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.25), # Non-rigid Transformation
    ])


class TioAdapter:
    """
    Yields (MRI, CT, Seg) tuples from the queue.
    """
    def __init__(self, loader, has_seg=False):
        self.loader = loader
        self.dataset = self 
        self.has_seg = has_seg
        
    def __len__(self):
        return len(self.loader)
    
    def __iter__(self):
        for batch in self.loader:
            mri = batch['mri'][tio.DATA]
            ct = batch['ct'][tio.DATA]

            # tio.LOCATION: [Batch, 6] -> (i_ini, j_ini, k_ini, i_fin, j_fin, k_fin)
            loc = batch[tio.LOCATION]
            
            # Custom attr vol_shape: [Batch, 1, 3] -> Squeeze to [Batch, 3]
            # vol_shape = batch['vol_shape']
            vol_shape = batch['vol_shape'].view(mri.shape[0], 3)

            
            if self.has_seg:
                seg = batch['seg'][tio.DATA]
                yield mri, ct, seg, loc, vol_shape
            else:
                yield mri, ct, None, loc, vol_shape

def get_dataloader(data_path_list, args):
    """
    Unified Lazy Loader for both MLP and CNN.
    data_path_list: List of dicts {'mri': str_path, 'ct': str_path, 'seg': str_path}
    """
    print(f"⚡ Creating Lazy Loader for {len(data_path_list)} subjects...")
    
    subjects = []
    for item in data_path_list:
        subject_dict = {
            'mri': tio.ScalarImage(item['mri']),
            'ct': tio.ScalarImage(item['ct']),
        }
        if args.use_seg and 'seg' in item:
            subject_dict['seg'] = tio.LabelMap(item['seg'])
        
        subject = tio.Subject(**subject_dict)
        
        subject['vol_shape'] = torch.tensor(subject.spatial_shape).float()
        subjects.append(subject)  
    
    # --- Transforms Pipeline ---
    # 1. Preprocess 
    preprocess = ProjectPreprocessing(
        use_seg=args.use_seg, seg_classes=args.seg_classes, patch_size=args.patch_size
    )    
    # 2. Augment (Optional)
    augment = get_augmentations() if args.augment else None
    
    # 3. Combine
    if augment:
        transforms = tio.Compose([preprocess, augment])
        print("   🎨 Augmentations Enabled (Lazy)")
    else:
        transforms = preprocess
        
    dataset = tio.SubjectsDataset(subjects, transform=transforms)
    
    # --- Queue ---
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    # sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    if args.model_type == "mlp":
        print("   🎲 Sampler: Uniform (Full Grid Coverage)")
        sampler = tio.UniformSampler(patch_size=patch_size)
    else:
        print("   ⚖️ Sampler: Weighted (Body Focus)")
        sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    # sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    
    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=max(args.patches_per_volume, 300), # Large buffer for diversity
        samples_per_volume=args.patches_per_volume,
        sampler=sampler,
        num_workers=2, # Pre-load in background
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    
    # Batch size logic
    # MLP uses batch size 1 here because it subsamples points later
    batch_size = args.cnn_batch_size if args.model_type == "cnn" else 1
    # MLP processes 1 patch per step (subsamples mlp_batch_size voxels from that 1 patch)
    # CNN can process multiple patches per step (uses all voxels for these patches)
    
    loader = DataLoader(queue, batch_size=batch_size, num_workers=0, pin_memory=False)
    return TioAdapter(loader, has_seg=args.use_seg)
    
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
    # pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    # targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    pred_2d = pred.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    targ_2d = target.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    
    # SSIM
    ssim_val = fused_ssim(pred_2d, targ_2d, train=False).item()

    # PSNR
    # mse = torch.mean((pred_2d - targ_2d) ** 2, dim=[1, 2, 3])
    # mse = torch.clamp(mse, min=1e-10) 
    # psnr_2d = 10.0 * torch.log10((data_range ** 2) / mse)
    # psnr_val = torch.mean(psnr_2d).item()
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3, 4])
    mse = torch.clamp(mse, min=1e-10) 
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    psnr_val = torch.mean(psnr).item()

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