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

def load_segmentation(root, subj_id, seg_filename="labels_moved_*.nii.gz", pad_vals=None):
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
class RandomPatchDataset(Dataset):
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

def get_dataloader(feats_mri, ct_target, args):
    if args.model_type == "mlp":
        print("Creating MLP Dataloader (Flattened voxels + Coords)...")
        total_channels = feats_mri.shape[0]
        feats_t = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, total_channels).float()
        
        D, H, W = ct_target.shape
        z = torch.linspace(0, 1, D)
        y = torch.linspace(0, 1, H)
        x = torch.linspace(0, 1, W)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords_t = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3).float()
        
        target_t = torch.from_numpy(ct_target).reshape(-1, 1).float()
        
        ds = TensorDataset(feats_t, coords_t, target_t)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    elif args.model_type == "cnn":
        print(f"Creating CNN Dataloader (Patch-based {args.patch_size}^3)...")
        ds = RandomPatchDataset(
            feats_mri, 
            ct_target, 
            patch_size=(args.patch_size, args.patch_size, args.patch_size), 
            samples_per_epoch=args.patches_per_epoch
        )
        return DataLoader(ds, batch_size=1, shuffle=True, num_workers=2, pin_memory=True)
    
    else:
        raise ValueError(f"Unknown model type: {args.model_type}")

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