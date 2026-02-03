import os
import gc
import random
import numpy as np
import torch
from fused_ssim import fused_ssim3d

def set_seed(seed=42):
    print(f"[DEBUG] Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # FOR TESTING & PRODUCTION TRAINING (SPEED):
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def anatomix_normalize(tensor, percentile_range = None, clip_range=None):
    if not torch.is_tensor(tensor):
        tensor = torch.as_tensor(tensor, dtype=torch.float32)
    
    # 1. CT path: Explicit Clipping (Windowing)
    if clip_range is not None:
        min_c, max_c = clip_range
        tensor = torch.clamp(tensor, min_c, max_c)
        denom = max_c - min_c
        if denom == 0:
            print(f"[WARNING] CT Window has 0 width: {clip_range}")
            return torch.zeros_like(tensor)
        return (tensor - min_c) / denom
    
    # 2. MRI path: Percentile Normalization (Instance-level)
    if percentile_range is not None:
        min_percentile, max_percentile = percentile_range
        v_min = torch.quantile(tensor.float(), min_percentile / 100.0)
        v_max = torch.quantile(tensor.float(), max_percentile / 100.0)
        tensor = torch.clamp(tensor, v_min, v_max)
    
        denom = v_max - v_min
        if denom == 0:
            print(f"[WARNING] MRI Volume is constant (Val: {v_min:.4f}). Returning zeros.")
            return torch.zeros_like(tensor)
            
        return (tensor - v_min) / denom

    # 3. just minmax normalization
    v_min = tensor.min()
    v_max = tensor.max()
    # tensor = torch.clamp(tensor, v_min, v_max)
    denom = v_max - v_min
    if denom == 0:
        print(f"[WARNING] MRI Volume is constant (Val: {v_min:.4f}). Returning zeros.")
        return torch.zeros_like(tensor)
        
    return (tensor - v_min) / denom

def unpad(data, original_shape, offset = 0):
    if original_shape is None: return data
    w_orig, h_orig, d_orig = original_shape
    return data[..., offset:offset+w_orig, offset:offset+h_orig, offset:offset+d_orig]

def compute_metrics(pred, target, data_range=1.0):
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(f"Expected (B, C, D, H, W), got {pred.shape}")
        
    b, c, d, h, w = pred.shape
    
    ssim_val = fused_ssim3d(pred.float(), target.float(), train=False).item()
    
    # PSNR
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3, 4])
    mse = torch.clamp(mse, min=1e-10) 
    psnr = 10 * torch.log10((data_range ** 2) / mse)

    # 2. Gradient Difference (Sharpness Metric)
    def get_gradients(img):
        dz = torch.abs(img[:, :, 1:, :, :] - img[:, :, :-1, :, :])
        dy = torch.abs(img[:, :, :, 1:, :] - img[:, :, :, :-1, :])
        dx = torch.abs(img[:, :, :, :, 1:] - img[:, :, :, :, :-1])
        return dz, dy, dx
    pred_dz, pred_dy, pred_dx = get_gradients(pred)
    targ_dz, targ_dy, targ_dx = get_gradients(target)
    grad_diff = (
        torch.mean(torch.abs(pred_dz - targ_dz)) +
        torch.mean(torch.abs(pred_dy - targ_dy)) +
        torch.mean(torch.abs(pred_dx - targ_dx))
    ).item()

    # 3. Bone Dice Coefficient (Structure Metric)
    bone_thresh = 0.8
    pred_bone = (pred > bone_thresh).float()
    targ_bone = (target > bone_thresh).float()
    intersection = (pred_bone * targ_bone).sum()
    union = pred_bone.sum() + targ_bone.sum()
    # Smooth Dice (Add epsilon to avoid division by zero)
    dice_score = (2.0 * intersection + 1e-5) / (union + 1e-5)
    dice_val = dice_score.item()
    
    return {
        "mae": torch.mean(torch.abs(pred - target)).item(),
        "psnr": torch.mean(psnr).item(),
        "ssim": ssim_val,
        "grad_diff": grad_diff, # 낮을수록 좋음 (Lower is better)
        "bone_dice": dice_val,  # 높을수록 좋음 (Higher is better)
    }
