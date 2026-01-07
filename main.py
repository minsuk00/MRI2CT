#!/usr/bin/env python3

import os
import gc
import sys
import time
import random
import warnings
import datetime
from types import SimpleNamespace
from glob import glob

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio
import copy

from fused_ssim import fused_ssim
from anatomix.model.network import Unet
from models import MLPTranslator, CNNTranslator

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")

# ==========================================
# 1. HELPER UTILITIES (SSIM, MATH, IO)
# ==========================================
class Config(SimpleNamespace):
    """Helper to allow dot notation access (args.lr)"""
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)
            
def set_seed(seed=42):
    print(f"[DEBUG] 🌱 Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def minmax(arr, minclip=None, maxclip=None):
    if torch.is_tensor(arr):
        if minclip is not None and maxclip is not None:
            arr = torch.clamp(arr, minclip, maxclip)
        denom = arr.max() - arr.min()
        if denom == 0: return torch.zeros_like(arr)
        return (arr - arr.min()) / denom
    else:
        if not (minclip is None) and not (maxclip is None):
            arr = np.clip(arr, minclip, maxclip)
        denom = arr.max() - arr.min()
        if denom == 0: return np.zeros_like(arr)
        return (arr - arr.min()) / denom

def pad_to_multiple_np(arr, multiple=16):
    D, H, W = arr.shape
    pad_D = (multiple - D % multiple) % multiple
    pad_H = (multiple - H % multiple) % multiple
    pad_W = (multiple - W % multiple) % multiple
    return np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant'), (pad_D, pad_H, pad_W)

def unpad_torch(data, pad_vals):
    if pad_vals is None: return data
    d_pad, h_pad, w_pad = pad_vals
    b, c, d, h, w = data.shape
    d_end = d - d_pad if d_pad > 0 else d
    h_end = h - h_pad if h_pad > 0 else h
    w_end = w - w_pad if w_pad > 0 else w
    return data[..., :d_end, :h_end, :w_end]

def compute_metrics(pred, target, data_range=1.0):
    b, c, d, h, w = pred.shape
    pred_2d = pred.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    targ_2d = target.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    
    # SSIM
    ssim_val = fused_ssim(pred_2d, targ_2d, train=False).item()

    # PSNR
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3, 4])
    mse = torch.clamp(mse, min=1e-10) 
    psnr = 10 * torch.log10((data_range ** 2) / mse)
    psnr_val = torch.mean(psnr).item()

    # MAE
    mae_val = torch.mean(torch.abs(pred - target)).item()
    
    return mae_val, psnr_val, ssim_val

def get_subject_paths(root, subj_id):
    paths = {}
    data_path = os.path.join(root, "data")
    
    # 1. CT
    ct = glob(os.path.join(data_path, subj_id, "ct_resampled.nii*"))
    if not ct: raise FileNotFoundError(f"CT missing for {subj_id}")
    paths['ct'] = ct[0]
    
    # 2. MRI
    mr = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))
    if not mr: raise FileNotFoundError(f"MRI missing for {subj_id}")
    paths['mri'] = mr[0]
            
    return paths

def load_image_pair(root, subj_id):
    """Used for Validation Loading"""
    data_path = os.path.join(root, "data")
    ct_path = glob(os.path.join(data_path, subj_id, "ct_resampled.nii*"))[0]
    mr_path = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))[0]
    
    mr_img = tio.ScalarImage(mr_path)
    ct_img = tio.ScalarImage(ct_path)
    
    mri = mr_img.data[0].numpy()
    ct = ct_img.data[0].numpy()
    
    mri = minmax(mri)
    ct = minmax(ct, minclip=-450, maxclip=450)
    
    mri, pad_vals = pad_to_multiple_np(mri, multiple=16)
    ct, _ = pad_to_multiple_np(ct, multiple=16)
    
    print(f"[DEBUG] Valid Data Loaded {subj_id} | MRI: {mri.shape}, CT: {ct.shape}")
    return mri, ct, pad_vals

# ==========================================
# 2. DATA PIPELINE (TORCHIO)
# ==========================================
class ProjectPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, **kwargs):
        super().__init__(**kwargs)
        self.patch_size = patch_size

    def apply_transform(self, subject):
        # 0. Safety Padding
        shape = subject['ct'].spatial_shape
        mult = 16
        pad_w = (mult - shape[0] % mult) % mult
        pad_h = (mult - shape[1] % mult) % mult
        pad_d = (mult - shape[2] % mult) % mult
        
        pad_w = max(pad_w, max(0, self.patch_size - shape[0]))
        pad_h = max(pad_h, max(0, self.patch_size - shape[1]))
        pad_d = max(pad_d, max(0, self.patch_size - shape[2]))

        if pad_w > 0 or pad_h > 0 or pad_d > 0:
            pad = (0, pad_w, 0, pad_h, 0, pad_d)
            pad_transform = tio.Pad(pad, padding_mode=0)
            subject = pad_transform(subject)
            if 'vol_shape' in subject:
                new_shape = subject['ct'].spatial_shape 
                subject['vol_shape'] = torch.tensor(new_shape).float().view(1, 1, 1, 3)

        # 1. Normalize CT
        ct_data = subject['ct'].data
        subject['ct'].set_data(minmax(ct_data, -450, 450).to(torch.float32))

        # 2. Normalize MRI
        mr_data = subject['mri'].data
        subject['mri'].set_data(minmax(mr_data).to(torch.float32))

        # 3. Probability Map
        prob = (subject['ct'].data > 0.01).to(torch.float32)
        subject.add_image(tio.LabelMap(tensor=prob, affine=subject['mri'].affine), 'prob_map')

        return subject

def get_augmentations():
    return tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomBiasField(p=0.5),
        tio.RandomNoise(std=0.02, p=0.25),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.25),
    ])

class TioAdapter:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = self 
        
    def __len__(self): 
        return len(self.loader)
        
    def __iter__(self):
        for batch in self.loader:
            mri = batch['mri'][tio.DATA]
            ct = batch['ct'][tio.DATA]
            loc = batch[tio.LOCATION]
            vol_shape = batch['vol_shape'].view(mri.shape[0], 3)
            
            yield mri, ct, loc, vol_shape

def get_dataloader(data_path_list, args):
    print(f"[DEBUG] ⚡ Creating Lazy Loader for {len(data_path_list)} subjects...")
    subjects = []
    for item in data_path_list:
        subject_dict = {
            'mri': tio.ScalarImage(item['mri']),
            'ct': tio.ScalarImage(item['ct']),
        }
        
        subject = tio.Subject(**subject_dict)
        subject['vol_shape'] = torch.tensor(subject.spatial_shape).float()
        subjects.append(subject)  
    
    preprocess = ProjectPreprocessing(patch_size=args.patch_size)    
    augment = get_augmentations() if args.augment else None
    
    if augment:
        transforms = tio.Compose([preprocess, augment])
    else:
        transforms = preprocess
        
    dataset = tio.SubjectsDataset(subjects, transform=transforms)
    
    # Queue
    patch_size = (args.patch_size, args.patch_size, args.patch_size)
    if args.model_type.lower() == "mlp":
        sampler = tio.UniformSampler(patch_size=patch_size)
    else:
        sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    
    queue = tio.Queue(
        subjects_dataset=dataset,
        max_length=max(args.patches_per_volume, 300), 
        samples_per_volume=args.patches_per_volume,
        sampler=sampler,
        num_workers=2, 
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    
    batch_size = args.cnn_batch_size if args.model_type.lower() == "cnn" else 1
    # loader = DataLoader(queue, batch_size=batch_size, num_workers=0, pin_memory=False)
    loader = tio.SubjectsLoader(queue, batch_size=batch_size, num_workers=0, pin_memory=False)
    
    return TioAdapter(loader)

# ==========================================
# 3. LOSS & SAMPLING
# ==========================================
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

def get_mlp_samples(features, target, locations, vol_shapes, num_samples=16384):
    """
    features: [B, C, W, H, D] # [1, 16, 96, 96, 96]
    target: [B, 1, D, H, W] # [1, 1, 96, 96, 96]
    locations: [B, 6] -> (i_ini, j_ini, k_ini, ...) # [ 34,  25,   0, 130, 121,  96]
    vol_shapes: [B, 3] -> (W, H, D) # [155., 122.,  91. -> 96. (updated)]
    """
    B, C, dim_w, dim_h, dim_d = features.shape

    # 1. Permute to [B, W, H, D, C] for easier flattening
    # Moves Channel to last dim. Spatial dims (W,H,D) stay in 1,2,3
    feats_flat = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
    targ_flat = target.permute(0, 2, 3, 4, 1).reshape(-1, 1)
    
    # 2. Generate GLOBAL Coordinates
    # Local grid matches feature dims: (W, H, D)
    w_local = torch.arange(dim_w, device=features.device).float()
    h_local = torch.arange(dim_h, device=features.device).float()
    d_local = torch.arange(dim_d, device=features.device).float()
    
    # 'ij' indexing preserves order: Dim0=W, Dim1=H, Dim2=D
    grid_w, grid_h, grid_d = torch.meshgrid(w_local, h_local, d_local, indexing='ij')
    
    # Stack in (x, y, z) order which corresponds to (W, H, D)
    # Shape: [1, W, H, D, 3]
    coords_local = torch.stack([grid_w, grid_h, grid_d], dim=-1).unsqueeze(0).repeat(B, 1, 1, 1, 1)
    
    # --- 3. Global Normalization ---
    # offsets and shapes are already in (x, y, z) / (W, H, D) order from utils.py
    
    off_w = locations[:, 0].view(B, 1, 1, 1).float().to(features.device)
    off_h = locations[:, 1].view(B, 1, 1, 1).float().to(features.device)
    off_d = locations[:, 2].view(B, 1, 1, 1).float().to(features.device)
    
    max_w = vol_shapes[:, 0].view(B, 1, 1, 1).float().to(features.device)
    max_h = vol_shapes[:, 1].view(B, 1, 1, 1).float().to(features.device)
    max_d = vol_shapes[:, 2].view(B, 1, 1, 1).float().to(features.device)
    
    # Normalize: (Local + Offset) / Max
    # Index 0 is W(x), 1 is H(y), 2 is D(z). Everything aligns now.
    global_x = (coords_local[..., 0] + off_w) / (max_w - 1)
    global_y = (coords_local[..., 1] + off_h) / (max_h - 1)
    global_z = (coords_local[..., 2] + off_d) / (max_d - 1)
    
    # Stack [B, W, H, D, 3]
    coords_global = torch.stack([global_x, global_y, global_z], dim=-1)
    coords_flat = coords_global.reshape(-1, 3)
    
    # 4. Random Subsampling
    # Optional: You can add the "Valid Mask" logic here if you want to skip air
    total_voxels = feats_flat.shape[0]
    actual_samples = min(num_samples, total_voxels)
    indices = torch.randperm(total_voxels)[:actual_samples]

    return feats_flat[indices], coords_flat[indices], targ_flat[indices]

# ==========================================
# 4. VISUALIZATION
# ==========================================
@torch.no_grad()
def visualize_ct_feature_comparison(pred_ct, gt_ct, gt_mri, model, subj_id, root_dir, epoch=None, use_wandb=False, idx=1):
    device = next(model.parameters()).device
    
    def extract_feats_np(volume_np):
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        feats = model(inp)
        return feats.squeeze(0).cpu().numpy()

    feats_gt = extract_feats_np(gt_ct)
    feats_pred = extract_feats_np(pred_ct)
    feats_mri = extract_feats_np(gt_mri)
    
    C, H, W, D = feats_gt.shape

    # PCA
    def sample_vox(feats, max_vox=200_000):
        X = feats.reshape(C, -1).T
        if X.shape[0] > max_vox:
            X = X[np.random.choice(X.shape[0], max_vox, replace=False)]
        return X

    X_both = np.concatenate([sample_vox(feats_mri), sample_vox(feats_gt), sample_vox(feats_pred)], axis=0)
    pca = PCA(n_components=3, svd_solver="randomized").fit(X_both)

    def project_pca(feats):
        X = feats.reshape(C, -1).T
        Y = pca.transform(X)
        Y = (Y - Y.min(0, keepdims=True)) / (Y.max(0, keepdims=True) - Y.min(0, keepdims=True) + 1e-8)
        return Y.reshape(H, W, D, 3)

    pca_mri  = project_pca(feats_mri)
    pca_gt   = project_pca(feats_gt)
    pca_pred = project_pca(feats_pred)

    gt_t = torch.from_numpy(feats_gt).unsqueeze(0)
    pred_t = torch.from_numpy(feats_pred).unsqueeze(0)
    
    cos_sim = F.cosine_similarity(gt_t, pred_t, dim=1).squeeze(0).numpy()
    cos_sim_n = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)

    residual = pred_ct - gt_ct
    slice_indices = np.linspace(0.1 * D, 0.9 * D, 5, dtype=int)
    fig, axes = plt.subplots(len(slice_indices), 8, figsize=(30, 3.5 * len(slice_indices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    
    for i, z in enumerate(slice_indices):
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (residual, "Residual", "seismic", (-0.5, 0.5)),
            (pca_mri, "PCA (MRI)", None, None),
            (pca_gt, "PCA (GT CT)", None, None),
            (pca_pred, "PCA (Pred)", None, None),
            (cos_sim_n, "Cosine Sim", "plasma", (0,1))
        ]
        for j, (data, title, cmap, clim) in enumerate(items):
            ax = axes[i, j]
            if cmap:
                im = ax.imshow(data[:, :, z], cmap=cmap, vmin=clim[0], vmax=clim[1])
                if j == 3: res_im = im 
                if j == 7: cos_im = im
            else:
                ax.imshow(data[:, :, z, :])
                
            if i == 0: ax.set_title(title)
            ax.axis("off")

    cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
    cbar.set_label("Residual Error")
    
    cbar2 = fig.colorbar(cos_im, ax=axes[:, 7], fraction=0.04, pad=0.01)
    cbar2.set_label("Cosine Similarity")
    
    fig.suptitle(f"Translation Analysis — {subj_id} (epoch {epoch})", fontsize=16, y=0.99)

    save_dir = os.path.join(root_dir, "results", "vis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subj_id}_epoch{epoch}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    
    if use_wandb: 
        wandb.log({f"val/viz_{idx}": wandb.Image(save_path)}, step=epoch)

def log_aug_viz(val_data, args, epoch, root_dir):
    """
    Simulates the training augmentation pipeline on a validation subject
    and logs the Before/After visualization to WandB.
    """
    try:
        mri_t = torch.from_numpy(val_data['mri']).float()
        if mri_t.ndim == 3: mri_t = mri_t.unsqueeze(0)
        ct_t = torch.from_numpy(val_data['ct']).float()
        if ct_t.ndim == 3: ct_t = ct_t.unsqueeze(0)

        subject = tio.Subject(mri=tio.ScalarImage(tensor=mri_t), ct=tio.ScalarImage(tensor=ct_t))
        preprocess = ProjectPreprocessing(patch_size=args.patch_size)
        clean_subj = preprocess(subject)
        augment_transform = get_augmentations()
        aug_subj = augment_transform(clean_subj)
        hist_str = " | ".join([t.name for t in aug_subj.history])

        clean_vol = clean_subj['mri'].data[0].numpy()
        aug_vol = aug_subj['mri'].data[0].numpy()
        z = clean_vol.shape[2] // 2
        sl_clean = np.rot90(clean_vol[:, :, z])
        sl_aug = np.rot90(aug_vol[:, :, z])
        sl_diff = sl_aug - sl_clean

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        axes[0].imshow(sl_clean, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Input (Normalized)")
        axes[1].imshow(sl_aug, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Augmented\n{hist_str}")
        im2 = axes[2].imshow(sl_diff, cmap='seismic', vmin=-0.5, vmax=0.5)
        axes[2].set_title("Difference (Aug - Input)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        for ax in axes: ax.axis('off')
        plt.tight_layout()

        save_path = os.path.join(root_dir, "results", "vis_aug.png")
        plt.savefig(save_path)
        plt.close(fig)
        wandb.log({"val/aug_example": wandb.Image(save_path)}, step=epoch)
    except Exception as e:
        print(f"[DEBUG] Aug Viz failed: {e}")

# ==========================================
# 5. CORE ENGINE (TRAIN/EVAL)
# ==========================================
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, model_type, feat_extractor, args):
    feat_extractor.eval() 
    model.train()
    
    total_loss = 0.0
    comp_accumulator = {}
    num_batches = len(loader)
    
    for mri, ct, location, vol_shape in tqdm(loader, leave=False, desc="Train Step"):
        mri, ct = mri.to(device), ct.to(device)
        
        # 1. Feature Extraction
        with torch.no_grad():
            features = feat_extractor(mri) 

        # 2. Forward Pass
        optimizer.zero_grad(set_to_none=True)
        
        with torch.amp.autocast(device_type="cuda" if "cuda" in device.type else "cpu"):
            if model_type.lower() == "mlp":
                f_pts, c_pts, t_pts = get_mlp_samples(features, ct, location, vol_shape, num_samples=args.mlp_batch_size)
                pred = model(f_pts, c_pts)
                loss, components = loss_fn(pred, t_pts)
            
            elif model_type.lower() == "cnn":
                pred = model(features)
                loss, components = loss_fn(pred, ct)
        
        if torch.isnan(loss):
            print("[ERROR] Loss is NaN!")
            exit()

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        for k, v in components.items():
            comp_accumulator[k] = comp_accumulator.get(k, 0.0) + v

    avg_loss = total_loss / num_batches
    avg_components = {k: v / num_batches for k, v in comp_accumulator.items()}
    
    return avg_loss, avg_components

@torch.no_grad()
def evaluate(model, feats_mri, ct, device, model_type, pad_vals):
    model.eval()
    
    if isinstance(ct, np.ndarray):
        ct_tensor = torch.from_numpy(ct).float().to(device)
    else:
        ct_tensor = ct.float().to(device)
    
    if ct_tensor.ndim == 3: ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)
    
    pred_ct_tensor = None

    if model_type.lower() == "mlp":
        C, dim_w, dim_h, dim_d = feats_mri.shape
        feats_flat = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, C).float().to(device)
        
        w_local = torch.arange(dim_w, device=device).float()
        h_local = torch.arange(dim_h, device=device).float()
        d_local = torch.arange(dim_d, device=device).float()
        grid_w, grid_h, grid_d = torch.meshgrid(w_local, h_local, d_local, indexing='ij')

        # Normalize 0..1
        norm_w = grid_w / (dim_w - 1)
        norm_h = grid_h / (dim_h - 1)
        norm_d = grid_d / (dim_d - 1)
        # Stack [W, H, D, 3] -> (x, y, z)
        coords = torch.stack([norm_w, norm_h, norm_d], dim=-1).reshape(-1, 3)
        
        preds = []
        batch_size = 100000
        for i in range(0, feats_flat.size(0), batch_size):
            f_batch = feats_flat[i:i+batch_size]
            c_batch = coords[i:i+batch_size]
            preds.append(model(f_batch, c_batch))
        
        pred_flat = torch.cat(preds, dim=0)
        pred_ct_tensor = pred_flat.reshape(1, 1, dim_w, dim_h, dim_d)

    elif model_type.lower() == "cnn":
        feats_t = torch.from_numpy(feats_mri).unsqueeze(0).float().to(device)
        pred_ct_tensor = model(feats_t)

    pred_ct_tensor_unpad = unpad_torch(pred_ct_tensor, pad_vals)
    ct_tensor_unpad = unpad_torch(ct_tensor, pad_vals)
    metrics = compute_metrics(pred_ct_tensor_unpad, ct_tensor_unpad)
    pred_ct = pred_ct_tensor.squeeze().cpu().numpy()
    return metrics, pred_ct

# ==========================================
# 6. ORCHESTRATION
# ==========================================
def discover_subjects(data_dir, target_list=None):
    if target_list:
        candidates = target_list
    else:
        candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    
    valid = []
    required = ["ct_resampled.nii.gz"]
    for subj in candidates:
        path = os.path.join(data_dir, subj)
        if all(os.path.exists(os.path.join(path, f)) for f in required):
            valid.append(subj)
    print(f"Found {len(valid)} subjects")
    return valid

def run_experiment(exp_config_dict):
    """
    Main entry point for a single experiment.
    """
    for var in ['model', 'optimizer', 'loader', 'features', 'pred']:
        if var in globals():
            print(f"[DEBUG] 🗑️ Explicitly deleting {var} from memory")
            del globals()[var]
    cleanup_gpu()
    
    # Merge configs
    final_conf_dict = copy.deepcopy(DEFAULT_CONFIG)
    final_conf_dict.update(exp_config_dict)
    args = Config(final_conf_dict)
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[DEBUG] 🟢 Starting Experiment. Device: {device}, Model: {args.model_type}")
    
    # 1. Subject Discovery
    subjects = discover_subjects(os.path.join(args.root_dir, "data"), target_list=args.subjects)
    if not subjects:
        print("[ERROR] No subjects found.")
        return

    # 2. Train/Val Split
    num_val = max(1, int(len(subjects) * args.val_split)) # By default, use at least 1 validation subject
    train_subjects = subjects[:-num_val]
    val_subjects = subjects[-num_val:]
    print(f"[DEBUG] Train: {len(train_subjects)} | Val: {len(val_subjects)}")

    # 3. W&B
    if args.wandb:
        run_name = f"{args.model_type}_Train{len(train_subjects)}"
        wandb.init(project=args.project_name, name=run_name, config=vars(args), reinit=True)

    # 4. Feature Extractor (Anatomix)
    feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
    ckpt_path = os.path.join(args.root_dir, "anatomix", "model-weights", "anatomix.pth")
    if os.path.exists(ckpt_path):
        feat_extractor.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    else:
        print(f"[WARNING] Anatomix weights not found at {ckpt_path}. Using random init.")
    feat_extractor.eval()

    # 5. Load Train Data
    train_paths = []
    shapes = []

    for subj_id in tqdm(train_subjects, desc="Loading Paths"):
        try:
            paths = get_subject_paths(args.root_dir, subj_id)
            train_paths.append(paths)
            
            # Header Scan for Statistics
            img = nib.load(paths['mri'])
            shapes.append(img.header.get_data_shape())
        except Exception as e:
            print(f"Skipping {subj_id}: {e}")
    
    if shapes:
        avg_shape = np.mean(np.array(shapes), axis=0).astype(int)
        print(f"📊 Mean Volume Shape: {tuple(int(x) for x in avg_shape)}")
        if np.any(avg_shape < args.patch_size):
            print(f"⚠️ Warning: Mean shape {tuple(avg_shape)} is smaller than patch size {args.patch_size} in some dims.")
            print(f"   Auto-padding is active to prevent crashes.")
    if not train_paths:
        print("❌ No valid training data.")
        return   
        
    # 6. Pre-load Val Data (Features)
    val_meta_list = []
    print("[DEBUG] Pre-loading Validation Data...")
    for subj_id in tqdm(val_subjects):
        try:
            mri, ct, pad_vals = load_image_pair(args.root_dir, subj_id)
            with torch.no_grad():
                inp = torch.from_numpy(mri[None, None]).float().to(device)
                feats = feat_extractor(inp).squeeze(0).cpu().numpy()
            
            val_meta_list.append({'id': subj_id, 'ct': ct, 'mri': mri, 'feats': feats, 'pad_vals': pad_vals})
        except Exception as e:
            print(f"Error loading val {subj_id}: {e}")

    cleanup_gpu()

    # 7. Create Loader & Model
    loader = get_dataloader(train_paths, args)

    total_channels = 16
    if args.model_type.lower() == "mlp":
        print(f"Building MLP: Depth={args.mlp_depth}, Hidden={args.mlp_hidden}, Fourier={'Yes' if args.fourier else 'No'}, Scale={args.sigma}")
        model = MLPTranslator(
            in_feat_dim=total_channels, 
            use_fourier=args.fourier, 
            fourier_scale=args.sigma,
            hidden_channels=args.mlp_hidden, 
            depth=args.mlp_depth,    
            dropout=args.dropout
        ).to(device)
        epochs = args.epochs_mlp
    else:
        print(f"Building CNN: Depth={args.cnn_depth}, Hidden={args.cnn_hidden}, Act={args.final_activation}")
        model = CNNTranslator(
            in_channels=total_channels, 
            hidden_channels=args.cnn_hidden, 
            depth=args.cnn_depth, 
            final_activation=args.final_activation,
            dropout=args.dropout
        ).to(device)
        epochs = args.epochs_cnn

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = CompositeLoss(weights={"l1": args.l1_w, "l2": args.l2_w, "ssim": args.ssim_w}).to(device)
    scaler = torch.amp.GradScaler()

    # 8. Loop
    print(f"[DEBUG] 🚀 Training for {epochs} epochs...")
    
    # Validation Helper
    def run_validation(epoch_idx, loss_val, viz_limit = 3):
        model.eval()
        val_metrics = {'mae': [], 'psnr': [], 'ssim': []}
        viz_count = 0 

        # NOTE: Augmentation is not actually applied during validation. This is just for visualization.
        if args.augment and args.wandb and val_meta_list:
            log_aug_viz(val_meta_list[0], args, epoch_idx, args.root_dir)
            
        for v_data in val_meta_list:
            (mae, psnr, ssim), pred_ct = evaluate(model, v_data['feats'], v_data['ct'], device, args.model_type, v_data['pad_vals'])
            val_metrics['mae'].append(mae)
            val_metrics['psnr'].append(psnr)
            val_metrics['ssim'].append(ssim)
            
            if args.wandb and viz_count < viz_limit:
                print(f"   🔎 Val Viz [{viz_count+1}/{viz_limit}] ({v_data['id']}): MAE={mae:.4f}, SSIM={ssim:.3f}, PSNR={psnr:.2f}")
                visualize_ct_feature_comparison(pred_ct, v_data['ct'], v_data['mri'], feat_extractor, v_data['id'], args.root_dir, epoch=epoch_idx, use_wandb=True)
                viz_count += 1

        avg_mae = np.mean(val_metrics['mae']) if val_metrics['mae'] else 0.0
        avg_psnr = np.mean(val_metrics['psnr']) if val_metrics['psnr'] else 0.0
        avg_ssim = np.mean(val_metrics['ssim']) if val_metrics['ssim'] else 0.0

        print(f"Ep {epoch_idx} | Loss: {loss_val:.5f} | Val MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f}")
        if args.wandb:
            wandb.log({"val/mae": avg_mae, "val/psnr": avg_psnr, "val/ssim": avg_ssim}, step=epoch_idx)

    # Pre-Training Sanity Check
    if args.sanity_check:
        run_validation(0, 0.0)

    start_time = time.time()
    epoch_iter = tqdm(range(1, epochs + 1), desc="Epochs", leave=True, dynamic_ncols=True)
    for epoch in epoch_iter:
        loss, loss_comps = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, args.model_type, feat_extractor, args)
        epoch_iter.set_postfix({"train_loss": f"{loss:.5f}"})
        
        if args.wandb:
            log = {"loss/total": loss}
            for k, v in loss_comps.items(): log[k.replace("loss_", "loss/")] = v
            wandb.log(log, step=epoch)

        if (epoch % args.val_interval == 0) or (epoch == epochs):
            run_validation(epoch, loss)

    # Save
    save_path = os.path.join(args.root_dir, "results", "models", f"{args.model_type}_{datetime.datetime.now():%Y%m%d_%H%M}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved model to {save_path}")
    
    if args.wandb: wandb.finish()
    print(f"⏱️ Total: {time.time()-start_time:.2f}s")


# ==========================================
# 7. EXPERIMENT CONFIGURATIONS
# ==========================================
DEFAULT_CONFIG = {
    # System
    "root_dir": "/home/minsukc/MRI2CT",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    # "wandb": False,
    "project_name": "mri2ct_single_file",
    
    # Data
    "subjects": ["1ABA005_3.0x3.0x3.0_resampled", "1HNA001_3.0x3.0x3.0_resampled"], # None for all
    # "subjects": None
    "val_split": 0.1,
    "augment": True,
    "patch_size": 96,
    "patches_per_volume": 100,

    # Training
    "lr": 1e-3,
    "val_interval": 1,
    "sanity_check": True,
    
    # Model Choice
    "model_type": "mlp", # "mlp" or "cnn"
    "epochs_mlp": 500,
    "epochs_cnn": 200,
    
    # MLP Specifics
    "mlp_batch_size": 131072, # Points to sample per patch (feature vector batch)
    "fourier": True,
    "sigma": 5.0,
    "mlp_depth": 4,
    "mlp_hidden": 256,
    "dropout": 0.0,
    
    # CNN Specifics
    "cnn_batch_size": 1,
    "cnn_depth": 9,
    "cnn_hidden": 128,
    "final_activation": "sigmoid",
    
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 1.0
}


if __name__ == "__main__":
    # Define experiments here
    experiments = [
        # Experiment 1: MLP Baseline
        {
            "model_type": "mlp",
            "lr": 1e-3,
            "mlp_hidden": 256,
            "epochs_mlp": 100,
        },
        # Experiment 2: CNN Baseline
        {
            "model_type": "cnn",
            "lr": 5e-4,
            "cnn_depth": 5,
            "epochs_cnn": 50,
        }
    ]

    print(f"📚 Found {len(experiments)} experiments to run.")
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*40}")
        print(f"STARTING EXPERIMENT {i+1}/{len(experiments)}")
        print(f"Config: {exp}")
        print(f"{'='*40}\n")
        
        try:
            run_experiment(exp)
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
            break
        except Exception as e:
            print(f"❌ Experiment {i+1} Failed: {e}")
            import traceback
            traceback.print_exc()