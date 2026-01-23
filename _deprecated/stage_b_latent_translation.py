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
import json
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable 
from sklearn.decomposition import PCA 
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio

from fused_ssim import fused_ssim
from anatomix.model.network import Unet
from stage_a_ct_decoding import CNNTranslator

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")

# ==========================================
# 1. HELPER UTILITIES
# ==========================================
class Config(SimpleNamespace):
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
    if not torch.is_tensor(arr):
        arr = torch.as_tensor(arr)
    if minclip is not None and maxclip is not None:
        arr = torch.clamp(arr, minclip, maxclip)
    denom = arr.max() - arr.min()
    if denom == 0: 
        return torch.zeros_like(arr)
    return (arr - arr.min()) / denom

# Robust Scaling for MRI
def robust_scale(arr, p_min=1, p_max=99):
    """Scales array based on percentiles to handle MRI outliers."""
    if not torch.is_tensor(arr):
        arr = torch.as_tensor(arr)
        
    # Calculate percentiles
    # NOTE: quantile requires float32/64
    v_min = torch.quantile(arr.float(), p_min / 100.0)
    v_max = torch.quantile(arr.float(), p_max / 100.0)
    
    # Clip to percentiles
    arr = torch.clamp(arr, v_min, v_max)
    
    # MinMax Scale to [0, 1] based on the clipped range
    denom = v_max - v_min
    if denom == 0:
        return torch.zeros_like(arr)
        
    return (arr - v_min) / denom

def unpad(data, original_shape):
    if original_shape is None: return data
    w, h, d = original_shape
    return data[..., :w, :h, :d]

def compute_metrics(pred, target, data_range=1.0):
    b, c, d, h, w = pred.shape
    pred_2d = pred.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    targ_2d = target.permute(0, 4, 1, 2, 3).reshape(-1, c, h, w)
    
    ssim_val = fused_ssim(pred_2d, targ_2d, train=False).item()
    mse = torch.mean((pred - target) ** 2)
    mse = torch.clamp(mse, min=1e-10) 
    psnr_val = (10 * torch.log10((data_range ** 2) / mse)).item()
    mae_val = torch.mean(torch.abs(pred - target)).item()
    
    return mae_val, psnr_val, ssim_val

def get_subject_paths(root, subj_id):
    data_path = os.path.join(root, "data")
    ct = glob(os.path.join(data_path, subj_id, "ct_resampled.nii*"))
    mr = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))
    
    if not ct: raise FileNotFoundError(f"CT missing for {subj_id}")
    if not mr: raise FileNotFoundError(f"MRI missing for {subj_id}")
    
    return {'ct': ct[0], 'mri': mr[0]}

def load_image_pair(root, subj_id, args):
    """Validation Loading"""
    paths = get_subject_paths(root, subj_id)
    ct_img = tio.ScalarImage(paths['ct'])
    mr_img = tio.ScalarImage(paths['mri'])
    
    # Normalize
    ct = minmax(ct_img.data[0], minclip=-450, maxclip=450).numpy()
    
    # Use robust scale for Validation MRI loading too
    mri_tensor = mr_img.data[0]
    mri = robust_scale(mri_tensor).numpy()
    
    orig_shape = ct.shape
    
    target_shape = [max(args.patch_size, (d + 31) // 32 * 32) for d in orig_shape]
    pad_width = [(0, t - o) for t, o in zip(target_shape, orig_shape)]
    
    ct_padded = np.pad(ct, pad_width, mode='constant', constant_values=0)
    mri_padded = np.pad(mri, pad_width, mode='constant', constant_values=0)
    
    return mri_padded, ct_padded, orig_shape

def discover_subjects(data_dir, target_list=None, region=None): 
    if target_list:
        candidates = target_list
        print(f"[DEBUG] 🎯 Using explicit target_list ({len(candidates)} subjects)")
    else:
        candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        if region:
            region = region.upper()
            candidates = [c for c in candidates if region in c]
            print(f"[DEBUG] 📍 No target_list found. Filtering for region: {region} ({len(candidates)} subjects)")
        else:
            print(f"[DEBUG] 📂 No target_list or region specified. Using all {len(candidates)} discovered subjects.")
    
    valid = []
    required = ["ct_resampled.nii.gz"]
    for subj in candidates:
        path = os.path.join(data_dir, subj)
        if all(os.path.exists(os.path.join(path, f)) for f in required):
            valid.append(subj)
    print(f"Found {len(valid)} subjects")
    return valid

# ==========================================
# 2. DATA PIPELINE
# ==========================================
class ProjectPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, enable_safety_padding=True, **kwargs):        
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.enable_safety_padding = enable_safety_padding

    def apply_transform(self, subject):
        # 1. Normalize CT (-450 to 450 HU -> 0 to 1)
        ct_data = subject['ct'].data
        subject['ct'].set_data(minmax(ct_data, -450, 450).float())
        
        # Normalize MRI using Robust Scaling (1-99 percentile)
        # This handles artifacts better than minmax
        mr_data = subject['mri'].data
        subject['mri'].set_data(robust_scale(mr_data).float())

        if self.enable_safety_padding:
            pad_amount = self.patch_size // 2
            safety_padder = tio.Pad(pad_amount, padding_mode=0)
            subject = safety_padder(subject)
        
        # 2. Smart Padding
        current_shape = subject['ct'].spatial_shape
        subject['original_shape'] = torch.tensor(current_shape)
        
        target_shape = []
        for dim in current_shape:
            mult_32 = (int(dim) + 31) // 32 * 32
            target_shape.append(max(self.patch_size, mult_32))
            
        padding_params = []
        for curr, targ in zip(current_shape, target_shape):
            padding_params.extend([0, int(targ - curr)])
            
        if any(p > 0 for p in padding_params):
            padder = tio.Pad(padding_params, padding_mode=0)
            subject = padder(subject)

        subject['vol_shape'] = torch.tensor(subject['ct'].spatial_shape).float()
        
        # 3. Probability Map (Based on CT body)
        prob = (subject['ct'].data > 0.01).to(torch.float32)
        subject.add_image(tio.LabelMap(tensor=prob, affine=subject['ct'].affine), 'prob_map')
            
        return subject

def get_augmentations_stage_b():
    spatial_transforms = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.25),
    ])

    intensity_transforms = tio.Compose([
        tio.RandomBiasField(p=0.5, include=['mri']), 
        tio.RandomNoise(std=0.02, p=0.25, include=['mri']),
        tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5, include=['mri'])
    ])

    return tio.Compose([spatial_transforms, intensity_transforms])

def get_dataloader(data_path_list, args):
    subjects = []
    for item in data_path_list:
        subjects.append(tio.Subject(
            mri=tio.ScalarImage(item['mri']),
            ct=tio.ScalarImage(item['ct'])
        ))

    preprocess = ProjectPreprocessing(patch_size=args.patch_size, enable_safety_padding=True)    
    augment = get_augmentations_stage_b() if args.augment else None
    
    transforms = tio.Compose([preprocess, augment]) if augment else preprocess
    dataset = tio.SubjectsDataset(subjects, transform=transforms)
    
    sampler = tio.WeightedSampler(patch_size=args.patch_size, probability_map='prob_map')
    
    queue = tio.Queue(
        subjects_dataset=dataset,
        samples_per_volume=args.patches_per_volume,
        max_length=max(args.patches_per_volume, args.data_queue_max_length), 
        sampler=sampler,
        num_workers=args.num_workers, 
        shuffle_subjects=True,
        shuffle_patches=True,
    )
    
    loader = tio.SubjectsLoader(queue, batch_size=args.batch_size, num_workers=0, pin_memory=False)
    return loader

# ==========================================
# 3. MODELS
# ==========================================
# class LatentResidualMapper(nn.Module):
#     def __init__(self, channels=16, hidden=64, layers=5, use_residual=True):
#         super().__init__()
#         self.use_residual = use_residual 

#         self.input_conv = nn.Sequential(
#             nn.Conv3d(channels, hidden, 3, padding=1),
#             nn.InstanceNorm3d(hidden),
#             nn.ReLU(inplace=True)
#         )
#         res_layers = []
#         for _ in range(layers):
#             res_layers.append(nn.Sequential(
#                 nn.Conv3d(hidden, hidden, 3, padding=1),
#                 nn.InstanceNorm3d(hidden),
#                 nn.ReLU(inplace=True),
#                 nn.Conv3d(hidden, hidden, 3, padding=1),
#                 nn.InstanceNorm3d(hidden),
#                 nn.ReLU(inplace=True)
#             ))
#         self.res_blocks = nn.ModuleList(res_layers)
#         self.output_conv = nn.Conv3d(hidden, channels, 3, padding=1)
        
#         # nn.init.zeros_(self.output_conv.weight)
#         # nn.init.zeros_(self.output_conv.bias)
#         if self.use_residual:
#             nn.init.zeros_(self.output_conv.weight)
#             nn.init.zeros_(self.output_conv.bias)
        
#         print(f"[DEBUG] 🧠 Latent Residual Mapper: {channels}ch -> {hidden}ch -> {channels}ch ({layers} blocks)")

#     def forward(self, z_mri):
#         x = self.input_conv(z_mri)
#         for block in self.res_blocks:
#             x = x + block(x) 
#         delta = self.output_conv(x)

#         if self.use_residual:
#             return z_mri + delta 
#         else:
#             return delta # NOTE: Here 'delta' is actually the full predicted z_ct

# NOTE: NEW - Replaced InstanceNorm with GroupNorm to preserve intensity shifts.
# NOTE: NEW - Added Zero Initialization to the tail.

class StageBMapper(nn.Module):
    def __init__(self, channels=16, hidden=64):
        super().__init__()
        
        # 1. Remove InstanceNorm. Use GroupNorm (safe for Batch=1).
        # GroupNorm(num_groups, num_channels)
        gn_groups = 8 
        
        self.head = nn.Sequential(
            nn.Conv3d(channels, hidden, 3, padding=1),
            nn.GroupNorm(gn_groups, hidden), 
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # 2. Body: Replace IN with GN.
        self.body = nn.Sequential(
            ContextResidualBlock(hidden, dilation=1, groups=gn_groups),
            ContextResidualBlock(hidden, dilation=2, groups=gn_groups),
            ContextResidualBlock(hidden, dilation=4, groups=gn_groups),
            ContextResidualBlock(hidden, dilation=8, groups=gn_groups),
            ContextResidualBlock(hidden, dilation=16, groups=gn_groups),
            ContextResidualBlock(hidden, dilation=1, groups=gn_groups),
        )
        
        # Collapse
        self.tail = nn.Conv3d(hidden, channels, 3, padding=1)
        
        # 3. CRITICAL: Zero Initialization
        # This ensures the model starts as an Identity map (Output = Input)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x):
        feat = self.head(x)
        feat = self.body(feat)
        out = self.tail(feat)
        return out
        # delta = self.tail(feat)
        # return x + delta

class ContextResidualBlock(nn.Module):
    def __init__(self, channels, dilation=1, groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, dilation=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.norm2 = nn.GroupNorm(groups, channels)
        
        self.act = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        residual = x
        
        # Conv 1 -> Norm -> Act
        out = self.act(self.norm1(self.conv1(x)))
        
        # Conv 2 -> Norm (NO ACT)
        out = self.norm2(self.conv2(out))
        
        # Add Residual
        out = out + residual
        return out

# ==========================================
# 4. TRAINING & LOSS (LATENT SPACE)
# ==========================================
# class LatentConsistencyLoss(nn.Module):
#     def __init__(self, weights={'mse': 1.0, 'cosine': 0.1}):
#         super().__init__()
#         self.weights = weights
#         self.mse = nn.MSELoss()
        
#     def forward(self, pred_z, target_z):
#         loss_mse = self.mse(pred_z, target_z)
        
#         b, c, d, h, w = pred_z.shape
#         p_flat = pred_z.view(b, c, -1)
#         t_flat = target_z.view(b, c, -1)
#         cosine_sim = F.cosine_similarity(p_flat, t_flat, dim=1)
#         loss_cos = 1.0 - torch.mean(cosine_sim)
        
#         total = (self.weights['mse'] * loss_mse) + (self.weights['cosine'] * loss_cos)
#         return total, {'mse': loss_mse.item(), 'cosine': loss_cos.item()}

class HybridConsistencyLoss(nn.Module):
    def __init__(self, weights={'z_mse': 10.0, 'img_l1': 1.0}):
        super().__init__()
        self.weights = weights
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss() 

    def forward(self, pred_z, target_z, pred_img, target_img):
        # 1. Latent Space Consistency (Stability)
        loss_z = self.mse(pred_z, target_z)
        
        # 2. Image Space Consistency (Perceptual Quality)
        # Gradient flows from here -> Decoder (Frozen) -> pred_z -> Mapper
        loss_img = self.l1(pred_img, target_img)
        
        total = (self.weights['z_mse'] * loss_z) + (self.weights['img_l1'] * loss_img)
        
        return total, {'z_mse': loss_z.item(), 'img_l1': loss_img.item()}

def train_latent_epoch(mapper, extractor, decoder, loader, opt, loss_fn, device, args):
    mapper.train()
    extractor.eval() 
    decoder.eval() # Strict Eval mode
    for p in decoder.parameters(): p.requires_grad = False
    
    total_loss = 0
    total_grad = 0.0 
    
    metrics_accum = defaultdict(float)
    progress = tqdm(range(args.steps_per_epoch), desc="Hybrid Train", leave=False)
    loader_iter = iter(loader)
    
    for _ in progress:
        opt.zero_grad()
        step_loss = 0.0
        
        for _ in range(args.accum_steps):
            try:
                batch = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                batch = next(loader_iter)

            batch_metrics = defaultdict(float)
            
            mri = batch['mri'][tio.DATA].to(device)
            ct_gt_img = batch['ct'][tio.DATA].to(device)
            
            with torch.no_grad():
                z_mri = extractor(mri)
                z_ct_gt = extractor(ct_gt_img)
                # print(f"Z Stats: Mean={z_mri.mean():.2f} | Std={z_mri.std():.2f}")
            
            z_ct_pred = mapper(z_mri)
            pred_ct_img = decoder(z_ct_pred)
            
            # loss, _ = loss_fn(z_ct_pred, z_ct_gt)
            loss, loss_dict = loss_fn(z_ct_pred, z_ct_gt, pred_ct_img, ct_gt_img)
            
            loss = loss / args.accum_steps
            loss.backward()
            step_loss += loss.item()
            
            for k, v in loss_dict.items():
                batch_metrics[k] += (v / args.accum_steps)

            # 1. Standard Deviation (Contrast/Scale)
            batch_metrics['z_mri_std'] += z_mri.std().item() / args.accum_steps
            batch_metrics['z_gt_std'] += z_ct_gt.std().item() / args.accum_steps
            batch_metrics['z_pred_std'] += z_ct_pred.detach().std().item() / args.accum_steps

            # 2. Mean (Shift/DC Offset) -> ADD THIS
            batch_metrics['z_mri_mean'] += z_mri.mean().item() / args.accum_steps
            batch_metrics['z_gt_mean'] += z_ct_gt.mean().item() / args.accum_steps
            batch_metrics['z_pred_mean'] += z_ct_pred.detach().mean().item() / args.accum_steps
            
        norm_val = torch.nn.utils.clip_grad_norm_(mapper.parameters(), 1.0)
        total_grad += norm_val.item()
        
        opt.step()
        total_loss += step_loss
        # progress.set_postfix({'loss': step_loss})
        # progress.set_postfix(loss_dict)
        for k, v in batch_metrics.items():
            metrics_accum[k] += v
            
        progress.set_postfix(batch_metrics)

    avg_loss = total_loss / args.steps_per_epoch
    avg_grad = total_grad / args.steps_per_epoch
    avg_metrics = {k: v / args.steps_per_epoch for k, v in metrics_accum.items()}
    
    # Return everything
    return avg_loss, avg_grad, avg_metrics

# ==========================================
# 5. VALIDATION & VIZ
# ==========================================
@torch.no_grad()
def visualize_stage_b(mri, gt_ct, pred_ct, z_mri, z_pred, z_gt, subj_id, root, epoch, use_wandb, idx):
    """
    Advanced Visualization:
    - Row 1: Image Space (Input, GT, Decoded)
    - Row 2: Feature Space (PCA projected to RGB)
    - Row 3: Feature Space (Mean Intensity)
    - Row 4: Latent Deltas (Target Shift, Model Shift, Remaining Error)
    """
    
    # 1. PCA Projection for Feature Visualization
    # Stack all latents: (3, C, W, H, D) -> (3, C, N)
    C, W, H, D = z_mri.shape
    
    # Concatenate along the "pixels" dimension for fitting PCA
    flat_mri = z_mri.reshape(C, -1).T.cpu().numpy()
    flat_gt = z_gt.reshape(C, -1).T.cpu().numpy()
    flat_pred = z_pred.reshape(C, -1).T.cpu().numpy()
    
    # Subsample for faster PCA fit
    combined = np.concatenate([flat_mri, flat_gt], axis=0)
    if combined.shape[0] > 50000:
        indices = np.random.choice(combined.shape[0], 50000, replace=False)
        sample = combined[indices]
    else:
        sample = combined
        
    pca = PCA(n_components=3)
    pca.fit(sample)
    
    def project_pca(flat_data):
        # Transform -> MinMax Normalize to [0, 1] for RGB
        proj = pca.transform(flat_data)
        p_min, p_max = proj.min(), proj.max()
        if p_max - p_min > 1e-6:
            proj = (proj - p_min) / (p_max - p_min)
        else:
            proj = np.zeros_like(proj)
        return proj.reshape(W, H, D, 3)

    pca_mri = project_pca(flat_mri)
    pca_gt = project_pca(flat_gt)
    pca_pred = project_pca(flat_pred)
    
    # 2. Metric Maps
    # Target Delta: What is the "Ground Truth" shift needed? (|GT - MRI|)
    z_target_delta = torch.mean(torch.abs(z_gt - z_mri), dim=0).cpu().numpy()

    # Model Delta: What shift did the model actually apply? (|Pred - MRI|)
    z_model_delta = torch.mean(torch.abs(z_pred - z_mri), dim=0).cpu().numpy()
    
    # Latent Error: What is the remaining error? (|Pred - GT|)
    z_error_map = torch.mean(torch.abs(z_pred - z_gt), dim=0).cpu().numpy()
    
    # Mean Feature Intensity (Channel-wise mean)
    mean_mri = torch.mean(z_mri, dim=0).cpu().numpy()
    mean_gt = torch.mean(z_gt, dim=0).cpu().numpy()
    mean_pred = torch.mean(z_pred, dim=0).cpu().numpy()

    # 3. Plotting
    z_slice = D // 2
    fig, axes = plt.subplots(4, 3, figsize=(15, 20))
    
    # -- Row 1: Image Space --
    axes[0, 0].imshow(mri[:, :, z_slice], cmap='gray')
    axes[0, 0].set_title("Input MRI", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_ct[:, :, z_slice], cmap='gray')
    axes[0, 1].set_title("GT CT", fontweight='bold')
    axes[0, 1].axis('off')

    if pred_ct is not None:
        # NOTE: Removed the 'seismic' overlay so it's clean grayscale
        axes[0, 2].imshow(pred_ct[:, :, z_slice], cmap='gray')
        axes[0, 2].set_title("Pred CT (Decoded)", fontweight='bold')
    else:
        axes[0, 2].text(0.5, 0.5, "Decoding Skipped", ha='center')
    axes[0, 2].axis('off')
    
    # -- Row 2: PCA Features (RGB) --
    axes[1, 0].imshow(np.rot90(pca_mri[:, :, z_slice]))
    axes[1, 0].set_title("Z_MRI (PCA)", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(np.rot90(pca_gt[:, :, z_slice]))
    axes[1, 1].set_title("Z_GT (PCA)", fontweight='bold')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(np.rot90(pca_pred[:, :, z_slice]))
    axes[1, 2].set_title("Z_Pred (PCA)", fontweight='bold')
    axes[1, 2].axis('off')

    # -- Row 3: Mean Features (Viridis) --
    # Scale all means to same range for fair comparison
    vmin = min(mean_mri.min(), mean_gt.min(), mean_pred.min())
    vmax = max(mean_mri.max(), mean_gt.max(), mean_pred.max())

    im_mm = axes[2, 0].imshow(mean_mri[:, :, z_slice], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 0].set_title("Z_MRI (Mean Intensity)")
    axes[2, 0].axis('off')
    divider = make_axes_locatable(axes[2, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_mm, cax=cax)

    im_mg = axes[2, 1].imshow(mean_gt[:, :, z_slice], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 1].set_title("Z_GT (Mean Intensity)")
    axes[2, 1].axis('off')
    divider = make_axes_locatable(axes[2, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_mg, cax=cax)

    im_mp = axes[2, 2].imshow(mean_pred[:, :, z_slice], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, 2].set_title("Z_Pred (Mean Intensity)")
    axes[2, 2].axis('off')
    divider = make_axes_locatable(axes[2, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_mp, cax=cax)

    # -- Row 4: Deltas and Errors (Inferno/Magma) --
    # Scale deltas to same range to see if model is being "lazy"
    d_max = max(z_target_delta.max(), z_model_delta.max())
    
    # 1. Target Delta
    im_td = axes[3, 0].imshow(z_target_delta[:, :, z_slice], cmap='inferno', vmin=0, vmax=d_max)
    axes[3, 0].set_title("Target Delta (|GT - MRI|)")
    axes[3, 0].axis('off')
    divider = make_axes_locatable(axes[3, 0])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_td, cax=cax)

    # 2. Model Delta
    im_md = axes[3, 1].imshow(z_model_delta[:, :, z_slice], cmap='inferno', vmin=0, vmax=d_max)
    axes[3, 1].set_title("Model Delta (|Pred - MRI|)")
    axes[3, 1].axis('off')
    divider = make_axes_locatable(axes[3, 1])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_md, cax=cax)

    # 3. Remaining Error
    im_err = axes[3, 2].imshow(z_error_map[:, :, z_slice], cmap='magma')
    axes[3, 2].set_title("Latent Error (|Pred - GT|)")
    axes[3, 2].axis('off')
    divider = make_axes_locatable(axes[3, 2])
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(im_err, cax=cax)
    
    fig.suptitle(f"Stage 2 Analysis: {subj_id} (Ep {epoch})", fontsize=16, y=0.99)
    
    save_dir = os.path.join(root, "results", "vis_stage_b")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subj_id}_ep{epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if use_wandb:
        wandb.log({f"stage_b_viz/val_{idx}": wandb.Image(save_path)}, step=epoch)


@torch.no_grad()
def validate_mapping(mapper, extractor, decoder, val_list, device, loss_fn, epoch, args):
    mapper.eval()
    extractor.eval()
    decoder.eval()
    
    metrics = {'mse_z': [], 'cosine_z': [], 'img_mae': [], 'img_psnr': []}
    val_running_loss = 0.0 
    
    # Only compute expensive metrics (SSIM/PSNR on CPU) occasionally
    run_detailed_metrics = (epoch % args.image_val_interval == 0)
    
    for idx, item in enumerate(val_list):
        # Load Data
        mri_t = torch.from_numpy(item['mri'])[None, None].float().to(device)
        ct_t = torch.from_numpy(item['ct'])[None, None].float().to(device)
        
        # 1. Feature Extraction
        z_mri = extractor(mri_t)
        z_ct_gt = extractor(ct_t)
        
        # 2. Mapping
        z_ct_pred = mapper(z_mri)
        
        # 3. Decode (REQUIRED for Loss Calculation now)
        rec_ct = decoder(z_ct_pred)
        
        # 4. Hybrid Loss (Fix: Pass all 4 arguments)
        loss_val, _ = loss_fn(z_ct_pred, z_ct_gt, rec_ct, ct_t)
        val_running_loss += loss_val.item()

        # 5. Basic Latent Metrics
        mse_val = F.mse_loss(z_ct_pred, z_ct_gt).item()
        metrics['mse_z'].append(mse_val)
        
        # Optimization: Flattening large volumes is slow, maybe skip cosine every step if speed issues arise
        p_flat = z_ct_pred.flatten(1)
        t_flat = z_ct_gt.flatten(1)
        cos_sim = F.cosine_similarity(p_flat, t_flat).mean().item()
        metrics['cosine_z'].append(cos_sim)
        
        # 6. Detailed Image Metrics (Only run periodically to save time)
        pred_ct_vis = None
        if run_detailed_metrics:
            orig = item['orig_shape']
            
            # Unpad / CPU Transfer is the slow part
            rec_unpad = unpad(rec_ct, orig)
            gt_unpad = unpad(ct_t, orig)
            
            mae, psnr, _ = compute_metrics(rec_unpad, gt_unpad)
            metrics['img_mae'].append(mae)
            metrics['img_psnr'].append(psnr)
            
            pred_ct_vis = unpad(rec_ct, orig).cpu().squeeze().numpy()

        # Visualization Logging
        if idx < args.viz_limit and run_detailed_metrics:
            print(f"[DEBUG] 📸 Viz Stage 2: {item['id']}")
            orig = item['orig_shape']
            mri_unpad = unpad(mri_t, orig).cpu().squeeze().numpy()
            ct_unpad = unpad(ct_t, orig).cpu().squeeze().numpy()
            
            visualize_stage_b(
                mri_unpad, 
                ct_unpad,
                pred_ct_vis, 
                z_mri.squeeze(0), z_ct_pred.squeeze(0), z_ct_gt.squeeze(0), 
                item['id'], args.root_dir, epoch, args.wandb,
                idx 
            )

    # Aggregation
    avg_cos = np.mean(metrics['cosine_z'])
    avg_mse = np.mean(metrics['mse_z'])
    avg_val_loss = val_running_loss / len(val_list)
    
    log_dict = {
        "val/loss": avg_val_loss,
        "val/z_cosine": avg_cos,
        "val/z_mse": avg_mse
    }
    
    print_str = f"✅ Ep {epoch} | Val Loss: {avg_val_loss:.5f} | Z-Cos: {avg_cos:.4f}"
    
    if run_detailed_metrics and len(metrics['img_mae']) > 0:
        avg_img_mae = np.mean(metrics['img_mae'])
        avg_img_psnr = np.mean(metrics['img_psnr'])
        log_dict["val/img_mae"] = avg_img_mae
        log_dict["val/img_psnr"] = avg_img_psnr
        print_str += f" | Img MAE: {avg_img_mae:.4f} | PSNR: {avg_img_psnr:.2f}"
    
    print(print_str)
    
    if args.wandb:
        wandb.log(log_dict, step=epoch)
# ==========================================
# 6. MAIN
# ==========================================
def run_stage_b(config_dict):
    cleanup_gpu()
    args = Config(config_dict)
    set_seed(args.seed)
    device = torch.device(args.device)
    
    print(f"\n[INFO] 🏁 Starting Stage 2: Latent Mapping")
    
    # 1. Data Setup
    subjects = discover_subjects(os.path.join(args.root_dir, "data"), target_list=args.subjects, region=args.region)
    if not subjects:
        print("[ERROR] No subjects found.")
        return
    
    random.shuffle(subjects)
    val_len = max(1, int(len(subjects) * args.val_split))
    train_sub = subjects[:-val_len]
    val_sub = subjects[-val_len:]
    print(f"[INFO] Train: {len(train_sub)} | Val: {len(val_sub)}")
    
    # 2. Preload Val
    val_list = []
    print("[INFO] Preloading Validation volumes...")
    for s in tqdm(val_sub):
        m, c, o = load_image_pair(args.root_dir, s, args)
        val_list.append({'id': s, 'mri': m, 'ct': c, 'orig_shape': o})
        
    # 3. Load Frozen Models
    print("[INFO] Loading Anatomix (Frozen)...")
    feat_extractor = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=5, ngf=20, norm="instance", interp="trilinear", pooling="Avg").to(device)
    feat_extractor = torch.compile(feat_extractor, mode="default")
    ckpt_ana = os.path.join(args.root_dir, "anatomix", "model-weights", "best_val_net_G.pth")
    feat_extractor.load_state_dict(torch.load(ckpt_ana, map_location=device))
    feat_extractor.eval()
    
    # Load Stage A Decoder
    print("[INFO] Loading Stage A Decoder (Frozen)...")
    # Must match config from Stage A
    decoder = CNNTranslator(in_channels=16, hidden_channels=args.dec_hidden, depth=args.dec_depth, final_activation="sigmoid").to(device)
    ckpt_dec = os.path.join(args.root_dir, "results", "models", "stage_a_decoder.pt")
    if os.path.exists(ckpt_dec):
        decoder.load_state_dict(torch.load(ckpt_dec, map_location=device))
        print(f"   -> Loaded decoder weights from {ckpt_dec}")
    else:
        # print(f"   -> [WARNING] Weights not found at {ckpt_dec}. Decoder is random!")
        raise FileNotFoundError(f"❌ [CRITICAL] Stage A Decoder not found at {ckpt_dec}. Training cannot proceed.")
    decoder.eval()
    
    # Freeze everything except mapper
    for p in feat_extractor.parameters(): p.requires_grad = False
    for p in decoder.parameters(): p.requires_grad = False
    
    # 4. Build Mapper
    print("[INFO] Building Latent Mapper...")
    # mapper = LatentResidualMapper(channels=16, hidden=args.map_hidden, layers=args.map_layers, use_residual=args.use_residual).to(device)
    mapper = StageBMapper(channels=16, hidden=args.map_hidden).to(device)
    opt = torch.optim.Adam(mapper.parameters(), lr=args.lr)
    # loss_fn = LatentConsistencyLoss(weights={'mse': args.mse, 'cosine': args.cosine}).to(device)
    loss_fn = HybridConsistencyLoss(weights={'z_mse': args.mse, 'img_l1': args.img_lambda}).to(device)
    
    # 5. Loader
    train_paths = [get_subject_paths(args.root_dir, s) for s in train_sub]
    loader = get_dataloader(train_paths, args)
    
    # 6. W&B
    if args.wandb:
        wandb.init(project=args.project_name, name="StageB_LatentMap", config=vars(args), reinit=True)
        wandb.save(os.path.abspath(__file__))

    # 7. Loop
    for epoch in range(1, args.epochs + 1):
        # loss, grad_norm = train_latent_epoch(mapper, feat_extractor, loader, opt, loss_fn, device, args)
        loss, grad_norm, train_metrics = train_latent_epoch(mapper, feat_extractor, decoder, loader, opt, loss_fn, device, args)
        print(f"Ep {epoch} | Loss: {loss:.5f} | Grad: {grad_norm:.4f}")
        
        if args.wandb: 
            log_payload = {
                "train/total_loss": loss,
                "train/grad_norm": grad_norm,
            }
            
            # Dynamically add all specific losses (z_mse, img_l1, etc.)
            for k, v in train_metrics.items():
                log_payload[f"train/{k}"] = v
                
            wandb.log(log_payload, step=epoch)   
            
        if epoch % args.val_interval == 0:
            validate_mapping(mapper, feat_extractor, decoder, val_list, device, loss_fn, epoch, args)
            
            save_path = os.path.join(args.root_dir, "results", "models", "stage_b_mapper.pt")
            torch.save(mapper.state_dict(), save_path)
            
    print("[INFO] Stage 2 Complete.")
    if args.wandb: wandb.finish()

# ==========================================
# CONFIG
# ==========================================
DEFAULT_CONFIG = {
    "root_dir": "/home/minsukc/MRI2CT",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    "project_name": "mri2ct_stage_b",
    
    "subjects": None, 
    "region": None,
    "val_split": 0.1,
    "patch_size": 96,
    "augment": True,
    
    "patches_per_volume": 30,
    "data_queue_max_length": 300,
    "num_workers": 2,
    "batch_size": 1,
    
    # Mapper Params
    "map_hidden": 128,
    # "map_layers": 3,
    # "use_residual": True, 
    # "use_residual": False, 
    
    # Decoder Params (Must match Stage A)
    "dec_hidden": 64,
    "dec_depth": 3,
    
    "epochs": 100,
    "lr": 3e-4,
    "steps_per_epoch": 50,
    "accum_steps": 8,
    "val_interval": 1,
    "image_val_interval": 1, 
    "viz_limit": 3,

    # "mse": 10.0,
    "mse": 1.0,
    "img_lambda": 1.0
    # "cosine": 0.0,
}

if __name__ == "__main__":
    run_stage_b(DEFAULT_CONFIG)