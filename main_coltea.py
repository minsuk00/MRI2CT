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
from collections import defaultdict 
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio
import copy

from fused_ssim import fused_ssim
from anatomix.model.network import Unet
# from models import MLPTranslator, CNNTranslator

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")

# ======================
# CNN Translator
# ======================
# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.norm1 = nn.InstanceNorm3d(channels, affine=True)
#         self.relu = nn.ReLU(inplace=True)
#         self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
#         self.norm2 = nn.InstanceNorm3d(channels, affine=True)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.norm1(out)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out = self.norm2(out)
#         out += residual 
#         # out = self.relu(out)
#         return out

# class CNNTranslator(nn.Module):
#     def __init__(self, in_channels=16, hidden_channels=64, depth=5, final_activation="sigmoid", dropout=0.0):
#         super().__init__()
#         print(f"[DEBUG] 🏗️ Building Robust CNNTranslator | Depth: {depth} | Hidden: {hidden_channels}")
        
#         # 1. Entry Block
#         self.entry = nn.Sequential(
#             nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
#             nn.InstanceNorm3d(hidden_channels),
#             nn.ReLU(inplace=True)
#         )
        
#         # 2. Residual Body
#         # Depth determines how many ResBlocks we stack
#         res_blocks = []
#         for _ in range(depth):
#             res_blocks.append(ResidualBlock(hidden_channels))
#         self.body = nn.Sequential(*res_blocks)
        
#         # 3. Exit Block
#         self.exit_conv = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)
        
#         self.final_act = final_activation

#     def forward(self, x):
#         x = self.entry(x)
#         x = self.body(x)
#         x = self.exit_conv(x)
        
#         if self.final_act == "sigmoid":
#             return torch.sigmoid(x)
#         elif self.final_act == "relu_clamp":
#             return torch.clamp(torch.relu(x), 0, 1)
#         return x

class CNNTranslator(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32, depth=3, final_activation="relu_clamp", dropout=0.0):
        """
        Args:
            in_channels (int): Input feature dimension.
            hidden_channels (int): Number of filters in hidden layers.
            depth (int): Total number of Conv3d layers.
            final_activation (str): "sigmoid", "relu_clamp", or "none".
        """
        super().__init__()
        self.final_activation = final_activation
        
        layers = []
        
        # --- 1. First Layer (Input -> Hidden) ---
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.InstanceNorm3d(hidden_channels, affine=True))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout3d(p=dropout))
        
        # --- 2. Middle Layers (Hidden -> Hidden) ---
        # We add (depth - 2) middle layers because first and last are handled separately
        for _ in range(depth - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.InstanceNorm3d(hidden_channels, affine=True)) 
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0:
                layers.append(nn.Dropout3d(p=dropout))
            
        # --- 3. Last Layer (Hidden -> 1) ---
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        
        if self.final_activation == "sigmoid":
            return torch.sigmoid(x)
        elif self.final_activation == "relu_clamp":
            return torch.clamp(torch.relu(x), 0, 1)
        elif self.final_activation == "none":
            return x
        else:
            raise ValueError(f"Unknown activation: {self.final_activation}")

            
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
    if not torch.is_tensor(arr):
        arr = torch.as_tensor(arr)
    
    if minclip is not None and maxclip is not None:
        arr = torch.clamp(arr, minclip, maxclip)
    
    denom = arr.max() - arr.min()
    if denom == 0: 
        print("MinMax - Denominator is 0. returning zero-like tensor")
        return torch.zeros_like(arr)
    return (arr - arr.min()) / denom
    
def unpad(data, original_shape):
    """
    Slices the input tensor to match original_shape.
    Assumes padding was applied at the END (anchor='start').
    data: (B, C, W, H, D) or (B, C, D, H, W) - we assume dim order matches shape
    """
    if original_shape is None:
        return data
    
    # original_shape comes in as (W, H, D) from TorchIO spatial_shape
    w_orig, h_orig, d_orig = original_shape
    # Slice the last 3 dimensions
    return data[..., :w_orig, :h_orig, :d_orig]

def cytran_metrics(pred, target,min_clip,max_clip):
    # 1. Denormalize to HU
    pred_hu = pred * (max_clip - min_clip) + min_clip
    targ_hu = target * (max_clip - min_clip) + min_clip

    # 2. Convert to Author Space (Clip < 0, /1000, -1)
    def to_author_space(x):
        x = torch.clamp(x, min=0)
        return (x / 1000.0) - 1.0

    pred_auth = to_author_space(pred_hu)
    targ_auth = to_author_space(targ_hu)

    # 3. Reshape to List of 2D Slices
    # (B, C, D, H, W) -> (D, H, W) assuming B=1, C=1
    pred_slices = pred_auth.squeeze(0).squeeze(0) # Shape: (Depth, H, W)
    targ_slices = targ_auth.squeeze(0).squeeze(0)

    # --- VECTORIZED SLICE-BY-SLICE CALCULATION ---
    
    # MAE per slice (Mean over H, W dims) -> then Mean over Depth
    # Note: For MAE, Global Mean and Slice Mean are mathematically the same, so this is safe.
    mae = torch.mean(torch.abs(pred_slices - targ_slices)).item()

    # RMSE per slice
    # 1. MSE per slice (Mean over H, W) -> Shape: (Depth,)
    mse_per_slice = torch.mean((pred_slices - targ_slices) ** 2, dim=[1, 2])
    # 2. RMSE per slice (Sqrt)
    rmse_per_slice = torch.sqrt(mse_per_slice)
    # 3. Average of RMSEs
    rmse = torch.mean(rmse_per_slice).item()

    # SSIM per slice (using skimage on CPU)
    # Move to Numpy for skimage
    pred_np = pred_slices.detach().cpu().numpy()
    targ_np = targ_slices.detach().cpu().numpy()
    
    ssim_list = []
    n_slices = pred_np.shape[0]
    
    # Calculate Data Range for SSIM (Author's Space is approx -1 to 2 or 3)
    # We infer it from the target volume to be safe
    data_range = targ_np.max() - targ_np.min()
    
    for i in range(n_slices):
        val = ssim(pred_np[i], targ_np[i], data_range=data_range)
        ssim_list.append(val)
        
    ssim_val = np.mean(ssim_list)

    return mae, rmse, ssim_val

def compute_metrics(pred, target, args, data_range=1.0):
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

    # ===== cytran metrics =====
    cytran_mae, cytran_rmse, cytran_ssim_val = cytran_metrics(pred, target, args.min_clip,args.max_clip)
    
    # return mae_val, psnr_val, ssim_val
    # return mae_val, psnr_val, ssim_val, cytran_mae, cytran_rmse, cytran_ssim_val 
    return {
        "mae": mae_val,
        "psnr": psnr_val,
        "ssim": ssim_val,
        "cytran_mae": cytran_mae,
        "cytran_rmse": cytran_rmse,
        "cytran_ssim": cytran_ssim_val,
    }

# def get_subject_paths(root, subj_id):
#     paths = {}
#     data_path = os.path.join(root, "data")
    
#     # 1. CT
#     ct = glob(os.path.join(data_path, subj_id, "ct_resampled.nii*"))
#     if not ct: raise FileNotFoundError(f"CT missing for {subj_id}")
#     paths['ct'] = ct[0]
    
#     # 2. MRI
#     mr = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))
#     if not mr: raise FileNotFoundError(f"MRI missing for {subj_id}")
#     paths['mri'] = mr[0]
            
#     return paths

def get_subject_paths(root, subj_id, split="train"):
    # root should be the path containing 'train', 'val', 'test' folders
    base_path = os.path.join(root, split, subj_id)
    
    paths = {}
    # 1. Target (Venous/Contrast) -> Mapped to 'ct' variable
    # ct_candidates = glob(os.path.join(base_path, "venous.nii*"))
    ct_candidates = glob(os.path.join(base_path, "arterial.nii*"))
    if not ct_candidates: 
        raise FileNotFoundError(f"Target (Contrast) missing for {subj_id} in {base_path}")
    paths['ct'] = ct_candidates[0]
    
    # 2. Input (Native) -> Mapped to 'mri' variable
    mr_candidates = glob(os.path.join(base_path, "native.nii*"))
    if not mr_candidates: 
        raise FileNotFoundError(f"Input (Native) missing for {subj_id} in {base_path}")
    paths['mri'] = mr_candidates[0]
            
    return paths

    
# def load_image_pair(root, subj_id, args):
#     """Used for Validation Loading"""
#     data_path = os.path.join(root, "data")
#     ct_path = glob(os.path.join(data_path, subj_id, "ct_resampled.nii*"))[0]
#     mr_path = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))[0]
    
#     mr_img = tio.ScalarImage(mr_path)
#     ct_img = tio.ScalarImage(ct_path)
    
#     # Normalize (Numpy)
#     # mri = minmax(mr_img.data[0]).numpy()
#     mri_tensor = mr_img.data[0]
#     mri = robust_scale(mri_tensor).numpy()
#     ct = minmax(ct_img.data[0], minclip=-450, maxclip=450).numpy()
    
#     # Capture Original Shape (W, H, D)
#     orig_shape = mri.shape
#     # Calculate Target Shape (Multiple of 16)
#     # target_shape = [(d + 15) // 16 * 16 for d in orig_shape] # round up to multiple of 16
#     # target_shape = [max(args.patch_size, (d + 15) // 16 * 16) for d in orig_shape]
#     target_shape = [max(args.patch_size, (d + 31) // 32 * 32) for d in orig_shape]
    
#     # Pad Numpy Arrays (pad at the end)
#     # np.pad takes list of (before, after) tuples
#     # We want (0, target - original) for each dim
#     pad_width = [(0, t - o) for t, o in zip(target_shape, orig_shape)]
    
#     mri_padded = np.pad(mri, pad_width, mode='constant', constant_values=0)
#     ct_padded = np.pad(ct, pad_width, mode='constant', constant_values=0)

#     assert mri_padded.shape == tuple(target_shape), "MRI Padding Mismatch!"
#     # Check Anchor Start: The top-left corner should be identical
#     assert np.allclose(mri_padded[:orig_shape[0], :orig_shape[1], :orig_shape[2]], mri), "Anchor Start Failed! Data shifted."
#     # Check Padding Area: The newly added area should be 0
#     if target_shape[0] > orig_shape[0]:
#         assert mri_padded[orig_shape[0]:, ...].sum() == 0, "Padding area is not zero!"
    
#     print(f"[DEBUG] Valid {subj_id} | Orig: {orig_shape} -> Padded: {target_shape}")
#     return mri_padded, ct_padded, orig_shape

def load_image_pair(root, subj_id, args, split="val"):
    """
    Used for Validation Loading and Training Monitor.
    Now supports explicit 'split' (train/val) and CT-specific normalization.
    """
    # 1. Get Paths using the helper we updated
    paths = get_subject_paths(root, subj_id, split=split)
    
    # 2. Load Images (TorchIO)
    # paths['mri'] maps to Native (Input)
    # paths['ct']  maps to Venous (Target)
    mr_img = tio.ScalarImage(paths['mri'])
    ct_img = tio.ScalarImage(paths['ct'])
    
    # 3. Normalize (Numpy)
    mri_tensor = mr_img.data[0]
    # Changed from robust_scale (for MRI) to minmax (for CT)
    # mri = minmax(mri_tensor, minclip=-450, maxclip=450).numpy()
    mri = minmax(mri_tensor, minclip=args.min_clip, maxclip=args.max_clip).numpy()
    
    # Target (Arterial)
    # ct = minmax(ct_img.data[0], minclip=-450, maxclip=450).numpy()
    ct = minmax(ct_img.data[0], minclip=args.min_clip, maxclip=args.max_clip).numpy()
    
    # 4. Smart Padding (unchanged logic)
    # Capture Original Shape (W, H, D)
    orig_shape = mri.shape
    
    # Calculate Target Shape (Multiple of 32 for UNet safety)
    target_shape = [max(args.patch_size, (d + args.res_mult-1) // args.res_mult * args.res_mult) for d in orig_shape]
    
    
    # Pad Numpy Arrays (pad at the end)
    pad_width = [(0, t - o) for t, o in zip(target_shape, orig_shape)]
    
    mri_padded = np.pad(mri, pad_width, mode='constant', constant_values=0)
    ct_padded = np.pad(ct, pad_width, mode='constant', constant_values=0)

    # Safety Checks
    assert mri_padded.shape == tuple(target_shape), "Padding Mismatch!"
    assert np.allclose(mri_padded[:orig_shape[0], :orig_shape[1], :orig_shape[2]], mri), "Anchor Start Failed!"
    
    print(f"[DEBUG] Loaded {subj_id} ({split}) | Orig: {orig_shape} -> Padded: {target_shape}")
    return mri_padded, ct_padded, orig_shape
    
# ==========================================
# 2. DATA PIPELINE (TORCHIO)
# ==========================================
class ProjectPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, enable_safety_padding=False, min_clip=-1000, max_clip=1000, res_mult = 32, **kwargs):       
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.enable_safety_padding = enable_safety_padding
        self.min_clip=min_clip
        self.max_clip=max_clip
        self.res_mult = res_mult

    def apply_transform(self, subject):
        # 1. Normalize CT & MRI
        ct_data = subject['ct'].data
        # subject['ct'].set_data(minmax(ct_data, -450, 450).float())
        subject['ct'].set_data(minmax(ct_data, self.min_clip, self.max_clip).float())
        mr_data = subject['mri'].data
        # subject['mri'].set_data(minmax(mr_data).float())
        # subject['mri'].set_data(robust_scale(mr_data).float())
        # subject['mri'].set_data(minmax(mr_data, -450, 450).float())
        subject['mri'].set_data(minmax(mr_data, self.min_clip, self.max_clip).float())


        if self.enable_safety_padding:
            safety_margin = self.patch_size // 2  # 48 voxels
            safety_padder = tio.Pad(safety_margin, padding_mode=0)
            subject = safety_padder(subject)

        # 2. Smart Padding (Multiple of 16 AND >= Patch Size)
        current_shape = subject['ct'].spatial_shape # (W, H, D)
        # Store original shape for unpadding later
        subject['original_shape'] = torch.tensor(current_shape)
        # Calculate target shape: max(patch_size, next_multiple_of_16)
        target_shape = []
        for dim in current_shape:
            # Ensure multiple of 16 AND at least patch_size
            # mult_16 = (int(dim) + 15) // 16 * 16
            # mult_16 = (int(dim) + 31) // 32 * 32
            mult_16 = (int(dim) + self.res_mult-1) // self.res_mult * self.res_mult
            target_shape.append(max(self.patch_size, mult_16))
            
        # Calculate diff: Target - Current
        # need padding tuple: (w_in, w_out, h_in, h_out, d_in, d_out)
        # set 'in' to 0 and 'out' to diff to anchor at start.
        padding_params = []
        for curr, targ in zip(current_shape, target_shape):
            diff = int(targ - curr)
            padding_params.extend([0, diff]) # Append (0, diff)
            
        if any(p > 0 for p in padding_params):
            padder = tio.Pad(padding_params, padding_mode=0)
            subject = padder(subject)

        # vol_shape: NEW Padded Size
        subject['vol_shape'] = torch.tensor(subject['ct'].spatial_shape).float()
        
        # 3. Probability Map
        prob = (subject['ct'].data > 0.01).to(torch.float32)
        subject.add_image(tio.LabelMap(tensor=prob, affine=subject['mri'].affine), 'prob_map')

        final_shape = subject['mri'].data.shape[1:]
        if final_shape[0] % self.res_mult != 0:
            print(f"[WARNING] Training volume width {final_shape[0]} is not multiple of {self.res_mult}!")
            
        return subject

def get_augmentations():
    # 1. Geometric Transforms: Apply to BOTH MRI and CT (to keep them aligned)
    spatial_transforms = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
        tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.25),
    ])

    # 2. Intensity Transforms: Apply ONLY to MRI (the input)
    #    Protect the CT (ground truth) from noise/bias artifacts.
    intensity_transforms = tio.Compose([
        # tio.RandomBiasField(p=0.5, include=['mri']), 
        tio.RandomNoise(std=0.02, p=0.25, include=['mri']),
        # tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5, include=['mri'])
    ])

    return tio.Compose([spatial_transforms, intensity_transforms])

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
        subjects.append(subject)  

    use_safety = (args.model_type.lower() == "cnn" and args.enable_safety_padding)
    preprocess = ProjectPreprocessing(patch_size=args.patch_size, enable_safety_padding=use_safety,min_clip=args.min_clip,max_clip=args.max_clip, res_mult = args.res_mult)    
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
        # sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    else:
        sampler = tio.WeightedSampler(patch_size=patch_size, probability_map='prob_map')
    
    queue = tio.Queue(
        subjects_dataset=dataset,
        samples_per_volume=args.patches_per_volume,
        max_length=max(args.patches_per_volume, args.data_queue_max_length), 
        sampler=sampler,
        num_workers=3, 
        # num_workers=0, 
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
    def __init__(self, weights={"l1": 1.0, "l2": 0.0, "ssim": 0.0, "perceptual": 0.0}):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target, feat_extractor=None):
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
        
        if self.weights.get("perceptual", 0) > 0:
            if feat_extractor is None:
                # print("[Warning] Perceptual Loss weight > 0, but no feature_extractor!!")
                raise ValueError("[Warning] Perceptual Loss weight > 0, but no feature_extractor!!")
            else:
                # 1. Get features of the PREDICTION (Gradients Flow Here)
                pred_feats = feat_extractor(pred)
                # 2. Get features of the TARGET (No Gradients needed for GT)
                with torch.no_grad():
                    target_feats = feat_extractor(target)
                
                val_perceptual = self.l1(pred_feats, target_feats)
                
                total_loss += self.weights["perceptual"] * val_perceptual
                loss_components["loss_perceptual"] = val_perceptual.item()
        
        return total_loss, loss_components

        
def get_mlp_samples(features, target, locations, vol_shapes, num_samples=16384):
    """
    Dimension-Agnostic Sampling.
    
    This function ignores whether the dimensions are (D,H,W) or (W,H,D).
    It simply matches:
      - Tensor spatial dim 0 -> Metadata index 0
      - Tensor spatial dim 1 -> Metadata index 1
      - Tensor spatial dim 2 -> Metadata index 2
    """
    # 1. Capture Raw Dimensions
    # features shape: (Batch, Channel, Dim0, Dim1, Dim2)
    B, C, *spatial_dims = features.shape
    device = features.device

    # 2. Permute Features for flattening
    # From (B, C, d0, d1, d2) -> (B, d0, d1, d2, C) -> Flatten
    feats_flat = features.permute(0, 2, 3, 4, 1).reshape(-1, C)
    
    # Target shape: (B, C, d0, d1, d2) -> Flatten same way
    targ_flat = target.permute(0, 2, 3, 4, 1).reshape(-1, 1)
    
    # 3. Generate Local Grids (Index-Matched)
    # We create a list of ranges: [0..d0-1, 0..d1-1, 0..d2-1]
    ranges = [torch.arange(d, device=device).float() for d in spatial_dims]
    
    # Meshgrid 'ij' ensures the output grids match input order
    # grids[0] varies along dim 0, grids[1] along dim 1, etc.
    grids = torch.meshgrid(*ranges, indexing='ij')

    # 4. Global Normalization (The Fix)
    norm_grids = []
    for i, grid in enumerate(grids):
        # Metadata is (B, 3). We grab column i to match tensor dim i.
        off = locations[:, i].view(B, 1, 1, 1).float().to(device)
        max_val = vol_shapes[:, i].view(B, 1, 1, 1).float().to(device)
        
        # Normalize: (local_index + offset) / (total_size - 1)
        # This maps the very first voxel to 0.0 and the very last (padded) voxel to 1.0
        norm_coord = (grid.unsqueeze(0) + off) / (max_val - 1)
        norm_grids.append(norm_coord)

    # 5. Stack Coordinates
    # Result: (B, d0, d1, d2, 3)
    coords_global = torch.stack(norm_grids, dim=-1)
    coords_flat = coords_global.reshape(-1, 3)
    
    # 6. Random Sampling
    total_voxels = feats_flat.shape[0]
    actual_samples = min(num_samples, total_voxels)
    
    # Randomly select indices
    indices = torch.randperm(total_voxels)[:actual_samples]

    return feats_flat[indices], coords_flat[indices], targ_flat[indices]

# ==========================================
# 4. VISUALIZATION
# ==========================================
@torch.no_grad()
def visualize_ct_feature_comparison(pred_ct, gt_ct, gt_mri, model, subj_id, root_dir, original_shape, epoch=None, use_wandb=False, idx=1):
    device = next(model.parameters()).device
    
    def extract_feats_np(volume_np):
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        feats = model(inp)
        return feats.squeeze(0).cpu().numpy()
        
    # NOTE: extract features from the PADDED volumes first
    feats_gt = extract_feats_np(gt_ct)
    feats_pred = extract_feats_np(pred_ct)
    feats_mri = extract_feats_np(gt_mri)
    
    # --- NOW unpad everything for visualization ---
    if original_shape:
        w, h, d = original_shape
        # Unpad Volumes
        gt_ct = gt_ct[:w, :h, :d]
        gt_mri = gt_mri[:w, :h, :d]
        pred_ct = pred_ct[:w, :h, :d]
        # Unpad Features (C, W, H, D) -> slice last 3 dimensions
        feats_gt = feats_gt[..., :w, :h, :d]
        feats_pred = feats_pred[..., :w, :h, :d]
        feats_mri = feats_mri[..., :w, :h, :d]
    
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
            (gt_mri, "Native (Input)", "gray", (0,1)),
            (gt_ct, "Arterial (GT)", "gray", (0,1)),
            (pred_ct, "Arterial (Pred)", "gray", (0,1)),
            (residual, "Residual", "seismic", (-0.5, 0.5)),
            (pca_mri, "PCA (Native)", None, None),
            (pca_gt, "PCA (GT Arterial)", None, None),
            (pca_pred, "PCA (Pred Arterial)", None, None),
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
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    # plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    
    if use_wandb: 
        wandb.log({f"viz/{'train' if idx==-1 else ('val_'+ str(idx))}": wandb.Image(save_path)}, step=epoch)

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
        preprocess = ProjectPreprocessing(patch_size=args.patch_size, min_clip=args.min_clip,max_clip=args.max_clip, res_mult = args.res_mult)
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
        plt.savefig(save_path, dpi=200)
        # plt.savefig(save_path)
        plt.close(fig)
        wandb.log({"val/aug_example": wandb.Image(save_path)}, step=epoch)
    except Exception as e:
        print(f"[DEBUG] Aug Viz failed: {e}")

# ==========================================
# 5. CORE ENGINE (TRAIN/EVAL)
# ==========================================
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, model_type, feat_extractor, args):
    model.train()
    if args.finetune_feat_extractor:
        feat_extractor.train()
    else:
        feat_extractor.eval()
    
    total_loss = 0.0
    total_grad_norm = 0.0
    comp_accumulator = {}
    
    # 1. Create the Infinite Iterator from your TioAdapter
    loader_iter = iter(loader)
    # 2. Set the target steps (e.g., 1000)
    target_steps = args.steps_per_epoch
    progress = tqdm(range(target_steps), desc="Train Step", leave=False, dynamic_ncols=True)
    
    for step in progress:
        optimizer.zero_grad()
        step_loss = 0.0
        
        # 3. Accumulation Loop
        for _ in range(args.accum_steps):
            try:
                # TioAdapter already yields (mri, ct, loc, vol_shape) as Tensors
                mri, ct, location, vol_shape = next(loader_iter)
            except StopIteration:
                # Restart the adapter's iterator
                loader_iter = iter(loader)
                mri, ct, location, vol_shape = next(loader_iter)
            
            mri, ct = mri.to(device), ct.to(device)

            # 4. Feature Extraction
            if args.finetune_feat_extractor:
                # Normal forward pass (Gradients ON)
                features = feat_extractor(mri)
            else:
                # Frozen pass (Gradients OFF)
                with torch.no_grad():
                    features = feat_extractor(mri)
            
            # 5. Forward Pass
            if model_type.lower() == "mlp":
                f_pts, c_pts, t_pts = get_mlp_samples(features, ct, location, vol_shape, num_samples=args.mlp_batch_size)
                pred = model(f_pts, c_pts)
                loss, components = loss_fn(pred, t_pts)
            elif model_type.lower() == "cnn":
                pred = model(features)
                loss, components = loss_fn(pred, ct, feat_extractor = feat_extractor)
            
            # Accumulate Loss Components
            for k, v in components.items():
                comp_accumulator[k] = comp_accumulator.get(k, 0.0) + (v / args.accum_steps)

            loss = loss / args.accum_steps
            loss.backward()
            step_loss += loss.item()

        # 6. Optimizer Step
        # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(feat_extractor.parameters()), 1.0)
        
        all_params = list(model.parameters())
        if args.finetune_feat_extractor:
            all_params += list(feat_extractor.parameters())
        # Clip norms for everything
        grad_norm = torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        
        total_grad_norm += grad_norm.item()
        total_loss += step_loss
        # progress.set_postfix({"loss": f"{step_loss:.5f}"})
        progress.set_postfix({
            "loss": f"{step_loss:.4f}",
            "grad": f"{grad_norm.item():.4f}"
        })

    # 7. Final Averages
    avg_loss = total_loss / target_steps
    avg_grad_norm = total_grad_norm / target_steps
    avg_components = {k: v / target_steps for k, v in comp_accumulator.items()}

    return avg_loss, avg_components, avg_grad_norm

@torch.no_grad()
def evaluate(model, feats_mri, ct, device, model_type, original_shape, args):
    model.eval()
    
    # Handle CT GT input (Numpy or Tensor)
    if isinstance(ct, np.ndarray):
        ct_tensor = torch.from_numpy(ct).float().to(device)
    else:
        ct_tensor = ct.float().to(device)
    if ct_tensor.ndim == 3: ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)
    
    pred_ct_tensor = None

    if model_type.lower() == "mlp":
        # feats_mri input is typically numpy (C, d0, d1, d2) from your validation loader
        # We convert to tensor (1, C, d0, d1, d2) just to get shapes easily
        if isinstance(feats_mri, np.ndarray):
            f_tensor = torch.from_numpy(feats_mri).unsqueeze(0).float().to(device)
        else:
            f_tensor = feats_mri.unsqueeze(0).float().to(device)

        B, C, *spatial_dims = f_tensor.shape
        
        # 1. Flatten Features: (1, d0, d1, d2, C) -> (N, C)
        feats_flat = f_tensor.permute(0, 2, 3, 4, 1).reshape(-1, C)
        
        # 2. Generate Linear Grids 0..1 (Global)
        # Since validation creates the FULL volume, we don't need 'locations'.
        # We just need 0.0 to 1.0 across the entire dimension.
        ranges = [torch.linspace(0, 1, d, device=device) for d in spatial_dims]
        
        # 3. Meshgrid
        grids = torch.meshgrid(*ranges, indexing='ij')
        
        # 4. Stack (d0, d1, d2, 3) -> Flatten -> (N, 3)
        coords = torch.stack(grids, dim=-1).reshape(-1, 3)
        
        # 5. Inference Loop (Chunked to save VRAM)
        preds = []
        batch_size = 800000 
        for i in range(0, feats_flat.size(0), batch_size):
            f_batch = feats_flat[i:i+batch_size]
            c_batch = coords[i:i+batch_size]
            preds.append(model(f_batch, c_batch))
        
        pred_flat = torch.cat(preds, dim=0)
        
        # Reshape back to (1, 1, d0, d1, d2)
        # We use *spatial_dims to unpack the list [d0, d1, d2]
        pred_ct_tensor = pred_flat.reshape(1, 1, *spatial_dims)

    elif model_type.lower() == "cnn":
        feats_t = torch.from_numpy(feats_mri).unsqueeze(0).float().to(device)
        pred_ct_tensor = model(feats_t)

    # Metrics
    pred_ct_tensor_unpad = unpad(pred_ct_tensor, original_shape)
    ct_tensor_unpad = unpad(ct_tensor, original_shape)
    
    metrics = compute_metrics(pred_ct_tensor_unpad, ct_tensor_unpad, args)
    pred_ct = pred_ct_tensor.squeeze().cpu().numpy()
    
    return metrics, pred_ct
    
# Validation Helper
def run_validation(epoch_idx, model, val_meta_list, device, feat_extractor, args, loss_fn, avg_train_loss=0.0, viz_limit=3, train_monitor_data=None):
    model.eval()
    # val_metrics = {'mae': [], 'psnr': [], 'ssim': [], 'loss': []}
    viz_count = 0 
    val_metrics = defaultdict(list)

    if args.wandb and train_monitor_data:
        print(f"   🔎 Visualizing Training Monitor ({train_monitor_data['id']})")
        try:
            # Run inference on the training subject
            _, pred_ct = evaluate(
                model, 
                train_monitor_data['feats'], 
                train_monitor_data['ct'], 
                device, 
                args.model_type, 
                train_monitor_data['orig_shape'],
                args
            )
            # Log it with a special index (999) so it stands out in WandB
            visualize_ct_feature_comparison(
                pred_ct, train_monitor_data['ct'], train_monitor_data['mri'], feat_extractor, 
                train_monitor_data['id'], args.root_dir, 
                original_shape=train_monitor_data['orig_shape'],
                epoch=epoch_idx, use_wandb=True, idx=-1 
            )
        except Exception as e:
            print(f"   ⚠️ Training Viz Failed: {e}")

    # NOTE: Augmentation is not actually applied during validation. This is just for visualization.
    if args.augment and args.wandb and val_meta_list:
        log_aug_viz(val_meta_list[0], args, epoch_idx, args.root_dir)
        
    for v_data in val_meta_list:
        # (mae, psnr, ssim), pred_ct = evaluate(model, v_data['feats'], v_data['ct'], device, args.model_type, v_data['orig_shape'])
        metrics_dict, pred_ct = evaluate(model, v_data['feats'], v_data['ct'], device, args.model_type, v_data['orig_shape'], args)

        with torch.no_grad():
            # Convert Prediction to Tensor
            pred_t = torch.from_numpy(pred_ct).float().to(device)
            if pred_t.ndim == 3: 
                pred_t = pred_t.unsqueeze(0).unsqueeze(0)

            # Convert Ground Truth to Tensor
            gt_t = torch.from_numpy(v_data['ct']).float().to(device)
            if gt_t.ndim == 3:
                gt_t = gt_t.unsqueeze(0).unsqueeze(0)

            val_loss_item, _ = loss_fn(pred_t, gt_t, feat_extractor=feat_extractor)
            # val_metrics['loss'].append(val_loss_item)
            # val_metrics['loss'].append(val_loss_item.item())
            metrics_dict['loss'] = val_loss_item.item()
        
        # val_metrics['mae'].append(mae)
        # val_metrics['psnr'].append(psnr)
        # val_metrics['ssim'].append(ssim)
        for k, v in metrics_dict.items():
            # if isinstance(v, torch.Tensor):
            #     v = v.item()
            #     print(k,"is on gpu")
            val_metrics[k].append(v)
        
        if args.wandb and viz_count < viz_limit:
            print(f"   🔎 Val Viz [{viz_count+1}/{viz_limit}] ({v_data['id']})")
            visualize_ct_feature_comparison(
                pred_ct, v_data['ct'], v_data['mri'], feat_extractor, 
                v_data['id'], args.root_dir, 
                original_shape=v_data['orig_shape'],
                epoch=epoch_idx, use_wandb=True, idx = viz_count,
            )
            viz_count += 1
    
    avg_metrics = {k: np.mean(v) for k, v in val_metrics.items()}
    # avg_mae = np.mean(val_metrics['mae']) if val_metrics['mae'] else 0.0
    # avg_psnr = np.mean(val_metrics['psnr']) if val_metrics['psnr'] else 0.0
    # avg_ssim = np.mean(val_metrics['ssim']) if val_metrics['ssim'] else 0.0
    # avg_val_loss = np.mean(val_metrics['loss']) if val_metrics['loss'] else 0.0

    log_str = f"Ep {epoch_idx} | Train Loss: {avg_train_loss:.5f}"
    for k, v in avg_metrics.items():
        log_str += f" | Val {k.upper()}: {v:.4f}"
    print(log_str)
    
    if args.wandb:
        wandb_log = {f"val/{k}": v for k, v in avg_metrics.items()}
        wandb.log(wandb_log, step=epoch_idx)
    
    # print(f"Ep {epoch_idx} | Avg Train Loss: {avg_train_loss:.5f} | Avg Val Loss: {avg_val_loss:.5f} | Val MAE: {avg_mae:.4f} | PSNR: {avg_psnr:.2f} | SSIM: {avg_ssim:.3f}")
    # if args.wandb:
    #     wandb.log({"val/loss": avg_val_loss, "val/mae": avg_mae, "val/psnr": avg_psnr, "val/ssim": avg_ssim}, step=epoch_idx)
        
# ==========================================
# 6. ORCHESTRATION
# ==========================================
# def discover_subjects(data_dir, target_list=None, region=None): # region: AB, TH, HN
#     if target_list:
#         candidates = target_list
#         print(f"[DEBUG] 🎯 Using explicit target_list ({len(candidates)} subjects)")
#     else:
#         candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
        
#         if region:
#             region = region.upper()
#             candidates = [c for c in candidates if region in c]
#             print(f"[DEBUG] 📍 No target_list found. Filtering for region: {region} ({len(candidates)} subjects)")
#         else:
#             print(f"[DEBUG] 📂 No target_list or region specified. Using all {len(candidates)} discovered subjects.")
    
#     valid = []
#     required = ["ct_resampled.nii.gz"]
#     for subj in candidates:
#         path = os.path.join(data_dir, subj)
#         if all(os.path.exists(os.path.join(path, f)) for f in required):
#             valid.append(subj)
#     print(f"Found {len(valid)} subjects")
#     return valid

def discover_subjects(root_dir, split):
    search_path = os.path.join(root_dir, split)
    
    if not os.path.exists(search_path):
        print(f"[ERROR] Split folder not found: {search_path}")
        return []

    candidates = sorted([d for d in os.listdir(search_path) if os.path.isdir(os.path.join(search_path, d))])
    
    valid = []
    required = ["native.nii.gz", "arterial.nii.gz"] 
    
    for subj in candidates:
        subj_path = os.path.join(search_path, subj)
        if all(glob(os.path.join(subj_path, f"{f.split('.')[0]}*")) for f in required):
            valid.append(subj)
            
    print(f"[{split.upper()}] Found {len(valid)} subjects in {search_path}")
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
    # Enables TF32 for significantly faster training on Ampere+ GPUs (makes the numerical precision a bit lower)
    torch.set_float32_matmul_precision('high')

    # Merge configs
    final_conf_dict = copy.deepcopy(DEFAULT_CONFIG)
    final_conf_dict.update(exp_config_dict)
    args = Config(final_conf_dict)

    config_json = json.dumps(vars(args), indent=4, sort_keys=True)
    print("\n" + "🔍" + " EXPERIMENT CONFIG ".center(50, "="))
    print(config_json)
    print("=" * 52 + "\n")
    
    # Setup
    set_seed(args.seed)
    device = torch.device(args.device)
    print(f"[DEBUG] 🟢 Starting Experiment. Device: {device}, Model: {args.model_type}")
    
    # 1. Subject Discovery
    # subjects = discover_subjects(os.path.join(args.root_dir, "data"), target_list=args.subjects, region=args.region)
    # if not subjects:
    #     print("[ERROR] No subjects found.")
    #     return

    # # 2. Train/Val Split
    # random.shuffle(subjects)
    # num_val = max(1, int(len(subjects) * args.val_split)) # By default, use at least 1 validation subject
    # train_subjects = subjects[:-num_val]
    # val_subjects = subjects[-num_val:]
    # print(f"[DEBUG] Train: {len(train_subjects)} | Val: {len(val_subjects)}")

    train_subjects = discover_subjects(args.data_root_dir, split="train")
    val_subjects = discover_subjects(args.data_root_dir, split="val")
    
    if not train_subjects:
        print("[ERROR] No training subjects found.")
        return
    print(f"[DEBUG] Train: {len(train_subjects)} | Val: {len(val_subjects)}")

    # 3. W&B
    if args.wandb:
        run_name = f"{args.model_type.upper()}_Train{len(train_subjects)}"
        wandb.init(project=args.project_name, name=run_name, config=vars(args), reinit=True)
        wandb.save(os.path.abspath(__file__))
        wandb.save(os.path.join(args.root_dir, "models.py"))

    # 4. Feature Extractor (Anatomix)
    print("[DEBUG] 🏗️ Building and Compiling Anatomix...")
    
    if args.anatomix_weights == "new":
        args.res_mult=32
        feat_extractor = Unet(
            dimension=3,
            input_nc=1,
            output_nc=16,
            num_downs=5,
            ngf=20,
            norm="instance",
            interp="trilinear",
            pooling="Avg",
        ).to(device)
        feat_extractor = torch.compile(feat_extractor, mode="default")
        ckpt_path = os.path.join(args.root_dir, "anatomix", "model-weights", "best_val_net_G.pth")
    elif args.anatomix_weights == "prev":
        args.res_mult=16
        feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
        ckpt_path = os.path.join(args.root_dir, "anatomix", "model-weights", "anatomix.pth")
    else:
        raise ValueError("define which anatomix weights to use: 'new' or 'prev'")

    if os.path.exists(ckpt_path):
        print(f"Loading weights from {ckpt_path}...")
        feat_extractor.load_state_dict(torch.load(ckpt_path, map_location=device), strict=True)
    else:
        print(f"[WARNING] ⚠️ Weights not found at {ckpt_path}...")
        return
    feat_extractor.eval()

    # 5. Load Train Data
    train_paths = []
    shapes = []

    for subj_id in tqdm(train_subjects, desc="Loading Paths"):
        try:
            # paths = get_subject_paths(args.root_dir, subj_id)
            paths = get_subject_paths(args.data_root_dir, subj_id, split="train")
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
            print(f"⚠️ Warning: Mean shape {tuple(avg_shape)} is smaller than patch size {args.patch_size} in some dims. (Auto-padding is active to prevent crashes)")
    if not train_paths:
        print("❌ No valid training data.")
        return   

    # Capture one Training Subject for Monitoring
    train_monitor_data = None
    if len(train_subjects) > 0:
        print(f"[DEBUG] 📸 Capturing Training Monitor (Subject: {train_subjects[0]})...")
        try:
            t_id = train_subjects[0]
            # Load it using the VALIDATION logic (Full volume, no patches)
            mri_t, ct_t, orig_shape_t = load_image_pair(args.data_root_dir, t_id, args, split="train")
            
            # Pre-calculate features (just like we do for validation)
            with torch.no_grad():
                inp = torch.from_numpy(mri_t[None, None]).float().to(device)
                feats_t = feat_extractor(inp).squeeze(0).cpu().numpy()
            
            train_monitor_data = {
                'id': f"{t_id}_TRAIN_MONITOR", 
                'ct': ct_t, 
                'mri': mri_t, 
                'feats': feats_t, 
                'orig_shape': orig_shape_t
            }
        except Exception as e:
            print(f"⚠️ Failed to capture training monitor: {e}")
        
    # 6. Pre-load Val Data (Features)
    val_meta_list = []
    print("[DEBUG] Pre-loading Validation Data...")
    for subj_id in tqdm(val_subjects):
        try:
            mri, ct, orig_shape = load_image_pair(args.data_root_dir, subj_id,args)
            with torch.no_grad():
                inp = torch.from_numpy(mri[None, None]).float().to(device)
                feats = feat_extractor(inp).squeeze(0).cpu().numpy()
            
            val_meta_list.append({'id': subj_id, 'ct': ct, 'mri': mri, 'feats': feats, 'orig_shape': orig_shape})
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
        # model = torch.compile(model, mode="default")
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
        # model = torch.compile(model, mode="default")

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    params = [
        # Group 1: The Translator (Main model) - Uses standard LR
        {'params': model.parameters(), 'lr': args.lr},
    ]

    if args.finetune_feat_extractor:
        print(f"[INFO] 🔓 Fine-tuning Feature Extractor with LR={args.lr_feat_extractor}")
        # Group 2: The Feature Extractor - Uses lower LR
        params.append({'params': feat_extractor.parameters(), 'lr': args.lr_feat_extractor})
        # Ensure it requires grad
        for param in feat_extractor.parameters():
            param.requires_grad = True
    else:
        print("[INFO] 🔒 Feature Extractor is FROZEN.")
        for param in feat_extractor.parameters():
            param.requires_grad = False
            
    optimizer = torch.optim.Adam(params)
    loss_fn = CompositeLoss(weights={
        "l1": args.l1_w, 
        "l2": args.l2_w, 
        "ssim": args.ssim_w, 
        "perceptual": args.perceptual_w
    }).to(device)    
    scaler = None
    # scaler = torch.amp.GradScaler()

    # 8. ======= Training Loop =======
    print(f"[DEBUG] 🚀 Training for {epochs} epochs...")
    start_time = time.time()
    
    # Pre-Training Sanity Check
    if args.sanity_check:
        run_validation(0, model, val_meta_list, device, feat_extractor, args, loss_fn, viz_limit=args.viz_limit, train_monitor_data=train_monitor_data)
    
    epoch_iter = tqdm(range(1, epochs + 1), desc="Epochs", leave=True, dynamic_ncols=True)
    for epoch in epoch_iter:
        epoch_start = time.time()
        
        loss, loss_comps, grad_norm = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, args.model_type, feat_extractor, args)

        epoch_duration = time.time() - epoch_start
        cumulative_time = time.time() - start_time
        epoch_iter.set_postfix({"train_loss": f"{loss:.5f}"})
        
        if args.wandb:
            log = {
                "train_loss/total": loss,
                "info/grad_norm": grad_norm,
                "info/epoch_duration": epoch_duration, 
                "info/cumulative_time": cumulative_time
            }
            for k, v in loss_comps.items(): log[k.replace("loss_", "train_loss/")] = v
            wandb.log(log, step=epoch)

        if (epoch % args.val_interval == 0) or (epoch == epochs):
            run_validation(epoch, model, val_meta_list, device, feat_extractor, args, loss_fn, avg_train_loss=loss, viz_limit=args.viz_limit, train_monitor_data=train_monitor_data)

        if epoch % args.model_save_interval == 0:
            ckpt_name = f"{args.model_type}_epoch{epoch}_{datetime.datetime.now():%Y%m%d_%H%M}.pt"
            ckpt_path = os.path.join(args.root_dir, "results", "models", "checkpoints", ckpt_name)
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")
    
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
    "data_root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/Coltea-Lung-CT-100W/data_nifti_3mm",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    # "wandb": False,
    # "project_name": "mri2ct-native-venous",
    "project_name": "mri2ct-native-arterial",
    "viz_limit": 2,
    # "res_mult": 32,
    "anatomix_weights": "new",
    
    # Data
    "subjects": ["1ABA005_3.0x3.0x3.0_resampled", "1ABA005_3.0x3.0x3.0_resampled"], # None for all
    # "subjects": None
    "region": None, # "AB", "TH", "HN"
    "val_split": 0.1,
    # "augment": True,
    "augment": False,
    "patch_size": 96,
    # "patches_per_volume": 100,
    "patches_per_volume": 20,
    "data_queue_max_length": 300,
    "min_clip": -1000,
    "max_clip": 1000,

    # Training
    "lr": 3e-4,
    "val_interval": 1,
    "model_save_interval": 50,
    "sanity_check": True,
    "accum_steps": 32,
    "steps_per_epoch": 1000,
    "finetune_feat_extractor": False,
    "lr_feat_extractor": 1e-5,
    
    # Model Choice
    "model_type": "mlp", # "mlp" or "cnn"
    "epochs_mlp": 500,
    "epochs_cnn": 200,
    "dropout": 0.2,
    
    # 일단 이게 제일 좋은 듯
    # MLP Specifics
    "mlp_batch_size": 131072, # Points to sample per patch (feature vector batch)
    "fourier": True,
    "sigma": 5.0,
    "mlp_depth": 6, 
    "mlp_hidden": 512,
    
    # CNN Specifics
    "cnn_batch_size": 1,
    "cnn_depth": 9,
    "cnn_hidden": 128,
    "final_activation": "sigmoid",
    "enable_safety_padding": True,
    
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 1.0,
    "perceptual_w": 0.0, # 0.0 to disable
}


if __name__ == "__main__":
    # Define experiments here
    experiments = [
        {
            "model_type": "cnn",
            # "cnn_depth": 5,
            # "cnn_hidden": 64,
            "cnn_depth": 9,
            "cnn_hidden": 128,
            # "epochs_cnn": 200,
            "epochs_cnn": 500,
            "subjects": None,
            # "region": "AB",
            "dropout": 0,
            # "dropout": 0.2,
            "patches_per_volume": 30,
            "augment": True,
            # "augment": False,
            "perceptual_w": 0.1,
            "min_clip": -450,
            "max_clip": 450,
            "cnn_batch_size": 4,
            "accum_steps": 1,
            "model_save_interval": 25,
            "steps_per_epoch": 50,
            
            "anatomix_weights": "prev", # "new", "prev"
        },
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