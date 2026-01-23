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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.axes_grid1 import make_axes_locatable 
import matplotlib.pyplot as plt
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio

from fused_ssim import fused_ssim
from anatomix.model.network import Unet
from models import CNNTranslator

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm3d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += residual 
        out = self.relu(out)
        return out

class CNNTranslator(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=64, depth=5, final_activation="sigmoid", dropout=0.0):
        super().__init__()
        print(f"[DEBUG] 🏗️ Building Robust CNNTranslator | Depth: {depth} | Hidden: {hidden_channels}")
        
        # 1. Entry Block
        self.entry = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.InstanceNorm3d(hidden_channels),
            nn.ReLU(inplace=True)
        )
        
        # 2. Residual Body
        # Depth determines how many ResBlocks we stack
        res_blocks = []
        for _ in range(depth):
            res_blocks.append(ResidualBlock(hidden_channels))
        self.body = nn.Sequential(*res_blocks)
        
        # 3. Exit Block
        self.exit_conv = nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1)
        
        self.final_act = final_activation

    def forward(self, x):
        x = self.entry(x)
        x = self.body(x)
        x = self.exit_conv(x)
        
        if self.final_act == "sigmoid":
            return torch.sigmoid(x)
        elif self.final_act == "relu_clamp":
            return torch.clamp(torch.relu(x), 0, 1)
        return x

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
    if not ct: raise FileNotFoundError(f"CT missing for {subj_id}")
    # mr = glob(os.path.join(data_path, subj_id, "registration_output", "moved_*.nii*"))
    mr = None
    return {'ct': ct[0], 'mri': mr[0] if mr else None}

def load_image_pair(root, subj_id, args):
    """Validation Loading"""
    paths = get_subject_paths(root, subj_id)
    ct_img = tio.ScalarImage(paths['ct'])
    
    # Normalize CT
    ct = minmax(ct_img.data[0], minclip=-450, maxclip=450).numpy()
    orig_shape = ct.shape
    
    # Pad to multiple of 32
    target_shape = [max(args.patch_size, (d + 31) // 32 * 32) for d in orig_shape]
    pad_width = [(0, t - o) for t, o in zip(target_shape, orig_shape)]
    ct_padded = np.pad(ct, pad_width, mode='constant', constant_values=0)
    
    return None, ct_padded, orig_shape

def discover_subjects(data_dir, target_list=None, region=None): # region: AB, TH, HN
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
        # 1. Normalize CT (Target & Input for Stage A)
        ct_data = subject['ct'].data
        subject['ct'].set_data(minmax(ct_data, -450, 450).float())

        if self.enable_safety_padding:
            pad_amount = self.patch_size // 2
            safety_padder = tio.Pad(pad_amount, padding_mode=0)
            subject = safety_padder(subject)
        
        # 2. Smart Padding
        current_shape = subject['ct'].spatial_shape
        subject['original_shape'] = torch.tensor(current_shape)
        
        target_shape = []
        for dim in current_shape:
            # Round up to multiple of 32
            mult_32 = (int(dim) + 31) // 32 * 32
            target_shape.append(max(self.patch_size, mult_32))
            
        padding_params = []
        for curr, targ in zip(current_shape, target_shape):
            padding_params.extend([0, int(targ - curr)])
            
        if any(p > 0 for p in padding_params):
            padder = tio.Pad(padding_params, padding_mode=0)
            subject = padder(subject)

        subject['vol_shape'] = torch.tensor(subject['ct'].spatial_shape).float()
        
        # 3. Probability Map (Focus on body, ignore air)
        prob = (subject['ct'].data > 0.01).to(torch.float32)
        subject.add_image(tio.LabelMap(tensor=prob, affine=subject['ct'].affine), 'prob_map')
            
        return subject

def get_augmentations_stage_a():
    # NOTE: Stage A is Reconstruction.
    # We apply Geometric transforms (Spatial) to ensure the decoder handles
    # rotated/shifted features correctly.
    # We DO NOT apply intensity transforms because we want to learn exact reconstruction.
    spatial_transforms = tio.Compose([
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
    ])
    return spatial_transforms

def get_dataloader(data_path_list, args):
    subjects = []
    for item in data_path_list:
        sub_kwargs = {'ct': tio.ScalarImage(item['ct'])}
        if item['mri']: sub_kwargs['mri'] = tio.ScalarImage(item['mri'])
        subjects.append(tio.Subject(**sub_kwargs))  

    preprocess = ProjectPreprocessing(patch_size=args.patch_size)    
    augment = get_augmentations_stage_a() if args.augment else None
    
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
# 3. VISUALIZATION (STAGE A)
# ==========================================
@torch.no_grad()
def visualize_reconstruction(pred_ct, gt_ct, subj_id, root_dir, epoch=None, use_wandb=False, idx=1):
    """
    Visualizes GT vs Reconstruction for Stage A.
    Uses make_axes_locatable for robust colorbar alignment.
    """
    residual = pred_ct - gt_ct
    
    # Pick slices
    D = gt_ct.shape[2]
    slice_indices = np.linspace(0.2 * D, 0.8 * D, 3, dtype=int)
    
    # NOTE: increased figsize height slightly
    fig, axes = plt.subplots(len(slice_indices), 3, figsize=(12, 5 * len(slice_indices)))
    if len(slice_indices) == 1: axes = axes[None, :] 
    
    for i, z in enumerate(slice_indices):
        # 1. GT
        axes[i, 0].imshow(gt_ct[:, :, z], cmap='gray', vmin=0, vmax=1)
        if i == 0: axes[i, 0].set_title("GT CT", fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')
        
        # 2. Pred
        axes[i, 1].imshow(pred_ct[:, :, z], cmap='gray', vmin=0, vmax=1)
        if i == 0: axes[i, 1].set_title("Reconstructed CT", fontsize=12, fontweight='bold')
        axes[i, 1].axis('off')
        
        # 3. Residual
        im = axes[i, 2].imshow(residual[:, :, z], cmap='seismic', vmin=-0.5, vmax=0.5)
        if i == 0: axes[i, 2].set_title("Residual (Pred - GT)", fontsize=12, fontweight='bold')
        axes[i, 2].axis('off')
        
        divider = make_axes_locatable(axes[i, 2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("Error (HU)")
        
    fig.suptitle(f"Reconstruction Check: {subj_id} (Ep {epoch})", fontsize=16, y=0.98)
    # NOTE: Removed tight_layout() to prevent "Incompatible Axes" warning
    # plt.tight_layout() 
    
    save_dir = os.path.join(root_dir, "results", "vis_stage_a")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subj_id}_ep{epoch}.png")
    
    # NOTE: bbox_inches='tight' does the trimming safely at save time
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    if use_wandb:
        wandb.log({f"recon_viz/val_{idx}": wandb.Image(save_path)}, step=epoch)

# ==========================================
# 4. TRAINING & EVAL LOOP (STAGE A)
# ==========================================
class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target):
        total = 0.0
        comps = {}
        
        if self.weights['l1'] > 0:
            l1 = self.l1(pred, target)
            total += self.weights['l1'] * l1
            comps['l1'] = l1.item()
            
        if self.weights['l2'] > 0:
            l2 = self.l2(pred, target)
            total += self.weights['l2'] * l2
            comps['l2'] = l2.item()
            
        if self.weights['ssim'] > 0:
            # Helper for 5D SSIM
            b, c, d, h, w = pred.shape
            p2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            t2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
            val_ssim = fused_ssim(p2d, t2d, train=True)
            loss_ssim = 1.0 - val_ssim
            total += self.weights['ssim'] * loss_ssim
            comps['ssim'] = loss_ssim.item()

        if self.weights.get('bone', 0) > 0:
            bone_mask = (target > 0.8).float()
            
            # Calculate L1 error ONLY on bone voxels
            diff = torch.abs(pred - target)
            bone_diff = diff * bone_mask
            
            # Avoid division by zero if no bone in patch
            num_bone_voxels = torch.sum(bone_mask)
            if num_bone_voxels > 0:
                loss_bone = torch.sum(bone_diff) / num_bone_voxels
            else:
                loss_bone = torch.tensor(0.0, device=pred.device)
                
            total += self.weights['bone'] * loss_bone
            comps['bone'] = loss_bone.item()
            
        return total, comps

def train_recon_epoch(decoder, extractor, loader, opt, loss_fn, device, args):
    decoder.train()
    extractor.eval()
    
    total_loss = 0
    total_grad_norm = 0.0 

    progress = tqdm(range(args.steps_per_epoch), desc="Recon Train", leave=False)
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
            
            ct = batch['ct'][tio.DATA].to(device)
            
            # 1. Extract Features from CT (Ground Truth Features)
            with torch.no_grad():
                z_ct = extractor(ct)
            
            # 2. Decode back to Image
            rec_ct = decoder(z_ct)
            
            # 3. Loss (Compare Reconstruction vs Input CT)
            loss, _ = loss_fn(rec_ct, ct)
            loss = loss / args.accum_steps
            loss.backward()
            step_loss += loss.item()
            
        norm_val = torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        total_grad_norm += norm_val.item()
        
        opt.step()
        total_loss += step_loss
        progress.set_postfix({'loss': step_loss})
    
    avg_loss = total_loss / args.steps_per_epoch
    avg_grad_norm = total_grad_norm / args.steps_per_epoch
    
    return avg_loss, avg_grad_norm

@torch.no_grad()
def validate_recon(decoder, extractor, val_list, device, loss_fn, epoch, args):
    decoder.eval()
    extractor.eval()
    
    metrics = {'mae': [], 'psnr': [], 'ssim': []}
    val_running_loss = 0.0 
    
    for idx, item in enumerate(val_list):
        # 1. Prepare Full Volume CT
        ct_np = item['ct']
        ct_tensor = torch.from_numpy(ct_np)[None, None].float().to(device)
        
        # 2. Forward Pass
        z_ct = extractor(ct_tensor)
        rec_ct_tensor = decoder(z_ct)
        
        # Compute Validation Loss on full volume
        loss_val, _ = loss_fn(rec_ct_tensor, ct_tensor)
        val_running_loss += loss_val.item()

        # 3. Unpad
        orig = item['orig_shape']
        rec_unpad = unpad(rec_ct_tensor, orig)
        gt_unpad = unpad(ct_tensor, orig)
        
        # 4. Metrics
        m, p, s = compute_metrics(rec_unpad, gt_unpad)
        metrics['mae'].append(m)
        metrics['psnr'].append(p)
        metrics['ssim'].append(s)
        
        # 5. Visualize (First few only)
        if idx < args.viz_limit:
            print(f"[DEBUG] 📸 Visualizing Recon {item['id']}")
            visualize_reconstruction(
                rec_unpad.cpu().squeeze().numpy(), 
                gt_unpad.cpu().squeeze().numpy(), 
                item['id'], args.root_dir, epoch, args.wandb, idx
            )
            
    avg_mae = np.mean(metrics['mae'])
    avg_ssim = np.mean(metrics['ssim'])
    avg_val_loss = val_running_loss / len(val_list) 
    
    print(f"✅ Ep {epoch} | Val Loss: {avg_val_loss:.5f} | Val MAE: {avg_mae:.4f} | SSIM: {avg_ssim:.4f}")
    
    if args.wandb:
        wandb.log({
            "val/loss": avg_val_loss,
            "val/mae": avg_mae, 
            "val/ssim": avg_ssim, 
            "val/psnr": np.mean(metrics['psnr'])
        }, step=epoch)

# ==========================================
# 5. MAIN
# ==========================================
def run_stage_a(config_dict):
    cleanup_gpu()
    args = Config(config_dict)
    set_seed(args.seed)
    device = torch.device(args.device)
    
    print(f"\n[INFO] 🏁 Starting Stage A: CT Reconstruction Sanity Check")
    
    # 1. Data Setup
    # subjects = [d for d in os.listdir(os.path.join(args.root_dir, "data")) 
    #             if os.path.isdir(os.path.join(args.root_dir, "data", d))]
    # if args.subjects: subjects = args.subjects
    subjects = discover_subjects(os.path.join(args.root_dir, "data"), target_list=args.subjects, region=args.region)
    if not subjects:
        print("[ERROR] No subjects found.")
        return
    
    # Simple Split
    random.shuffle(subjects)
    val_len = max(1, int(len(subjects) * args.val_split))
    train_sub = subjects[:-val_len]
    val_sub = subjects[-val_len:]
    print(f"[INFO] Train: {len(train_sub)} | Val: {len(val_sub)}")
    
    # 2. Preload Validation Data
    val_list = []
    print("[INFO] Preloading Validation volumes...")
    for s in tqdm(val_sub):
        try:
            _, ct, orig = load_image_pair(args.root_dir, s, args)
            val_list.append({'id': s, 'ct': ct, 'orig_shape': orig})
        except Exception as e:
            print(f"[WARN] Failed to load {s}: {e}")
            
    # 3. Models
    print("[INFO] Loading Anatomix (Frozen)...")
    feat_extractor = Unet(
        dimension=3, input_nc=1, output_nc=16, num_downs=5, ngf=20, 
        norm="instance", interp="trilinear", pooling="Avg"
    ).to(device)
    feat_extractor = torch.compile(feat_extractor, mode="default")
    
    ckpt = os.path.join(args.root_dir, "anatomix", "model-weights", "best_val_net_G.pth")
    feat_extractor.load_state_dict(torch.load(ckpt, map_location=device))
    feat_extractor.eval()
    for p in feat_extractor.parameters(): p.requires_grad = False
    
    print(f"[INFO] Building Decoder (CNNTranslator)...")
    # NOTE: Using CNNTranslator as Decoder: 16 Input Channels -> 1 Output Channel
    decoder = CNNTranslator(
        in_channels=16, 
        hidden_channels=args.dec_hidden, 
        depth=args.dec_depth, 
        final_activation="sigmoid"
    ).to(device)
    
    opt = torch.optim.Adam(decoder.parameters(), lr=args.lr)
    loss_fn = CompositeLoss(weights={'l1': args.l1_w, 'l2': args.l2_w, 'ssim': args.ssim_w, 'bone': args.bone_w}).to(device)

    
    # 4. Loader
    train_paths = [get_subject_paths(args.root_dir, s) for s in train_sub]
    loader = get_dataloader(train_paths, args)
    
    # 5. W&B
    if args.wandb:
        wandb.init(project=args.project_name, name=f"CNN_Train{len(train_paths)}", config=vars(args), reinit=True)
        wandb.save(os.path.abspath(__file__))

    # 6. Loop
    best_ssim = 0.0
    for epoch in range(1, args.epochs + 1):
        loss, grad_norm = train_recon_epoch(decoder, feat_extractor, loader, opt, loss_fn, device, args)
        print(f"Ep {epoch} | Train Loss: {loss:.5f} | Grad: {grad_norm:.4f}")
        
        if args.wandb: 
            wandb.log({
                "train/loss": loss, 
                "train/grad_norm": grad_norm
            }, step=epoch)        
        if epoch % args.val_interval == 0:
            validate_recon(decoder, feat_extractor, val_list, device, loss_fn, epoch, args)
            
            # Save Checkpoint
            save_path = os.path.join(args.root_dir, "results", "models", "stage_a_decoder.pt")
            torch.save(decoder.state_dict(), save_path)
            
    print("[INFO] Stage A Complete.")
    if args.wandb: wandb.finish()

# ==========================================
# CONFIG & RUN
# ==========================================
DEFAULT_CONFIG = {
    "root_dir": "/home/minsukc/MRI2CT",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    "project_name": "mri2ct_stage_a",
    
    # Data
    "subjects": None, 
    "region": None,
    "val_split": 0.1,
    "patch_size": 96,
    "augment": True, # Spatial Only
    
    # Loader
    "patches_per_volume": 15,
    "data_queue_max_length": 300,
    "num_workers": 4,
    "batch_size": 1,
    
    # Model (Decoder)
    "dec_hidden": 64,
    "dec_depth": 3, 
    # "dec_hidden": 64,
    # "dec_depth": 5, 
    
    # Train
    "epochs": 100,
    # "epochs": 50,
    # "lr": 1e-3,
    "lr": 3e-4,
    # "steps_per_epoch": 200,
    "steps_per_epoch": 100,
    "accum_steps": 1,
    "val_interval": 1,
    "viz_limit": 3,
    "grad_clip": 1.0, 
    
    # Loss
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 1.0,
    # "ssim_w": 0.0,
    "bone_w": 0.1,
}

if __name__ == "__main__":
    run_stage_a(DEFAULT_CONFIG)