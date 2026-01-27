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
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import nibabel as nib
import wandb
import torchio as tio
from fused_ssim import fused_ssim
from monai.inferers import sliding_window_inference

from anatomix.model.network import Unet

# ==========================================
# 0. GLOBAL SETUP & UTILS
# ==========================================
# Enables TF32 for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

def set_seed(seed=42):
    print(f"[DEBUG] 🌱 Setting global seed to {seed}")
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
    # NOTE: Reshaping for 2D SSIM calculation
    pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
    
    ssim_val = fused_ssim(pred_2d, targ_2d, train=False).item()
    
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

def get_subject_paths(root, relative_path):
    """
    root: base directory (e.g., .../3.0x3.0x3.0mm)
    relative_path: 'train/1ABA005' or just '1ABA005' if using flat structure
    """
    # Construct full path
    subj_dir = os.path.join(root, relative_path)
    
    ct_path = os.path.join(subj_dir, "ct.nii.gz")
    mr_path = os.path.join(subj_dir, "registration_output", "moved_mr.nii.gz")
    
    # Fallback for checking existence
    if not os.path.exists(ct_path) or not os.path.exists(mr_path):
        raise FileNotFoundError(f"Missing files in {subj_dir}")
        
    return {'ct': ct_path, 'mri': mr_path}

class Config(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)

# ==========================================
# 1. MODELS & LOSS
# ==========================================
class CNNTranslator(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32, depth=3, final_activation="relu_clamp", dropout=0.0):
        super().__init__()
        self.final_activation = final_activation
        layers = []
        
        # Input
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        if dropout > 0: layers.append(nn.Dropout3d(p=dropout))
        
        # Hidden
        for _ in range(depth - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0: layers.append(nn.Dropout3d(p=dropout))
        
        # Output
        layers.append(nn.Conv3d(hidden_channels, 1, kernel_size=3, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        if self.final_activation == "sigmoid": return torch.sigmoid(x)
        elif self.final_activation == "relu_clamp": return torch.clamp(torch.relu(x), 0, 1)
        elif self.final_activation == "none": return x
        else: raise ValueError(f"Unknown activation: {self.final_activation}")

class CompositeLoss(nn.Module):
    def __init__(self, weights):
        super().__init__()
        self.weights = weights
        self.l1 = nn.L1Loss()
        self.l2 = nn.MSELoss()

    def forward(self, pred, target, feat_extractor=None, use_sliding_window=False):
        total_loss = 0.0
        loss_components = {}
        
        if self.weights.get("l1", 0) > 0:
            val = self.l1(pred, target)
            total_loss += self.weights["l1"] * val
            loss_components["loss_l1"] = val.item()
            
        if self.weights.get("l2", 0) > 0:
            val = self.l2(pred, target)
            total_loss += self.weights["l2"] * val
            loss_components["loss_l2"] = val.item()

        if self.weights.get("ssim", 0) > 0:
            b, c, d, h, w = pred.shape
            pred_2d = pred.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
            targ_2d = target.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w).float()
            val = 1.0 - fused_ssim(pred_2d, targ_2d, train=True)
            total_loss += self.weights["ssim"] * val
            loss_components["loss_ssim"] = val.item()

        if self.weights.get("perceptual", 0) > 0:
            if feat_extractor is None: 
                raise ValueError("Feat extractor missing for perceptual loss")
            if use_sliding_window:
                print("Skipping perceptual loss calculation during validation. NOTE: val loss will differ from train loss.")
            else:
                pred_feats = feat_extractor(pred)
                with torch.no_grad(): target_feats = feat_extractor(target)
                val = self.l1(pred_feats, target_feats)
                total_loss += self.weights["perceptual"] * val
                loss_components["loss_perceptual"] = val.item()
            
        return total_loss, loss_components

# ==========================================
# 2. DATA PIPELINE
# ==========================================
class DataPreprocessing(tio.Transform):
    def __init__(self, patch_size=96, enable_safety_padding=False, res_mult=32, **kwargs):        
        super().__init__(**kwargs)
        self.patch_size = patch_size
        self.enable_safety_padding = enable_safety_padding
        self.res_mult = res_mult

    def apply_transform(self, subject):
        subject['ct'].set_data(anatomix_normalize(subject['ct'].data, clip_range=(-450, 450)).float())
        subject['mri'].set_data(anatomix_normalize(subject['mri'].data, percentile_range=(0,99.99)).float())
        # subject['mri'].set_data(anatomix_normalize(subject['mri'].data).float())
        
        # Save original shape
        subject['original_shape'] = torch.tensor(subject['ct'].spatial_shape)
        
        pad_offset=0
        # Padding logic
        if self.enable_safety_padding:
            pad_val = self.patch_size//2
            subject = tio.Pad(pad_val, padding_mode=0)(subject)
            pad_offset=pad_val

        subject['pad_offset'] = pad_offset

        current_shape = subject['ct'].spatial_shape
        padding_params = []
        for dim in current_shape:
            target = max(self.patch_size, (int(dim) + self.res_mult - 1) // self.res_mult * self.res_mult)
            padding_params.extend([0, int(target - dim)])
            
        if any(p > 0 for p in padding_params):
            subject = tio.Pad(padding_params, padding_mode=0)(subject)
            
        # Probability Map for Sampler
        if 'prob_map' not in subject:
            prob = (subject['ct'].data > 0.01).to(torch.float32)
            subject.add_image(tio.LabelMap(tensor=prob, affine=subject['mri'].affine), 'prob_map')

        spatial_shape = subject['mri'].spatial_shape
        if any(d % self.res_mult != 0 for d in spatial_shape):
             print(f"[WARNING] Volume shape {spatial_shape} is not a multiple of {self.res_mult}!")
            
        return subject

# def get_augmentations():
#     return tio.Compose([
#         tio.Compose([
#             tio.RandomFlip(axes=(0, 1, 2), p=0.5),
#             tio.RandomAffine(scales=(0.9, 1.1), degrees=10, translation=5, p=0.5),
#             tio.RandomElasticDeformation(num_control_points=7, max_displacement=7, p=0.25),
#         ]), # Geometric (Both)
#         tio.Compose([
#             tio.RandomBiasField(p=0.5, include=['mri']), 
#             tio.RandomNoise(std=0.02, p=0.25, include=['mri']),
#             tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.5, include=['mri'])
#         ])  # Intensity (MRI only)
#     ])

def get_augmentations():
    return tio.Compose([
        # Applied to BOTH MRI and CT identically
        tio.OneOf({
            tio.RandomElasticDeformation(
                num_control_points=7, 
                max_displacement=4, 
                locked_borders=2, 
                image_interpolation='bspline' 
            ): 0.3, 
            tio.RandomAffine(
                scales=(0.95, 1.1), 
                degrees=7, 
                translation=4,
                default_pad_value='minimum'
            ): 0.7,
        }, p=0.8), 
        tio.RandomFlip(axes=(0, 1, 2), p=0.5),
        tio.Clamp(0, 1),
        
        tio.Compose([
            tio.RandomBiasField(coefficients=0.5, p=0.4),
            tio.RandomGamma(log_gamma=(-0.3, 0.3), p=0.4),
        ], include=['mri']) ,
        tio.Clamp(0, 1),
    ])
    
# ==========================================
# 3. TRAINER CLASS
# ==========================================
class Trainer:
    def __init__(self, config_dict):
        # 1. Config Setup
        self.cfg = Config(config_dict)
        set_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"[DEBUG] 🚀 Initializing Trainer on {self.device}")

        # 2. Setup Components
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()
        
        # 3. State Tracking
        self.start_epoch = 0
        self.global_start_time = None 
        self._load_resume()

    def _setup_wandb(self):
        if not self.cfg.wandb: return
        
        run_name = f"{self.cfg.model_type.upper()}_Train{len(self.train_subjects)}"
        os.makedirs(self.cfg.log_dir, exist_ok=True)
        
        print(f"[DEBUG] 📡 Initializing WandB: {run_name}")
        wandb.init(
            project=self.cfg.project_name, 
            name=run_name, 
            config=vars(self.cfg),
            notes=self.cfg.wandb_note,
            reinit=True,
            dir=self.cfg.log_dir,
            id=self.cfg.resume_wandb_id, 
            resume="allow",
        )
        if not self.cfg.resume_wandb_id:
            current_code_dir = os.path.dirname(os.path.abspath(__file__))
            wandb.run.log_code(root=current_code_dir, include_fn=lambda path: path.endswith(".py"))

    def _setup_data(self):
        print(f"[DEBUG] 📂 Searching for data in: {self.cfg.root_dir}")
        
        # Helper to scan a folder
        def scan_split(split_name):
            split_dir = os.path.join(self.cfg.root_dir, split_name)
            if not os.path.exists(split_dir): return []
            return sorted([
                os.path.join(split_name, d) # Store as relative path 'train/1ABA005'
                for d in os.listdir(split_dir) 
                if os.path.isdir(os.path.join(split_dir, d))
            ])

        train_candidates = scan_split("train")
        val_candidates = scan_split("val")
        
        # Logic for 'subjects' (Single Image Optimization)
        if self.cfg.subjects:
            print(f"[DEBUG] 🎯 Filtering specific subjects: {self.cfg.subjects}")
            # Filter candidates that end with the requested ID
            # e.g., if requested '1ABA005', match 'train/1ABA005'
            self.train_subjects = [c for c in train_candidates + val_candidates if os.path.basename(c) in self.cfg.subjects]
            self.val_subjects = self.train_subjects # Validate on the same subject for overfitting
        else:
            # Standard Mode: Use the existing splits
            self.train_subjects = train_candidates
            self.val_subjects = val_candidates
        
        print(f"[DEBUG] 📊 Data Split - Train: {len(self.train_subjects)} | Val: {len(self.val_subjects)}")

        if self.cfg.analyze_shapes:
            shapes = []
            for s in tqdm(self.train_subjects[:30], desc="Analyzing Shapes (Sample)"):
                try:
                    p = get_subject_paths(self.cfg.root_dir, s)
                    sh = nib.load(p['mri']).header.get_data_shape()
                    shapes.append(sh)
                except Exception: pass
            
            if shapes:
                avg_shape = np.mean(np.array(shapes), axis=0).astype(int)
                print(f"📊 Mean Volume Shape: {tuple(int(x) for x in avg_shape)}")
        
        # 3. Helper to create paths
        def _make_subj_list(subjs):
            return [tio.Subject(
                mri=tio.ScalarImage(p['mri']), 
                ct=tio.ScalarImage(p['ct']),
                subj_id=os.path.basename(s) # Extract just ID for logging
            ) for s in subjs for p in [get_subject_paths(self.cfg.root_dir, s)]]

        # 5. Train Loader (Queue)
        train_objs = _make_subj_list(self.train_subjects)
        use_safety = (self.cfg.model_type.lower() == "cnn" and self.cfg.enable_safety_padding)
        
        preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=use_safety, res_mult=self.cfg.res_mult)
        transforms = tio.Compose([preprocess, get_augmentations()]) if self.cfg.augment else preprocess
        
        train_ds = tio.SubjectsDataset(train_objs, transform=transforms)
        sampler = tio.WeightedSampler(patch_size=self.cfg.patch_size, probability_map='prob_map')
        
        queue = tio.Queue(
            subjects_dataset=train_ds,
            samples_per_volume=self.cfg.patches_per_volume,
            max_length=max(self.cfg.patches_per_volume, self.cfg.data_queue_max_length),
            sampler=sampler,
            num_workers=self.cfg.data_queue_num_workers,
            shuffle_patches=True,
            shuffle_subjects=True
        )
        self.train_loader = tio.SubjectsLoader(queue, batch_size=self.cfg.batch_size, num_workers=0)
        
        # Create infinite iterator
        def _inf_gen(loader):
            while True:
                for batch in loader: yield batch
        self.train_iter = _inf_gen(self.train_loader)

        # 6. Val Loader (Full Volume)
        val_objs = _make_subj_list(self.val_subjects)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess) 
        self.val_loader = torch.utils.data.DataLoader(val_ds, batch_size=1, shuffle=False)

    def _setup_models(self):
        # 1. Anatomix (Feature Extractor)
        print(f"[DEBUG] 🏗️ Building Anatomix ({self.cfg.anatomix_weights})...")
        if self.cfg.anatomix_weights == "v1":
            self.cfg.res_mult = 16 
            self.feat_extractor = Unet(3, 1, 16, 4, 16).to(self.device)
            # ckpt = os.path.join(self.cfg.root_dir, "anatomix", "model-weights", "anatomix.pth")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
        elif self.cfg.anatomix_weights == "v2":
            self.cfg.res_mult = 32
            self.feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(self.device)
            # Optimize inference speed
            self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
            # ckpt = os.path.join(self.cfg.root_dir, "anatomix", "model-weights", "best_val_net_G.pth")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G.pth"
        else:
            raise ValueError("Invalid anatomix_weights")
            
        if os.path.exists(ckpt):
            self.feat_extractor.load_state_dict(torch.load(ckpt, map_location=self.device), strict=True)
            print(f"[DEBUG] Loaded Anatomix weights from {ckpt}")
        else:
            print(f"[WARNING] ⚠️ Anatomix weights NOT FOUND at {ckpt}")

        if not self.cfg.finetune_feat_extractor:
            for p in self.feat_extractor.parameters(): p.requires_grad = False
            self.feat_extractor.eval()
        
        # 2. CNN Translator
        print(f"[DEBUG] 🏗️ Building CNN (D={self.cfg.cnn_depth}, H={self.cfg.cnn_hidden})...")
        model = CNNTranslator(
            in_channels=16,
            hidden_channels=self.cfg.cnn_hidden,
            depth=self.cfg.cnn_depth,
            final_activation=self.cfg.final_activation,
            dropout=self.cfg.dropout
        ).to(self.device)
        # self.model = torch.compile(model, mode="reduce-overhead")
        # self.model = torch.compile(model, mode="default")
        if self.cfg.model_compile_mode:
            print(f"[DEBUG] 🚀 Compiling model with mode: {self.cfg.model_compile_mode}")
            self.model = torch.compile(model, mode=self.cfg.model_compile_mode)
        else:
            print(f"[DEBUG] 🐢 specific compile mode not set or None. Skipping compilation.")
            self.model = model

    def _setup_opt(self):
        params = [{'params': self.model.parameters(), 'lr': self.cfg.lr}]
        if self.cfg.finetune_feat_extractor:
            params.append({'params': self.feat_extractor.parameters(), 'lr': self.cfg.lr_feat_extractor})
            
        self.optimizer = torch.optim.Adam(params)
        
        # Auto-Pilot Scheduler
        # patience=10: Waits 10 validation checks (10 * 5 = 50 epochs) before dropping.
        # factor=0.5: When stuck, cuts LR in half.
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=5, 
            min_lr=1e-6,
        )

        self.loss_fn = CompositeLoss(weights={
            "l1": self.cfg.l1_w, "l2": self.cfg.l2_w, 
            "ssim": self.cfg.ssim_w, "perceptual": self.cfg.perceptual_w
        }).to(self.device)
        self.scaler = torch.cuda.amp.GradScaler()
        
    def _load_resume(self):
        if not self.cfg.resume_wandb_id: return
        
        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        run_folders = glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}"))
        if not run_folders:
            print("[RESUME] ❌ Run folder not found.")
            return

        all_ckpts = []
        for f in run_folders:
            ckpts = glob(os.path.join(f, "files", "*.pt"))
            all_ckpts.extend(ckpts)
            
        if not all_ckpts:
            print("[RESUME] ⚠️ No checkpoints found inside run folder.")
            return

        resume_path = max(all_ckpts, key=os.path.getmtime)
        print(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            self.start_epoch = checkpoint['epoch'] + 1
    
    def save_checkpoint(self, epoch, is_final=False):
        filename = f"{self.cfg.model_type}_{'FINAL' if is_final else f'epoch{epoch:05d}_{datetime.datetime.now():%Y%m%d_%H%M}'}.pt"
        save_dir = wandb.run.dir if self.cfg.wandb else os.path.join(self.cfg.root_dir, "results", "models")
        os.makedirs(save_dir, exist_ok=True)
        
        save_dict = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': vars(self.cfg)
        }
        
        if self.cfg.finetune_feat_extractor:
             save_dict['feat_extractor_state_dict'] = self.feat_extractor.state_dict()
             
        path = os.path.join(save_dir, filename)
        torch.save(save_dict, path)
        print(f"[SAVE] 💾 Checkpoint saved: {path}")
        if self.cfg.wandb:
            wandb.log({"info/checkpoint_path": path}, commit=False)
            
    # ==========================================
    # VISUALIZATION
    # ==========================================
    def _log_aug_viz(self, epoch):
        try:
            # 1. Get Data
            subj_id = self.val_subjects[0] 
            paths = get_subject_paths(self.cfg.root_dir, subj_id)
            
            subj = tio.Subject(mri=tio.ScalarImage(paths['mri']), ct=tio.ScalarImage(paths['ct']))
            prep = DataPreprocessing(patch_size=self.cfg.patch_size, res_mult=self.cfg.res_mult)
            subj = prep(subj)

            # 2. Augment
            aug = get_augmentations()(subj)
            hist_str = " | ".join([t.name for t in aug.history])

            # 3. Slice & Plot
            z = subj['mri'].shape[-1] // 2
            
            # NOTE: If aug changes shape, this line will crash.
            orig_sl = subj['mri'].data[0, ..., z].numpy()
            aug_sl = aug['mri'].data[0, ..., z].numpy()

            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(np.rot90(orig_sl), cmap='gray', vmin=0, vmax=1); ax[0].set_title(f"Original")
            ax[1].imshow(np.rot90(aug_sl), cmap='gray', vmin=0, vmax=1); ax[1].set_title(f"Augmented\n{hist_str}")
            
            # The simple Diff you wanted
            ax[2].imshow(np.rot90(aug_sl - orig_sl), cmap='seismic', vmin=-0.5, vmax=0.5); ax[2].set_title("Diff")
            
            wandb.log({"val/aug_viz": wandb.Image(fig)}, step=epoch)
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Aug Viz failed: {e}")

    
    @torch.no_grad()
    def _visualize_full(self, pred, ct, mri, feats_mri, subj_id, shape, epoch, idx, offset=0):
        """
        Full 8-column visualization with PCA, Cosine Sim, and Residuals.
        """
        # 1. Extract Features for Comparison
        def extract_np(vol_tensor):
            inp = vol_tensor.to(self.device)
            if inp.ndim == 4: inp = inp.unsqueeze(0) # Handle missing batch dim
            return self.feat_extractor(inp).squeeze(0).cpu().numpy()

        feats_gt = extract_np(ct)
        feats_pred = extract_np(pred)
        # feats_mri is already extracted, just convert to numpy
        feats_mri_np = feats_mri.squeeze(0).cpu().numpy()
        
        # 2. Unpad Volumes
        w, h, d = shape
        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()
        
        # 3. Unpad Features (C, W, H, D)
        feats_gt = feats_gt[..., offset:offset+w, offset:offset+h, offset:offset+d]
        feats_pred = feats_pred[..., offset:offset+w, offset:offset+h, offset:offset+d]
        feats_mri_np = feats_mri_np[..., offset:offset+w, offset:offset+h, offset:offset+d]

        C, H, W, D_dim = feats_gt.shape
        
        # 4. Define Items
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
        ]

        # 5. PCA Logic
        if self.cfg.viz_pca:
            def sample_vox(f, max_v=200_000):
                X = f.reshape(C, -1).T
                if X.shape[0] > max_v: X = X[np.random.choice(X.shape[0], max_v, replace=False)]
                return X
            
            X_all = np.concatenate([sample_vox(feats_mri_np), sample_vox(feats_gt), sample_vox(feats_pred)], axis=0)
            pca = PCA(n_components=3, svd_solver="randomized").fit(X_all)
            
            def proj(f):
                Y = pca.transform(f.reshape(C, -1).T)
                Y = (Y - Y.min(0, keepdims=True)) / (Y.max(0, keepdims=True) - Y.min(0, keepdims=True) + 1e-8)
                return Y.reshape(H, W, D_dim, 3)

            items.extend([
                (proj(feats_mri_np), "PCA (MRI)", None, None),
                (proj(feats_gt), "PCA (GT CT)", None, None),
                (proj(feats_pred), "PCA (Pred)", None, None),
            ])

        # 6. Cosine Similarity
        gt_t = torch.from_numpy(feats_gt).unsqueeze(0)
        pred_t = torch.from_numpy(feats_pred).unsqueeze(0)
        cos_sim = F.cosine_similarity(gt_t, pred_t, dim=1).squeeze(0).numpy()
        cos_sim_n = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
        items.append((cos_sim_n, "Cosine Sim", "plasma", (0,1)))

        # 7. Plotting
        num_cols = len(items)
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)
        
        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(4 * num_cols, 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        
        # Handle single row edge case
        if len(slice_indices) == 1: axes = axes.reshape(1, -1)

        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                if data.ndim == 3: # (H, W, D)
                    im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                    if title == "Residual": res_im = im
                    if title == "Cosine Sim": cos_im = im
                else: # (H, W, D, 3) RGB
                    ax.imshow(data[:, :, z_slice, :])
                
                if i == 0: ax.set_title(title)
                ax.axis("off")

        # Colorbars
        if 'res_im' in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")
        
        cbar2 = fig.colorbar(cos_im, ax=axes[:, num_cols-1], fraction=0.04, pad=0.01)
        cbar2.set_label("Cosine Similarity")

        fig.suptitle(f"Viz — {subj_id} (Ep {epoch})", fontsize=16, y=0.99)
        
        if self.cfg.wandb:
            wandb.log({f"viz/{'train' if idx==-1 else ('val_'+ str(idx))}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)

    @torch.no_grad()
    def _visualize_lite(self, pred, ct, mri, subj_id, shape, epoch, idx, offset=0):
        """
        Lightweight visualization: MRI, GT, Pred, Residual. 
        """
        # 1. Unpad Volumes
        w, h, d = shape
        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()
        
        # 2. Define Items (Standard 4-column view)
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
        ]

        # 3. Plotting
        D_dim = gt_ct.shape[-1]
        num_cols = len(items)
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)
        
        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        
        if len(slice_indices) == 1: axes = axes.reshape(1, -1)

        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                
                if title == "Residual": res_im = im
                if i == 0: ax.set_title(title)
                ax.axis("off")

        if 'res_im' in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")
        
        fig.suptitle(f"Viz LITE — {subj_id} (Ep {epoch})", fontsize=16, y=0.99)
        
        if self.cfg.wandb:
            wandb.log({f"viz/{('val_'+ str(idx))}": wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
    # Added method to visualize training patches
    @torch.no_grad()
    def _log_training_patch(self, mri, ct, pred, epoch, step):
        """
        Visualizes MRI, Prediction, and CT (GT) for the first patch in the batch.
        """
        # 1. Prepare Data (Batch 0, Channel 0)
        img_in = mri[0, 0].detach().cpu().float().numpy()
        img_gt = ct[0, 0].detach().cpu().float().numpy()
        img_pred = pred[0, 0].detach().cpu().float().numpy()
        
        # Center indices
        cx, cy, cz = np.array(img_in.shape) // 2

        # 3 Rows (MRI, Pred, CT), 3 Cols (Axial, Sagittal, Coronal)
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        # Helper to plot a row
        def plot_row(row_idx, vol, title_prefix, vmin=None, vmax=None):
            # Axial
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{title_prefix} Ax")
            # Sagittal
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"{title_prefix} Sag")
            # Coronal
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 2].set_title(f"{title_prefix} Cor")
            
            # Add stats text to the left of the row
            axes[row_idx, 0].text(-5, 10, f"Min: {vol.min():.2f}\nMax: {vol.max():.2f}", 
                                  fontsize=8, color='white', backgroundcolor='black')

        # Row 1: MRI (Auto-scaled intensity)
        plot_row(0, img_in, "MRI")

        # Row 2: Prediction (Fixed 0-1 range to match CT)
        plot_row(1, img_pred, "Pred", vmin=0, vmax=1)

        # Row 3: CT (Fixed 0-1 range)
        plot_row(2, img_gt, "GT CT", vmin=0, vmax=1)

        # Cleanup
        for ax in axes.flatten():
            ax.axis('off')
        
        plt.tight_layout()
        
        wandb.log({f"train/patch_viz": wandb.Image(fig)}, step=epoch)
        plt.close(fig)
        
    # ==========================================
    # CORE LOGIC
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        if self.cfg.finetune_feat_extractor: self.feat_extractor.train()
        else: self.feat_extractor.eval()
        
        total_loss = 0.0
        total_grad = 0.0
        comp_accum = {}
        
        pbar = tqdm(range(self.cfg.steps_per_epoch), desc=f"Train Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            
            for _ in range(self.cfg.accum_steps):
                batch = next(self.train_iter)
                mri = batch['mri'][tio.DATA].to(self.device, non_blocking=True)
                ct = batch['ct'][tio.DATA].to(self.device, non_blocking=True)


                # Use 'dtype=torch.bfloat16' for Ampere+ GPUs (3090, 4090, A100, A6000)
                # Use 'dtype=torch.float16' for older GPUs (2080, V100, Titan)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    if self.cfg.finetune_feat_extractor:
                        features = self.feat_extractor(mri)
                    else:
                        with torch.no_grad(): features = self.feat_extractor(mri)
                    
                    pred = self.model(features)
                    
                    if self.cfg.wandb and step_idx == 0:
                        self._log_training_patch(mri, ct, pred, epoch, step_idx)
                    
                    fe_ref = self.feat_extractor if self.cfg.perceptual_w > 0 else None
                    loss, comps = self.loss_fn(pred, ct, feat_extractor=fe_ref)
                    
                    for k, v in comps.items():
                        comp_accum[k] = comp_accum.get(k, 0.0) + (v / self.cfg.accum_steps)
                    
                    loss = loss / self.cfg.accum_steps
                    self.scaler.scale(loss).backward()
                    step_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                list(self.model.parameters()) + (list(self.feat_extractor.parameters()) if self.cfg.finetune_feat_extractor else []), 
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += step_loss
            total_grad += grad_norm.item()
            pbar.set_postfix({"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"})
            
        return total_loss / self.cfg.steps_per_epoch, \
               {k: v / self.cfg.steps_per_epoch for k, v in comp_accum.items()}, \
               total_grad / self.cfg.steps_per_epoch

    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        val_metrics = defaultdict(list)
        
        # 1. Validation Loop
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch['mri'][tio.DATA].to(self.device)
            ct = batch['ct'][tio.DATA].to(self.device)
            orig_shape = batch['original_shape'][0].tolist()
            subj_id = batch['subj_id'][0]
            pad_offset = batch['pad_offset'][0].item() if 'pad_offset' in batch else 0
            
            # Sliding Window (Lite) vs Full Volume (Standard)
            feats = None
            if self.cfg.val_sliding_window:
                def combined_forward(x):
                    return self.model(self.feat_extractor(x))

                pred = sliding_window_inference(
                    inputs=mri, 
                    roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size), 
                    sw_batch_size=self.cfg.val_sw_batch_size, 
                    predictor=combined_forward,
                    overlap=self.cfg.val_sw_overlap,
                    mode="gaussian",
                    device=self.device
                )
            else:
                feats = self.feat_extractor(mri)
                pred = self.model(feats)
            
            # Metrics
            pred_unpad = unpad(pred, orig_shape, pad_offset)
            ct_unpad = unpad(ct, orig_shape, pad_offset)
            met = compute_metrics(pred_unpad, ct_unpad)
            
            # Loss (Composite)
            l_val, _ = self.loss_fn(pred, ct, feat_extractor=self.feat_extractor, use_sliding_window = self.cfg.val_sliding_window)
            met['loss'] = l_val.item()
            
            for k, v in met.items():
                val_metrics[k].append(v)
            
            # Viz
            if self.cfg.wandb and i < self.cfg.viz_limit:
                if self.cfg.val_sliding_window:
                    self._visualize_lite(pred, ct, mri, subj_id, orig_shape, epoch, idx=i, offset=pad_offset)
                else:
                    self._visualize_full(pred, ct, mri, feats, subj_id, orig_shape, epoch, idx=i, offset=pad_offset)
        
        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
             self._log_aug_viz(epoch)

        # 3. Log
        avg_met = {k: np.mean(v) for k, v in val_metrics.items()}
        if self.cfg.wandb:
            wandb.log({f"val/{k}": v for k, v in avg_met.items()}, step=epoch)

        return avg_met

    def train(self):
        print(f"[DEBUG] 🏁 Starting Loop: Ep {self.start_epoch} -> {self.cfg.total_epochs}")
        self.global_start_time = time.time()
        
        if self.cfg.sanity_check and not self.cfg.resume_wandb_id:
            print("[DEBUG] running sanity check...")
            avg_met = self.validate(0)
            tqdm.write(
                    f"Ep -1 | Train: 0.0000 | Val: {avg_met.get('loss',0):.4f} | "
                    f"SSIM: {avg_met.get('ssim',0):.4f} | PSNR: {avg_met.get('psnr',0):.2f}"
                )

        global_pbar = tqdm(
            range(self.start_epoch, self.cfg.total_epochs),
            desc="🚀 Total Progress",
            initial=self.start_epoch,
            total=self.cfg.total_epochs,
            dynamic_ncols=True,
            unit="ep"
        )
            
        for epoch in global_pbar:
            ep_start = time.time()
            
            loss, comps, gn = self.train_epoch(epoch)
            
            if epoch % self.cfg.val_interval == 0 or (epoch+1) == self.cfg.total_epochs:
                avg_met = self.validate(epoch)
                val_loss = avg_met.get('loss', 0)

                self.scheduler.step(val_loss)
                
                tqdm.write(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {val_loss:.4f} | "
                    f"SSIM: {avg_met.get('ssim',0):.4f} | PSNR: {avg_met.get('psnr',0):.2f} | "
                    f"Bone: {avg_met.get('bone_dice',0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = time.time() - self.global_start_time

            if self.cfg.wandb:
                current_lr = self.optimizer.param_groups[0]['lr']
                log = {
                    "train_loss/total": loss, 
                    "info/grad_norm": gn, 
                    "info/epoch_duration": ep_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/lr": current_lr 
                }
                for k, v in comps.items(): log[k.replace("loss_", "train_loss/")] = v
                wandb.log(log, step=epoch)
                
            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)
                
        self.save_checkpoint(self.cfg.total_epochs, is_final=True)
        if self.cfg.wandb: wandb.finish()
        print(f"✅ Training Complete. Total Time: {time.time() - self.global_start_time:.1f}s")

# ==========================================
# 4. CONFIG & EXECUTION
# ==========================================
DEFAULT_CONFIG = {
    # System
    # "root_dir": "/home/minsukc/MRI2CT",
    "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/3.0x3.0x3.0mm", 
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    "project_name": "mri2ct",
    
    # Data
    "subjects": None,
    "region": None, # "AB", "TH", "HN"
    "val_split": 0.1,
    "augment": True,
    "patch_size": 96,
    "patches_per_volume": 40,
    "data_queue_max_length": 400,
    "data_queue_num_workers": 4,
    "anatomix_weights": "v2", # "v1", "v2"
    "res_mult": 32 ,
    "analyze_shapes": True, 
    
    # Training
    "lr": 3e-4,
    "val_interval": 1,
    "sanity_check": True,
    "accum_steps": 2,
    "model_save_interval": 50,
    "viz_limit": 6,
    "viz_pca": False,
    "steps_per_epoch": 25,
    "finetune_feat_extractor": False,
    "lr_feat_extractor": 1e-5,
    
    # Model Choice
    "model_type": "cnn",
    "model_compile_mode": None, # "default", "reduce-overhead", None
    "total_epochs": 5001,
    "dropout": 0,
    
    # CNN Specifics
    "batch_size": 4,
    "cnn_depth": 9,
    "cnn_hidden": 128,
    "final_activation": "sigmoid",
    "enable_safety_padding": True,

    # Sliding Window & Viz Options
    "val_sliding_window": True, 
    "val_sw_batch_size": 4, 
    "val_sw_overlap": 0.25, 
    
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 1.0,
    "perceptual_w": 0.0,

    "wandb_note": "test_run",
    "resume_wandb_id": None, 
}

EXPERIMENT_CONFIG = [
    {
        "total_epochs": 5000,
        "sanity_check": False,
        
        "accum_steps": 4,
        "batch_size": 4,
        "steps_per_epoch": 100,
        "val_interval": 1,
        "viz_limit": 10,
        
    
        # "wandb_note": "long_run_anatomix_v2",
        
        "anatomix_weights": "v1",
        "wandb_note": "long_run_anatomix_v1",
        
        # "resume_wandb_id": "l2rpr7g5", 
    },
]

if __name__ == "__main__":
    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")
    
    for i, exp in enumerate(EXPERIMENT_CONFIG):
        print(f"\n{'='*40}")
        print(f"STARTING EXPERIMENT {i+1}/{len(EXPERIMENT_CONFIG)}")
        print(f"Config: {exp}")
        print(f"{'='*40}\n")
        
        try:
            # Merge Configs
            conf = copy.deepcopy(DEFAULT_CONFIG)
            conf.update(exp)
            
            # Execute
            trainer = Trainer(conf)
            trainer.train()
            
            # Clean up
            del trainer
            cleanup_gpu()
            
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
            cleanup_gpu()
            break
        except Exception as e:
            print(f"❌ Experiment {i+1} Failed: {e}")
            import traceback
            traceback.print_exc()