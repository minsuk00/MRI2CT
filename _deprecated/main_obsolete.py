#!/usr/bin/env python3

import os
import sys
import glob
import gc
import time
import random
import argparse
import datetime
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torchio as tio
import wandb
from tqdm import tqdm

# --- External Dependencies (Assumed to exist in environment) ---
try:
    from fused_ssim import fused_ssim
    from anatomix.model.network import Unet
except ImportError:
    print("Warning: 'fused_ssim' or 'anatomix' not found. Ensure they are in your PYTHONPATH.")

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")

# ==========================================
# 0. GLOBAL CONFIG & CONSTANTS
# ==========================================
ROOT_DIR = "/home/minsukc/MRI2CT"
CKPT_PATH = os.path.join(ROOT_DIR, "anatomix", "model-weights", "anatomix.pth")
DATA_DIR = os.path.join(ROOT_DIR, "data")


# ==========================================
# 1. UTILITIES (IO, MATH, TOOLS)
# ==========================================
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
    if torch.cuda.is_available():
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

def load_image_pair(root, subj_id):
    ct_path = glob.glob(os.path.join(root, subj_id, "ct_resampled.nii*"))[0]
    mr_path = glob.glob(os.path.join(root, subj_id, "registration_output", "moved_*.nii*"))[0]
    
    mr_img, ct_img = tio.ScalarImage(mr_path), tio.ScalarImage(ct_path)
    mri, ct = mr_img.data[0].numpy(), ct_img.data[0].numpy()
    
    mri = minmax(mri)
    ct = minmax(ct, minclip=-450, maxclip=450)
    
    mri, pad_vals = pad_to_multiple_np(mri, multiple=16)
    ct, _ = pad_to_multiple_np(ct, multiple=16)
    
    # print(f"Data Loaded | MRI: {mri.shape}, CT: {ct.shape}")
    return mri, ct, pad_vals

def load_segmentation(root, subj_id, seg_filename="labels_moved*.nii.gz", pad_vals=None):
    paths_to_check = os.path.join(root, subj_id, "registration_output", seg_filename)
    seg_glob = glob.glob(paths_to_check)
    
    if not seg_glob:
        raise FileNotFoundError(f"Segmentation not found for {subj_id} in {paths_to_check}")

    seg_path = seg_glob[0]
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


# ==========================================
# 2. DATASETS & LOADERS
# ==========================================
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
        max_start = np.maximum(self.dims - self.patch_size, 0)
        
        start_d = np.random.randint(0, max_start[0] + 1)
        start_h = np.random.randint(0, max_start[1] + 1)
        start_w = np.random.randint(0, max_start[2] + 1)
        
        start = np.array([start_d, start_h, start_w])
        end = start + self.patch_size
        
        f_patch = self.feats[:, start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        t_patch = self.target[start[0]:end[0], start[1]:end[1], start[2]:end[2]]
        
        return torch.from_numpy(f_patch).float(), torch.from_numpy(t_patch).unsqueeze(0).float()

class StochasticVoxelDataset(Dataset):
    """
    Optimized MLP Dataset.
    Instead of 1 voxel per call, returns a CHUNK of voxels per call.
    """
    def __init__(self, data_list, dataset_len, chunk_size=4096):
        self.subjects = []
        self.weights = []
        self.chunk_size = chunk_size
        self.dataset_len = dataset_len # Virtual length
        
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
    Input: List of (Chunk, C) tensors -> Output: (Batch*Chunk, C) tensor
    """
    feats, coords, targs = zip(*batch)
    
    # Stack -> [Batch, Chunk, C] -> View -> [Batch*Chunk, C]
    feats = torch.stack(feats).view(-1, feats[0].shape[-1])
    coords = torch.stack(coords).view(-1, coords[0].shape[-1])
    targs = torch.stack(targs).view(-1, targs[0].shape[-1])
    
    return feats, coords, targs

class TioAdapter:
    def __init__(self, loader):
        self.loader = loader
        self.dataset = self 
    def __len__(self):
        return len(self.loader)
    def __iter__(self):
        for batch in self.loader:
            yield batch['input'][tio.DATA], batch['target'][tio.DATA]

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
        loader_batch_size = args.batch_size // chunk_size
        # We want steps_per_epoch update steps
        dataset_len = args.steps_per_epoch * loader_batch_size
        
        ds = StochasticVoxelDataset(data_list, dataset_len=dataset_len, chunk_size=chunk_size)
        
        return DataLoader(
            ds, 
            batch_size=loader_batch_size, 
            shuffle=True, 
            num_workers=4, 
            pin_memory=True,
            collate_fn=collate_flatten 
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
                feats = np.pad(feats, ((0,0), (pad_d1, pad_d2), (pad_h1, pad_h2), (pad_w1, pad_w2)), mode='constant') 
                target = np.pad(target, ((pad_d1, pad_d2), (pad_h1, pad_h2), (pad_w1, pad_w2)), mode='constant')

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
            max_length=max(args.patches_per_epoch, 30),
            samples_per_volume=args.patches_per_epoch,
            sampler=sampler,
            num_workers=0, # Faster for in-memory
            shuffle_subjects=True,
            shuffle_patches=True,
        )
        
        loader = DataLoader(queue, batch_size=1, num_workers=0, pin_memory=False)
        return TioAdapter(loader)


# ==========================================
# 3. MODELS
# ==========================================
class FourierFeatureMapping(nn.Module):
    """
    NeurIPS 2020: "Fourier Features Let Networks Learn High Frequency Functions..."
    """
    def __init__(self, input_dim=3, mapping_size=128, scale=10.0):
        super().__init__()
        self.mapping_size = mapping_size
        self.register_buffer('B', torch.randn((mapping_size, input_dim)) * scale)
        
    def forward(self, x):
        # x: [N, 3] -> x_proj: [N, M]
        x_proj = (2 * np.pi * x) @ self.B.T
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class MLPTranslator(nn.Module):
    def __init__(self, in_feat_dim=16, use_fourier=True, fourier_scale=10.0, hidden=256, out_dim=1):
        super().__init__()
        self.use_fourier = use_fourier
        
        if self.use_fourier:
            self.rff = FourierFeatureMapping(input_dim=3, mapping_size=128, scale=fourier_scale)
            combined_dim = in_feat_dim + 256
        else:
            combined_dim = in_feat_dim
        
        print(f"Combined feature dimension: {combined_dim}")
        
        self.net = nn.Sequential(
            nn.Linear(combined_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(), # Output 0-1
        )

    def forward(self, feats, coords):
        if self.use_fourier:
            pos_enc = self.rff(coords)
            x = torch.cat([feats, pos_enc], dim=1)
        else:
            x = feats
        return self.net(x)

class CNNTranslator(nn.Module):
    def __init__(self, in_channels=16, hidden_channels=32, depth=3, final_activation="relu_clamp"):
        """
        Args:
            depth (int): Total number of Conv3d layers.
            final_activation (str): "sigmoid", "relu_clamp", or "none".
        """
        super().__init__()
        self.final_activation = final_activation
        
        layers = []
        
        # --- 1. First Layer (Input -> Hidden) ---
        layers.append(nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        
        # --- 2. Middle Layers (Hidden -> Hidden) ---
        for _ in range(depth - 2):
            layers.append(nn.Conv3d(hidden_channels, hidden_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU(inplace=True))
            
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
# 4. LOSS FUNCTIONS
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


# ==========================================
# 5. VISUALIZATION
# ==========================================
@torch.no_grad()
def visualize_ct_feature_comparison(
    pred_ct, gt_ct, gt_mri, model, subj_id, 
    root_dir, epoch=None, use_wandb=False
):
    device = next(model.parameters()).device
    
    def extract_feats_np(volume_np):
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        feats = model(inp)
        return feats.squeeze(0).cpu().numpy()

    feats_gt = extract_feats_np(gt_ct)
    feats_pred = extract_feats_np(pred_ct)
    feats_mri = extract_feats_np(gt_mri)
    
    C, H, W, D = feats_gt.shape

    # --- PCA ---
    def sample_vox(feats, max_vox=200_000):
        X = feats.reshape(C, -1).T
        if X.shape[0] > max_vox:
            X = X[np.random.choice(X.shape[0], max_vox, replace=False)]
        return X

    X_both = np.concatenate([
        sample_vox(feats_mri),
        sample_vox(feats_gt),
        sample_vox(feats_pred)
    ], axis=0)

    pca = PCA(n_components=3, svd_solver="randomized").fit(X_both)

    def project_pca(feats):
        X = feats.reshape(C, -1).T
        Y = pca.transform(X)
        Y = (Y - Y.min(0, keepdims=True)) / (Y.max(0, keepdims=True) - Y.min(0, keepdims=True) + 1e-8)
        return Y.reshape(H, W, D, 3)

    pca_mri  = project_pca(feats_mri)
    pca_gt   = project_pca(feats_gt)
    pca_pred = project_pca(feats_pred)

    # --- Similarity & Residual ---
    gt_t = torch.from_numpy(feats_gt).unsqueeze(0)
    pred_t = torch.from_numpy(feats_pred).unsqueeze(0)
    
    cos_sim = F.cosine_similarity(gt_t, pred_t, dim=1).squeeze(0).numpy()
    cos_sim_n = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min() + 1e-8)
    
    residual = pred_ct - gt_ct

    # --- Plotting ---
    slice_indices = np.linspace(0.1 * D, 0.9 * D, 5, dtype=int)
    fig, axes = plt.subplots(len(slice_indices), 8, figsize=(30, 3.5 * len(slice_indices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    
    for i, z in enumerate(slice_indices):
        items = [
            (gt_mri, "GT MRI", "gray", (0,1)),
            (gt_ct, "GT CT", "gray", (0,1)),
            (pred_ct, "Pred CT", "gray", (0,1)),
            (residual, "Residual (Pred-GT)", "seismic", (-0.5, 0.5)),
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
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        wandb.log({"val/visualization": wandb.Image(save_path)}, step=epoch)


# ==========================================
# 6. ENGINE (TRAIN / EVAL)
# ==========================================
def train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, model_type):
    model.train()
    total_loss = 0.0
    comp_accumulator = {}
    
    if isinstance(loader, list):
        dataset_size = len(loader)
    elif hasattr(loader, 'dataset'):
        dataset_size = len(loader.dataset)
    else:
        dataset_size = len(loader)

    for batch in tqdm(loader, leave=False):
        if model_type == "mlp":
            feats, coords, yb = batch
            feats, coords, yb = feats.to(device), coords.to(device), yb.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                pred = model(feats, coords)
                loss, components = loss_fn(pred, yb)
                
        elif model_type == "cnn":
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            
            with torch.amp.autocast(device_type=device.type):
                pred = model(xb)
                loss, components = loss_fn(pred, yb)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        curr_batch_size = yb.size(0)
        total_loss += loss.item() * curr_batch_size
        
        for k, v in components.items():
            comp_accumulator[k] = comp_accumulator.get(k, 0.0) + (v * curr_batch_size)

    avg_loss = total_loss / dataset_size
    avg_components = {k: v / dataset_size for k, v in comp_accumulator.items()}

    return avg_loss, avg_components

@torch.no_grad()
def evaluate(model, feats_mri, ct, device, model_type, pad_vals):
    model.eval()

    if isinstance(ct, np.ndarray):
        ct_tensor = torch.from_numpy(ct).float().to(device)
    else:
        ct_tensor = ct.float().to(device)
    if ct_tensor.ndim == 3:
        ct_tensor = ct_tensor.unsqueeze(0).unsqueeze(0)

    
    if model_type == "mlp":
        C, D, H, W = feats_mri.shape
        feats_flat = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, C).float().to(device)
        
        z = torch.linspace(0, 1, D, device=device)
        y = torch.linspace(0, 1, H, device=device)
        x = torch.linspace(0, 1, W, device=device)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing='ij')
        coords = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)
        
        preds = []
        batch_size = 100_000
        for i in range(0, feats_flat.size(0), batch_size):
            f_batch = feats_flat[i:i+batch_size]
            c_batch = coords[i:i+batch_size]
            preds.append(model(f_batch, c_batch))
        
        pred_flat = torch.cat(preds, dim=0)
        pred_ct_tensor = pred_flat.reshape(1, 1, D, H, W)

    elif model_type == "cnn":
        feats_t = torch.from_numpy(feats_mri).unsqueeze(0).float().to(device)
        pred_ct_tensor = model(feats_t)

    pred_ct_tensor_unpad = unpad_torch(pred_ct_tensor, pad_vals)
    ct_tensor_unpad = unpad_torch(ct_tensor, pad_vals)
    metrics = compute_metrics(pred_ct_tensor_unpad, ct_tensor_unpad)

    pred_ct_tensor = pred_ct_tensor.squeeze().cpu().numpy()
    return metrics, pred_ct_tensor


# ==========================================
# 7. MAIN SCRIPT
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

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Basics
    parser.add_argument("-W", "--no_wandb", action="store_true", help="DISABLE W&B")
    parser.add_argument("-E", "--epochs", type=int)
    parser.add_argument("-V", "--val_interval", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch_size", type=int, default=131072)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file (optional)")
    
    # Subject Selection
    parser.add_argument("--subjects", nargs='+', default=None)
    parser.add_argument("--val_split", type=float, default=0.1, help="Fraction of subjects to use for validation")
    
    # Model
    parser.add_argument("-M", "--model_type", type=str, choices=["mlp", "cnn"], default="mlp")
    parser.add_argument("--no_fourier", action="store_true")
    parser.add_argument("--sigma", type=float, default=10.0)
    
    # Seg
    parser.add_argument("--use_seg", action="store_true")
    parser.add_argument("--seg_name", type=str, default="labels_moved.nii.gz")
    parser.add_argument("--seg_classes", type=int, default=60)

    # MLP specific (Defaults from yaml)
    parser.add_argument("--epochs_mlp", type=int, default=200)
    parser.add_argument("--steps_per_epoch", type=int, default=1000)

    # CNN specific (Defaults from yaml)
    parser.add_argument("--epochs_cnn", type=int, default=50)
    parser.add_argument("--patch_size", type=int, default=96)
    parser.add_argument("--patches_per_epoch", type=int, default=50)
    parser.add_argument("--cnn_depth", type=int, default=3)
    parser.add_argument("--cnn_hidden", type=int, default=32)
    parser.add_argument("--final_activation", type=str, default="relu_clamp")

    # Loss
    parser.add_argument("--l1_w", type=float, default=1.0)
    parser.add_argument("--l2_w", type=float, default=0.0)
    parser.add_argument("--ssim_w", type=float, default=1.0)
    
    args = parser.parse_args()
    
    if args.epochs is not None:
        print(f"⚙️ Epochs set via CLI override: {args.epochs}")
    elif args.model_type == "mlp":
        args.epochs = args.epochs_mlp
        print(f"⚙️ Epochs set via Config (MLP): {args.epochs}")
    elif args.model_type == "cnn":
        args.epochs = args.epochs_cnn
        print(f"⚙️ Epochs set via Config (CNN): {args.epochs}")
    else:
        args.epochs = 200
        print(f"⚠️ Epochs defaulted: {args.epochs}")
    
    return args

def main():
    from tqdm import tqdm
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🟢 Using device: {device}")

    # 1. Discover Subjects
    subjects = discover_subjects(DATA_DIR, target_list=args.subjects)
    
    if not subjects:
        print("❌ No subjects found.")
        return
    print(f"📋 Found {len(subjects)} total subjects.")

    # 2. Train/Val Split
    random.shuffle(subjects)
    num_val = max(1, int(len(subjects) * args.val_split))
    train_subjects = subjects[:-num_val]
    val_subjects = subjects[-num_val:]
    
    print(f"   Train: {len(train_subjects)} | Val: {len(val_subjects)}")
    print(f"   Validation Subjects: {val_subjects}")

    use_wandb = not args.no_wandb
    if use_wandb:
        run_name = f"MULTI_{args.model_type}_N{len(train_subjects)}"
        wandb.init(project="mri2ct_multi", name=run_name, config=vars(args))

    # 3. Initialize Feature Extractor
    print(f"🏗️ Loading Anatomix from {CKPT_PATH}...")
    feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
    if os.path.exists(CKPT_PATH):
        feat_extractor.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
    else:
        print("⚠️ Warning: Anatomix checkpoint not found!")
    feat_extractor.eval()

    # 4. Load Data (Train & Val)
    train_data_list = []
    val_meta_list = [] 

    # Helper to load and process one subject
    def process_subject(subj_id):
        try:
            mri, ct, pad_vals = load_image_pair(DATA_DIR, subj_id)
            with torch.no_grad():
                inp_mri = torch.from_numpy(mri[None, None]).float().to(device)
                feats = feat_extractor(inp_mri).squeeze(0).cpu().numpy() # [16, D, H, W]
            
            if args.use_seg:
                try:
                    seg = load_segmentation(DATA_DIR, subj_id, args.seg_name, pad_vals)
                    seg_one_hot = one_hot_encode(seg, num_classes=args.seg_classes)
                    feats = np.concatenate([feats, seg_one_hot], axis=0)
                except Exception as e:
                    print(f"⚠️ Seg failed for {subj_id}: {e}. Skipping subject.")
                    return None

            return {'id': subj_id, 'mri': mri, 'ct': ct, 'feats': feats, 'pad_vals': pad_vals}
        except Exception as e:
            print(f"❌ Error loading {subj_id}: {e}")
            return None

    # Load Train
    print("🚀 Pre-loading Training Data...")
    for subj_id in tqdm(train_subjects, desc="Train Loading"):
        data = process_subject(subj_id)
        if data:
            train_data_list.append((data['feats'], data['ct']))
    
    # Load Val
    print("🚀 Pre-loading Validation Data...")
    for subj_id in tqdm(val_subjects, desc="Val Loading"):
        data = process_subject(subj_id)
        if data:
            val_meta_list.append(data)

    cleanup_gpu()
    
    if not train_data_list:
        print("❌ No valid training data.")
        return

    # 5. Create Loader & Model
    loader = get_dataloader_multi(train_data_list, args)
    total_channels = train_data_list[0][0].shape[0]
    print(f"✅ Data Ready. Input Channels: {total_channels}")

    if args.model_type == "mlp":
        model = MLPTranslator(
            in_feat_dim=total_channels, 
            use_fourier=not args.no_fourier, 
            fourier_scale=args.sigma
        ).to(device)
    else:
        print(f"Building CNN: Depth={args.cnn_depth}, Hidden={args.cnn_hidden}, Act={args.final_activation}")
        model = CNNTranslator(
            in_channels=total_channels, 
            hidden_channels=args.cnn_hidden, 
            depth=args.cnn_depth, 
            final_activation=args.final_activation
        ).to(device)

    # 6. Optimizer & Loss
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = CompositeLoss(weights={
        "l1": args.l1_w, 
        "l2": args.l2_w, 
        "ssim": args.ssim_w
    }).to(device)
    scaler = torch.amp.GradScaler()

    # --- Validation Function ---
    def run_validation(epoch_idx, current_loss):
        """Runs evaluation on ALL validation subjects."""
        model.eval()
        val_metrics = {'mae': [], 'psnr': [], 'ssim': []}
        
        viz_done = False 

        for v_data in val_meta_list:
            try:
                # Note: evaluate returns UNPADDED metrics but PADDED prediction
                (mae, psnr, ssim), pred_ct_padded = evaluate(
                    model, v_data['feats'], v_data['ct'], device, args.model_type, pad_vals=v_data['pad_vals']
                )
                
                val_metrics['mae'].append(mae)
                val_metrics['psnr'].append(psnr)
                val_metrics['ssim'].append(ssim)
                
                # Visualize the FIRST validation subject
                if not viz_done:
                    print(f"   🔎 Val ({v_data['id']}): MAE={mae:.4f}, PSNR={psnr:.2f}")
                    if use_wandb:
                        visualize_ct_feature_comparison(
                            pred_ct_padded, v_data['ct'], v_data['mri'], feat_extractor, 
                            v_data['id'], ROOT_DIR, epoch=epoch_idx, use_wandb=True
                        )
                    viz_done = True
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    tqdm.write(f"⚠️ OOM during Validation of {v_data['id']}. Skipping.")
                    cleanup_gpu()
                else:
                    raise e
        
        # Log Average Metrics
        avg_mae = np.mean(val_metrics['mae']) if val_metrics['mae'] else 0.0
        avg_psnr = np.mean(val_metrics['psnr']) if val_metrics['psnr'] else 0.0
        avg_ssim = np.mean(val_metrics['ssim']) if val_metrics['ssim'] else 0.0
        
        print(f"Epoch {epoch_idx:03d} | Train Loss: {current_loss:.5f} | Val MAE: {avg_mae:.4f} | Val PSNR: {avg_psnr:.2f}")
        
        if use_wandb:
            wandb.log({
                "val/mae": avg_mae, 
                "val/psnr": avg_psnr, 
                "val/ssim": avg_ssim
            }, step=epoch_idx)

    # --- Pre-Training Sanity Check ---
    print("🎨 Running initial sanity check (Epoch 0)...")
    run_validation(0, 0.0)

    # 7. Training Loop
    start_time = time.time()
    print(f"🚀 Training for {args.epochs} epochs...")
    epoch_iter = tqdm(range(1, args.epochs + 1), desc="Epochs", leave=True, dynamic_ncols=True)
    
    for epoch in epoch_iter:
        loss, loss_comps = train_one_epoch(model, loader, optimizer, loss_fn, scaler, device, args.model_type)
        epoch_iter.set_postfix({"train_loss": f"{loss:.5f}"})
        
        if use_wandb:
            log = {"loss/total": loss}
            for k, v in loss_comps.items(): log[k.replace("loss_", "loss/")] = v
            wandb.log(log, step=epoch)

        # Periodic Validation
        if ((epoch) % args.val_interval == 0) or (epoch == args.epochs):
            run_validation(epoch, loss)

    # Save Final Model
    save_path = os.path.join(ROOT_DIR, "results", "models", f"multi_{args.model_type}_N{len(train_subjects)}_{datetime.datetime.now():%Y%m%d_%H%M}.pt")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"💾 Saved model to {save_path}")
    
    if use_wandb:
        wandb.finish()
    print(f"⏱️ Total: {time.time()-start_time:.2f}s")

if __name__ == "__main__":
    main()