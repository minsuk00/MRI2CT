import os
import glob
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchio as tio
from torch.utils.data import Dataset, DataLoader, TensorDataset
from skimage.metrics import peak_signal_noise_ratio as psnr2d
from skimage.metrics import structural_similarity as ssim2d
import random

# --- SSIM Setup ---
try:
    from fused_ssim import fused_ssim
    HAS_FUSED_SSIM = True
except ImportError:
    HAS_FUSED_SSIM = False
    print("⚠️ fused_ssim not found. Using native PyTorch fallback.")

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

def unpad_np(arr, pad_vals):
    pad_D, pad_H, pad_W = pad_vals
    s_d = slice(None, -pad_D) if pad_D > 0 else slice(None)
    s_h = slice(None, -pad_H) if pad_H > 0 else slice(None)
    s_w = slice(None, -pad_W) if pad_W > 0 else slice(None)
    return arr[s_d, s_h, s_w]

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
def compute_metrics_cpu(pred, target, pad_vals):
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
            
        ssims.append(ssim2d(t_slice, p_slice, data_range=1.0))
        
    return mae, np.mean(psnrs) if psnrs else 0.0, np.mean(ssims) if ssims else 0.0

def gaussian_window(window_size, sigma):
    gauss = torch.Tensor([np.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian_window(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim_native(img1, img2, window_size=11, size_average=True):
    channel = img1.size(1)
    window = create_window(window_size, channel).to(img1.device).type_as(img1)
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
    C1 = 0.01**2
    C2 = 0.03**2
    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

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
                
                if HAS_FUSED_SSIM:
                    pred32 = pred_2d.float()
                    targ32 = targ_2d.float()
                    ssim_score = fused_ssim(pred32, targ32, train=True)
                else:
                    ssim_score = ssim_native(pred_2d, targ_2d)
                
                val_ssim_loss = 1.0 - ssim_score
                total_loss += self.weights["ssim"] * val_ssim_loss
                loss_components["loss_ssim"] = val_ssim_loss.item()
        
        return total_loss, loss_components