#!/usr/bin/env python3
"""
MRI→CT translation using Anatomix features + MLP translator.
Trains on a single subject and logs progress to Weights & Biases.
"""

import argparse
import gc
import glob
import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from anatomix.model.network import Unet
from skimage.metrics import peak_signal_noise_ratio as psnr2d
from skimage.metrics import structural_similarity as ssim2d
from sklearn.decomposition import PCA
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb

import time

start_time = time.time()

# -----------------------------
# 0. Setup
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)
print(f"🟢 Using device: {device}")

ROOT_DIR = "/home/minsukc/MRI2CT"
CKPT_PATH = os.path.join(ROOT_DIR, "anatomix", "model-weights", "anatomix.pth")
DATA_DIR = os.path.join(ROOT_DIR, "data")

# CLI Args
parser = argparse.ArgumentParser(description="Train MRI→CT translator")
parser.add_argument(
    "-W", "--no_wandb", action="store_false", help="Enable Weights & Biases logging"
)
parser.add_argument("-E", "--epochs", type=int, default=200)
parser.add_argument("-V", "--val_interval", type=int, default=10)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--batch_size", type=int, default=131_072)
parser.add_argument("-S", "--subject", type=str, default="1ABA103_3x3x3_resampled")
parser.add_argument(
    "-M",
    "--model_type",
    type=str,
    default="mlp",
    choices=["mlp", "cnn"],
    help="Choose translation model: mlp or cnn",
)
parser.add_argument(
    "--posenc",
    type=str,
    default="fourier",
    choices=["none", "raw", "fourier"],
    help="Type of positional encoding: none | raw | fourier"
)


args = parser.parse_args()

USE_WANDB = args.no_wandb
EPOCHS = args.epochs
LR = args.lr
BATCH_SIZE = args.batch_size
SUBJ_ID = args.subject
MODEL_TYPE = args.model_type
VAL_INTERVAL = args.val_interval
POSENC = args.posenc

NOW = datetime.now().strftime("%Y%m%d_%H%M%S")
if USE_WANDB:
    wandb.init(
        project="mri2ct",
        name=f"run_{SUBJ_ID}_{NOW}",
        config=dict(
            epochs=EPOCHS,
            learning_rate=LR,
            batch_size=BATCH_SIZE,
            subject_id=SUBJ_ID,
            model="MLPTranslator",
            feature_extractor="Anatomix_Unet",
        ),
    )
    print("🟢 W&B logging enabled")
else:
    wandb = None
    print("⚪ W&B logging disabled (debug mode)")


# -----------------------------
# 1. Helper functions
# -----------------------------
def log_wandb(metrics: dict):
    """Safely log to wandb if enabled."""
    if USE_WANDB:
        wandb.log(metrics)


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
    return np.pad(arr,
                  ((0, pad_D), (0, pad_H), (0, pad_W)),
                  mode='constant'), (pad_D, pad_H, pad_W)

def unpad_np(arr, pad_vals):
    pad_D, pad_H, pad_W = pad_vals
    D_end = None if pad_D == 0 else -pad_D
    H_end = None if pad_H == 0 else -pad_H
    W_end = None if pad_W == 0 else -pad_W
    return arr[:D_end, :H_end, :W_end]

def fourier_encode(coords, num_bands=4):
    """
    coords: [N,3] in [0,1] range
    returns: [N, 3 * 2 * num_bands] Fourier features
    """
    coords = coords.unsqueeze(-1)  # [N,3,1]
    freq_bands = 2 ** torch.arange(num_bands, device=coords.device).float()  # [num_bands]
    freq_bands = freq_bands.view(1,1,-1)  # [1,1,num_bands]
    
    phases = coords * freq_bands * 2 * np.pi  # [N,3,num_bands] #TODO: multiply σ?
    sin = torch.sin(phases)
    cos = torch.cos(phases)
    return torch.cat([sin, cos], dim=-1).reshape(coords.size(0), -1)  # [N, 3 * 2 * num_bands]

def build_positional_encoding(ct, device, mode="fourier", num_bands=4):
    D, H, W = ct.shape
    
    z = torch.linspace(0,1,D, device=device)
    y = torch.linspace(0,1,H, device=device)
    x = torch.linspace(0,1,W, device=device)
    zz, yy, xx = torch.meshgrid(z,y,x, indexing='ij')
    pos = torch.stack([xx,yy,zz], dim=-1).reshape(-1,3)  # [N,3]

    if mode == "none":
        return None                      # no positional encoding

    if mode == "raw":
        return pos                        # [N,3]

    if mode == "fourier":
        return fourier_encode(pos, num_bands=num_bands)  # [N, 24]

    raise ValueError(f"Unknown pos encoding: {mode}")


def find_image(root, subj_id, name):
    base = os.path.join(root, subj_id)
    matches = glob.glob(os.path.join(base, f"{name}.nii*"))
    if not matches:
        raise FileNotFoundError(f"{name}.nii* not found for {subj_id}")
    return matches[0]


def load_image_pair(root, subj_id):
    ct_path = find_image(root, subj_id, "ct_resampled")
    # mr_path = find_image(root, subj_id, "mr_resampled")  # for unregistered mr
    mr_path = glob.glob(os.path.join(root, subj_id, "registration_output", "moved_*.nii*"))[0]  # for registered mr
    mr_img, ct_img = tio.ScalarImage(mr_path), tio.ScalarImage(ct_path)
    # mri, ct = mr_img.data[0].numpy(), ct_img.data[0].numpy()
    # mri, ct = minmax(mri), minmax(ct, minclip=-450, maxclip=450)
    # print("MRI shape:", mri.shape, "CT shape:", ct.shape)
    # return mri, ct
    mri, ct = mr_img.data[0].numpy(), ct_img.data[0].numpy()
    mri, ct = minmax(mri), minmax(ct, minclip=-450, maxclip=450)
    
    # pad both volumes to satisfy UNet stride multiples
    mri, pad_vals = pad_to_multiple_np(mri, multiple=16)
    ct, _        = pad_to_multiple_np(ct, multiple=16)  # use same shape behavior
    
    print("MRI padded shape:", mri.shape, "CT padded shape:", ct.shape)
    
    return mri, ct, pad_vals



def compute_metrics(pred, target):
    """pred/target: [H,W,D] in [0,1]"""
    mae = np.mean(np.abs(pred - target))
    psnrs, ssims = [], []
    for z in range(pred.shape[2]):
        psnrs.append(psnr2d(target[..., z], pred[..., z], data_range=1.0))
        ssims.append(ssim2d(target[..., z], pred[..., z], data_range=1.0))
    return mae, np.mean(psnrs), np.mean(ssims)


@torch.no_grad()
def visualize_ct_feature_comparison(
    pred_ct, gt_ct, gt_mri, model, subj_id, epoch=None, use_wandb=USE_WANDB
):
    """
    Visualize Anatomix PCA + similarity maps comparing GT and Predicted CT,
    with GT MRI added as reference.
    Automatically logs to W&B with epoch slider support.
    """
    device = next(model.parameters()).device

    # --- Extract Anatomix features ---
    def extract_feats_np(volume_np):
        inp = torch.from_numpy(volume_np[None, None]).float().to(device)
        feats = model(inp)
        return feats.squeeze(0).cpu().numpy()  # [C,H,W,D]

    feats_gt = extract_feats_np(gt_ct)
    feats_pred = extract_feats_np(pred_ct)
    print("✅ Extracted Anatomix features for GT and Predicted CT")

    feats_mri = extract_feats_np(gt_mri)
    print("✅ Extracted Anatomix features for MRI")

    C, H, W, D = feats_gt.shape

    # --- Shared PCA (for both GT + Pred) ---
    def sample_vox(feats, max_vox=200_000):
        X = feats.reshape(C, -1).T
        if X.shape[0] > max_vox:
            X = X[np.random.choice(X.shape[0], max_vox, replace=False)]
        return X

    # X_both = np.concatenate([sample_vox(feats_gt), sample_vox(feats_pred)], axis=0)
    X_both = np.concatenate([
        sample_vox(feats_mri),
        sample_vox(feats_gt),
        sample_vox(feats_pred)
    ], axis=0)

    pca = PCA(n_components=3, svd_solver="randomized").fit(X_both)
    print("✅ Shared PCA basis fitted")

    def project_pca(feats):
        X = feats.reshape(C, -1).T
        Y = pca.transform(X)
        Y = (Y - Y.min(0, keepdims=True)) / (
            Y.max(0, keepdims=True) - Y.min(0, keepdims=True) + 1e-8
        )
        return Y.reshape(H, W, D, 3)

    pca_gt = project_pca(feats_gt)
    pca_mri = project_pca(feats_mri)
    pca_pred = project_pca(feats_pred)

    # --- Cosine & L2 similarity ---
    gt_t = torch.from_numpy(feats_gt).unsqueeze(0)
    pred_t = torch.from_numpy(feats_pred).unsqueeze(0)
    cos_sim = F.cosine_similarity(gt_t, pred_t, dim=1).squeeze(0).numpy()
    cos_sim_n = (cos_sim - cos_sim.min()) / (cos_sim.max() - cos_sim.min())
    l2_dist = torch.norm(gt_t - pred_t, dim=1).squeeze(0).numpy()
    l2_sim = 1 - (l2_dist - l2_dist.min()) / (l2_dist.max() - l2_dist.min())
    print("✅ Computed cosine and L2 similarities")

    # --- Layout ---
    # slice_indices = np.linspace(0, D - 1, 5, dtype=int)
    slice_indices = np.linspace(0.1 * D, 0.9 * D, 5, dtype=int)
    vmin, vmax = 0.0, 1.0
    # fig, axes = plt.subplots(
    #     len(slice_indices), 7, figsize=(26, 3.5 * len(slice_indices))
    # )
    fig, axes = plt.subplots(len(slice_indices), 8, figsize=(30, 3.5 * len(slice_indices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    for i, z in enumerate(slice_indices):
        # 1️⃣ GT MRI
        axes[i, 0].imshow(gt_mri[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 0].set_title(f"GT MRI (z={z})")
        axes[i, 0].axis("off")

        # 2️⃣ GT CT
        axes[i, 1].imshow(gt_ct[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 1].set_title("GT CT")
        axes[i, 1].axis("off")

        # 3️⃣ Pred CT
        axes[i, 2].imshow(pred_ct[:, :, z], cmap="gray", vmin=vmin, vmax=vmax)
        axes[i, 2].set_title("Pred CT")
        axes[i, 2].axis("off")
        
        # 4️⃣ PCA(MRI)
        axes[i, 3].imshow(pca_mri[:, :, z, :])
        axes[i, 3].set_title("PCA (MRI)")
        axes[i, 3].axis("off")

        # 4️⃣ PCA(GT)
        axes[i, 4].imshow(pca_gt[:, :, z, :])
        axes[i, 4].set_title("PCA (GT CT)")
        axes[i, 4].axis("off")

        # 5️⃣ PCA(Pred)
        axes[i, 5].imshow(pca_pred[:, :, z, :])
        axes[i, 5].set_title("PCA (Pred)")
        axes[i, 5].axis("off")

        # 6️⃣ Cosine Sim
        im1 = axes[i, 6].imshow(cos_sim_n[:, :, z], cmap="plasma", vmin=vmin, vmax=vmax)
        axes[i, 6].set_title("Cosine Sim (0–1)")
        axes[i, 6].axis("off")

        # 7️⃣ L2 Sim
        im2 = axes[i, 7].imshow(l2_sim[:, :, z], cmap="plasma", vmin=vmin, vmax=vmax)
        axes[i, 7].set_title("L2 Sim (1 - Norm)")
        axes[i, 7].axis("off")

    # Shared colorbar inside figure
    cbar = fig.colorbar(im2, ax=axes[:, -1], fraction=0.035, pad=0.01)
    cbar.set_label("Feature Similarity (0–1)")

    # Title
    epoch_str = f" (epoch {epoch})" if epoch is not None else ""
    fig.suptitle(
        f"Anatomix PCA + Similarity — {subj_id}{epoch_str}", fontsize=18, y=0.995
    )

    # Save & log
    SAVE_DIR = os.path.join(
        ROOT_DIR, "results", "single_pair_optimization", "feature_vis"
    )
    os.makedirs(SAVE_DIR, exist_ok=True)
    save_path = os.path.join(SAVE_DIR, f"{subj_id}_epoch{epoch}.png")
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        wandb.log({"feature_vis": wandb.Image(save_path)})
    # print(f"💾 Saved visualization (epoch {epoch}): {save_path}")


# -----------------------------
# 2. Load data & features
# -----------------------------
# mri, ct = load_image_pair(DATA_DIR, SUBJ_ID)
mri, ct, pad_vals = load_image_pair(DATA_DIR, SUBJ_ID)

feat_extractor = Unet(dimension=3, input_nc=1, output_nc=16, num_downs=4, ngf=16).to(
    device
)
feat_extractor.load_state_dict(torch.load(CKPT_PATH, map_location=device), strict=True)
feat_extractor.eval()
print("✅ Loaded Anatomix pretrained model")


@torch.no_grad()
def extract_feats(volume_np):
    inp = torch.from_numpy(volume_np[None, None]).float().to(device)
    return feat_extractor(inp).squeeze(0).cpu().numpy()


feats_mri, feats_ct = extract_feats(mri), extract_feats(ct)
cleanup_gpu()
print(f"✅ MRI feats: {feats_mri.shape}, CT feats: {feats_ct.shape}")

# -----------------------------
# 3. Prepare dataset
# -----------------------------
# if MODEL_TYPE == "mlp":
#     X = torch.from_numpy(feats_mri).permute(1, 2, 3, 0).reshape(-1, 16).float().to(device)
#     Y = torch.from_numpy(ct).reshape(-1, 1).float().to(device)
#     loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
#     print(f"Total voxels: {len(X):,}")
# if MODEL_TYPE == "mlp":
#     # --- Anatomix features ---
#     feats = torch.from_numpy(feats_mri).permute(1,2,3,0).reshape(-1,16).float().to(device)  
#     # shape = [N,16]

#     # --- 3D normalized coordinates for Fourier features ---
#     D, H, W = ct.shape
#     z = np.linspace(0,1,D)
#     y = np.linspace(0,1,H)
#     x = np.linspace(0,1,W)
#     zz, yy, xx = np.meshgrid(z,y,x, indexing='ij')
#     pos = np.stack([xx,yy,zz], axis=-1).astype(np.float32)  # [D,H,W,3]
#     pos = torch.from_numpy(pos).reshape(-1,3).to(device)     # [N,3]

#     # --- Fourier positional features ---
#     fourier_pos = fourier_encode(pos, num_bands=4)          # [N, 24]

#     # --- Final MLP input: Anatomix + Fourier ---
#     X = torch.cat([feats, fourier_pos], dim=1)               # [N, 16 + 24 = 40]

#     Y = torch.from_numpy(ct).reshape(-1,1).float().to(device)

#     # loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=True)
#     loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)
#     print(f"Total voxels: {len(X):,}")
if MODEL_TYPE == "mlp":
    feats = torch.from_numpy(feats_mri).permute(1,2,3,0).reshape(-1,16).float().to(device)

    posenc = build_positional_encoding(ct, device, mode=POSENC)

    if posenc is None:
        X = feats                                      # [N,16]
        in_dim = 16
    else:
        X = torch.cat([feats, posenc], dim=1)          # [N, 16+3] or [N,16+24]
        in_dim = X.size(1)

    Y = torch.from_numpy(ct).reshape(-1,1).float().to(device)

    loader = DataLoader(TensorDataset(X, Y), batch_size=BATCH_SIZE, shuffle=False)


elif MODEL_TYPE == "cnn":
    mri_t = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0).float()  # [1,1,H,W,D]
    ct_t  = torch.from_numpy(ct ).unsqueeze(0).unsqueeze(0).float()
    loader = [(mri_t.to(device), ct_t.to(device))]  # single batch


# -----------------------------
# 4. Translator model
# -----------------------------
class MLPTranslator(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)

class CNNTranslator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


if MODEL_TYPE == "mlp":
    model_t = MLPTranslator(in_dim=in_dim).to(device)
elif MODEL_TYPE == "cnn":
    model_t = CNNTranslator().to(device)

# model_t = MLPTranslator().to(device)
optimizer = torch.optim.Adam(model_t.parameters(), lr=LR)
# loss_fn = nn.MSELoss()
loss_fn = nn.L1Loss()


# -----------------------------
# 5. Training / Evaluation
# -----------------------------
scaler = torch.amp.GradScaler()


def train_one_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0

    if isinstance(loader, list):
        dataset_size = len(loader)
    else:
        dataset_size = len(loader.dataset)

    for xb, yb in tqdm(loader, leave=False):
        xb, yb = xb.to(device), yb.to(device)
        with torch.amp.autocast(device_type=device.type):
            pred = model(xb)
            loss = loss_fn(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * xb.size(0)

    return total_loss / dataset_size



@torch.no_grad()
def evaluate(model, feats_mri, ct):
    model.eval()

    # if MODEL_TYPE == "mlp":
    #     X_full = torch.from_numpy(feats_mri).permute(1,2,3,0).reshape(-1,16).to(device)
    #     preds = []
    #     for i in range(0, X_full.size(0), 100_000):
    #         preds.append(model(X_full[i:i+100_000]).cpu().numpy())
    #     pred_ct = np.concatenate(preds, axis=0).reshape(ct.shape)
    # if MODEL_TYPE == "mlp":
    #     # Flatten Anatomix features
    #     feats_flat = torch.from_numpy(feats_mri).permute(1,2,3,0).reshape(-1,16).float().to(device)
    
    #     # Build coordinates
    #     D, H, W = ct.shape
    #     z = torch.linspace(0,1,D, device=device)
    #     y = torch.linspace(0,1,H, device=device)
    #     x = torch.linspace(0,1,W, device=device)
    #     zz, yy, xx = torch.meshgrid(z,y,x, indexing='ij')
    #     pos = torch.stack([xx,yy,zz], dim=-1).reshape(-1,3)  # [N,3]
    
    #     # Fourier positional encoding
    #     fourier_pos = fourier_encode(pos, num_bands=4)       # [N,24]
    
    #     # Combined MLP input: [16 + 24] = 40 channels
    #     X_full = torch.cat([feats_flat, fourier_pos], dim=1)
    
    #     preds = []
    #     for i in range(0, X_full.size(0), 100000):
    #         preds.append(model(X_full[i:i+100000]).cpu().numpy())
    
    #     pred_ct = np.concatenate(preds, axis=0).reshape(ct.shape)
    if MODEL_TYPE == "mlp":
        feats_flat = torch.from_numpy(feats_mri).permute(1,2,3,0).reshape(-1,16).float().to(device)
        posenc = build_positional_encoding(ct, device, mode=POSENC)
    
        if posenc is None:
            X_full = feats_flat
        else:
            X_full = torch.cat([feats_flat, posenc], dim=1)
    
        preds = []
        for i in range(0, X_full.size(0), 100000):
            preds.append(model(X_full[i:i+100000]).cpu().numpy())
    
        pred_ct = np.concatenate(preds, axis=0).reshape(ct.shape)

    elif MODEL_TYPE == "cnn":
        mri_t = torch.from_numpy(mri).unsqueeze(0).unsqueeze(0).float().to(device)
        pred_ct = model(mri_t).cpu().numpy()[0,0]

    return compute_metrics(pred_ct, ct), pred_ct


# -----------------------------
# 6. Training loop
# -----------------------------
for epoch in tqdm(range(1, EPOCHS + 1), desc="Training Progress"):
    train_loss = train_one_epoch(model_t, loader, optimizer, loss_fn)
    log_wandb({"train_loss": train_loss, "epoch": epoch})

    if (epoch - 1) % VAL_INTERVAL == 0:
        (mae, psnr, ssim), pred_ct_padded = evaluate(model_t, feats_mri, ct)
        log_wandb({"mae": mae, "psnr": psnr, "ssim": ssim, "epoch": epoch})
        print(f"Epoch {epoch:03d} → MAE={mae:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.3f}")
        visualize_ct_feature_comparison(
            pred_ct_padded, ct, mri,
            feat_extractor, SUBJ_ID,
            epoch=epoch, use_wandb=USE_WANDB
        )


print("✅ Training complete!")

# -----------------------------
# 7. Save & Final Evaluation
# -----------------------------
SAVE_DIR = os.path.join(ROOT_DIR, "results", "single_pair_optimization", "models")
os.makedirs(SAVE_DIR, exist_ok=True)
SAVE_PATH = os.path.join(SAVE_DIR, f"mri2ct_simple_model_{NOW}.pt")
torch.save(model_t.state_dict(), SAVE_PATH)
print(f"💾 Saved model to {SAVE_PATH}")

(mae, psnr, ssim), pred_ct_padded = evaluate(model_t, feats_mri, ct)
log_wandb({"mae": mae, "psnr": psnr, "ssim": ssim, "epoch": epoch})
print(f"Epoch {epoch:03d} → MAE={mae:.4f}, PSNR={psnr:.2f}, SSIM={ssim:.3f}")
visualize_ct_feature_comparison(pred_ct_padded, ct, mri, feat_extractor, SUBJ_ID, epoch=EPOCHS, use_wandb=USE_WANDB)

if USE_WANDB:
    wandb.finish()

end_time = time.time()
elapsed = end_time - start_time
mins = elapsed / 60
hrs = elapsed / 3600
print(f"⏱️ Total runtime: {elapsed:.2f} s  ({mins:.2f} min, {hrs:.2f} hr)")

