import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from sklearn.decomposition import PCA
import wandb
import torchio as tio
from utils import ProjectPreprocessing, get_augmentations

@torch.no_grad()
def visualize_ct_feature_comparison(
    pred_ct, gt_ct, gt_mri, model, subj_id, 
    root_dir, epoch=None, use_wandb=False, idx = 1
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

    save_dir = os.path.join(root_dir, "results", "single_pair_optimization", "vis")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{subj_id}_epoch{epoch}.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if use_wandb:
        wandb.log({f"val/visualization_{idx}": wandb.Image(save_path)}, step=epoch)


def log_aug_viz(val_data, args, epoch, root_dir):
    """
    Simulates the training augmentation pipeline on a validation subject
    and logs the Before/After visualization to WandB.
    """
    try:
        # 1. Reconstruct TorchIO Subject from raw numpy
        # Note: We need 4D tensors (C, D, H, W)
        mri_t = torch.from_numpy(val_data['mri']).float()
        if mri_t.ndim == 3: mri_t = mri_t.unsqueeze(0)
        
        ct_t = torch.from_numpy(val_data['ct']).float()
        if ct_t.ndim == 3: ct_t = ct_t.unsqueeze(0)

        subject = tio.Subject(
            mri=tio.ScalarImage(tensor=mri_t),
            ct=tio.ScalarImage(tensor=ct_t)
        )

        # 2. Apply "Clean" Preprocessing (MinMax, Padding)
        preprocess = ProjectPreprocessing(
            use_seg=False, # Seg not needed for visual check
            patch_size=args.patch_size
        )
        clean_subj = preprocess(subject)

        # 3. Apply Augmentation (if enabled)
        # Note: We force p=1.0 for visualization so we actually see changes,
        # but using the pipeline from utils ensures we use the same transforms.
        augment_transform = get_augmentations()
        
        # We need to run it. If p values are low in get_augmentations, 
        # we might not see changes every time. That's fine, it reflects reality.
        aug_subj = augment_transform(clean_subj)
        hist_str = " | ".join([t.name for t in aug_subj.history])

        # 4. Visualization
        # Extract middle slice
        clean_vol = clean_subj['mri'].data[0].numpy()
        aug_vol = aug_subj['mri'].data[0].numpy()
        
        z = clean_vol.shape[2] // 2
        sl_clean = np.rot90(clean_vol[:, :, 2])
        sl_aug = np.rot90(aug_vol[:, :, 2])
        sl_diff = sl_aug - sl_clean

        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(sl_clean, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title("Input (Normalized)")
        
        axes[1].imshow(sl_aug, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title(f"Augmented\n{hist_str}")
        
        # Difference map (seismic to show shifts)
        im2 = axes[2].imshow(sl_diff, cmap='seismic', vmin=-0.5, vmax=0.5)
        axes[2].set_title("Difference (Aug - Input)")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

        
        for ax in axes: ax.axis('off')
        
        plt.tight_layout()
        save_path = os.path.join(root_dir, "results", "vis_aug_check.png")
        plt.savefig(save_path)
        plt.close(fig)

        wandb.log({"val/augmentation_example": wandb.Image(save_path)}, step=epoch)
        
    except Exception as e:
        print(f"⚠️ Augmentation Viz Failed: {e}")