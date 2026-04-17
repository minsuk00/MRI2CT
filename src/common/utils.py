import gc
import random

import numpy as np
import psutil
import torch
from fused_ssim import fused_ssim3d

import wandb


def apply_synchronized_cutout(mri, ct, cutout_obj, seg=None):
    """
    Applies the same spatial cutout to MRI, CT, and optionally Seg by concatenating them.
    mri, ct: (B, 1, D, H, W) tensors.
    cutout_obj: An initialized monai.transforms.CutOut instance.
    seg: Optional (B, 1, D, H, W) segmentation tensor.
    Returns: (mri_aug, ct_aug, seg_aug) where seg_aug is None if seg is None.
    """
    # Robustness: Update batch size on the fly to match current input
    # MONAI CutOut expects weights for each sample in the batch
    cutout_obj.batch_size = mri.shape[0]

    if seg is not None:
        # Concatenate MRI, CT, and Seg (B, 3, D, H, W)
        # Cast seg to float for cat compatibility, restore original dtype after
        combined = torch.cat([mri, ct, seg.float()], dim=1)
        combined_aug = cutout_obj(combined)
        mri_aug, ct_aug, seg_aug = torch.split(combined_aug, 1, dim=1)
        return mri_aug, ct_aug, seg_aug.to(seg.dtype)
    else:
        # Concatenate MRI and CT (B, 2, D, H, W)
        combined = torch.cat([mri, ct], dim=1)
        combined_aug = cutout_obj(combined)
        mri_aug, ct_aug = torch.split(combined_aug, 1, dim=1)
        return mri_aug, ct_aug, None


def send_notification(message, topic="mri2ct_minsukc"):
    """
    Sends a notification via ntfy.sh and email to minsukc@umich.edu.
    You can subscribe to this topic at ntfy.sh/<topic> to get alerts on phone/desktop.
    """
    try:
        import requests

        print(f"[INFO] Sending notification to ntfy.sh/{topic} (and email to minsukc@umich.edu)...")
        headers = {"Title": "MRI2CT Training Alert", "Priority": "high", "Tags": "warning,skull", "Email": "minsukc@umich.edu"}
        requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"), headers=headers)
    except Exception as e:
        print(f"[WARNING] Failed to send notification: {e}")


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def log_model_summary(model_dict, save_path):
    """
    Computes total, trainable, and frozen parameters across multiple models.
    Saves a detailed layer-by-layer manifest to save_path.

    model_dict: Dict mapping model name to the model instance.
    """
    with open(save_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write(f"MODEL PARAMETER SUMMARY\n")
        f.write("=" * 60 + "\n\n")

        grand_total = 0
        grand_trainable = 0

        for name, model in model_dict.items():
            if model is None:
                continue

            # Unpack compiled models if needed
            target_model = getattr(model, "_orig_mod", model)

            total = sum(p.numel() for p in target_model.parameters())
            trainable = sum(p.numel() for p in target_model.parameters() if p.requires_grad)
            frozen = total - trainable

            grand_total += total
            grand_trainable += trainable

            f.write(f"--- Model: {name} ---\n")
            f.write(f"  Total Params:     {total:,}\n")
            f.write(f"  Trainable Params: {trainable:,}\n")
            f.write(f"  Frozen Params:    {frozen:,}\n\n")

            # Build param_name → module type lookup
            param_to_type = {}
            for mod_name, mod in target_model.named_modules():
                for pname, _ in mod.named_parameters(recurse=False):
                    full_name = f"{mod_name}.{pname}" if mod_name else pname
                    param_to_type[full_name] = type(mod).__name__

            f.write("  Trainable Layers Manifest:\n")
            found_trainable = False
            for layer_name, p in target_model.named_parameters():
                if p.requires_grad:
                    found_trainable = True
                    layer_type = param_to_type.get(layer_name, "?")
                    f.write(f"    - {layer_name} [{layer_type}] ({p.numel():,} params)\n")

            if not found_trainable:
                f.write("    (No trainable layers)\n")
            f.write("\n")

        f.write("=" * 60 + "\n")
        f.write("FINAL AGGREGATE SUMMARY\n")
        f.write("-" * 60 + "\n")
        f.write(f"AGGREGATE TOTAL:     {grand_total:,}\n")
        f.write(f"AGGREGATE TRAINABLE: {grand_trainable:,} ({grand_trainable / max(1, grand_total) * 100:.2f}%)\n")
        f.write(f"AGGREGATE FROZEN:    {grand_total - grand_trainable:,}\n")
        f.write("=" * 60 + "\n")

    print(f"[Utils] 📝 Model summary saved to {save_path}")
    print(f"[Model Summary] 📊 Total Params: {grand_total:,} | Trainable: {grand_trainable:,} ({grand_trainable / max(1, grand_total) * 100:.2f}%)")
    return grand_total, grand_trainable


def clean_state_dict(state_dict):
    """Removes '_orig_mod.' prefix from keys if present (added by torch.compile)."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    return new_state_dict


def set_seed(seed=42):
    print(f"[DEBUG] Setting global seed to {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # FOR TESTING & PRODUCTION TRAINING (SPEED):
    # torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def cleanup_gpu():
    gc.collect()
    torch.cuda.empty_cache()


def get_ram_info():
    """
    Calculates accurate RAM usage using PSS (Proportional Set Size).
    Optimized for speed to avoid bottlenecking the training loop.
    """
    # 1. Use 48GB as the hardcoded baseline
    total_allowed_bytes = 48 * 1024 * 1024 * 1024

    main_p = psutil.Process()
    total_pss = 0

    try:
        # PSS calculation is the most accurate but can be slow on some kernels.
        # We try to get it for the main process and all children.
        full_info = main_p.memory_full_info()
        total_pss += getattr(full_info, "pss", full_info.rss)

        children = main_p.children(recursive=True)
        for child in children:
            try:
                # Use .rss as a faster fallback if .pss is too slow or unavailable
                child_info = child.memory_full_info()
                total_pss += getattr(child_info, "pss", child_info.rss)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except Exception:
        # Fallback to faster RSS if PSS fails
        total_pss = main_p.memory_info().rss
        total_pss += sum(c.memory_info().rss for c in main_p.children(recursive=True))

    total_gb = total_pss / (1024**3)
    usage_percent = (total_pss / total_allowed_bytes) * 100

    return {"percent": usage_percent, "total_gb": total_gb, "main_rss_gb": main_p.memory_info().rss / (1024**3), "num_children": len(main_p.children())}


def anatomix_normalize(tensor, percentile_range=None, clip_range=None):
    # if not torch.is_tensor(tensor):
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
        v_min = torch.quantile(tensor, min_percentile / 100.0)
        v_max = torch.quantile(tensor, max_percentile / 100.0)
        tensor = torch.clamp(tensor, v_min, v_max)

        denom = v_max - v_min
        if denom == 0:
            print(f"[WARNING] MRI Volume is constant (Val: {v_min.item():.4f}). Returning zeros.")
            return torch.zeros_like(tensor)

        return (tensor - v_min) / denom

    # 3. just minmax normalization
    v_min = tensor.min()
    v_max = tensor.max()
    # tensor = torch.clamp(tensor, v_min, v_max)
    denom = v_max - v_min
    if denom == 0:
        print(f"[WARNING] MRI Volume is constant (Val: {v_min.item():.4f}). Returning zeros.")
        return torch.zeros_like(tensor)

    return (tensor - v_min) / denom


def unpad(data, original_shape, offset=0):
    if original_shape is None:
        return data
    w_orig, h_orig, d_orig = original_shape
    return data[..., offset : offset + w_orig, offset : offset + h_orig, offset : offset + d_orig]


def compute_metrics(pred, target, data_range=1.0):
    if pred.ndim != 5 or target.ndim != 5:
        raise ValueError(f"Expected (B, C, D, H, W), got {pred.shape}")

    b, c, d, h, w = pred.shape

    # Ensure contiguous for custom CUDA kernels
    try:
        ssim_val = fused_ssim3d(pred.float().contiguous(), target.float().contiguous(), train=False).item()
    except Exception as e:
        print(f"⚠️ fused_ssim3d CUDA error ({e}), falling back to CPU...")
        ssim_val = fused_ssim3d(pred.cpu().float().contiguous(), target.cpu().float().contiguous(), train=False).item()

    # PSNR
    mse = torch.mean((pred - target) ** 2, dim=[1, 2, 3, 4])
    mse = torch.clamp(mse, min=1e-10)
    psnr = 10 * torch.log10((data_range**2) / mse)

    # 2. Gradient Difference (Sharpness Metric)
    def get_gradients(img):
        dz = torch.abs(img[:, :, 1:, :, :] - img[:, :, :-1, :, :])
        dy = torch.abs(img[:, :, :, 1:, :] - img[:, :, :, :-1, :])
        dx = torch.abs(img[:, :, :, :, 1:] - img[:, :, :, :, :-1])
        return dz, dy, dx

    pred_dz, pred_dy, pred_dx = get_gradients(pred)
    targ_dz, targ_dy, targ_dx = get_gradients(target)
    grad_diff = (torch.mean(torch.abs(pred_dz - targ_dz)) + torch.mean(torch.abs(pred_dy - targ_dy)) + torch.mean(torch.abs(pred_dx - targ_dx))).item()

    # 3. Bone Dice Coefficient (Structure Metric)
    bone_thresh = 0.7
    pred_bone = (pred > bone_thresh).float()
    targ_bone = (target > bone_thresh).float()
    intersection = (pred_bone * targ_bone).sum()
    union = pred_bone.sum() + targ_bone.sum()
    # Smooth Dice (Add epsilon to avoid division by zero)
    dice_score = (2.0 * intersection + 1e-5) / (union + 1e-5)
    dice_val = dice_score.item()

    mae_val = torch.mean(torch.abs(pred - target)).item()

    return {
        "mae": mae_val,
        "mae_hu": mae_val * 2048.0,
        "psnr": torch.mean(psnr).item(),
        "ssim": ssim_val,
        "grad_diff": grad_diff,
        "dice_score_bone_threshold": dice_val,  # Thresholded metric (naive)
    }


def compute_metrics_body(pred, target, mask):
    """Like compute_metrics but restricted to body voxels (from mask.nii.gz).
    - MAE, PSNR: computed on masked voxels only
    - SSIM: full volume with background zeroed in both (no penalty for matching zeros)
    Returns: {mae, mae_hu, psnr, ssim}
    """
    mask_bool = mask.bool()

    pred_m = pred * mask_bool.float()
    targ_m = target * mask_bool.float()
    try:
        ssim_val = fused_ssim3d(pred_m.float().contiguous(), targ_m.float().contiguous(), train=False).item()
    except Exception:
        ssim_val = fused_ssim3d(pred_m.cpu().float().contiguous(), targ_m.cpu().float().contiguous(), train=False).item()

    pred_v = pred[mask_bool]
    targ_v = target[mask_bool]
    mae_val = torch.mean(torch.abs(pred_v - targ_v)).item()
    mse = torch.mean((pred_v - targ_v) ** 2).clamp(min=1e-10)
    psnr = (10 * torch.log10(torch.tensor(1.0) / mse)).item()

    return {"mae": mae_val, "mae_hu": mae_val * 2048.0, "psnr": psnr, "ssim": ssim_val}


def visualize_lite(pred, ct, mri, subj_id, shape, step, epoch, offset=0, log_name=None, metrics=None, body_metrics=None):
    import matplotlib.pyplot as plt
    import numpy as np

    gt_ct = ct.squeeze()
    if gt_ct.ndim > 3:
        gt_ct = gt_ct[0]

    gt_mri = mri.squeeze()
    if gt_mri.ndim > 3:
        gt_mri = gt_mri[0]

    pred_ct = pred.squeeze()
    if pred_ct.ndim > 3:
        pred_ct = pred_ct[0]

    if isinstance(gt_ct, torch.Tensor):
        gt_ct = gt_ct.cpu().numpy()
    if isinstance(gt_mri, torch.Tensor):
        gt_mri = gt_mri.cpu().numpy()
    if isinstance(pred_ct, torch.Tensor):
        pred_ct = pred_ct.cpu().numpy()

    items = [
        (gt_mri, "GT MRI", "gray", (0, 1)),
        (gt_ct, "GT CT", "gray", (0, 1)),
        (pred_ct, "Pred CT", "gray", (0, 1)),
        (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
    ]

    D_dim = gt_ct.shape[-1]
    num_cols = len(items)
    slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)

    fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3.5 * len(slice_indices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    if len(slice_indices) == 1:
        axes = axes.reshape(1, -1)

    for i, z_slice in enumerate(slice_indices):
        for j, (data, title, cmap, clim) in enumerate(items):
            ax = axes[i, j]
            im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
            if title == "Residual":
                res_im = im
            if i == 0:
                ax.set_title(title)
            ax.axis("off")

    if "res_im" in locals():
        cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
        cbar.set_label("Residual Error")

    fig.suptitle(f"Subject: {subj_id} | Ep {epoch} | Step {step}", fontsize=16, y=0.99)

    if log_name is not None:
        caption = None
        if metrics:
            caption = " | ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
            if body_metrics:
                caption += "\n[body] " + " | ".join(f"{k}: {v:.4f}" for k, v in body_metrics.items())
        wandb.log({log_name: wandb.Image(fig, caption=caption)}, step=step)

    plt.close(fig)
    return fig
