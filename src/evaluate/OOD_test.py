"""
OOD inference on CHAOS MR volumes across multiple model families.

Models: see `_family_dispatch.MODELS` — unet_300k, amix_v1_4_300k, maisi, cwdm, mcddpm.

For each CHAOS subject in SUBJ_LIST, runs all models on T1 in/out-phase + T2 SPIR.

Output: evaluation_results/OOD_inference/subject_<id>/<model_name>/
  - pred_ct_<vol>.nii.gz  (HU)
  - viz_<vol>.png         (5-slice MRI | Pred CT figure)

MAISI note: the model conditions on `spacing_tensor = ct_spacing * 100`. CHAOS
ships MR only, so we substitute the MRI voxel sizes here. The model never saw
this distribution at training time — predictions are best-effort, not faithful.
"""

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from common.utils import unpad

from src.evaluate._family_dispatch import (
    GPFS,
    MODELS,
    infer_for_family,
    load_for_family,
    preprocess_for_family,
    to_hu,
)


# ─── configuration ───────────────────────────────────────────────────────────
SUBJ_LIST = ["20", "34"]

# Subset of MODELS to actually run (skip the slow diffusion ones for repeat passes).
# Set to None to run every entry in MODELS.
MODELS_TO_RUN = ["unet_300k", "amix_v1_4_300k"]

OUTPUT_BASE = os.path.join(PROJECT_ROOT, "evaluation_results", "OOD_inference")


def get_chaos_vols(subj_id):
    return {
        "T1_inphase":  os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T1DUAL/inphase.nii.gz"),
        "T1_outphase": os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T1DUAL/outphase.nii.gz"),
        "T2_spir":     os.path.join(GPFS, f"CHAOS/nifti/Train_Sets/MR/{subj_id}/T2SPIR/t2spir.nii.gz"),
    }


# ─── visualisation ────────────────────────────────────────────────────────────
def save_viz(mri_np, pred_hu_np, vol_name, model_name, out_dir):
    # Predicted CT: clip to amix window [-1024, 1024] → [0, 1] for display.
    pred_disp = np.clip(pred_hu_np, -1024.0, 1024.0)
    pred_disp = (pred_disp + 1024.0) / 2048.0

    # MRI: stretch to [0, 1] regardless of original normalization range.
    mri_lo, mri_hi = float(mri_np.min()), float(mri_np.max())
    if mri_hi - mri_lo > 1e-6:
        mri_disp = (mri_np - mri_lo) / (mri_hi - mri_lo)
    else:
        mri_disp = np.zeros_like(mri_np)

    D = mri_np.shape[2]
    slices = np.linspace(int(0.1 * D), int(0.9 * D), 5, dtype=int)
    items = [(mri_disp, "Input MRI"), (pred_disp, "Pred CT")]

    fig, axes = plt.subplots(len(slices), len(items), figsize=(3 * len(items), 3.5 * len(slices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)
    if len(slices) == 1:
        axes = axes.reshape(1, -1)
    for i, z in enumerate(slices):
        for j, (vol, title) in enumerate(items):
            ax = axes[i, j]
            ax.imshow(vol[:, :, z], cmap="gray", vmin=0.0, vmax=1.0)
            if i == 0:
                ax.set_title(title, fontsize=10)
            ax.axis("off")
    fig.suptitle(f"Model: {model_name} | Volume: {vol_name}", fontsize=13, y=1.01)
    out_path = os.path.join(out_dir, f"viz_{vol_name}.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved viz → {out_path}")


# ─── main ─────────────────────────────────────────────────────────────────────
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for subj_id in SUBJ_LIST:
        print(f"\n{'#' * 60}\nSubject: {subj_id}\n{'#' * 60}")
        chaos_vols = get_chaos_vols(subj_id)

        for model_name, info in MODELS.items():
            if MODELS_TO_RUN is not None and model_name not in MODELS_TO_RUN:
                continue
            family = info["family"]
            print(f"\n{'=' * 60}\nModel: {model_name} (family={family})\n{'=' * 60}")
            ckpt_path = info["ckpt_path"]

            family_tag, bundle, cfg, epoch = load_for_family(family, ckpt_path, device)
            print(f"  Loaded (epoch={epoch}) from {os.path.basename(ckpt_path)}")

            out_dir = os.path.join(OUTPUT_BASE, f"subject_{subj_id}", model_name)
            os.makedirs(out_dir, exist_ok=True)

            for vol_name, vol_path in chaos_vols.items():
                print(f"  [{vol_name}] preprocessing...")
                mri_tensor, orig_shape, affine, voxel_sizes = preprocess_for_family(family, vol_path, cfg)
                mri_tensor = mri_tensor.unsqueeze(0).to(device)
                print(f"    Shape: {list(mri_tensor.shape[2:])}, voxel_sizes: {voxel_sizes.round(2).tolist()}, running inference...")

                pred = infer_for_family(family, bundle, cfg, mri_tensor, voxel_sizes, device)

                pred_unpad = unpad(pred.float(), orig_shape)
                if family == "mcddpm":
                    pred_hu = pred_unpad.cpu().numpy().squeeze()
                else:
                    pred_hu = to_hu(family, pred_unpad).cpu().numpy().squeeze()

                mri_unpad = unpad(mri_tensor.float(), orig_shape)
                mri_vis = mri_unpad.cpu().numpy().squeeze()

                ct_path = os.path.join(out_dir, f"pred_ct_{vol_name}.nii.gz")
                nib.save(nib.Nifti1Image(pred_hu.astype(np.float32), affine), ct_path)
                print(f"    Saved CT → {ct_path}")

                save_viz(mri_vis, pred_hu, vol_name, model_name, out_dir)

            del bundle
            torch.cuda.empty_cache()

    print("\nDone.")


if __name__ == "__main__":
    main()
