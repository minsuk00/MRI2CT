"""
OOD inference on MRNet knee MR volumes across all 5 model families.

Models: see `_family_dispatch.MODELS` — unet_300k, amix_v1_4_300k, maisi, cwdm, mcddpm.

For the chosen subject, runs every model on the 3 reoriented planes
(axial / coronal / sagittal, all 1.5 mm iso) and saves:

  <out_dir>/subject<id>/
    unet_300k/       pred_ct_{axial,coronal,sagittal}.nii.gz
    amix_v1_4_300k/  ...
    maisi/           ...
    cwdm/            ...
    mcddpm/          ...
    comparison_axial.png    (3 anat views × 6 cols: MRI + 5 model preds)
    comparison_coronal.png
    comparison_sagittal.png

All models were trained on SynthRAD abdomen/thorax CT — knee anatomy is *very*
OOD; predictions are exploratory, not faithful. The pre-existing `ct_predictions/`
folder (legacy UNet+Amix outputs from the old 2-model version of this script) is
left untouched.
"""

import argparse
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
    MODELS,
    infer_for_family,
    load_for_family,
    preprocess_for_family,
    to_hu,
)


PLANES = ["axial", "coronal", "sagittal"]


def get_mrnet_vols(data_dir, subj_id):
    subj_dir = os.path.join(data_dir, f"subject{subj_id}")
    return {plane: os.path.join(subj_dir, f"{plane}.nii.gz") for plane in PLANES}


def _stretch01(x):
    """Rescale array to [0, 1] for display."""
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-6:
        return np.zeros_like(x)
    return (x - lo) / (hi - lo)


def _ct_to_disp(hu):
    """Clip predicted HU to amix window [-1024, 1024] → [0, 1] for display."""
    return np.clip((hu + 1024.0) / 2048.0, 0.0, 1.0)


def save_comparison_figure(mri_np, preds_by_model, plane_name, subj_id, out_path):
    """3 rows (axial/coronal/sagittal mid-slices) × N+1 cols (MRI + per-model pred).

    Inputs are (X, Y, Z) numpy arrays of the same shape (one input plane volume +
    one prediction per model). All viewed as the same 3 orthogonal mid-slices.
    """
    model_names = list(preds_by_model.keys())
    nx, ny, nz = mri_np.shape
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

    mri_disp = _stretch01(mri_np)
    pred_disps = {name: _ct_to_disp(p) for name, p in preds_by_model.items()}

    views = [
        ("Axial",    mid_z, lambda v, idx: np.rot90(v[:, :, idx])),
        ("Coronal",  mid_y, lambda v, idx: np.rot90(v[:, idx, :])),
        ("Sagittal", mid_x, lambda v, idx: np.rot90(v[idx, :, :])),
    ]
    columns = [("Input MRI", mri_disp)] + [(name, pred_disps[name]) for name in model_names]

    ncols = len(columns)
    nrows = len(views)
    fig, axes = plt.subplots(nrows, ncols, figsize=(2.6 * ncols, 2.6 * nrows))
    plt.subplots_adjust(wspace=0.04, hspace=0.08)

    for i, (view_name, slice_idx, slice_fn) in enumerate(views):
        for j, (col_title, vol) in enumerate(columns):
            ax = axes[i, j]
            ax.imshow(slice_fn(vol, slice_idx), cmap="gray", vmin=0.0, vmax=1.0)
            if i == 0:
                ax.set_title(col_title, fontsize=11, pad=6)
            if j == 0:
                ax.set_ylabel(f"{view_name}\nslice {slice_idx}", fontsize=10, labelpad=6)
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"MRNet subject{subj_id} — input plane: {plane_name}", fontsize=14, y=0.99)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved comparison → {out_path}")


def main():
    parser = argparse.ArgumentParser(description="OOD MRNet knee inference with all 5 models.")
    parser.add_argument("--subj_id", type=str, default="0016")
    parser.add_argument("--data_dir", type=str, default=os.path.join(PROJECT_ROOT, "MRNet_Knee_1.5mm"))
    parser.add_argument("--out_dir", type=str, default=None,
                        help="Output base dir; defaults to --data_dir (writes alongside the input).")
    args = parser.parse_args()
    if args.out_dir is None:
        args.out_dir = args.data_dir

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Subject: subject{args.subj_id}")

    vols = get_mrnet_vols(args.data_dir, args.subj_id)
    for plane, path in vols.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing MRNet volume: {path}")

    subj_out_dir = os.path.join(args.out_dir, f"subject{args.subj_id}")
    os.makedirs(subj_out_dir, exist_ok=True)

    # preds[plane][model_name] = pred_hu_np  (unpadded, (X, Y, Z))
    preds = {plane: {} for plane in PLANES}
    # mri_unpadded[plane] = mri_np from any one preprocess (we'll save the first family's mri).
    mri_unpadded = {}
    affines = {}

    for model_name, info in MODELS.items():
        family = info["family"]
        print(f"\n{'=' * 60}\nModel: {model_name} (family={family})\n{'=' * 60}")
        ckpt_path = info["ckpt_path"]

        family_tag, bundle, cfg, epoch = load_for_family(family, ckpt_path, device)
        print(f"  Loaded (epoch={epoch}) from {os.path.basename(ckpt_path)}")

        model_out_dir = os.path.join(subj_out_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        for plane, vol_path in vols.items():
            print(f"  [{plane}] preprocessing...")
            mri_tensor, orig_shape, affine, voxel_sizes = preprocess_for_family(family, vol_path, cfg)
            mri_tensor = mri_tensor.unsqueeze(0).to(device)
            print(f"    Shape: {list(mri_tensor.shape[2:])}, voxel_sizes: {voxel_sizes.round(2).tolist()}, running inference...")

            pred = infer_for_family(family, bundle, cfg, mri_tensor, voxel_sizes, device)

            pred_unpad = unpad(pred.float(), orig_shape)
            if family == "mcddpm":
                pred_hu = pred_unpad.cpu().numpy().squeeze()
            else:
                pred_hu = to_hu(family, pred_unpad).cpu().numpy().squeeze()

            ct_path = os.path.join(model_out_dir, f"pred_ct_{plane}.nii.gz")
            nib.save(nib.Nifti1Image(pred_hu.astype(np.float32), affine), ct_path)
            print(f"    Saved CT → {ct_path}")

            preds[plane][model_name] = pred_hu
            if plane not in mri_unpadded:
                mri_unpadded[plane] = unpad(mri_tensor.float(), orig_shape).cpu().numpy().squeeze()
                affines[plane] = affine

        del bundle
        torch.cuda.empty_cache()

    # Build one 3×6 figure per input plane (MRI + 5 model preds, 3 mid-slice views).
    print(f"\n{'=' * 60}\nBuilding comparison figures\n{'=' * 60}")
    for plane in PLANES:
        out_path = os.path.join(subj_out_dir, f"comparison_{plane}.png")
        save_comparison_figure(
            mri_unpadded[plane], preds[plane], plane, args.subj_id, out_path
        )

    print(f"\nDone. Outputs under {subj_out_dir}")


if __name__ == "__main__":
    main()
