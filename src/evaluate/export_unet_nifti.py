import argparse
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.inferers import sliding_window_inference

# Add project root and src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from anatomix.model.network import Unet

from common.data import DataPreprocessing, get_subject_paths
from common.utils import unpad


def clean_state_dict(state_dict):
    """Removes '_orig_mod.' prefix from keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    return new_state_dict


def create_comparison_figure(mri, gt_ct, pred, subj_id, out_dir, model_name="UNet"):
    """Generates a comparison PNG with 5 equidistant slices and index labels."""
    D_dim = gt_ct.shape[-1]
    slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)

    def normalize_ct(vol):
        return np.clip((vol + 1024) / 2048.0, 0, 1)

    gt_ct_viz = normalize_ct(gt_ct)
    pred_viz = normalize_ct(pred)

    mri_viz = mri.copy()
    mri_viz = np.clip(mri_viz, 0, np.percentile(mri_viz, 99))
    mri_viz = mri_viz / (mri_viz.max() + 1e-8)

    items = [
        (mri_viz, "Input MRI", "gray", (0, 1)),
        (gt_ct_viz, "Ground Truth CT", "gray", (0, 1)),
        (pred_viz, f"Pred CT ({model_name})", "gray", (0, 1)),
        (pred_viz - gt_ct_viz, "Residual", "seismic", (-0.2, 0.2)),
    ]

    num_cols = len(items)
    fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3 * len(slice_indices)))
    plt.subplots_adjust(wspace=0.05, hspace=0.15)

    if len(slice_indices) == 1:
        axes = axes.reshape(1, -1)

    for i, z_slice in enumerate(slice_indices):
        for j, (data, title, cmap, clim) in enumerate(items):
            ax = axes[i, j]
            im = ax.imshow(np.rot90(data[:, :, z_slice]), cmap=cmap, vmin=clim[0], vmax=clim[1])

            if i == 0:
                ax.set_title(title, fontsize=14)
            
            # Add Slice Index Label to the first column
            if j == 0:
                ax.text(-10, data.shape[1]//2, f"Slice {z_slice}", rotation=90, 
                        va='center', ha='right', fontsize=12, fontweight='bold')
            
            ax.axis("off")
            
    fig.suptitle(f"Comparison: {subj_id} ({model_name})", fontsize=18, y=0.98)

    out_path = os.path.join(out_dir, "comparison_slices.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"🖼️ Saved comparison figure to {out_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export NIfTI CT predictions from UNet model")
    parser.add_argument("--subj_id", type=str, required=True, help="Subject ID (e.g., 1ABA005)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to UNet baseline checkpoint")
    parser.add_argument("--name", type=str, required=True, help="Folder name for the run (e.g., dice_0.05)")
    parser.add_argument("--out_dir", type=str, default="val_viz", help="Output base directory")
    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm_registered")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.7, help="Sliding window overlap (Default: 0.7)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    # val_viz / name / subj_id
    out_subj_dir = os.path.join(args.out_dir, args.name, args.subj_id)
    os.makedirs(out_subj_dir, exist_ok=True)

    # 1. Load Data
    print(f"📂 Loading data for {args.subj_id}...")
    subj_path_full = None
    for split in ["train", "val", "test"]:
        cand = os.path.join(args.data_dir, split, args.subj_id)
        if os.path.exists(cand):
            subj_path_full = os.path.join(split, args.subj_id)
            break

    if not subj_path_full:
        raise FileNotFoundError(f"Subject {args.subj_id} not found in {args.data_dir}")

    paths = get_subject_paths(args.data_dir, subj_path_full)

    mri_img = tio.ScalarImage(paths["mri"])
    ct_img = tio.ScalarImage(paths["ct"])
    affine = ct_img.affine

    preprocess = DataPreprocessing(patch_size=args.patch_size, enable_safety_padding=False, res_mult=32)
    subj = tio.Subject(mri=mri_img, ct=ct_img)
    subj_prep = preprocess(subj)

    mri_tensor = subj_prep["mri"][tio.DATA].unsqueeze(0).to(device)
    orig_shape = subj_prep["original_shape"].tolist()
    pad_offset = int(subj_prep["pad_offset"]) if "pad_offset" in subj_prep else 0

    # Save inputs
    mri_unpad = unpad(subj_prep["mri"][tio.DATA].unsqueeze(0), orig_shape, pad_offset).squeeze().cpu().numpy()
    ct_unpad = unpad(subj_prep["ct"][tio.DATA].unsqueeze(0), orig_shape, pad_offset).squeeze().cpu().numpy()
    ct_hu = (ct_unpad * 2048.0) - 1024.0

    nib.save(nib.Nifti1Image(mri_unpad, affine), os.path.join(out_subj_dir, "input_mri.nii.gz"))
    nib.save(nib.Nifti1Image(ct_hu, affine), os.path.join(out_subj_dir, "gt_ct.nii.gz"))

    # 2. UNet Model Inference
    print(f"🏗️ Loading UNet model from {args.ckpt}...")
    ckpt = torch.load(args.ckpt, map_location=device)
    
    # Extract config from ckpt if available
    ngf = ckpt["config"].get("ngf", 16)
    num_downs = ckpt["config"].get("num_downs", 4)
    
    model = Unet(dimension=3, input_nc=1, output_nc=1, num_downs=num_downs, ngf=ngf, final_act="sigmoid").to(device)
    state_dict = clean_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state_dict)
    model.eval()

    print("🔮 Running sliding window inference...")
    with torch.amp.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=torch.bfloat16):
        pred = sliding_window_inference(
            inputs=mri_tensor,
            roi_size=(args.patch_size, args.patch_size, args.patch_size),
            sw_batch_size=4,
            predictor=model,
            overlap=args.overlap,
            device=device,
        )

    pred_unpad = unpad(pred, orig_shape, pad_offset).squeeze().float().cpu().numpy()
    pred_hu = (pred_unpad * 2048.0) - 1024.0
    
    out_path = os.path.join(out_subj_dir, "pred_ct_unet.nii.gz")
    nib.save(nib.Nifti1Image(pred_hu, affine), out_path)
    print(f"✅ Saved prediction to {out_path}")

    # 3. Generate Visual Comparison
    print("🎨 Generating comparison figure...")
    create_comparison_figure(mri_unpad, ct_hu, pred_hu, args.subj_id, out_subj_dir, model_name="UNet")


if __name__ == "__main__":
    main()
