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
SRC_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(SRC_ROOT)
sys.path.append(PROJECT_ROOT)
sys.path.append(SRC_ROOT)

from anatomix.model.network import Unet

from common.data import DataPreprocessing
from common.utils import unpad


def clean_state_dict(state_dict):
    """Removes '_orig_mod.' prefix from keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    return new_state_dict


def create_comparison_figure(mri, pred_unet, pred_amix, subj_id, plane, out_path):
    """Generates a comparison PNG with middle slices of all 3 orthogonal views."""
    # Data is (X, Y, Z)
    nx, ny, nz = mri.shape
    mid_x, mid_y, mid_z = nx // 2, ny // 2, nz // 2

    # Normalize CT predictions (HU -1024 to 1024) to [0, 1] for visualization
    def normalize_ct(vol):
        return np.clip((vol + 1024) / 2048.0, 0, 1)

    pred_unet_viz = normalize_ct(pred_unet)
    pred_amix_viz = normalize_ct(pred_amix)

    # Normalize MRI (0-99th percentile)
    mri_viz = np.clip(mri, 0, np.percentile(mri, 99))
    mri_viz = mri_viz / (mri_viz.max() + 1e-8)

    # Columns: MRI, UNet, Anatomix
    volumes = [(mri_viz, "Input MRI"), (pred_unet_viz, "UNet Prediction"), (pred_amix_viz, "Anatomix Prediction")]

    # Rows: Axial (Z-slice), Coronal (Y-slice), Sagittal (X-slice)
    views = [("Axial", mid_z, lambda v, idx: np.rot90(v[:, :, idx])), ("Coronal", mid_y, lambda v, idx: np.rot90(v[:, idx, :])), ("Sagittal", mid_x, lambda v, idx: np.rot90(v[idx, :, :]))]

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    plt.subplots_adjust(wspace=0.05, hspace=0.1)

    for i, (view_name, slice_idx, slice_fn) in enumerate(views):
        for j, (vol_data, vol_name) in enumerate(volumes):
            ax = axes[i, j]
            slice_data = slice_fn(vol_data, slice_idx)
            ax.imshow(slice_data, cmap="gray", vmin=0, vmax=1)

            # Labeling
            if i == 0:
                ax.set_title(vol_name, fontsize=16, pad=10)
            if j == 0:
                ax.set_ylabel(f"{view_name}\nSlice {slice_idx}", fontsize=14, labelpad=10)

            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle(f"MRNet Evaluation: {subj_id} ({plane.capitalize()})\nMiddle Slices Comparison", fontsize=22, y=0.98)
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"🖼️ Saved 3-view comparison figure to {out_path}")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Evaluate SynthRAD models on MRNet Knee MRI")
    parser.add_argument("--subj_id", type=str, default="0016")
    parser.add_argument("--name", type=str, default="ct_predictions", help="Subfolder name for these predictions")
    parser.add_argument("--unet_ckpt", type=str, required=True)
    parser.add_argument("--amix_ckpt", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="MRNet_Knee_1.5mm")
    parser.add_argument("--out_dir", type=str, default="MRNet_Knee_1.5mm", help="Output directory")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    subj_dir = os.path.join(args.data_dir, f"subject{args.subj_id}")
    out_subj_dir = os.path.join(args.out_dir, f"subject{args.subj_id}", args.name)
    os.makedirs(out_subj_dir, exist_ok=True)

    # 1. Load Models
    print("🏗️ Loading UNet Baseline...")
    unet_model = Unet(dimension=3, input_nc=1, output_nc=1, num_downs=4, ngf=16, final_act="sigmoid").to(device)
    unet_ckpt = torch.load(args.unet_ckpt, map_location=device)
    unet_model.load_state_dict(clean_state_dict(unet_ckpt["model_state_dict"]))
    unet_model.eval()

    print("🏗️ Loading Anatomix MRI2CT...")
    amix_ckpt = torch.load(args.amix_ckpt, map_location=device)
    translator_state = clean_state_dict(amix_ckpt["model_state_dict"])

    first_conv_weight = translator_state.get("model.0.weight", None)
    translator_input_nc = first_conv_weight.shape[1] if first_conv_weight is not None else 16

    feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
    translator = Unet(dimension=3, input_nc=translator_input_nc, output_nc=1, num_downs=4, ngf=16, final_act="sigmoid").to(device)
    translator.load_state_dict(translator_state)

    if "feat_extractor_state_dict" in amix_ckpt:
        feat_extractor.load_state_dict(clean_state_dict(amix_ckpt["feat_extractor_state_dict"]))
    else:
        default_amix = os.path.join(PROJECT_ROOT, "anatomix/model-weights/best_val_net_G_v2.pth")
        if os.path.exists(default_amix):
            feat_extractor.load_state_dict(clean_state_dict(torch.load(default_amix, map_location=device)))

    feat_extractor.eval()
    translator.eval()

    def amix_forward(x):
        f = feat_extractor(x)
        if translator_input_nc > 16:
            f = torch.cat([f, x], dim=1)
        return translator(f)

    # 2. Process each plane
    planes = ["axial", "coronal", "sagittal"]
    roi_size = (args.patch_size, args.patch_size, args.patch_size)

    for plane in planes:
        mri_path = os.path.join(subj_dir, f"{plane}.nii.gz")
        if not os.path.exists(mri_path):
            print(f"⚠️ Skipping {plane}, file not found: {mri_path}")
            continue

        print(f"\n📂 Processing {plane.upper()}...")
        mri_img = tio.ScalarImage(mri_path)
        affine = mri_img.affine

        dummy_ct = tio.ScalarImage(tensor=torch.zeros_like(mri_img.data), affine=affine)
        subj = tio.Subject(mri=mri_img, ct=dummy_ct)

        preprocess = DataPreprocessing(patch_size=args.patch_size, res_mult=32)
        subj_prep = preprocess(subj)

        mri_tensor = subj_prep["mri"][tio.DATA].unsqueeze(0).to(device)
        orig_shape = subj_prep["original_shape"].tolist()
        pad_offset = int(subj_prep["pad_offset"])

        with torch.autocast(device_type="cuda" if "cuda" in str(device) else "cpu", dtype=torch.bfloat16):
            p_unet = sliding_window_inference(mri_tensor, roi_size, 4, unet_model, overlap=args.overlap)
            p_amix = sliding_window_inference(mri_tensor, roi_size, 4, amix_forward, overlap=args.overlap)

        p_unet_unpad = unpad(p_unet, orig_shape, pad_offset).squeeze().float().cpu().numpy()
        p_unet_hu = (p_unet_unpad * 2048.0) - 1024.0

        p_amix_unpad = unpad(p_amix, orig_shape, pad_offset).squeeze().float().cpu().numpy()
        p_amix_hu = (p_amix_unpad * 2048.0) - 1024.0

        nib.save(nib.Nifti1Image(p_unet_hu, affine), os.path.join(out_subj_dir, f"pred_ct_unet_{plane}.nii.gz"))
        nib.save(nib.Nifti1Image(p_amix_hu, affine), os.path.join(out_subj_dir, f"pred_ct_amix_{plane}.nii.gz"))

        mri_data = mri_img.data.squeeze().numpy()
        viz_path = os.path.join(out_subj_dir, f"comparison_3view_{plane}.png")
        create_comparison_figure(mri_data, p_unet_hu, p_amix_hu, args.subj_id, plane, viz_path)

    print(f"\n✅ Done! Results saved to {out_subj_dir}")


if __name__ == "__main__":
    main()
