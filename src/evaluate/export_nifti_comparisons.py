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

from mri2ct.data import DataPreprocessing, get_subject_paths
from mri2ct.utils import unpad


def clean_state_dict(state_dict):
    """Removes '_orig_mod.' prefix from keys if present."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    return new_state_dict


def create_comparison_figure(mri, gt_ct, pred_unet, pred_amix, subj_id, out_dir):
    """Generates a comparison PNG with 5 equidistant slices."""
    D_dim = gt_ct.shape[-1]
    slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)

    # We want to show CTs in standard window [-1000, 1000] mapping to [0, 1] for visualization
    def normalize_ct(vol):
        return np.clip((vol + 1024) / 2048.0, 0, 1)

    gt_ct_viz = normalize_ct(gt_ct)
    pred_unet_viz = normalize_ct(pred_unet)
    pred_amix_viz = normalize_ct(pred_amix)

    # Normalize MRI for visualization (0-99th percentile)
    mri_viz = mri.copy()
    mri_viz = np.clip(mri_viz, 0, np.percentile(mri_viz, 99))
    mri_viz = mri_viz / (mri_viz.max() + 1e-8)

    items = [
        (mri_viz, "Input MRI", "gray", (0, 1)),
        (gt_ct_viz, "Ground Truth CT", "gray", (0, 1)),
        (pred_unet_viz, "UNet Baseline", "gray", (0, 1)),
        (pred_amix_viz, "Anatomix", "gray", (0, 1)),
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
            ax.axis("off")
    fig.suptitle(f"Comparison: {subj_id}", fontsize=18, y=0.98)

    out_path = os.path.join(out_dir, "comparison_slices.png")
    plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"🖼️ Saved comparison figure to {out_path}")


# 1ABA025
# 1BA005
# 1HNA013
# 1PA009
# 1THA011


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Export NIfTI CT predictions from UNet and Anatomix models")
    parser.add_argument("--subj_id", type=str, required=True, help="Subject ID (e.g., 1ABA005)")
    parser.add_argument("--name", type=str, required=True, default="name_placeholder", help="Sub-directory name for this experiment")

    parser.add_argument(
        "--unet_ckpt",
        type=str,
        help="Path to UNet baseline checkpoint",
        default="/home/minsukc/MRI2CT/wandb/wandb/run-20260312_000905-xmwpkopk/files/unet_baseline_epoch00330.pt",
        # default="/home/minsukc/MRI2CT/wandb/wandb/run-20260312_000925-bpaq1s1o/files/unet_baseline_epoch00501.pt",
    )
    parser.add_argument(
        "--amix_ckpt",
        type=str,
        help="Path to Anatomix MRI2CT checkpoint",
        default="/home/minsukc/MRI2CT/wandb/wandb/run-20260312_000905-4lyodgtl/files/anatomix_translator_epoch00330.pt",
        # default="/home/minsukc/MRI2CT/wandb/wandb/run-20260312_000914-jxk30spy/files/anatomix_translator_epoch00501.pt",
    )

    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm_registered")
    parser.add_argument("--out_dir", type=str, default="val_viz", help="Output base directory")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--patch_size", type=int, default=128)
    parser.add_argument("--overlap", type=float, default=0.7)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    out_subj_dir = os.path.join(args.out_dir, args.name, args.subj_id)
    os.makedirs(out_subj_dir, exist_ok=True)

    # 1. Load Data
    print(f"📂 Loading data for {args.subj_id}...")
    # Find subject split (train/val/test)
    subj_path_full = None
    for split in ["train", "val", "test"]:
        cand = os.path.join(args.data_dir, split, args.subj_id)
        if os.path.exists(cand):
            subj_path_full = os.path.join(split, args.subj_id)
            break

    if not subj_path_full:
        raise FileNotFoundError(f"Subject {args.subj_id} not found in {args.data_dir}")

    paths = get_subject_paths(args.data_dir, subj_path_full, use_registered=True)

    # Save original affine
    mri_img = tio.ScalarImage(paths["mri"])
    ct_img = tio.ScalarImage(paths["ct"])
    affine = ct_img.affine

    # Preprocess
    subj = tio.Subject(mri=mri_img, ct=ct_img)
    preprocess = DataPreprocessing(patch_size=args.patch_size, enable_safety_padding=False, res_mult=32)
    subj_prep = preprocess(subj)

    mri_tensor = subj_prep["mri"][tio.DATA].unsqueeze(0).to(device)  # [1, 1, X, Y, Z]
    orig_shape = subj_prep["original_shape"].tolist()
    pad_offset = int(subj_prep["pad_offset"]) if "pad_offset" in subj_prep else 0

    # Save inputs
    mri_unpad = unpad(subj_prep["mri"][tio.DATA].unsqueeze(0), orig_shape, pad_offset).squeeze().numpy()
    ct_unpad = unpad(subj_prep["ct"][tio.DATA].unsqueeze(0), orig_shape, pad_offset).squeeze().numpy()

    # Convert GT CT back to HU
    ct_hu = (ct_unpad * 2048.0) - 1024.0

    nib.save(nib.Nifti1Image(mri_unpad, affine), os.path.join(out_subj_dir, "input_mri.nii.gz"))
    nib.save(nib.Nifti1Image(ct_hu, affine), os.path.join(out_subj_dir, "gt_ct.nii.gz"))

    roi_size = (args.patch_size, args.patch_size, args.patch_size)

    # 2. UNet Baseline Inference
    print("🏗️ Running UNet Baseline...")
    unet_model = Unet(dimension=3, input_nc=1, output_nc=1, num_downs=4, ngf=16, final_act="sigmoid").to(device)

    unet_ckpt = torch.load(args.unet_ckpt, map_location=device)
    unet_state = clean_state_dict(unet_ckpt["model_state_dict"])
    unet_model.load_state_dict(unet_state, strict=True)
    unet_model.eval()

    with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=torch.bfloat16):
        pred_unet = sliding_window_inference(
            inputs=mri_tensor,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=unet_model,
            overlap=args.overlap,
            device=device,
        )

    pred_unet_unpad = unpad(pred_unet, orig_shape, pad_offset).squeeze().float().cpu().numpy()
    pred_unet_hu = (pred_unet_unpad * 2048.0) - 1024.0
    nib.save(nib.Nifti1Image(pred_unet_hu, affine), os.path.join(out_subj_dir, "pred_ct_unet.nii.gz"))
    del unet_model, pred_unet

    # 3. Anatomix Inference
    print("🏗️ Running Anatomix MRI2CT...")
    feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
    translator = Unet(dimension=3, input_nc=16, output_nc=1, num_downs=4, ngf=16, final_act="sigmoid").to(device)

    amix_ckpt = torch.load(args.amix_ckpt, map_location=device)
    translator_state = clean_state_dict(amix_ckpt["model_state_dict"])
    translator.load_state_dict(translator_state, strict=True)

    if "feat_extractor_state_dict" in amix_ckpt:
        feat_state = clean_state_dict(amix_ckpt["feat_extractor_state_dict"])
        feat_extractor.load_state_dict(feat_state, strict=True)
    else:
        # Load from default anatomix weights if frozen
        default_amix = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v2.pth"
        if os.path.exists(default_amix):
            feat_state = clean_state_dict(torch.load(default_amix, map_location=device))
            feat_extractor.load_state_dict(feat_state, strict=True)
        else:
            print("⚠️ Warning: Could not find frozen feature extractor weights in checkpoint or default path.")

    feat_extractor.eval()
    translator.eval()

    def amix_forward(x):
        return translator(feat_extractor(x))

    with torch.autocast(device_type="cuda" if "cuda" in args.device else "cpu", dtype=torch.bfloat16):
        pred_amix = sliding_window_inference(
            inputs=mri_tensor,
            roi_size=roi_size,
            sw_batch_size=4,
            predictor=amix_forward,
            overlap=args.overlap,
            device=device,
        )

    pred_amix_unpad = unpad(pred_amix, orig_shape, pad_offset).squeeze().float().cpu().numpy()
    pred_amix_hu = (pred_amix_unpad * 2048.0) - 1024.0
    nib.save(nib.Nifti1Image(pred_amix_hu, affine), os.path.join(out_subj_dir, "pred_ct_amix.nii.gz"))

    # 4. Generate Visual Comparison
    print("🎨 Generating side-by-side comparison figure...")
    create_comparison_figure(mri_unpad, ct_hu, pred_unet_hu, pred_amix_hu, args.subj_id, out_subj_dir)

    print(f"✅ All NIfTI files and visuals saved to {out_subj_dir}/")


if __name__ == "__main__":
    main()
