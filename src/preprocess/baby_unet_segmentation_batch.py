import argparse
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.inferers import sliding_window_inference
from tqdm import tqdm

# Add project root and src to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from anatomix.segmentation.segmentation_utils import load_model_v1_2

from common.utils import anatomix_normalize


def discover_subjects(data_root):
    valid_subjects = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue

        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        for subj_id in candidates:
            subj_path = os.path.join(split_path, subj_id)
            # Check for ct.nii or ct.nii.gz
            has_ct = os.path.exists(os.path.join(subj_path, "ct.nii")) or os.path.exists(os.path.join(subj_path, "ct.nii.gz"))

            if has_ct:
                valid_subjects.append({"id": subj_id, "path": subj_path})
    return valid_subjects


def visualize_result(ct_vol, seg_vol, out_path, subj_id):
    """Saves a PNG visualization of the segmentation in 3 planes."""
    if ct_vol.ndim == 4:
        ct_vol = ct_vol[0]

    H, W, D = ct_vol.shape
    # Middle slices for each plane
    cx, cy, cz = H // 2, W // 2, D // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 15))

    # Plane titles and slices
    planes = [
        ("Axial", np.rot90(ct_vol[:, :, cz]), np.rot90(seg_vol[:, :, cz])),
        ("Sagittal", np.rot90(ct_vol[cx, :, :]), np.rot90(seg_vol[cx, :, :])),
        ("Coronal", np.rot90(ct_vol[:, cy, :]), np.rot90(seg_vol[:, cy, :])),
    ]

    for i, (name, ct_slice, seg_slice) in enumerate(planes):
        # Row 0: CT
        axes[0, i].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"{name} CT")
        axes[0, i].axis("off")

        # Row 1: Seg Mask
        axes[1, i].imshow(seg_slice, cmap="tab20", interpolation="nearest", vmin=0, vmax=12)
        axes[1, i].set_title(f"{name} Seg")
        axes[1, i].axis("off")

        # Row 2: Overlay
        axes[2, i].imshow(ct_slice, cmap="gray", vmin=0, vmax=1)
        masked = np.ma.masked_where(seg_slice == 0, seg_slice)
        axes[2, i].imshow(masked, cmap="tab20", alpha=0.5, interpolation="nearest", vmin=0, vmax=12)
        axes[2, i].set_title(f"{name} Overlay")
        axes[2, i].axis("off")

    plt.suptitle(f"Segmentation Visualization: {subj_id}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path)
    plt.close()


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description="Batch Baby U-Net Segmentation for MRI2CT")
    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm_registered")
    parser.add_argument("--weights_path", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/wandb/run-20260222_160128-p0i9chz3/files/seg_baby_unet_epoch_499.pth")
    parser.add_argument("--force", action="store_true", help="Overwrite existing ct_seg.nii")
    parser.add_argument("--batch_size", type=int, default=4, help="Sliding window batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of subjects to process")
    parser.add_argument("--viz", action="store_true", help="Generate PNG visualizations")
    parser.add_argument("--viz_dir", type=str, default="segmentation_viz", help="Directory for visualizations")
    parser.add_argument("--overlap", type=float, default=0.7, help="Sliding window overlap")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"🚀 Using device: {device}")

    if args.viz:
        os.makedirs(args.viz_dir, exist_ok=True)

    # 1. Load Model (11 organs + Brain = 12 classes total)
    print(f"🏗️ Loading Baby U-Net from {args.weights_path}...")
    # n_classes=11 results in 12 output channels in load_model_v1_2
    model = load_model_v1_2(pretrained_ckpt=args.weights_path, n_classes=11, device=device, compile_model=False)
    model.eval()
    model.to(device=device, dtype=torch.bfloat16)
    print("✅ Model loaded.")

    # 2. Discover Subjects
    subjects = discover_subjects(args.data_dir)
    to_process = []
    for s in subjects:
        p_final = os.path.join(s["path"], "ct_seg.nii")
        if not os.path.exists(p_final) or args.force:
            to_process.append(s)

    if not to_process:
        print("All subjects already processed (or none found).")
        return

    if args.limit:
        to_process = to_process[: args.limit]

    print(f"Found {len(subjects)} subjects. Processing {len(to_process)} subjects.")

    # 3. Process
    for subj in tqdm(to_process, desc="Segmenting"):
        subj_path = subj["path"]
        ct_path = os.path.join(subj_path, "ct.nii")
        if not os.path.exists(ct_path):
            ct_path = os.path.join(subj_path, "ct.nii.gz")

        try:
            # Use torchio for consistent loading and orientation
            image = tio.ScalarImage(ct_path)

            # Normalization (CT windowing)
            # anatomix_normalize returns a new tensor
            norm_data = anatomix_normalize(image.data, clip_range=(-1024, 1024))

            # Prepare for sliding window inference [1, 1, H, W, D]
            input_t = norm_data.unsqueeze(0).to(device, dtype=torch.bfloat16)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                output = sliding_window_inference(inputs=input_t, roi_size=(128, 128, 128), sw_batch_size=args.batch_size, predictor=model, overlap=args.overlap, device=device)

                # Argmax to get labels [H, W, D]
                seg = torch.argmax(output, dim=1).squeeze(0)

            # Convert to int8 and save using nibabel for strict dtype control
            seg_np = seg.cpu().numpy().astype(np.int8)

            out_path = os.path.join(subj_path, "ct_seg.nii")
            out_img = nib.Nifti1Image(seg_np, image.affine)
            nib.save(out_img, out_path)

            if args.viz:
                viz_path = os.path.join(args.viz_dir, f"{subj['id']}_seg.png")
                visualize_result(norm_data.cpu().numpy(), seg_np, viz_path, subj["id"])

        except Exception as e:
            print(f"\n❌ Error processing {subj['id']}: {e}")

    print("\n✅ Batch segmentation complete.")


if __name__ == "__main__":
    main()
