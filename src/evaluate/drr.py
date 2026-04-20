import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from nanodrr.camera import make_k_inv, make_rt_inv
from nanodrr.data import Subject
from nanodrr.drr import render
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SDD = 1020.0
DELX = 2.0
HEIGHT = WIDTH = 256
NUM_ANGLES = 4

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DRR_OUTPUT_DIR = PROJECT_ROOT / "evaluation_results" / "DRR"

# --- configure inline ---
SUBJ_ID = "1THB011"
GT_CT_PATH = f"/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat/{SUBJ_ID}/ct.nii"
PRED_CT_PATH = f"/home/minsukc/MRI2CT/dataset/predictions/1.5x1.5x1.5mm_registered/20260419_0307_amix/epoch_276/pred_{SUBJ_ID}.nii.gz"
# ------------------------

TRANSLATION = [0.0, 850.0, 0.0]  # mm, relative to isocenter
ORIENTATION = "AP"
RES_VLIM = 0.5  # symmetric colorbar limit for residual map (attenuation units)


def render_drr(ct_path, device, num_angles=NUM_ANGLES, height=HEIGHT, width=WIDTH):
    subject = Subject.from_filepath(ct_path).to(device)
    k_inv = make_k_inv(SDD, DELX, DELX, 0.0, 0.0, height, width, device=device)
    sdd_tensor = torch.tensor([SDD], device=device)
    thetas = torch.linspace(0, 180, num_angles, device=device)
    rotations = torch.stack([thetas, torch.zeros_like(thetas), torch.zeros_like(thetas)], dim=1)
    translations = torch.tensor([TRANSLATION], device=device).repeat(num_angles, 1)
    rt_inv = make_rt_inv(rotations, translations, orientation=ORIENTATION, isocenter=subject.isocenter)
    img = render(subject, k_inv, rt_inv, sdd_tensor, height, width)
    return img, thetas.cpu().tolist()  # [num_angles, 1, H, W], list of angles in degrees


def compute_metrics(gt, pred, data_range):
    """gt, pred: [H, W] numpy arrays."""
    mae = np.mean(np.abs(gt - pred))
    psnr = peak_signal_noise_ratio(gt, pred, data_range=data_range)
    ssim = structural_similarity(gt, pred, data_range=data_range)
    return mae, psnr, ssim


def make_drr_figure(gt_np, pred_np, thetas, title="DRR Comparison", caption=None):
    """Build and return the 3-row DRR comparison figure (GT / sCT / Residual).

    Args:
        gt_np: [N, H, W] numpy array of GT DRR projections
        pred_np: [N, H, W] numpy array of sCT DRR projections
        thetas: list of N angle values in degrees
        title: figure suptitle
        caption: optional bottom caption string
    Returns:
        matplotlib Figure (caller is responsible for plt.close)
    """
    num_angles = gt_np.shape[0]
    vmin = np.stack([gt_np, pred_np]).min()
    vmax = np.stack([gt_np, pred_np]).max()
    data_range = vmax - vmin
    residual = gt_np - pred_np

    angle_labels = [f"{a:.0f}°" for a in thetas]

    fig = plt.figure(figsize=(num_angles * 4 + 0.6, 12))
    gs = GridSpec(
        3,
        num_angles + 1,
        figure=fig,
        width_ratios=[1] * num_angles + [0.03],
        hspace=0.08,
        wspace=0.05,
        left=0.07,
        right=0.97,
        top=0.94,
        bottom=0.08,
    )

    axes = [[fig.add_subplot(gs[r, c]) for c in range(num_angles)] for r in range(3)]
    cax_gray = fig.add_subplot(gs[0:2, -1])
    cax_res = fig.add_subplot(gs[2, -1])

    fig.suptitle(title, fontsize=14, fontweight="bold")
    row_labels = ["GT CT", "sCT", "Residual\n(GT − sCT)"]

    for col in range(num_angles):
        gray_im = axes[0][col].imshow(gt_np[col], cmap="gray", vmin=vmin, vmax=vmax)
        axes[1][col].imshow(pred_np[col], cmap="gray", vmin=vmin, vmax=vmax)
        res_im = axes[2][col].imshow(residual[col], cmap="RdBu_r", vmin=-RES_VLIM, vmax=RES_VLIM)

        mae, psnr, ssim = compute_metrics(gt_np[col], pred_np[col], data_range)
        axes[2][col].set_xlabel(
            f"MAE={mae:.4f}\nPSNR={psnr:.1f}dB  SSIM={ssim:.3f}",
            fontsize=7.5,
            labelpad=3,
        )
        axes[0][col].set_title(angle_labels[col], fontsize=11)
        for row in range(3):
            axes[row][col].set_xticks([])
            axes[row][col].set_yticks([])

    for row, label in enumerate(row_labels):
        axes[row][0].annotate(
            label,
            xy=(0, 0.5),
            xycoords="axes fraction",
            xytext=(-6, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=10,
            fontweight="bold",
            rotation=90,
        )

    fig.colorbar(gray_im, cax=cax_gray, label="attenuation")
    fig.colorbar(res_im, cax=cax_res, label="GT − sCT")

    if caption:
        fig.text(0.5, 0.02, caption, ha="center", fontsize=7, color="gray")

    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_ct", default=GT_CT_PATH)
    parser.add_argument("--pred_ct", default=PRED_CT_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Rendering GT CT...")
    img_gt, thetas = render_drr(args.gt_ct, device)
    print("Rendering predicted CT...")
    img_pred, _ = render_drr(args.pred_ct, device)

    gt_np = img_gt[:, 0].cpu().numpy()
    pred_np = img_pred[:, 0].cpu().numpy()

    vmin = np.stack([gt_np, pred_np]).min()
    vmax = np.stack([gt_np, pred_np]).max()
    print(f"DRR intensity range — vmin={vmin:.4f}  vmax={vmax:.4f}  range={vmax - vmin:.4f}")

    cam_info = f"SDD={SDD}mm  delx=dely={DELX}mm  res={HEIGHT}×{WIDTH}  translation={TRANSLATION}mm  orientation={ORIENTATION}"
    caption = f"pred: {Path(args.pred_ct).resolve()}\n{cam_info}"

    fig = make_drr_figure(gt_np, pred_np, thetas, caption=caption)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    subj_tag = Path(args.pred_ct).stem.replace(".nii", "")
    DRR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = DRR_OUTPUT_DIR / f"{timestamp}_{subj_tag}.png"

    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
