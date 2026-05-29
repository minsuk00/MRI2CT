"""Six-baseline MRI->CT comparison figure for one or more subjects.

Usage:
    python src/evaluate/visualize_six_baselines.py 1ABB116
    python src/evaluate/visualize_six_baselines.py 1ABB116 1THB011 1HNC117 1PC011 1BC050

Reads precomputed predictions and metrics from the 20260520 sweep; no recompute.
"""

import argparse
import os
import sys
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.data import get_region_key  # noqa: E402


SYNTHRAD_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"

MODELS = [
    ("UNet", "unet", "/home/minsukc/MRI2CT/wandb/runs/20260507_0952_9xmodnhn/validate_20260520_061949_job50505874"),
    ("Amix v1.4", "amix_v1_4", "/home/minsukc/MRI2CT/wandb/runs/20260509_1413_6hjye9gp/validate_20260520_061949_job50505873"),
    ("MAISI", "maisi", "/home/minsukc/MRI2CT/wandb/runs/20260511_0244_5hprtpwl/validate_20260520_002407_job50492057"),
    ("cWDM", "cwdm", "/home/minsukc/MRI2CT/wandb/wandb/run-20260518_175723-smg8thkr/files/checkpoints/validate_cwdm_ddim100_arr50492831"),
    ("MC-IDDPM", "mcddpm", "/home/minsukc/MRI2CT/wandb/runs/20260515_0836_a3g28rez/validate_mcddpm_paper50_ov0.25_mc1_arr50493246"),
]

METRICS_ROOT = "/home/minsukc/MRI2CT/evaluation_results/six_baselines_20260520_064315"
DEFAULT_OUT_DIR = os.path.join(METRICS_ROOT, "figures")

HU_WINDOWS = {
    "brain": (-100, 100),
    "abdomen": (-1000, 1000),
    "thorax": (-1000, 1000),
    "head_neck": (-1000, 1000),
    "pelvis": (-1000, 1000),
}


def find_pred_path(model_root, subj_id):
    """Locate sample.nii.gz for a subject, handling flat and sharded layouts."""
    flat = os.path.join(model_root, subj_id, "sample.nii.gz")
    if os.path.exists(flat):
        return flat
    sharded = glob(os.path.join(model_root, "shard_*_of_*", subj_id, "sample.nii.gz"))
    if sharded:
        return sharded[0]
    raise FileNotFoundError(f"No sample.nii.gz for {subj_id} under {model_root}")


def find_target_path(model_root, subj_id):
    flat = os.path.join(model_root, subj_id, "target.nii.gz")
    if os.path.exists(flat):
        return flat
    sharded = glob(os.path.join(model_root, "shard_*_of_*", subj_id, "target.nii.gz"))
    if sharded:
        return sharded[0]
    raise FileNotFoundError(f"No target.nii.gz for {subj_id} under {model_root}")


def load_metrics_table(metrics_path):
    """Parse a `validate_metrics_combined.txt` file into {subj_id: {col: val}}.

    Files have `# ...` comments, a `==== Per subject ====` banner, then a
    whitespace-separated header line starting with `subj_id`.
    """
    rows = {}
    header = None
    with open(metrics_path) as f:
        for line in f:
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or stripped.startswith("="):
                continue
            parts = stripped.split()
            if header is None:
                if parts[0] != "subj_id":
                    continue
                header = parts
                continue
            if len(parts) < 2:
                continue
            row = {}
            for i, col in enumerate(header[1:], start=1):
                try:
                    row[col] = float(parts[i])
                except (ValueError, IndexError):
                    row[col] = None
            rows[parts[0]] = row
    return rows


def load_all_metrics():
    out = {}
    for _, model_key, _ in MODELS:
        path = os.path.join(METRICS_ROOT, model_key, "validate_metrics_combined.txt")
        out[model_key] = load_metrics_table(path)
    return out


def metrics_caption(row):
    if row is None:
        return "(no metrics)"
    fields = [
        ("PSNR", "psnr", "{:.2f}"),
        ("SSIM", "ssim", "{:.3f}"),
        ("MAE HU", "mae_hu", "{:.2f}"),
        ("Dice-all", "dice_score_all", "{:.3f}"),
        ("Dice-bone", "dice_score_bone", "{:.3f}"),
    ]
    parts = []
    for label, key, fmt in fields:
        v = row.get(key)
        parts.append(f"{label}: {fmt.format(v)}" if v is not None else f"{label}: -")
    return "\n".join(parts)


def render_subject(subj_id, metrics_all, out_dir, num_slices, ct_window=None, suffix=""):
    region = get_region_key(subj_id)
    if ct_window is not None:
        ct_vmin, ct_vmax = ct_window
    else:
        ct_vmin, ct_vmax = HU_WINDOWS.get(region, (-1000, 1000))

    mri_path = sorted(glob(os.path.join(SYNTHRAD_ROOT, subj_id, "moved_mr*.nii*")))
    if not mri_path:
        raise FileNotFoundError(f"No moved_mr*.nii* in {SYNTHRAD_ROOT}/{subj_id}")
    # Validators save predictions in RAS (MONAI Orientationd); raw MRI is LPS
    # in this dataset, so canonicalize to match.
    mri = nib.as_closest_canonical(nib.load(mri_path[0])).get_fdata()

    unet_root = MODELS[0][2]
    ct_gt = nib.load(find_target_path(unet_root, subj_id)).get_fdata()

    preds = []
    for name, key, root in MODELS:
        path = find_pred_path(root, subj_id)
        vol = nib.load(path).get_fdata()
        # cWDM saves predictions in normalized [0, 1] space; convert to HU.
        if key == "cwdm" and float(vol.min()) >= -1.0 and float(vol.max()) <= 2.0:
            vol = vol * 2048.0 - 1024.0
        preds.append((name, key, vol))

    D = ct_gt.shape[-1]
    slice_idx = np.linspace(0.1 * D, 0.9 * D, num_slices, dtype=int)

    columns = [("MRI", mri, "per-volume", None), ("GT CT", ct_gt, "ct", None)]
    for name, key, vol in preds:
        columns.append((name, vol, "ct", metrics_all[key].get(subj_id)))

    n_cols = len(columns)
    n_rows = num_slices
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3.5 * n_rows), squeeze=False)
    plt.subplots_adjust(wspace=0.05, hspace=0.1, top=0.92)

    mri_vmin, mri_vmax = float(mri.min()), float(mri.max())

    for r, z in enumerate(slice_idx):
        for c, (name, vol, kind, row) in enumerate(columns):
            ax = axes[r, c]
            if kind == "ct":
                ax.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=ct_vmin, vmax=ct_vmax)
            else:
                ax.imshow(np.rot90(vol[:, :, z]), cmap="gray", vmin=mri_vmin, vmax=mri_vmax)
            ax.axis("off")
            if r == 0:
                if kind == "ct" and row is not None:
                    ax.set_title(f"{name}\n{metrics_caption(row)}", fontsize=9)
                else:
                    ax.set_title(name, fontsize=11)

    fig.suptitle(
        f"Subject: {subj_id}  ({region})   |   CT window: [{ct_vmin}, {ct_vmax}] HU",
        fontsize=14,
        y=0.985,
    )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{subj_id}{suffix}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[viz] wrote {out_path}")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Six-baseline MRI->CT comparison figure.")
    parser.add_argument("subjects", nargs="+", help="Subject IDs, e.g. 1ABB116 1THB011")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--num_slices", type=int, default=5)
    parser.add_argument(
        "--ct_window",
        default=None,
        help="Override CT window as 'vmin,vmax' (HU). Suffix added to filename.",
    )
    parser.add_argument(
        "--suffix",
        default="",
        help="Suffix appended to output filename (e.g. '_wide').",
    )
    args = parser.parse_args()

    ct_window = None
    if args.ct_window:
        ct_window = tuple(float(v) for v in args.ct_window.split(","))
        if len(ct_window) != 2:
            raise ValueError("--ct_window must be 'vmin,vmax'")

    metrics_all = load_all_metrics()
    for subj in args.subjects:
        try:
            render_subject(subj, metrics_all, args.out_dir, args.num_slices, ct_window, args.suffix)
        except Exception as e:
            print(f"[viz] FAILED {subj}: {e}")


if __name__ == "__main__":
    main()
