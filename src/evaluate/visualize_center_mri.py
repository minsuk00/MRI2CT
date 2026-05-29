"""Visualize input MRIs across brain centers (A=1BA, B=1BB, C=1BC) to inspect
whether the MRIs themselves are a source of domain shift / OOD.

MRIs are normalized per-volume to [0,1] with min-max, exactly as the training
pipeline (`ScaleIntensityd(minv=0, maxv=1)` in src/common/data.py), and displayed
on a shared [0,1] scale so centers are directly comparable.

Outputs:
  <prefix>_center_mri_slices.png      3 centers x 3 subjects, middle axial slice
  <prefix>_center_mri_hist.png        normalized-intensity histogram, body voxels, by center
"""

import argparse
import os
import sys
from glob import glob

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.data import get_subject_paths  # noqa: E402

GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
DEFAULT_ROOT = f"{GPFS}/SynthRAD/1.5mm_registered_flat_masked"
CENTERS = {"A": "1BA", "B": "1BB", "C": "1BC"}
CENTER_COLORS = {"A": "#d95f02", "B": "#1b9e77", "C": "#7570b3"}


def load_canonical(path):
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj, dtype=np.float32), img.affine


def minmax01(vol):
    """Per-volume min-max to [0,1], matching MONAI ScaleIntensityd(minv=0, maxv=1)."""
    lo, hi = float(vol.min()), float(vol.max())
    if hi <= lo:
        return np.zeros_like(vol)
    return (vol - lo) / (hi - lo)


# world-axis row in the affine that defines each anatomical plane's slicing axis
_PLANE_WORLD_ROW = {"axial": 2, "coronal": 1, "sagittal": 0}  # S, A, R


def mid_slice(vol, affine, plane="axial"):
    """Middle slice in the given anatomical plane, oriented superior-up for display."""
    ax = int(np.argmax(np.abs(affine[_PLANE_WORLD_ROW[plane], :3])))
    sl = np.take(vol, vol.shape[ax] // 2, axis=ax)
    return np.rot90(sl)  # both axial and coronal slices read upright after one CCW rot


def list_center_subjects(root, prefix):
    """All resolvable subjects for a center prefix (e.g. '1BA'), sorted."""
    out = []
    for p in sorted(glob(os.path.join(root, f"{prefix}*"))):
        if not os.path.isdir(p):
            continue
        d = os.path.basename(p)
        try:
            get_subject_paths(root, d)
        except Exception:
            continue
        out.append(d)
    return out


def pick_subjects(root, prefix, n):
    return list_center_subjects(root, prefix)[:n]


def render_all_center(args):
    """Grid of mid-plane slices for every subject of one center."""
    c = args.all_center
    plane = args.all_plane
    subs = list_center_subjects(args.root_dir, CENTERS[c])
    print(f"Center {c} ({CENTERS[c]}): {len(subs)} subjects")

    ncols = args.ncols
    nrows = int(np.ceil(len(subs) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(1.9 * ncols, 2.1 * nrows))
    axes = np.atleast_2d(axes)
    for i in range(nrows * ncols):
        ax = axes[i // ncols, i % ncols]
        ax.set_xticks([]); ax.set_yticks([])
        if i >= len(subs):
            ax.axis("off")
            continue
        mri, aff = load_canonical(get_subject_paths(args.root_dir, subs[i])["mri"])
        ax.imshow(mid_slice(minmax01(mri), aff, plane), cmap="gray", vmin=0.0, vmax=1.0)
        ax.set_title(subs[i], fontsize=8)
    fig.suptitle(f"Center {c} ({CENTERS[c]}) — all {len(subs)} subjects, {plane} middle slice, "
                 f"per-volume min-max [0,1]", fontsize=14, y=1 - 0.5 / nrows)
    fig.subplots_adjust(left=0.01, right=0.99, top=1 - 1.4 / nrows, bottom=0.01,
                        wspace=0.05, hspace=0.22)
    out = os.path.join(args.out_dir, f"{args.prefix}_center{c}_all_{plane}.png")
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print("Wrote:\n  " + out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root_dir", default=DEFAULT_ROOT)
    p.add_argument("--per_center", type=int, default=3)
    p.add_argument("--out_dir", default="evaluation_results/brain_unet_centerwise_vs_random")
    p.add_argument("--prefix", default="unet_brain")
    p.add_argument("--all_center", default=None, choices=list(CENTERS),
                   help="If set (A/B/C), render ALL subjects of that center in one grid and exit.")
    p.add_argument("--all_plane", default="sagittal", choices=list(_PLANE_WORLD_ROW))
    p.add_argument("--ncols", type=int, default=10)
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    if args.all_center:
        render_all_center(args)
        return

    # resolve subjects per center
    subjects = {c: pick_subjects(args.root_dir, pre, args.per_center) for c, pre in CENTERS.items()}
    for c, subs in subjects.items():
        print(f"Center {c} ({CENTERS[c]}): {subs}")

    # load + normalize each volume once; collect body voxels for the histogram
    vols = {}          # subj -> (mri_norm, affine)
    hist_data = {}     # center -> normalized body voxels
    for c, subs in subjects.items():
        center_vals = []
        for subj in subs:
            paths = get_subject_paths(args.root_dir, subj)
            mri, aff = load_canonical(paths["mri"])
            mri_n = minmax01(mri)
            vols[subj] = (mri_n, aff)
            if "body_mask" in paths:
                m, _ = load_canonical(paths["body_mask"])
                center_vals.append(mri_n[m > 0.5])
            else:
                center_vals.append(mri_n[mri_n > 1e-3])
        hist_data[c] = np.concatenate(center_vals) if center_vals else np.array([])

    # ---- slice grids (rows = centers, cols = subjects), one per plane ----
    n = args.per_center
    out_paths = []
    for plane in ("axial", "coronal", "sagittal"):
        fig, axes = plt.subplots(len(CENTERS), n, figsize=(3.2 * n, 3.5 * len(CENTERS)))
        axes = np.atleast_2d(axes)
        for r, (c, subs) in enumerate(subjects.items()):
            for col in range(n):
                ax = axes[r, col]
                ax.set_xticks([]); ax.set_yticks([])
                if col >= len(subs):
                    ax.axis("off")
                    continue
                subj = subs[col]
                mri_n, aff = vols[subj]
                ax.imshow(mid_slice(mri_n, aff, plane), cmap="gray", vmin=0.0, vmax=1.0)
                ax.set_title(subj, fontsize=10)
                if col == 0:
                    ax.set_ylabel(f"Center {c}\n({CENTERS[c]})", fontsize=12, rotation=0,
                                  labelpad=34, va="center")
        fig.suptitle(f"Input MRI by center — {plane} middle slice, "
                     f"per-volume min-max [0,1] (shared display scale)",
                     fontsize=14, y=1 - 0.18 / len(CENTERS))
        fig.subplots_adjust(left=0.08, right=0.99, top=1 - 0.6 / len(CENTERS), bottom=0.02,
                            wspace=0.05, hspace=0.18)
        out = os.path.join(args.out_dir, f"{args.prefix}_center_mri_slices_{plane}.png")
        fig.savefig(out, dpi=130)
        plt.close(fig)
        out_paths.append(out)

    # ---- histogram by center (body voxels), plane-independent ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 80)
    for c, vals in hist_data.items():
        if vals.size == 0:
            continue
        ax2.hist(vals, bins=bins, histtype="step", density=True, linewidth=2,
                 color=CENTER_COLORS[c], label=f"Center {c} ({CENTERS[c]}), n={len(subjects[c])}")
    ax2.set_xlabel("normalized MRI intensity (body voxels)")
    ax2.set_ylabel("density")
    ax2.set_title("Normalized input-MRI intensity distribution by center")
    ax2.legend()
    fig2.tight_layout()
    out2 = os.path.join(args.out_dir, f"{args.prefix}_center_mri_hist.png")
    fig2.savefig(out2, dpi=130)
    plt.close(fig2)

    print("Wrote:")
    for pth in out_paths + [out2]:
        print("  " + pth)


if __name__ == "__main__":
    main()
