"""
Analyze 128^3 corner patches across all thorax center A (1THA*) and center B (1THB*)
volumes to test whether center B has more background-only corner patches.

Hypothesis: center B has a smaller FOV, so 128^3 corner patches may be pure background,
causing instance-norm models to output noisy gray blobs (haven't seen all-background
patches during training on center A).

Usage:
    micromamba run -n mrct python src/evaluate/analyze_corner_patches.py
    micromamba run -n mrct python src/evaluate/analyze_corner_patches.py \
        --root /gpfs/.../1.5mm_registered_flat --patch_size 128 --bg_thresh 0.05
"""

import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np

ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat"
PATCH_SIZE = 128
# Threshold for "background": normalized CT value < 0.05 → roughly < -922 HU
BG_THRESH = 0.05


def load_ct(subject_dir):
    """Load CT volume, try .nii.gz then .nii. Returns numpy array (H, W, D) float32."""
    for name in ("ct.nii.gz", "ct.nii"):
        path = os.path.join(subject_dir, name)
        if os.path.exists(path):
            img = nib.load(path)
            return img.get_fdata(dtype=np.float32), img.header.get_zooms()[:3]
    return None, None


def normalize_ct(vol, clip_lo=-1024.0, clip_hi=1024.0):
    """Anatomix-style CT normalization: clip then scale to [0, 1]."""
    vol = np.clip(vol, clip_lo, clip_hi)
    return (vol - clip_lo) / (clip_hi - clip_lo)


def extract_corner_patches(vol, patch_size):
    """
    Extract 8 corner patches of shape (patch_size^3) from a 3-D volume.
    Returns list of 8 arrays, each shape (P, P, P).
    Each corner is the cube touching the nearest face of the volume.
    """
    P = patch_size
    H, W, D = vol.shape
    # Clamp patch size to volume dimensions
    ph = min(P, H)
    pw = min(P, W)
    pd = min(P, D)

    corners = []
    for hi in (0, H - ph):
        for wi in (0, W - pw):
            for di in (0, D - pd):
                patch = vol[hi : hi + ph, wi : wi + pw, di : di + pd]
                corners.append(patch)
    return corners


def is_background_patch(patch, bg_thresh):
    """Return True if ALL voxels in the patch are below bg_thresh."""
    return float(patch.max()) < bg_thresh


def analyze_center(subjects, root, patch_size, bg_thresh):
    """
    For each subject, load CT, normalize, extract 8 corner patches,
    check how many are background-only.

    Returns dict with aggregate stats and per-subject details.
    """
    results = []
    for subj_id in subjects:
        subj_dir = os.path.join(root, subj_id)
        vol, zooms = load_ct(subj_dir)
        if vol is None:
            print(f"  [SKIP] {subj_id}: no ct.nii(.gz) found")
            continue

        norm = normalize_ct(vol)
        corners = extract_corner_patches(norm, patch_size)

        bg_flags = [is_background_patch(p, bg_thresh) for p in corners]
        n_bg = sum(bg_flags)
        max_vals = [float(p.max()) for p in corners]

        results.append(
            {
                "subj_id": subj_id,
                "shape": vol.shape,
                "zooms": tuple(round(float(z), 2) for z in zooms),
                "n_corners": len(corners),
                "n_bg_corners": n_bg,
                "has_any_bg_corner": n_bg > 0,
                "corner_maxvals": max_vals,
            }
        )

    if not results:
        return results, {}

    n_subjects = len(results)
    n_with_bg = sum(r["has_any_bg_corner"] for r in results)
    total_corners = sum(r["n_corners"] for r in results)
    total_bg_corners = sum(r["n_bg_corners"] for r in results)

    shapes = np.array([r["shape"] for r in results])
    stats = {
        "n_subjects": n_subjects,
        "n_with_any_bg_corner": n_with_bg,
        "pct_subjects_with_bg": 100.0 * n_with_bg / n_subjects,
        "total_corners": total_corners,
        "total_bg_corners": total_bg_corners,
        "pct_corners_bg": 100.0 * total_bg_corners / total_corners,
        "shape_mean": shapes.mean(axis=0).round(1).tolist(),
        "shape_min": shapes.min(axis=0).tolist(),
        "shape_max": shapes.max(axis=0).tolist(),
    }

    return results, stats


def get_subjects(root, prefix):
    dirs = sorted(
        d for d in os.listdir(root) if d.startswith(prefix) and os.path.isdir(os.path.join(root, d))
    )
    return dirs


def print_per_subject(results, max_show=10):
    """Print per-subject table; truncate long lists."""
    print(f"  {'Subject':<12}  {'Shape':>18}  {'BG/8':>6}  {'Corner max-vals'}")
    print("  " + "-" * 80)
    shown = 0
    bg_subjects = [r for r in results if r["has_any_bg_corner"]]
    non_bg = [r for r in results if not r["has_any_bg_corner"]]
    # Show all subjects with BG patches first, then up to max_show - len(bg) non-BG
    display = bg_subjects + non_bg[: max(0, max_show - len(bg_subjects))]
    for r in display:
        maxvals_str = ", ".join(f"{v:.3f}" for v in r["corner_maxvals"])
        flag = " *** BG ***" if r["has_any_bg_corner"] else ""
        print(f"  {r['subj_id']:<12}  {str(r['shape']):>18}  {r['n_bg_corners']:>3}/{r['n_corners']:<3}  [{maxvals_str}]{flag}")
        shown += 1
    remaining = len(results) - shown
    if remaining > 0:
        print(f"  ... ({remaining} more subjects not shown)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--patch_size", type=int, default=PATCH_SIZE)
    parser.add_argument("--bg_thresh", type=float, default=BG_THRESH,
                        help="Normalized CT threshold below which a voxel is considered background")
    args = parser.parse_args()

    print(f"Root:        {args.root}")
    print(f"Patch size:  {args.patch_size}^3")
    print(f"BG thresh:   {args.bg_thresh}  (~{args.bg_thresh * 2048 - 1024:.0f} HU)")

    subjs_A = get_subjects(args.root, "1THA")
    subjs_B = get_subjects(args.root, "1THB")
    print(f"\nCenter A (1THA*): {len(subjs_A)} subjects")
    print(f"Center B (1THB*): {len(subjs_B)} subjects")

    for center_label, subjects in [("Center A (1THA*)", subjs_A), ("Center B (1THB*)", subjs_B)]:
        print(f"\n{'=' * 70}")
        print(f"  {center_label}")
        print(f"{'=' * 70}")
        results, stats = analyze_center(subjects, args.root, args.patch_size, args.bg_thresh)

        if not stats:
            print("  No subjects processed.")
            continue

        print(f"  Subjects processed:          {stats['n_subjects']}")
        print(f"  Subjects w/ ≥1 BG corner:    {stats['n_with_any_bg_corner']}  ({stats['pct_subjects_with_bg']:.1f}%)")
        print(f"  Background corners (all 8):   {stats['total_bg_corners']} / {stats['total_corners']}  ({stats['pct_corners_bg']:.1f}%)")
        print(f"  Volume shape  mean: {stats['shape_mean']}")
        print(f"                 min: {stats['shape_min']}")
        print(f"                 max: {stats['shape_max']}")
        print()
        print_per_subject(results)

    print(f"\n{'=' * 70}")
    print("Done.")


if __name__ == "__main__":
    main()
