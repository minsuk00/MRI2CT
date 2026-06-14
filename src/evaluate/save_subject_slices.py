"""Per-subject middle-axial-slice PNGs, one clean image per source.

For every subject in the split, writes
    figures/subject_slices/<subj>/{mri,gtct,amix,unet,maisi,mcddpm,cwdm,koalAI}.png
each a borderless grayscale image of the middle axial slice (z = D//2). CT panels
use a region-appropriate HU window (brain/head_neck -100..100, else -1024..1024);
MRI uses its own 1-99 percentile window.

Reads the unified volume tree (volumes/<model>/<subj>/sample.nii.gz, GT from
volumes/amix/<subj>/target.nii.gz) + the dataset MRI (moved_mr*).

Usage:
    python src/evaluate/save_subject_slices.py --eval_root /gpfs/.../full_eval_20260601
"""
import argparse
import os
import sys
from glob import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.common.data import get_region_key, get_split_subjects  # noqa: E402

SYNTHRAD_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
MODELS = ["amix", "unet", "maisi", "mcddpm", "cwdm", "koalAI"]
HU_WINDOWS = {"brain": (-100, 100), "head_neck": (-100, 100),
              "abdomen": (-1024, 1024), "thorax": (-1024, 1024), "pelvis": (-1024, 1024)}


def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def mid_slice(vol):
    """Middle axial slice (last axis), rotated for upright display."""
    z = vol.shape[-1] // 2
    return np.rot90(vol[:, :, z])


def save_png(path, sl, vmin, vmax):
    plt.imsave(path, sl, cmap="gray", vmin=vmin, vmax=vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    args = ap.parse_args()

    subjects = get_split_subjects(args.split_file, args.split_name)
    out_root = os.path.join(args.eval_root, "figures", "subject_slices")
    n_ok = 0
    for subj in subjects:
        region = get_region_key(subj)
        vmin, vmax = HU_WINDOWS.get(region, (-1024, 1024))
        out_dir = os.path.join(out_root, subj)
        os.makedirs(out_dir, exist_ok=True)

        # Numeric prefixes so the per-source PNGs sort in display order:
        # 1_mri, 2_gtct, 3_amix, 4_unet, 5_maisi, 6_mcddpm, 7_cwdm, 8_koalAI.
        # MRI (own percentile window)
        mri_p = sorted(glob(os.path.join(SYNTHRAD_ROOT, subj, "moved_mr*.nii*")))
        if mri_p:
            mri = load_vol(mri_p[0])
            mlo, mhi = float(np.percentile(mri, 1)), float(np.percentile(mri, 99))
            save_png(os.path.join(out_dir, "1_mri.png"), mid_slice(mri), mlo, mhi)

        # GT CT
        gt_p = os.path.join(args.eval_root, "volumes", "amix", subj, "target.nii.gz")
        if not os.path.exists(gt_p):
            cand = glob(os.path.join(args.eval_root, "volumes", "*", subj, "target.nii.gz"))
            gt_p = cand[0] if cand else None
        if gt_p:
            save_png(os.path.join(out_dir, "2_gtct.png"), mid_slice(load_vol(gt_p)), vmin, vmax)

        # Each model's prediction (numbered from 3 in MODELS order)
        for i, mdl in enumerate(MODELS, start=3):
            sp = os.path.join(args.eval_root, "volumes", mdl, subj, "sample.nii.gz")
            if os.path.exists(sp):
                save_png(os.path.join(out_dir, f"{i}_{mdl}.png"), mid_slice(load_vol(sp)), vmin, vmax)

        n_ok += 1
        if n_ok % 25 == 0:
            print(f"[slices] {n_ok}/{len(subjects)}", flush=True)

    print(f"[slices] done — wrote per-source PNGs for {n_ok} subjects under {out_root}")


if __name__ == "__main__":
    main()
