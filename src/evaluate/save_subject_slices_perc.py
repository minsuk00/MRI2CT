"""Per-subject middle-axial-slice PNGs for the UNet perceptual-loss ablation.

For each subject, writes figures/subject_slices/<subj>/{1_mri,2_gtct,3_noperc,
4_perceptual,5_perc_nodice}.png — one borderless grayscale middle-axial slice per
source. CT panels use a region-appropriate HU window (brain/head_neck -100..100,
else -1024..1024); MRI uses its own 1-99 percentile window. Predictions read from
<raw_dir>/<tag>/<subj>/sample.nii.gz; GT CT + MRI from the dataset.

Default subjects = those present in the full_eval subject_slices dir (so the
variant slices line up with the 6-model report); override with --subjects.

Usage:
    python src/evaluate/save_subject_slices_perc.py --eval_root /gpfs/.../perc_ablation_20260603
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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from common.data import get_region_key, get_split_subjects  # noqa: E402

DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
FULL_EVAL_SLICES = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260601/figures/subject_slices"
# (tag, numbered filename stem) in column order
VARIANTS = [("9xmodnhn_ep400", "noperc"), ("06e850ny_ep400", "perceptual"), ("ye820cq0_ep400", "perc_nodice")]
HU_WINDOWS = {"brain": (-100, 100), "head_neck": (-100, 100),
              "abdomen": (-1024, 1024), "thorax": (-1024, 1024), "pelvis": (-1024, 1024)}


def load_vol(path):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=np.float32)


def mid_slice(vol):
    z = vol.shape[-1] // 2
    return np.rot90(vol[:, :, z])


def save_png(path, sl, vmin, vmax):
    plt.imsave(path, sl, cmap="gray", vmin=vmin, vmax=vmax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--split_file", default="splits/center_wise_split.txt")
    ap.add_argument("--split_name", default="val")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Subject IDs; default = full_eval subject_slices dirs.")
    args = ap.parse_args()

    if args.subjects:
        subjects = args.subjects
    elif os.path.isdir(FULL_EVAL_SLICES):
        subjects = sorted(os.path.basename(d.rstrip("/")) for d in glob(os.path.join(FULL_EVAL_SLICES, "*/")))
    else:
        subjects = get_split_subjects(args.split_file, args.split_name)

    raw_dir = os.path.join(args.eval_root, "raw")
    out_root = os.path.join(args.eval_root, "figures", "subject_slices")
    n_ok = 0
    for subj in subjects:
        region = get_region_key(subj)
        vmin, vmax = HU_WINDOWS.get(region, (-1024, 1024))
        out_dir = os.path.join(out_root, subj)
        os.makedirs(out_dir, exist_ok=True)

        mri_p = sorted(glob(os.path.join(DATA_ROOT, subj, "moved_mr*.nii*")))
        if mri_p:
            mri = load_vol(mri_p[0])
            mlo, mhi = float(np.percentile(mri, 1)), float(np.percentile(mri, 99))
            save_png(os.path.join(out_dir, "1_mri.png"), mid_slice(mri), mlo, mhi)

        gt_p = os.path.join(DATA_ROOT, subj, "ct.nii")
        if not os.path.exists(gt_p):
            gt_p = os.path.join(DATA_ROOT, subj, "ct.nii.gz")
        if os.path.exists(gt_p):
            save_png(os.path.join(out_dir, "2_gtct.png"), mid_slice(load_vol(gt_p)), vmin, vmax)

        for i, (tag, stem) in enumerate(VARIANTS, start=3):
            sp = os.path.join(raw_dir, tag, subj, "sample.nii.gz")
            if os.path.exists(sp):
                save_png(os.path.join(out_dir, f"{i}_{stem}.png"), mid_slice(load_vol(sp)), vmin, vmax)
        n_ok += 1

    print(f"[slices] wrote per-source PNGs for {n_ok} subjects under {out_root}")


if __name__ == "__main__":
    main()
