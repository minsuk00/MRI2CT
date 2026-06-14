"""Convert normalized [0,1] prediction volumes to HU, in place (idempotent).

cWDM's validate.py historically saved sample.nii.gz / target.nii.gz in [0,1] (over
[-1024,1024]) instead of HU, breaking the cross-model "volumes are HU" contract. This
rescales any volume whose intensity range looks normalized ([-1,2]) to HU via
val*span+lo; volumes already in HU are left untouched.

Usage:
    python src/evaluate/convert_normalized_to_hu.py \
        --glob '/gpfs/.../full_eval_20260601/raw/cwdm/shard_*/*/sample.nii.gz' \
        '/gpfs/.../full_eval_20260601/raw/cwdm/shard_*/*/target.nii.gz' \
        --lo -1024 --hi 1024
"""
import argparse
import glob as globlib

import nibabel as nib
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", nargs="+", required=True, help="One or more glob patterns.")
    ap.add_argument("--lo", type=float, default=-1024.0)
    ap.add_argument("--hi", type=float, default=1024.0)
    args = ap.parse_args()
    span = args.hi - args.lo

    paths = []
    for g in args.glob:
        paths += globlib.glob(g)
    paths = sorted(set(paths))
    print(f"[convert] {len(paths)} files matched")

    converted = skipped = 0
    for p in paths:
        img = nib.load(p)
        arr = np.asarray(img.dataobj, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if lo >= -50.0 and hi <= 2.5:  # normalized [0,1]-ish → convert
            hu = arr * span + args.lo
            nib.save(nib.Nifti1Image(hu, img.affine, img.header), p)
            converted += 1
        else:
            skipped += 1  # already HU
    print(f"[convert] converted {converted}, skipped(already HU) {skipped}")


if __name__ == "__main__":
    main()
