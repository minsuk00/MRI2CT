"""Causal oracle test for report 09's claim (density, not localization).

We apply a MONOTONIC intensity recalibration to the sCT -- histogram-match the sCT
to the GT CT within the body -- which by construction CANNOT move any voxel in
space (it is a single global value->value LUT). We then re-run the SAME segmenter
on the recalibrated sCT and measure bone-union Dice.

Logic: if the U-Net localized bone correctly and only got the HU magnitude wrong,
then fixing the intensity distribution (no spatial change) must recover bone Dice
toward the real-CT ceiling. Any residual gap is genuine localization error.

Recovery = (Dice_recal - Dice_sCT) / (Dice_ceiling - Dice_sCT), per subject.
Balanced subset (default 6 per region). Needs GPU. Prints the recovery split.
"""
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import torch
from skimage.exposure import match_histograms

from seg_infer import load_seg_model, segment, DATA, EVAL  # reuse exact model + inference

RUN = os.path.join("/home/minsukc/MRI2CT", "evaluation_results/unet_failure_20260619")
BONE = [7, 27, 28, 29, 30]


def canon(p, dt=np.float32):  # bare-array loader (seg_infer.canon returns (array, affine))
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def get_region_key(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_region", type=int, default=6)
    args = ap.parse_args()

    subs = sorted(p.name for p in os.scandir(os.path.join(EVAL, "seg", "unet")) if p.is_dir())
    by_reg = {}
    for s in subs:
        by_reg.setdefault(get_region_key(s), []).append(s)
    pick = []
    for r, ss in by_reg.items():
        pick += ss[: args.per_region]
    print(f"[recalib] {len(pick)} subjects ({args.per_region}/region)", flush=True)

    device = "cuda"
    model = load_seg_model(device)
    rows = []
    for i, s in enumerate(pick):
        gt_ct = canon(os.path.join(DATA, s, "ct.nii"))
        sct = canon(os.path.join(EVAL, "volumes", "unet", s, "sample.nii.gz"))
        body = canon(os.path.join(DATA, s, "mask.nii")) > 0
        gt_seg = canon(os.path.join(DATA, s, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        real_seg = canon(os.path.join(EVAL, "seg", "realct", s, "seg.nii.gz"), np.int16)
        sct_seg = canon(os.path.join(EVAL, "seg", "unet", s, "seg.nii.gz"), np.int16)

        # monotonic histogram match sCT -> GT within body
        recal = sct.copy()
        recal[body] = match_histograms(sct[body], gt_ct[body])
        recal[~body] = -1024.0
        recal_seg = segment(model, recal, device)

        gb = np.isin(gt_seg, BONE)
        d_ceil = dice(np.isin(real_seg, BONE), gb)
        d_sct = dice(np.isin(sct_seg, BONE), gb)
        d_recal = dice(np.isin(recal_seg, BONE), gb)
        rows.append({"subj": s, "region": get_region_key(s),
                     "dice_ceiling": d_ceil, "dice_sct": d_sct, "dice_recal": d_recal})
        if (i + 1) % 5 == 0:
            print(f"  {i + 1}/{len(pick)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUN, "verify_recalib.csv"), index=False)
    ce, sc, re_ = df.dice_ceiling.mean(), df.dice_sct.mean(), df.dice_recal.mean()
    rec = (re_ - sc) / (ce - sc) if (ce - sc) > 1e-6 else float("nan")
    print("\n== CAUSAL ORACLE: monotonic intensity recalibration, then re-segment ==")
    print(f"  bone Dice  sCT {sc:.3f}  --recalibrate(intensity only)-->  {re_:.3f}   ceiling {ce:.3f}")
    print(f"  -> recovers {rec*100:.0f}% of the sCT-to-ceiling bone-Dice gap from a purely intensity (no-spatial) fix")
    print(f"  residual gap to ceiling after fix: {(ce - re_):.3f} Dice (genuine localization component)")


if __name__ == "__main__":
    main()
