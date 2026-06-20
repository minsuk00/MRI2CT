"""Self-contained CADS-group error decomposition table for the U-Net sCT.
Reads the raw volumes directly (GT CT, sCT, body mask, GT CADS seg) for all 207
center-wise val subjects and prints the group table + whole-body micro/macro MAE.
No intermediate CSV.

  error = sCT - GT (HU), inside the body mask, per GT CADS label -> 4 groups.
  micro = pool all subjects' voxels, then average   (Sum|err| / Sum n)
  macro = MAE per subject, then average over subjects  (matches synthrad_mae)
"""

import glob
import os
from multiprocessing import Pool

import nibabel as nib
import numpy as np
import pandas as pd

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"

BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]
GROUPS = ["bone (5 labels)", "air-organs (airway+lung)", "soft (other CADS)", "unlabeled (CADS=0)"]

# map each of the 35 CADS labels (0-34) to a group index into GROUPS
GRP_OF_LABEL = np.full(35, 2, dtype=np.int8)  # default: soft
GRP_OF_LABEL[BONE] = 0
GRP_OF_LABEL[AIRORG] = 1
GRP_OF_LABEL[0] = 3  # unlabeled (CADS Background)


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def process(s):
    """Per-subject group sums: n, Sum|err|, Sum(err) for each of the 4 groups."""
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception:
        return None

    err = (sct - gt)[body]
    ae = np.abs(err)
    grp = GRP_OF_LABEL[seg[body]]  # group index per body voxel

    n = np.bincount(grp, minlength=4)
    sabs = np.bincount(grp, weights=ae, minlength=4)
    serr = np.bincount(grp, weights=err, minlength=4)
    return {"subj": s, "n": n, "sabs": sabs, "serr": serr}


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    R = [r for r in Pool(8).map(process, subs) if r]
    print(f"{len(R)} subjects")

    N = np.stack([r["n"] for r in R])  # (subj, 4)
    SABS = np.stack([r["sabs"] for r in R])
    SERR = np.stack([r["serr"] for r in R])

    tot_n, tot_abs = N.sum(), SABS.sum()
    micro_mae = SABS.sum(0) / N.sum(0)  # pooled per group
    bias = SERR.sum(0) / N.sum(0)
    voxshare = 100 * N.sum(0) / tot_n
    errmass = 100 * SABS.sum(0) / tot_abs

    # macro: per-subject group MAE, averaged over subjects that have the group
    with np.errstate(invalid="ignore", divide="ignore"):
        per_subj_mae = SABS / N  # (subj, 4), nan where group absent
    macro_mae = np.nanmean(per_subj_mae, axis=0)

    out = pd.DataFrame(
        {
            "CADS group": GROUPS,
            "% body vox": voxshare,
            "micro MAE": micro_mae,
            "macro MAE": macro_mae,
            "bias": bias,
            "% of body error": errmass,
        }
    )

    body_micro = tot_abs / tot_n
    body_macro = (SABS.sum(1) / N.sum(1)).mean()  # per-subject body MAE, then averaged

    pd.set_option("display.float_format", lambda v: f"{v:.1f}")
    print(out.to_string(index=False))
    print(f"\nwhole body:  micro MAE {body_micro:.1f}   macro MAE {body_macro:.1f} (= synthrad_mae)")
    print(f"check: sum %vox {voxshare.sum():.1f}   sum %error {errmass.sum():.1f}")


if __name__ == "__main__":
    main()
