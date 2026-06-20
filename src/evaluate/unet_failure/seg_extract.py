"""Per-subject extraction for the seg-downstream U-Net failure analysis (report 09).

For each eval subject we have, on one shared per-subject grid:
  - GT CADS seg  : dataset/<S>/cads_grouped_35_labels_seg.nii.gz  (labels 0..34)
  - babyseg(realCT): seg/realct/<S>/seg.nii.gz   (di54npq3 on the real CT  -> ceiling)
  - babyseg(sCT) : seg/unet/<S>/seg.nii.gz       (di54npq3 on the U-Net sCT)
  - GT CT HU     : dataset/<S>/ct.nii  (raw HU)
  - sCT HU       : full_eval/volumes/unet/<S>/sample.nii.gz  (HU, clipped to [-1024,1024])
  - body mask    : dataset/<S>/mask.nii

Methodology (exactly how each value is produced):
  - Dice(A,B) = 2|A&B| / (|A|+|B|), every set intersected with the body mask.
    Label absent in GT for a subject -> NaN (excluded from that subject's means).
  - dice_ceiling = Dice(babyseg(realCT)==l, GT==l): the segmenter's own reachable
    score on the real CT. dice_sct = Dice(babyseg(sCT)==l, GT==l). The drop
    (ceiling - sCT) is the part attributable to the synthetic CT (localization).
  - per-GT-label HU bias/MAE inside the GT CADS ROI (GT==l & body), raw HU:
    bias = mean(sCT) - mean(GT), MAE = mean|sCT - GT|. Segmenter-independent
    (density). Bone undershoot shows here.
  - bone confusion: over GT-bone voxels (GT in {7,27,28,29,30} & body), the label
    babyseg assigns, pooled across subjects, for realCT vs sCT -> shows bone being
    relabeled (e.g. as muscle/soft) on the sCT.

Coarse tissue groups (headline, robust): bone={7,27,28,29,30},
air/lung={9 airway,13 lungs}, soft = all other in-body labels (1..34 minus those).

Outputs: seg_per_subject.csv, seg_per_label.csv (long), bone_confusion.npz.
"""
import os
import argparse
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")

CLASS_NAMES = [
    "Background", "Brain - other", "CSF", "Eyes & optic pathway", "Face & oral soft tissue",
    "Gray matter", "Head & neck glands", "Skull", "White matter", "Airway", "Breast",
    "Esophagus", "Heart", "Lungs", "Thoracic cavity", "Abdominal cavity", "Adrenals",
    "Bowel", "Gallbladder", "Kidneys", "Liver", "Pancreas", "Spleen", "Stomach", "Bladder",
    "Prostate & seminal vesicle", "Blood vessels", "Bone - other", "Limb & girdle bones",
    "Spine", "Thoracic cage", "Gland - other", "Muscle", "Spinal cord", "Subcutaneous tissue",
]
BONE = [7, 27, 28, 29, 30]
AIR = [9, 13]
SOFT = [l for l in range(1, 35) if l not in BONE and l not in AIR]
NL = 35


def get_region_key(subj_id):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if subj_id[1:3].upper() in m:
        return m[subj_id[1:3].upper()]
    if subj_id[1:2].upper() in m:
        return m[subj_id[1:2].upper()]
    return "abdomen"


def canon(path, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(path)).dataobj, dtype=dt)


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def process(s):
    try:
        gt_seg = canon(os.path.join(DATA, s, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        real_seg = canon(os.path.join(EVAL, "seg", "realct", s, "seg.nii.gz"), np.int16)
        sct_seg = canon(os.path.join(EVAL, "seg", "unet", s, "seg.nii.gz"), np.int16)
        gt_ct = canon(os.path.join(DATA, s, "ct.nii"))
        sct_ct = canon(os.path.join(EVAL, "volumes", "unet", s, "sample.nii.gz"))
        body = canon(os.path.join(DATA, s, "mask.nii")) > 0
    except Exception as e:
        return None, None, None, None, f"{s}: {e}"

    region = get_region_key(s)
    gtb, rsb, scb = gt_seg[body], real_seg[body], sct_seg[body]
    gct, sct = gt_ct[body], sct_ct[body]

    def grp(arr, labs):
        return np.isin(arr, labs)

    row = {"subj": s, "region": region, "n_body": int(body.sum())}
    for nm, labs in [("bone", BONE), ("air", AIR), ("soft", SOFT)]:
        gmask = grp(gtb, labs)
        row[f"dice_ceil_{nm}"] = dice(grp(rsb, labs), gmask)
        row[f"dice_sct_{nm}"] = dice(grp(scb, labs), gmask)
        row[f"n_{nm}"] = int(gmask.sum())
    # bone-union HU density (raw HU inside GT bone)
    bmask = grp(gtb, BONE)
    if bmask.any():
        row["bone_gt_hu"] = float(gct[bmask].mean())
        row["bone_pred_hu"] = float(sct[bmask].mean())
        row["bone_bias"] = float((sct[bmask] - gct[bmask]).mean())
        row["bone_mae"] = float(np.abs(sct[bmask] - gct[bmask]).mean())
    else:
        row["bone_gt_hu"] = row["bone_pred_hu"] = row["bone_bias"] = row["bone_mae"] = np.nan

    # per-label rows
    lrows = []
    for l in range(1, NL):
        gmask = gtb == l
        n = int(gmask.sum())
        if n < 50:
            continue
        d_ceil = dice(rsb == l, gmask)
        d_sct = dice(scb == l, gmask)
        gh = float(gct[gmask].mean())
        ph = float(sct[gmask].mean())
        lrows.append({
            "subj": s, "region": region, "label": l, "name": CLASS_NAMES[l],
            "is_bone": l in BONE, "n_vox": n,
            "dice_ceil": d_ceil, "dice_sct": d_sct,
            "gt_hu": gh, "pred_hu": ph, "bias": ph - gh,
            "mae": float(np.abs(sct[gmask] - gct[gmask]).mean()),
        })

    # bone confusion: over GT-bone voxels, what label does babyseg assign
    conf_real = np.bincount(rsb[bmask], minlength=NL)[:NL] if bmask.any() else np.zeros(NL, int)
    conf_sct = np.bincount(scb[bmask], minlength=NL)[:NL] if bmask.any() else np.zeros(NL, int)
    return row, lrows, conf_real.astype(np.int64), conf_sct.astype(np.int64), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(RUN, exist_ok=True)

    subs = sorted(os.path.basename(p.path) for p in os.scandir(os.path.join(EVAL, "seg", "unet")) if p.is_dir())
    if args.limit:
        subs = subs[: args.limit]
    print(f"[seg_extract] {len(subs)} subjects, {args.workers} workers", flush=True)

    rows, lrows, errs = [], [], []
    conf_real = np.zeros(NL, np.int64)
    conf_sct = np.zeros(NL, np.int64)
    with Pool(args.workers) as pool:
        for i, (row, lr, cr, cs, err) in enumerate(pool.imap_unordered(process, subs)):
            if err:
                errs.append(err)
                continue
            rows.append(row)
            lrows.extend(lr)
            conf_real += cr
            conf_sct += cs
            if (i + 1) % 25 == 0:
                print(f"  {i + 1}/{len(subs)}", flush=True)

    pd.DataFrame(rows).to_csv(os.path.join(RUN, "seg_per_subject.csv"), index=False)
    pd.DataFrame(lrows).to_csv(os.path.join(RUN, "seg_per_label.csv"), index=False)
    np.savez(os.path.join(RUN, "bone_confusion.npz"),
             conf_real=conf_real, conf_sct=conf_sct, names=np.array(CLASS_NAMES))
    print(f"[seg_extract] done. {len(rows)} ok, {len(errs)} errors", flush=True)
    for e in errs[:10]:
        print("  ERR", e, flush=True)


if __name__ == "__main__":
    main()
