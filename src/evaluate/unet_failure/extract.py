"""Per-subject + per-CADS-label error extraction for the U-Net MR->CT baseline.

Fresh extraction (does NOT reuse error_anatomy/extract.py). For each of the 207
center-wise validation subjects: load the saved U-Net prediction (HU), the RAW
ground-truth CT (full HU to ~2976), the 35-label CADS segmentation, and the body
mask, all reoriented to canonical RAS so they share the prediction's grid.

Validation was scored on GT *clipped* to [-1024, 1024] (data.py ScaleIntensityRanged
clip=True, hu_range=2048), so every error is computed in BOTH frames:
  - Frame C (clipped, primary): clip(pred,-1024,1024) - clip(gt,-1024,1024).
    Matches how the model was validated/selected -> genuine in-range model failure.
  - Frame R (raw): pred - raw_gt. The excess (R - C) over bone is the clipped-target
    ceiling component, unrecoverable by construction.
Tissue / cortical masks are always defined on the RAW GT HU.

Outputs under OUT: per_subject.csv, per_label.csv, bone_hist.npz.
Sign convention: err = pred - gt; negative bias = undershoot.
"""
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA_ROOT = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
VOL_ROOT = os.path.join(EVAL, "volumes/unet")
PER_SUBJECT_CSV = os.path.join(EVAL, "metrics/per_subject.csv")
OUT = os.path.join(REPO, "evaluation_results/unet_failure_20260619")

BONE_LABELS = {7, 27, 28, 29, 30}
LABEL_NAMES = {
    0: "background", 1: "brain_other", 2: "csf", 3: "eyes", 4: "face_oral", 5: "gray_matter",
    6: "hn_glands", 7: "skull", 8: "white_matter", 9: "airway", 10: "breast", 11: "esophagus",
    12: "heart", 13: "lungs", 14: "thoracic_cavity", 15: "abdominal_cavity", 16: "adrenals",
    17: "bowel", 18: "gallbladder", 19: "kidneys", 20: "liver", 21: "pancreas", 22: "spleen",
    23: "stomach", 24: "bladder", 25: "prostate_sv", 26: "blood_vessels", 27: "bone_other",
    28: "limb_girdle", 29: "spine", 30: "thoracic_cage", 31: "gland_other", 32: "muscle",
    33: "spinal_cord", 34: "subcutaneous",
}
LABEL_CADS_REGION = {
    1: "H&N", 2: "H&N", 3: "H&N", 4: "H&N", 5: "H&N", 6: "H&N", 7: "Skeleton", 8: "H&N", 9: "H&N",
    10: "Thorax", 11: "Thorax", 12: "Thorax", 13: "Thorax", 14: "Thorax", 15: "Abdomen", 16: "Abdomen",
    17: "Abdomen", 18: "Abdomen", 19: "Abdomen", 20: "Abdomen", 21: "Abdomen", 22: "Abdomen", 23: "Abdomen",
    24: "Pelvis", 25: "Pelvis", 26: "Vessels", 27: "Skeleton", 28: "Skeleton", 29: "Skeleton",
    30: "Skeleton", 31: "Whole-body", 32: "Whole-body", 33: "H&N", 34: "Whole-body",
}

AIR_HI = -300.0     # GT HU < -300            -> air
SOFT_HI = 200.0     # -300..200               -> soft ; >200 -> bone
CAP = 1024.0        # cortical threshold / clip ceiling
MIN_VOX = 50        # per-label min voxels

# 2D bone-HU joint histogram (raw GT x pred), accumulated across subjects
H_BINS = 256
GT_EDGES = np.linspace(-200.0, 3000.0, H_BINS + 1)
PRED_EDGES = np.linspace(-1024.0, 1200.0, H_BINS + 1)


def get_region_key(subj_id):
    mapping = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if not subj_id or len(subj_id) < 2:
        return "abdomen"
    if subj_id[1:3].upper() in mapping:
        return mapping[subj_id[1:3].upper()]
    if subj_id[1:2].upper() in mapping:
        return mapping[subj_id[1:2].upper()]
    return "abdomen"


def _load(p, dtype=np.float32):
    # Predictions are canonical RAS; raw CT/seg/mask are native. as_closest_canonical
    # is a pure axis flip/permute (no interpolation -> HU + int labels preserved).
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dtype)


def _mean(a):
    return float(a.mean()) if a.size else np.nan


def _pct(a, q):
    return float(np.percentile(a, q)) if a.size else np.nan


def process_subject(subj):
    region = get_region_key(subj)
    sdir = os.path.join(DATA_ROOT, subj)
    try:
        gt = _load(os.path.join(sdir, "ct.nii"))
        seg = _load(os.path.join(sdir, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        body = _load(os.path.join(sdir, "mask.nii")) > 0
        pred = _load(os.path.join(VOL_ROOT, subj, "sample.nii.gz"))
    except Exception as e:
        return None, [], None, f"{subj}: load fail {e}"
    if not (gt.shape == seg.shape == body.shape == pred.shape):
        return None, [], None, f"{subj}: shape mismatch gt{gt.shape} seg{seg.shape} mask{body.shape} pred{pred.shape}"
    # guard: prediction must be in HU, not normalized [0,1]
    if pred.min() >= -50.0 and pred.max() <= 2.5:
        return None, [], None, f"{subj}: pred looks normalized (min {pred.min():.2f} max {pred.max():.2f})"

    gtb = gt[body]
    pb = pred[body]
    segb = seg[body]
    gtb_c = np.clip(gtb, -1024.0, 1024.0)
    pb_c = np.clip(pb, -1024.0, 1024.0)
    err_r = pb - gtb
    err_c = pb_c - gtb_c
    aerr_r = np.abs(err_r)
    aerr_c = np.abs(err_c)

    air = gtb < AIR_HI
    soft = (gtb >= AIR_HI) & (gtb <= SOFT_HI)
    bone = gtb > SOFT_HI
    cort = gtb > CAP
    mid = (gtb > SOFT_HI) & (gtb <= CAP)

    def tissue(name, m):
        return {
            f"mae_{name}_raw": _mean(aerr_r[m]), f"bias_{name}_raw": _mean(err_r[m]),
            f"mae_{name}_clip": _mean(aerr_c[m]), f"bias_{name}_clip": _mean(err_c[m]),
            f"n_{name}": int(m.sum()),
            f"aerr_sum_{name}_raw": float(aerr_r[m].sum()), f"aerr_sum_{name}_clip": float(aerr_c[m].sum()),
        }

    row = {
        "subj_id": subj, "region": region, "n_body": int(body.sum()), "n_total": int(gt.size),
        "mae_raw": _mean(aerr_r), "mae_clip": _mean(aerr_c),
        "total_aerr_sum_raw": float(aerr_r.sum()), "total_aerr_sum_clip": float(aerr_c.sum()),
    }
    for nm, m in [("air", air), ("soft", soft), ("bone", bone), ("midbone", mid), ("cortical", cort)]:
        row.update(tissue(nm, m))
    # bone pred-HU distribution + ceiling diagnostics
    row.update({
        "pred_bone_mean": _mean(pb[bone]), "pred_bone_p50": _pct(pb[bone], 50),
        "pred_bone_p95": _pct(pb[bone], 95), "pred_bone_max": float(pb[bone].max()) if bone.any() else np.nan,
        "gt_bone_mean": _mean(gtb[bone]), "gt_bone_p95": _pct(gtb[bone], 95),
        "gt_bone_max": float(gtb[bone].max()) if bone.any() else np.nan,
        "gt_bone_clip_mean": _mean(gtb_c[bone]),
        "pred_near_ceiling_frac": float((pb[bone] >= 850).mean()) if bone.any() else np.nan,
        "pred_capped_frac": float((pb[bone] >= 1000).mean()) if bone.any() else np.nan,
        "gt_cortical_frac": float(cort.sum()) / max(int(bone.sum()), 1),
        # oracle: total abs-error if bone voxels were predicted perfectly (error->0)
        "bone_zeroed_aerr_sum_raw": float(aerr_r.sum() - aerr_r[bone].sum()),
        "bone_zeroed_aerr_sum_clip": float(aerr_c.sum() - aerr_c[bone].sum()),
    })

    # 2D bone HU joint histogram (raw GT x pred)
    if bone.any():
        h2d, _, _ = np.histogram2d(gtb[bone], pb[bone], bins=[GT_EDGES, PRED_EDGES])
    else:
        h2d = np.zeros((H_BINS, H_BINS))

    # per-label rows
    label_rows = []
    for lab in (int(v) for v in np.unique(segb) if v != 0):
        m = segb == lab
        n = int(m.sum())
        if n < MIN_VOX:
            continue
        label_rows.append({
            "subj_id": subj, "region": region, "label": lab,
            "name": LABEL_NAMES.get(lab, str(lab)), "cads_region": LABEL_CADS_REGION.get(lab, "?"),
            "is_bone": lab in BONE_LABELS, "n": n,
            "mae_raw": _mean(aerr_r[m]), "mae_clip": _mean(aerr_c[m]),
            "bias_raw": _mean(err_r[m]), "bias_clip": _mean(err_c[m]),
            "gt_mean": _mean(gtb[m]), "gt_clip_mean": _mean(gtb_c[m]), "pred_mean": _mean(pb[m]),
            "gt_p95": _pct(gtb[m], 95), "pred_p95": _pct(pb[m], 95),
            "gt_frac_gt1024": float((gtb[m] > CAP).mean()),
            "pred_capped_frac": float((pb[m] >= 1000).mean()),
            "aerr_sum_clip": float(aerr_c[m].sum()),
        })
    return row, label_rows, h2d, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="debug: only first N subjects")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(PER_SUBJECT_CSV)
    subs = df[df.model == "unet"]["subj_id"].drop_duplicates().tolist()
    if args.limit:
        subs = subs[:args.limit]
    print(f"[extract] {len(subs)} unet subjects, {args.workers} workers", flush=True)

    summ, labels, errs = [], [], []
    hist = np.zeros((H_BINS, H_BINS))
    with Pool(args.workers) as pool:
        for i, (row, lrows, h2d, err) in enumerate(pool.imap_unordered(process_subject, subs)):
            if err:
                errs.append(err)
                continue
            summ.append(row)
            labels.extend(lrows)
            hist += h2d
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(subs)} done", flush=True)

    sdf = pd.DataFrame(summ)
    ldf = pd.DataFrame(labels)
    sdf.to_csv(os.path.join(args.out, "per_subject.csv"), index=False)
    ldf.to_csv(os.path.join(args.out, "per_label.csv"), index=False)
    np.savez(os.path.join(args.out, "bone_hist.npz"), hist=hist, gt_edges=GT_EDGES, pred_edges=PRED_EDGES)
    print(f"[extract] wrote per_subject.csv ({len(sdf)} rows), per_label.csv ({len(ldf)} rows), bone_hist.npz", flush=True)
    if errs:
        print(f"[extract] {len(errs)} errors:")
        for e in errs[:20]:
            print("   ", e)


if __name__ == "__main__":
    main()
