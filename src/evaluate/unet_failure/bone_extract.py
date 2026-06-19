"""Bone deep-dive extraction for the U-Net MR->CT baseline (report 06).

For each of the 207 center-wise val subjects, computes everything needed to answer
two questions honestly:
  1. Is bone the biggest issue / biggest gain? -> a COMPARATIVE oracle: substitute GT
     HU inside air/soft/bone/cortical/skull and recompute the EXACT reported metrics
     (body PSNR, body MAE [clip], full-HU MAE) so air/soft/bone are compared head-to-head.
  2. WHY does bone fail? -> universality of undershoot, localization-vs-magnitude,
     loss imbalance, and the MR->CT information limit (rank-based Spearman of MR vs CT
     within bone vs soft, plus pooled 2D (MR-rank, CT-HU) histograms).

Outputs: bone_subject.csv, mrct_hist.npz.
Sign: err = pred - gt; negative bias = undershoot. GT = raw ct.nii (full HU).
"""
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from scipy import ndimage
from scipy.stats import rankdata, spearmanr
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA_ROOT = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
VOL_ROOT = os.path.join(EVAL, "volumes/unet")
PER_SUBJECT_CSV = os.path.join(EVAL, "metrics/per_subject.csv")
OUT = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
SKULL_LABEL = 7
SOFT_HI, BONE_HI, CAP = 200.0, 200.0, 1024.0
AIR_HI = -300.0
RNG = np.random.RandomState(0)
SUBSAMPLE = 200_000  # per-subject cap for Spearman / MR-CT hist

# pooled 2D (MR percentile-rank in [0,1]) x (CT HU) histograms
MR_BINS = 50
HU_BINS = 120
MR_EDGES = np.linspace(0.0, 1.0, MR_BINS + 1)
HU_EDGES = np.linspace(-1024.0, 2600.0, HU_BINS + 1)


def get_region_key(subj_id):
    mapping = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if subj_id[1:3].upper() in mapping:
        return mapping[subj_id[1:3].upper()]
    if subj_id[1:2].upper() in mapping:
        return mapping[subj_id[1:2].upper()]
    return "abdomen"


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def metricsA(pred_hu, gt_hu, body):
    """Replicate compute_metrics_body: clip[-1024,1024]->[0,1], zero outside body,
    body_mae=|.|.mean()*2048, body_psnr=10log10(1/mse) over FULL volume."""
    p01 = (np.clip(pred_hu, -1024, 1024) + 1024) / 2048.0
    g01 = (np.clip(gt_hu, -1024, 1024) + 1024) / 2048.0
    pm, gm = p01 * body, g01 * body
    mae = float(np.abs(pm - gm).mean() * 2048.0)
    mse = max(float(((pm - gm) ** 2).mean()), 1e-10)
    return mae, 10 * np.log10(1.0 / mse)


def smae(pred_hu, gt_hu, body):
    """Full-HU body-voxel mean MAE (unclipped; == released synthrad_mae, verified in 05).
    Unclipped also makes a tissue 'fix' give exactly zero error in the fixed voxels."""
    return float(np.abs(pred_hu - gt_hu)[body].mean())


def process(subj):
    region = get_region_key(subj)
    sdir = os.path.join(DATA_ROOT, subj)
    try:
        gt = canon(os.path.join(sdir, "ct.nii"))
        seg = canon(os.path.join(sdir, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        body = canon(os.path.join(sdir, "mask.nii")) > 0
        mr = canon(os.path.join(sdir, "moved_mr.nii"))
        pred = canon(os.path.join(VOL_ROOT, subj, "sample.nii.gz"))
    except Exception as e:
        return None, None, None, f"{subj}: load {e}"
    if not (gt.shape == seg.shape == body.shape == mr.shape == pred.shape):
        return None, None, None, f"{subj}: shape mismatch"

    air = gt < AIR_HI
    soft = (gt >= AIR_HI) & (gt <= SOFT_HI)
    bone = gt > BONE_HI
    cort = gt > CAP
    skull = seg == SKULL_LABEL

    # ---- comparative oracle (fix one tissue at a time) ----
    def fix(mask):
        pf = pred.copy()
        pf[mask] = gt[mask]
        return pf
    row = {"subj_id": subj, "region": region}
    for nm, msk in [("base", None), ("air", air), ("soft", soft),
                    ("bone", bone), ("cortical", cort), ("skull", skull)]:
        pf = pred if msk is None else fix(msk)
        mae, psnr = metricsA(pf, gt, body)
        row[f"{nm}_bmae"], row[f"{nm}_psnr"] = mae, psnr
        row[f"{nm}_smae"] = smae(pf, gt, body)

    gtb, pb, mrb = gt[body], pred[body], mr[body]
    err = pb - gtb
    aerr = np.abs(err)
    bone_b = gtb > BONE_HI
    cort_b = gtb > CAP
    soft_b = (gtb >= AIR_HI) & (gtb <= SOFT_HI)

    # ---- universality of undershoot ----
    row.update({
        "n_body": int(body.sum()), "n_bone": int(bone_b.sum()), "n_cort": int(cort_b.sum()),
        "bias_bone_raw": float(err[bone_b].mean()) if bone_b.any() else np.nan,
        "bias_bone_clip": float((np.clip(pb, -1024, 1024) - np.clip(gtb, -1024, 1024))[bone_b].mean()) if bone_b.any() else np.nan,
        "bias_cortical_raw": float(err[cort_b].mean()) if cort_b.any() else np.nan,
        "bias_cortical_clip": float((np.clip(pb, -1024, 1024) - np.clip(gtb, -1024, 1024))[cort_b].mean()) if cort_b.any() else np.nan,
        "frac_bone_under": float((pb[bone_b] < gtb[bone_b]).mean()) if bone_b.any() else np.nan,
        "pred_bone_mean": float(pb[bone_b].mean()) if bone_b.any() else np.nan,
        "pred_bone_max": float(pb[bone_b].max()) if bone_b.any() else np.nan,
        "gt_bone_mean": float(gtb[bone_b].mean()) if bone_b.any() else np.nan,
        # loss imbalance
        "aerr_sum_bone": float(aerr[bone_b].sum()), "total_aerr": float(aerr.sum()),
        # per-voxel severity
        "mae_bone": float(aerr[bone_b].mean()) if bone_b.any() else np.nan,
        "mae_soft": float(aerr[soft_b].mean()) if soft_b.any() else np.nan,
    })

    # ---- localization vs magnitude (on full grid) ----
    pred_bone_full = body & (pred > BONE_HI)
    gt_bone_full = body & (gt > BONE_HI)
    inter = (pred_bone_full & gt_bone_full).sum()
    row["shape_dice"] = float(2 * inter / max(pred_bone_full.sum() + gt_bone_full.sum(), 1))
    row["missed_frac"] = float((gt_bone_full & ~pred_bone_full).sum() / max(gt_bone_full.sum(), 1))
    row["fp_frac"] = float((pred_bone_full & ~gt_bone_full).sum() / max(pred_bone_full.sum(), 1))
    interior = ndimage.binary_erosion(gt_bone_full, iterations=1)
    boundary = gt_bone_full & ~interior
    row["mae_bone_interior"] = float(np.abs(pred - gt)[interior].mean()) if interior.any() else np.nan
    row["mae_bone_boundary"] = float(np.abs(pred - gt)[boundary].mean()) if boundary.any() else np.nan

    # ---- MR -> CT information limit (rank-based, within subject) ----
    # MR percentile-rank within body (scale-invariant; MR is per-volume minmax).
    mr_rank = (rankdata(mrb) - 1) / max(len(mrb) - 1, 1)
    def rho(mask):
        if mask.sum() < 100:
            return np.nan
        r, _ = spearmanr(mrb[mask], gtb[mask])
        return float(r)
    row["rho_bone"] = rho(bone_b)
    row["rho_soft"] = rho(soft_b)
    row["rho_all"] = rho(np.ones_like(bone_b, dtype=bool))
    # conditional CT-HU spread within bone vs soft (how much HU varies at fixed-ish MR)
    row["ctstd_bone"] = float(gtb[bone_b].std()) if bone_b.any() else np.nan
    row["ctstd_soft"] = float(gtb[soft_b].std()) if soft_b.any() else np.nan

    # pooled 2D hists (subsample to bound memory)
    def hist2d(mask):
        idx = np.where(mask)[0]
        if len(idx) == 0:
            return np.zeros((MR_BINS, HU_BINS))
        if len(idx) > SUBSAMPLE:
            idx = RNG.choice(idx, SUBSAMPLE, replace=False)
        h, _, _ = np.histogram2d(mr_rank[idx], gtb[idx], bins=[MR_EDGES, HU_EDGES])
        return h
    h_bone = hist2d(bone_b)
    h_soft = hist2d(soft_b)
    return row, h_bone, h_soft, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = pd.read_csv(PER_SUBJECT_CSV)
    subs = df[df.model == "unet"]["subj_id"].drop_duplicates().tolist()
    if args.limit:
        subs = subs[:args.limit]
    print(f"[bone_extract] {len(subs)} subjects, {args.workers} workers", flush=True)

    rows, errs = [], []
    hb = np.zeros((MR_BINS, HU_BINS))
    hs = np.zeros((MR_BINS, HU_BINS))
    with Pool(args.workers) as pool:
        for i, (row, h_bone, h_soft, err) in enumerate(pool.imap_unordered(process, subs)):
            if err:
                errs.append(err)
                continue
            rows.append(row)
            hb += h_bone
            hs += h_soft
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(subs)} done", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "bone_subject.csv"), index=False)
    np.savez(os.path.join(args.out, "mrct_hist.npz"), bone=hb, soft=hs,
             mr_edges=MR_EDGES, hu_edges=HU_EDGES)
    print(f"[bone_extract] wrote bone_subject.csv ({len(out)} rows), mrct_hist.npz", flush=True)
    if errs:
        print(f"[bone_extract] {len(errs)} errors:", *errs[:10], sep="\n  ")


if __name__ == "__main__":
    main()
