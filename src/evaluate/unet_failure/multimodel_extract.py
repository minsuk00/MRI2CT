"""Cross-model bone diagnosis: run the same bone analysis on ALL models to see whether
the bone undershoot is universal across architectures (especially koalAI) or specific
to the U-Net.

For each (model, subject) over the 207 center-wise val subjects, compute (body mask only):
  - tissue MAE/bias for air/soft/bone (clipped frame) + cortical bias (clip & raw)
  - comparative oracle: base/air/soft/bone -> body PSNR + body MAE (== reported metrics)
  - bone HU stats: pred mean/max, gt mean, fraction undershot
  - per-model bone calibration histogram (true HU x pred HU)
Also records exact CADS-label coverage of the body (model-independent) for the math note.

GT = raw ct.nii (RAS). Predictions live in full_eval_20260617/volumes/<model>/<subj>/sample.nii.gz.
Note: cwdm/maisi are under training-budget parity -> read as lower bounds.
"""
import os
import argparse
import numpy as np
import nibabel as nib
import pandas as pd
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
VOL = os.path.join(EVAL, "volumes")
PER_SUBJECT_CSV = os.path.join(EVAL, "metrics/per_subject.csv")
OUT = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
MODELS = ["unet", "amix", "koalAI", "mcddpm", "cwdm", "maisi"]
SCEN = ["base", "air", "soft", "bone"]

# per-model bone calibration histogram (true HU x pred HU)
GTE = np.linspace(-200.0, 3000.0, 121)
PRE = np.linspace(-1024.0, 2000.0, 121)


def get_region(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def metricsA(pred, gt, body):
    p01 = (np.clip(pred, -1024, 1024) + 1024) / 2048.0
    g01 = (np.clip(gt, -1024, 1024) + 1024) / 2048.0
    pm, gm = p01 * body, g01 * body
    mae = float(np.abs(pm - gm).mean() * 2048.0)
    mse = max(float(((pm - gm) ** 2).mean()), 1e-10)
    return mae, 10 * np.log10(1.0 / mse)


def process(subj):
    region = get_region(subj)
    sdir = os.path.join(DATA, subj)
    try:
        gt = canon(os.path.join(sdir, "ct.nii"))
        seg = canon(os.path.join(sdir, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        body = canon(os.path.join(sdir, "mask.nii")) > 0
    except Exception as e:
        return [], {}, f"{subj}: gt load {e}"

    n_body = int(body.sum())
    n_labeled = int((seg[body] > 0).sum())     # exact CADS coverage (model-independent)
    air = gt < -300
    soft = (gt >= -300) & (gt <= 200)
    bone = gt > 200
    cort = gt > 1024
    gtb = gt[body]
    air_b, soft_b, bone_b, cort_b = air[body], soft[body], bone[body], cort[body]

    rows = []
    hists = {}
    for model in MODELS:
        pp = os.path.join(VOL, model, subj, "sample.nii.gz")
        if not os.path.exists(pp):
            continue
        try:
            pred = canon(pp)
        except Exception:
            continue
        if pred.shape != gt.shape:
            continue
        if pred.min() >= -50.0 and pred.max() <= 2.5:   # guard: normalized, not HU
            continue
        pb = pred[body]
        errc = np.clip(pb, -1024, 1024) - np.clip(gtb, -1024, 1024)
        aerrc = np.abs(errc)
        errr = pb - gtb

        # oracle scenarios
        def fix(mask):
            pf = pred.copy(); pf[mask] = gt[mask]; return pf
        row = {"model": model, "subj_id": subj, "region": region,
               "n_body": n_body, "n_labeled": n_labeled}
        for nm, msk in [("base", None), ("air", air), ("soft", soft), ("bone", bone)]:
            pf = pred if msk is None else fix(msk)
            mae, psnr = metricsA(pf, gt, body)
            row[f"{nm}_bmae"], row[f"{nm}_psnr"] = mae, psnr
        # tissue MAE/bias (clip)
        for nm, m in [("air", air_b), ("soft", soft_b), ("bone", bone_b)]:
            row[f"mae_{nm}"] = float(aerrc[m].mean()) if m.any() else np.nan
            row[f"bias_{nm}"] = float(errc[m].mean()) if m.any() else np.nan
        row["bias_cortical_clip"] = float(errc[cort_b].mean()) if cort_b.any() else np.nan
        row["bias_cortical_raw"] = float(errr[cort_b].mean()) if cort_b.any() else np.nan
        row["mae_cortical_clip"] = float(aerrc[cort_b].mean()) if cort_b.any() else np.nan
        # bone HU stats
        row["pred_bone_mean"] = float(pb[bone_b].mean()) if bone_b.any() else np.nan
        row["pred_bone_max"] = float(pb[bone_b].max()) if bone_b.any() else np.nan
        row["pred_bone_p99"] = float(np.percentile(pb[bone_b], 99)) if bone_b.any() else np.nan
        row["gt_bone_mean"] = float(gtb[bone_b].mean()) if bone_b.any() else np.nan
        row["frac_bone_under"] = float((pb[bone_b] < gtb[bone_b]).mean()) if bone_b.any() else np.nan
        rows.append(row)
        # calibration hist over bone voxels
        if bone_b.any():
            h, _, _ = np.histogram2d(gtb[bone_b], pb[bone_b], bins=[GTE, PRE])
            hists[model] = h
    return rows, hists, None


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
    print(f"[multimodel] {len(subs)} subjects x {len(MODELS)} models, {args.workers} workers", flush=True)

    allrows, errs = [], []
    hist = {m: np.zeros((len(GTE) - 1, len(PRE) - 1)) for m in MODELS}
    with Pool(args.workers) as pool:
        for i, (rows, hists, err) in enumerate(pool.imap_unordered(process, subs)):
            if err:
                errs.append(err); continue
            allrows.extend(rows)
            for m, h in hists.items():
                hist[m] += h
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(subs)} done", flush=True)

    out = pd.DataFrame(allrows)
    out.to_csv(os.path.join(args.out, "multimodel_bone.csv"), index=False)
    np.savez(os.path.join(args.out, "multimodel_calib.npz"),
             gt_edges=GTE, pred_edges=PRE, **{m: hist[m] for m in MODELS})
    print(f"[multimodel] wrote multimodel_bone.csv ({len(out)} rows) + multimodel_calib.npz", flush=True)
    print("models present:", sorted(out.model.unique()))
    if errs:
        print(f"[multimodel] {len(errs)} errors:", *errs[:8], sep="\n  ")


if __name__ == "__main__":
    main()
