"""Oracle counterfactual: replace the UNet prediction with ground-truth HU inside a
tissue category, then recompute the EXACT reported metrics. Answers "if tissue X were
perfect, how much would PSNR/MAE improve?" -> proves whether bone is the real bottleneck.

Track A body_psnr / body_mae replicate compute_metrics_body (clip [-1024,1024] -> [0,1],
zero outside body mask, mean over FULL volume, psnr=10log10(1/mse)).
Track B synthrad_mae replicates body-voxel mean over [-1024,3000] (the full-HU truth).
"""
import os, argparse, numpy as np, nibabel as nib, pandas as pd
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA_ROOT = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL_ROOT = os.path.join(REPO, "evaluation_results/full_eval_20260609/volumes")
PER_SUBJECT_CSV = os.path.join(REPO, "evaluation_results/full_eval_20260609/metrics/per_subject.csv")
SKULL_LABEL = 7


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def metricsA(pred_hu, gt_hu, body):
    """Track A: body_mae_hu (full-vol zeroed mean *2048) and body_psnr (clip 1024)."""
    p01 = (np.clip(pred_hu, -1024, 1024) + 1024) / 2048.0
    g01 = (np.clip(gt_hu, -1024, 1024) + 1024) / 2048.0
    pm = p01 * body
    gm = g01 * body
    mae = np.abs(pm - gm).mean() * 2048.0
    mse = max(((pm - gm) ** 2).mean(), 1e-10)
    psnr = 10 * np.log10(1.0 / mse)
    return mae, psnr


def synthrad_mae(pred_hu, gt_hu, body):
    """Track B: body-voxel mean MAE over [-1024,3000]."""
    p = np.clip(pred_hu, -1024, 3000)
    g = np.clip(gt_hu, -1024, 3000)
    return float(np.abs(p - g)[body].mean())


def process(args):
    subj, region, model = args
    sdir = os.path.join(DATA_ROOT, subj)
    try:
        gt = canon(os.path.join(sdir, "ct.nii"))
        seg = canon(os.path.join(sdir, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
        body = canon(os.path.join(sdir, "mask.nii")) > 0
        pred = canon(os.path.join(VOL_ROOT, model, subj, "sample.nii.gz"))
    except Exception as e:
        return None
    if not (gt.shape == seg.shape == body.shape == pred.shape):
        return None

    air = gt < -300
    soft = (gt >= -300) & (gt <= 200)
    bone = gt > 200
    cort = gt > 1024
    skull = seg == SKULL_LABEL

    def fix(mask):
        pf = pred.copy()
        pf[mask] = gt[mask]
        return pf

    out = {"model": model, "subj_id": subj, "region": region}
    scenarios = {
        "base": pred, "fix_air": fix(air), "fix_soft": fix(soft),
        "fix_bone": fix(bone), "fix_cortical": fix(cort), "fix_skull": fix(skull),
    }
    for name, pf in scenarios.items():
        mae, psnr = metricsA(pf, gt, body)
        out[f"{name}_psnr"] = psnr
        out[f"{name}_bmae"] = mae
        out[f"{name}_smae"] = synthrad_mae(pf, gt, body)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616"))
    ap.add_argument("--models", nargs="+", default=["unet"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(PER_SUBJECT_CSV)
    subs = df[df.model == "unet"][["subj_id", "region"]].drop_duplicates().values.tolist()
    if args.limit:
        subs = subs[:args.limit]
    jobs = [(s, r, m) for (s, r) in subs for m in args.models]
    print(f"[oracle] {len(jobs)} jobs", flush=True)

    rows = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(process, jobs)):
            if r:
                rows.append(r)
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(args.out, "oracle_fix.csv"), index=False)
    print(f"[oracle] wrote oracle_fix.csv ({len(out)} rows)", flush=True)

    # sanity vs reported body_psnr
    chk = out.merge(df[["model", "subj_id", "body_psnr", "synthrad_mae"]], on=["model", "subj_id"], how="left")
    chk["dpsnr"] = (chk["base_psnr"] - chk["body_psnr"]).abs()
    chk["dsmae"] = (chk["base_smae"] - chk["synthrad_mae"]).abs()
    print("[gate] |base_psnr - csv body_psnr| mean/max:",
          round(chk.dpsnr.mean(), 4), round(chk.dpsnr.max(), 4))
    print("[gate] |base_smae - csv synthrad_mae| mean/max:",
          round(chk.dsmae.mean(), 4), round(chk.dsmae.max(), 4))

    print("\n===== ORACLE FIX: mean body_psnr (Track A) by region =====")
    cols = ["base_psnr", "fix_air_psnr", "fix_soft_psnr", "fix_bone_psnr", "fix_cortical_psnr", "fix_skull_psnr"]
    for m in args.models:
        sub = out[out.model == m]
        print(f"\n--- {m} ---")
        print(sub.groupby("region")[cols].mean().round(2).to_string())
        print("ALL:", sub[cols].mean().round(2).to_dict())
    print("\n===== ORACLE FIX: mean synthrad_mae (full-HU, lower=better) =====")
    cols2 = ["base_smae", "fix_air_smae", "fix_soft_smae", "fix_bone_smae", "fix_cortical_smae"]
    for m in args.models:
        sub = out[out.model == m]
        print(f"--- {m} ---")
        print(sub.groupby("region")[cols2].mean().round(1).to_string())


if __name__ == "__main__":
    main()
