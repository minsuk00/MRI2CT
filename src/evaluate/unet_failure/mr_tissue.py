"""Proof that MR cannot encode bone density: pool the within-subject MR intensity
percentile-rank for air / soft / bone / cortical voxels (~10 subjects per region),
and measure the distribution overlap. High bone-vs-soft / cortical-vs-air overlap
means a given MR brightness maps to many CT densities -> MR is uninformative for bone.

Writes mr_tissue_pool.npz (rank samples per tissue) + mr_tissue_stats.json (medians, overlaps).
"""
import os
import json
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import rankdata
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
RNG = np.random.RandomState(0)


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def work(s):
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        mr = canon(f"{DATA}/{s}/moved_mr.nii")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
    except Exception:
        return None
    gtb, mrb = gt[body], mr[body]
    rk = (rankdata(mrb) - 1) / max(len(mrb) - 1, 1)   # MR percentile-rank within this subject's body
    out = {}
    for nm, msk in [("air", gtb < -300), ("soft", (gtb >= -300) & (gtb <= 200)),
                    ("bone", gtb > 200), ("cort", gtb > 1024)]:
        if msk.sum() < 50:
            out[nm] = None
            continue
        r = rk[msk]
        if len(r) > 50000:
            r = RNG.choice(r, 50000, replace=False)
        out[nm] = r
    return out


def overlap(a, b, bins):
    ha, _ = np.histogram(a, bins=bins, density=True)
    hb, _ = np.histogram(b, bins=bins, density=True)
    return float(np.minimum(ha, hb).sum() * (bins[1] - bins[0]))


def main():
    ps = pd.read_csv(os.path.join(EVAL, "metrics/per_subject.csv"))
    ps = ps[ps.model == "unet"].copy()
    ps["region"] = ps.subj_id.map(reg)
    subs = ps.groupby("region").head(10).subj_id.tolist()
    res = [r for r in Pool(8).map(work, subs) if r]
    pool = {k: np.concatenate([r[k] for r in res if r.get(k) is not None])
            for k in ["air", "soft", "bone", "cort"]}
    np.savez(os.path.join(RUN, "mr_tissue_pool.npz"), **pool)

    bins = np.linspace(0, 1, 41)
    stats = {"n_subjects": len(res),
             "median": {k: float(np.median(v)) for k, v in pool.items()},
             "overlap_bone_air": overlap(pool["bone"], pool["air"], bins),
             "overlap_cort_air": overlap(pool["cort"], pool["air"], bins),
             "overlap_bone_soft": overlap(pool["bone"], pool["soft"], bins),
             "overlap_soft_air": overlap(pool["soft"], pool["air"], bins)}
    json.dump(stats, open(os.path.join(RUN, "mr_tissue_stats.json"), "w"), indent=2)
    print("[mr_tissue] pooled", len(res), "subjects")
    print("  median MR-rank:", {k: round(v, 2) for k, v in stats["median"].items()})
    print("  overlap bone-soft", round(stats["overlap_bone_soft"], 2),
          "| cort-air", round(stats["overlap_cort_air"], 2),
          "| bone-air", round(stats["overlap_bone_air"], 2))


if __name__ == "__main__":
    main()
