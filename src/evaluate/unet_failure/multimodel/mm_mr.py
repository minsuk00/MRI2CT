"""Report 11 per-model sampled passes (need MR / histogram matching):

  loose : MR->sCT scatter in in-mask AIR voxels (does the model translate un-masked
          MR signal into HU in the loose external band?)  -- from cads_loose.py
  blur  : magnitude-matched bone-edge sharpness ratio (sCT edges vs real CT after
          histogram matching) -- from verify_blur.py

  python mm_mr.py --model <...>

Writes to OUTROOT/<model>/: loose_stats.json, verify_blur.csv,
figures/c_loose_scatter.png, figures/vf2_blur.png
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from skimage.exposure import match_histograms
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mm_common as C  # noqa: E402

REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
PER_SUBJ = 40000        # air voxels sampled per subject (loose)
N_PER_REGION = 6        # subjects per region (loose)
PER_REGION_BLUR = 5     # subjects per region (blur)


def gradmag(v):
    return np.sqrt(sum(x ** 2 for x in np.gradient(v)))


def loose(model, RUN, FIG):
    su = pd.read_csv(os.path.join(RUN, "cads_subject.csv"))
    rng = np.random.default_rng(0)
    subs = []
    for _, g in su.groupby("region"):
        subs += list(g.subj.sample(min(N_PER_REGION, len(g)), random_state=0))

    MR, SC = [], []
    for s in subs:
        try:
            gt = C.canon(f"{C.DATA}/{s}/ct.nii")
            mr = C.canon(f"{C.DATA}/{s}/moved_mr.nii")
            body = C.canon(f"{C.DATA}/{s}/mask.nii") > 0
            sct = C.canon(C.sct_path(model, s))
        except Exception as e:
            print("skip", s, e)
            continue
        air = body & (gt < -300)
        idx = np.flatnonzero(air)
        if idx.size > PER_SUBJ:
            idx = rng.choice(idx, PER_SUBJ, replace=False)
        MR.append(mr.ravel()[idx])
        SC.append(sct.ravel()[idx])
    mr = np.concatenate(MR)
    sc = np.concatenate(SC)

    mr_hi = float(np.percentile(mr, 99))
    keep = mr <= mr_hi
    mr, sc = mr[keep], sc[keep]
    r = float(np.corrcoef(mr, sc)[0, 1])
    frac_fill = float((sc > -400).mean())

    fig, ax = plt.subplots(figsize=(9, 5))
    h = ax.hist2d(mr, sc, bins=[70, 70], range=[[0, mr_hi], [-1050, 300]], cmap="magma", cmin=1)
    fig.colorbar(h[3], ax=ax, label="voxel count")
    edges = np.linspace(0, mr_hi, 26)
    cen = 0.5 * (edges[:-1] + edges[1:])
    med = [np.median(sc[(mr >= edges[k]) & (mr < edges[k + 1])])
           if ((mr >= edges[k]) & (mr < edges[k + 1])).any() else np.nan for k in range(len(cen))]
    ax.plot(cen, med, color="cyan", lw=2.4, label="sCT median")
    ax.axhline(-1000, color="lime", ls="--", lw=1.6, label="ground truth (air)")
    ax.set_xlabel("MR intensity (input)")
    ax.set_ylabel("sCT HU")
    ax.set_title(f"{C.MODEL_LABEL[model]}: in-mask voxels where truth is air (GT < -300); "
                 f"sCT vs MR  (pooled r = {r:.2f}, {len(subs)} subjects)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c_loose_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    stats = {"n_subj": len(subs), "n_vox": int(mr.size), "pooled_r_mr_sct": r,
             "frac_air_filled_gt_m400": frac_fill, "mr_hi_p99": mr_hi}
    json.dump(stats, open(os.path.join(RUN, "loose_stats.json"), "w"), indent=2)
    print(f"[mm_mr:{model}] loose", stats)


def blur(model, RUN, FIG):
    subs = C.subjects()
    pick, seen = [], {}
    for s in subs:
        r = C.reg(s)
        if seen.get(r, 0) < PER_REGION_BLUR:
            pick.append(s)
            seen[r] = seen.get(r, 0) + 1
    rows = []
    for s in pick:
        gt = C.canon(f"{C.DATA}/{s}/ct.nii")
        sct = C.canon(C.sct_path(model, s))
        body = C.canon(f"{C.DATA}/{s}/mask.nii") > 0
        seg = C.canon(f"{C.DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
        recal = sct.copy()
        recal[body] = match_histograms(sct[body], gt[body])
        bone = np.isin(seg, C.BONE) & body
        shell = binary_dilation(bone) & ~binary_erosion(bone) & body
        gr, gc = gradmag(gt)[shell].mean(), gradmag(recal)[shell].mean()
        rows.append({"subj": s, "region": C.reg(s), "grad_real": gr, "grad_recal": gc, "ratio": gc / gr})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUN, "verify_blur.csv"), index=False)

    rr = df.groupby("region")["ratio"].mean().reindex(REG)
    overall = df.ratio.mean()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(rr)), rr.values, color=C.MODEL_COLOR[model])
    ax.axhline(1.0, color="#16a34a", ls="--", lw=1.2, label="as sharp as real CT")
    ax.set_xticks(range(len(rr)))
    ax.set_xticklabels(rr.index)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("bone-edge sharpness:\nmagnitude-matched sCT / real CT")
    ax.set_title(f"{C.MODEL_LABEL[model]}: bone-edge sharpness deficit (magnitude controlled)")
    ax.legend()
    for i, v in enumerate(rr.values):
        if not np.isnan(v):
            ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.savefig(os.path.join(FIG, "vf2_blur.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[mm_mr:{model}] blur overall recal/real edge-sharpness = {overall:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=C.MODELS)
    args = ap.parse_args()
    C.ensure(args.model)
    RUN, FIG = C.run_dir(args.model), C.fig_dir(args.model)
    loose(args.model, RUN, FIG)
    blur(args.model, RUN, FIG)


if __name__ == "__main__":
    main()
