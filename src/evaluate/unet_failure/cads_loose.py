"""Mechanism figure for report 10 section 1: the loose-region error is the U-Net
translating un-masked MR signal into HU. Pools in-mask AIR voxels (GT < -300)
across a sample of subjects and plots sCT vs MR intensity (truth is air, but the
sCT climbs with MR). Also copies a representative loose-mask zoom example.

Writes RUN/figures/c_loose_scatter.png, RUN/figures/c_loose_example.png and
RUN/cads_loose_stats.json (pooled correlation + filled-fraction)."""
import os
import json
import shutil
import numpy as np
import pandas as pd
import nibabel as nib

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
EXAMPLE_SRC = os.path.join(REPO, "temp/loose_mask/abdomen_1ABB143.png")
PER_SUBJ = 40000        # air voxels sampled per subject
N_PER_REGION = 6


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    su = pd.read_csv(os.path.join(RUN, "cads_subject.csv"))
    rng = np.random.default_rng(0)
    subs = []
    for _, g in su.groupby("region"):
        subs += list(g.subj.sample(min(N_PER_REGION, len(g)), random_state=0))

    MR, SC = [], []
    for s in subs:
        try:
            gt = canon(f"{DATA}/{s}/ct.nii")
            mr = canon(f"{DATA}/{s}/moved_mr.nii")
            body = canon(f"{DATA}/{s}/mask.nii") > 0
            sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
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
    med = [np.median(sc[(mr >= edges[k]) & (mr < edges[k + 1])]) if ((mr >= edges[k]) & (mr < edges[k + 1])).any() else np.nan for k in range(len(cen))]
    ax.plot(cen, med, color="cyan", lw=2.4, label="sCT median")
    ax.axhline(-1000, color="lime", ls="--", lw=1.6, label="ground truth (air)")
    ax.set_xlabel("MR intensity (input)")
    ax.set_ylabel("sCT HU")
    ax.set_title(f"In-mask voxels where the truth is air (GT < -300): sCT climbs with MR  (pooled r = {r:.2f}, "
                 f"{len(subs)} subjects)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "c_loose_scatter.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    if os.path.exists(EXAMPLE_SRC):
        shutil.copy(EXAMPLE_SRC, os.path.join(FIG, "c_loose_example.png"))

    stats = {"n_subj": len(subs), "n_vox": int(mr.size), "pooled_r_mr_sct": r,
             "frac_air_filled_gt_m400": frac_fill, "mr_hi_p99": mr_hi}
    json.dump(stats, open(os.path.join(RUN, "cads_loose_stats.json"), "w"), indent=2)
    print("[cads_loose]", stats)


if __name__ == "__main__":
    main()
