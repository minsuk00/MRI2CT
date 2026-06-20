"""HU-based tissue breakdown (air / soft / bone by density) for report 09 -- the
segmenter-free complement to the seg-downstream analysis. The CADS segmenter has
no "air" label, so true air (gas, ~-1000 HU) can only be defined by HU value.

Within the body mask, classify each voxel by HU:
  air  : HU < -300
  soft : -300 <= HU <= 150
  bone : HU > 150
and compare GT CT vs U-Net sCT (no segmentation involved):
  - per-class HU bias = mean(sCT - GT) over the GT-class ROI, and MAE; sign tells
    direction (air should overshoot UP, bone undershoot DOWN, soft ~0).
  - 3x3 GT-class -> sCT-class confusion (fraction of each GT class the sCT renders
    as each class): shows e.g. true air filled in as soft tissue.

Outputs hu_region.csv, hu_stats.json, and figures hu1_bias.png, hu2_confusion.png,
hu3_calibration.png (GT vs sCT 2D histogram), hu4_hist.png (per-tissue HU dists).
"""
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
TIS = ["air", "soft", "bone"]
AIR_HI, SOFT_HI = -300.0, 150.0
EDGES = np.linspace(-1024.0, 3000.0, 202)   # shared HU bins; top=3000 so true high-HU bone is not dropped
CENT = 0.5 * (EDGES[:-1] + EDGES[1:])
NB = len(EDGES) - 1


def get_region_key(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if s[1:3].upper() in m:
        return m[s[1:3].upper()]
    if s[1:2].upper() in m:
        return m[s[1:2].upper()]
    return "abdomen"


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def classify(hu):
    c = np.ones(hu.shape, np.int8)  # soft
    c[hu < AIR_HI] = 0
    c[hu > SOFT_HI] = 2
    return c


def process(s):
    try:
        gt = canon(os.path.join(DATA, s, "ct.nii"))
        sct = canon(os.path.join(EVAL, "volumes", "unet", s, "sample.nii.gz"))
        body = canon(os.path.join(DATA, s, "mask.nii")) > 0
    except Exception as e:
        return None, None, f"{s}: {e}"
    gtb, scb = gt[body], sct[body]
    gc, sc = classify(gtb), classify(scb)
    row = {"subj": s, "region": get_region_key(s), "n_body": int(body.sum())}
    for i, nm in enumerate(TIS):
        m = gc == i
        n = int(m.sum())
        row[f"frac_{nm}"] = n / max(len(gtb), 1)
        if n:
            row[f"bias_{nm}"] = float((scb[m] - gtb[m]).mean())
            row[f"mae_{nm}"] = float(np.abs(scb[m] - gtb[m]).mean())
            row[f"gt_hu_{nm}"] = float(gtb[m].mean())
            row[f"pred_hu_{nm}"] = float(scb[m].mean())
        else:
            for k in ("bias", "mae", "gt_hu", "pred_hu"):
                row[f"{k}_{nm}"] = np.nan
    conf = np.zeros((3, 3), np.int64)
    np.add.at(conf, (gc, sc), 1)
    # joint GT-vs-sCT 2D histogram (calibration) + per-GT-tissue 1D HU dists
    h2d, _, _ = np.histogram2d(gtb, scb, bins=[EDGES, EDGES])
    hgt = np.stack([np.histogram(gtb[gc == i], bins=EDGES)[0] for i in range(3)])
    hpr = np.stack([np.histogram(scb[gc == i], bins=EDGES)[0] for i in range(3)])
    return row, conf, h2d, hgt, hpr, None


def main():
    os.makedirs(FIG, exist_ok=True)
    subs = sorted(p.name for p in os.scandir(os.path.join(EVAL, "volumes", "unet")) if p.is_dir())
    print(f"[hu_tissue] {len(subs)} subjects", flush=True)
    rows, errs = [], []
    conf = np.zeros((3, 3), np.int64)
    h2d = np.zeros((NB, NB))
    hgt = np.zeros((3, NB))
    hpr = np.zeros((3, NB))
    with Pool(8) as pool:
        for row, c, h, hg, hp, err in pool.imap_unordered(process, subs):
            if err:
                errs.append(err)
                continue
            rows.append(row)
            conf += c
            h2d += h
            hgt += hg
            hpr += hp
    df = pd.DataFrame(rows)

    # per region + overall bias/mae/frac
    reg = df.groupby("region")[[f"bias_{t}" for t in TIS] + [f"mae_{t}" for t in TIS]
                               + [f"frac_{t}" for t in TIS]].mean().reindex(REG)
    reg.to_csv(os.path.join(RUN, "hu_region.csv"))

    overall = {t: {"bias": float(df[f"bias_{t}"].mean()), "mae": float(df[f"mae_{t}"].mean()),
                   "frac": float(df[f"frac_{t}"].mean()),
                   "gt_hu": float(df[f"gt_hu_{t}"].mean()), "pred_hu": float(df[f"pred_hu_{t}"].mean())}
               for t in TIS}
    conff = conf / conf.sum(1, keepdims=True)  # row-normalized: GT-class -> sCT-class
    stats = {"n_subjects": int(len(df)), "thresholds": {"air_hi": AIR_HI, "soft_hi": SOFT_HI},
             "overall": overall,
             "confusion_rows_gt_cols_sct": conff.tolist(),
             "air_kept_as_air": float(conff[0, 0]), "air_to_soft": float(conff[0, 1]),
             "bone_kept_as_bone": float(conff[2, 2]), "bone_to_soft": float(conff[2, 1])}
    json.dump(stats, open(os.path.join(RUN, "hu_stats.json"), "w"), indent=2)

    # --- fig hu1: per-tissue HU bias, overall + per region ---
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(TIS))
    w = 0.13
    for j, r in enumerate(REG):
        if r in reg.index:
            ax.bar(x + (j - 2) * w, [reg.loc[r, f"bias_{t}"] for t in TIS], w, label=r)
    ax.plot(x, [overall[t]["bias"] for t in TIS], "k_", ms=28, mew=2.5, label="overall")
    ax.axhline(0, color="#9ca3af", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{t}\n({overall[t]['gt_hu']:.0f}->{overall[t]['pred_hu']:.0f} HU)" for t in TIS])
    ax.set_ylabel("HU bias  (sCT - GT;  +up = overshoot, -down = undershoot)")
    ax.set_title("Per-tissue HU bias (sCT - GT) by region")
    ax.legend(fontsize=8, ncol=3)
    fig.savefig(os.path.join(FIG, "hu1_bias.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- fig hu2: GT->sCT HU-class confusion ---
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    im = ax.imshow(conff, cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{conff[i, j]*100:.1f}%", ha="center", va="center",
                    color="white" if conff[i, j] > 0.5 else "#111827", fontsize=11)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"sCT {t}" for t in TIS])
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"GT {t}" for t in TIS])
    ax.set_title("HU-tissue confusion (row = true GT class,\ncol = what the sCT renders it as)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.savefig(os.path.join(FIG, "hu2_confusion.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- fig hu3: GT vs sCT 2D calibration histogram ---
    fig, ax = plt.subplots(figsize=(6.0, 5.2))
    im = ax.imshow(np.log1p(h2d.T), origin="lower", cmap="magma", aspect="auto",
                   extent=[EDGES[0], EDGES[-1], EDGES[0], EDGES[-1]])
    ax.plot([EDGES[0], EDGES[-1]], [EDGES[0], EDGES[-1]], "w--", lw=1, label="perfect (y=x)")
    for thr in (AIR_HI, SOFT_HI):
        ax.axvline(thr, color="#22d3ee", lw=0.7, ls=":")
    ax.set_xlabel("GT CT HU")
    ax.set_ylabel("U-Net sCT HU")
    ax.set_title("GT vs predicted HU, all body voxels (207 subjects)")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(im, fraction=0.046, pad=0.04, label="log(1+count)")
    fig.savefig(os.path.join(FIG, "hu3_calibration.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- fig hu4: per-tissue HU distributions, GT vs sCT ---
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for i, (nm, ax) in enumerate(zip(TIS, axes)):
        gnorm = hgt[i] / max(hgt[i].sum(), 1)
        pnorm = hpr[i] / max(hpr[i].sum(), 1)
        ax.fill_between(CENT, gnorm, color="#9ca3af", alpha=0.6, label="GT")
        ax.plot(CENT, pnorm, color="#dc2626", lw=1.6, label="sCT")
        ax.axvline(overall[nm]["gt_hu"], color="#374151", ls="--", lw=0.9)
        ax.axvline(overall[nm]["pred_hu"], color="#dc2626", ls="--", lw=0.9)
        ax.set_title(f"{nm}: GT {overall[nm]['gt_hu']:.0f} -> sCT {overall[nm]['pred_hu']:.0f} HU "
                     f"(bias {overall[nm]['bias']:+.0f})", fontsize=10)
        ax.set_xlabel("HU")
        if i == 0:
            ax.set_ylabel("density (within GT class)")
        ax.legend(fontsize=8)
    fig.suptitle("HU distribution per true tissue class: GT (grey) vs U-Net sCT (red), dashed = means", y=1.04)
    fig.savefig(os.path.join(FIG, "hu4_hist.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)

    print(f"[hu_tissue] {len(df)} subjects, {len(errs)} errors")
    for t in TIS:
        o = overall[t]
        print(f"  {t:5s} frac {o['frac']*100:5.1f}%  GT {o['gt_hu']:7.1f} -> pred {o['pred_hu']:7.1f}  "
              f"bias {o['bias']:+7.1f}  mae {o['mae']:6.1f}")
    print(f"  air kept-as-air {stats['air_kept_as_air']*100:.1f}%  air->soft {stats['air_to_soft']*100:.1f}%")
    print(f"  bone kept-as-bone {stats['bone_kept_as_bone']*100:.1f}%  bone->soft {stats['bone_to_soft']*100:.1f}%")


if __name__ == "__main__":
    main()
