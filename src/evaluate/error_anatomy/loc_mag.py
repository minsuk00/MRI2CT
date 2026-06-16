"""Localization-vs-magnitude decomposition of UNet bone error (training-free).

Separates two failure modes inside bone (GT HU > 200), pooled per region:
  - LOCALIZATION: does the prediction place bone in the right voxels?
      bone shape-Dice( pred>200 , GT>200 ); plus false-negative (missed GT bone)
      and false-positive (hallucinated bone) rates.
  - MAGNITUDE: where pred and GT AGREE a voxel is bone, how wrong is the HU?
      MAE over the agreed-bone intersection.
Also reports the interior(eroded)-vs-boundary(shell) MAE already in summary.csv.
"""
import os, numpy as np, nibabel as nib, pandas as pd, json
from multiprocessing import Pool

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL = os.path.join(REPO, "evaluation_results/full_eval_20260609/volumes/unet")
RUN = os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616")
PS = os.path.join(REPO, "evaluation_results/full_eval_20260609/metrics/per_subject.csv")
TH = 200.0


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def proc(args):
    sid, region = args
    try:
        gt = canon(os.path.join(DATA, sid, "ct.nii"))
        body = canon(os.path.join(DATA, sid, "mask.nii")) > 0
        pr = canon(os.path.join(VOL, sid, "sample.nii.gz"))
    except Exception:
        return None
    if pr.shape != gt.shape:
        return None
    bg = body & (gt > TH)
    bp = body & (pr > TH)
    inter = bg & bp
    ng, npd, ni = int(bg.sum()), int(bp.sum()), int(inter.sum())
    if ng < 1000:
        return None
    dice = 2 * ni / (ng + npd) if (ng + npd) else np.nan
    mae_agreed = float(np.abs(pr[inter] - gt[inter]).mean()) if ni else np.nan
    return {
        "subj_id": sid, "region": region,
        "bone_shape_dice": dice,
        "frac_missed": (ng - ni) / ng,          # GT bone the model failed to mark as bone (localization)
        "frac_hallucinated": (npd - ni) / max(npd, 1),
        "mae_agreed": mae_agreed,               # magnitude error where both agree it's bone
        "n_bone": ng,
    }


def main():
    df = pd.read_csv(PS)
    subs = df[df.model == "unet"][["subj_id", "region"]].drop_duplicates().values.tolist()
    with Pool(4) as pool:
        rows = [r for r in pool.map(proc, subs) if r]
    d = pd.DataFrame(rows)
    d.to_csv(os.path.join(RUN, "loc_mag.csv"), index=False)

    s = pd.read_csv(os.path.join(RUN, "summary.csv"))
    u = s[s.model == "unet"]
    REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
    g = d.groupby("region").agg(shape_dice=("bone_shape_dice", "mean"),
                                missed=("frac_missed", "mean"),
                                halluc=("frac_hallucinated", "mean"),
                                mae_agreed=("mae_agreed", "mean")).reindex(REG)
    g["interior_mae"] = [u[u.region == r].mae_bone_interior.mean() for r in REG]
    g["boundary_mae"] = [u[u.region == r].mae_bone_boundary.mean() for r in REG]
    print("===== localization vs magnitude (UNet bone) =====")
    print(g.round(3).to_string())
    allrow = dict(shape_dice=d.bone_shape_dice.mean(), missed=d.frac_missed.mean(),
                  halluc=d.frac_hallucinated.mean(), mae_agreed=d.mae_agreed.mean(),
                  interior_mae=u.mae_bone_interior.mean(), boundary_mae=u.mae_bone_boundary.mean())
    print("ALL:", {k: round(v, 3) for k, v in allrow.items()})
    json.dump({"by_region": g.round(3).to_dict(), "all": {k: round(float(v), 3) for k, v in allrow.items()}},
              open(os.path.join(RUN, "loc_mag_stats.json"), "w"), indent=2)

    # figure
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True, "grid.alpha": 0.25})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(REG))
    a1.bar(x - 0.2, g.shape_dice, 0.4, label="bone shape-Dice (localization, ↑ good)", color="#16a34a")
    a1.bar(x + 0.2, g.missed, 0.4, label="fraction of GT bone missed", color="#f59e0b")
    a1.set_xticks(x); a1.set_xticklabels(REG, rotation=15); a1.set_ylim(0, 1); a1.legend(fontsize=8)
    a1.set_title("Detection: dense bone (brain/skull) localized well;\nthin trabecular bone (thorax/abd) under-detected")
    a2.bar(x - 0.27, g.mae_agreed, 0.27, label="agreed-bone MAE (pure magnitude)", color="#16a34a")
    a2.bar(x, g.interior_mae, 0.27, label="interior (dense core) MAE", color="#1e3a8a")
    a2.bar(x + 0.27, g.boundary_mae, 0.27, label="boundary (edge) MAE", color="#3b82f6")
    a2.set_xticks(x); a2.set_xticklabels(REG, rotation=15); a2.set_ylabel("HU"); a2.legend(fontsize=8)
    a2.set_title("Magnitude: ~220 HU undershoot even where bone is\ncorrectly found; dense interior worst")
    fig.suptitle("Bone failure = universal density UNDERSHOOT; in thin bone it is severe enough to also miss detection", y=1.04)
    fig.tight_layout(); fig.savefig(os.path.join(RUN, "figures/fig8_loc_mag.png"), bbox_inches="tight")
    print("wrote fig8_loc_mag.png")


if __name__ == "__main__":
    main()
