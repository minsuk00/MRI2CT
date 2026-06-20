"""Aggregate the seg-downstream per-subject/per-label extraction (report 09).

Builds:
  - seg_region.csv     : per-region coarse-tissue Dice (ceiling vs sCT) + bone HU bias.
  - seg_label_table.csv: per-CADS-label Dice ceiling/sCT/gap + HU bias/MAE (mean over
                         subjects where the label is present), sorted by sCT Dice.
  - seg_stats.json     : overall coarse numbers, bone HU summary, bone-confusion
                         fractions, and correctness gates.

Gates:
  G1  babyseg(realCT) bone-union Dice (ceiling) is sane (>= 0.80) -> the segmenter
      itself finds bone reliably, so a low sCT bone Dice is sCT-attributable.
  G2  bone HU bias is a real undershoot (mean bias <= -40 HU, well beyond the ~0 HU
      soft-tissue bias) -> density story holds.
All values are means over subjects (macro). Dice means skip NaN (label absent).
"""
import os
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
BONE = [7, 27, 28, 29, 30]


def main():
    ps = pd.read_csv(os.path.join(RUN, "seg_per_subject.csv"))
    pl = pd.read_csv(os.path.join(RUN, "seg_per_label.csv"))
    cf = np.load(os.path.join(RUN, "bone_confusion.npz"), allow_pickle=True)
    names = list(cf["names"])

    # ---- coarse per region ----
    cols = []
    for nm in ["bone", "air", "soft"]:
        cols += [f"dice_ceil_{nm}", f"dice_sct_{nm}"]
    reg = ps.groupby("region")[cols].mean()
    for nm in ["bone", "air", "soft"]:
        reg[f"gap_{nm}"] = reg[f"dice_ceil_{nm}"] - reg[f"dice_sct_{nm}"]
    reg["bone_bias"] = ps.groupby("region")["bone_bias"].mean()
    reg["bone_mae"] = ps.groupby("region")["bone_mae"].mean()
    reg = reg.reindex([r for r in REG if r in reg.index])
    reg.to_csv(os.path.join(RUN, "seg_region.csv"))

    # ---- per-label table ----
    agg = pl.groupby(["label", "name", "is_bone"]).agg(
        dice_ceil=("dice_ceil", "mean"),
        dice_sct=("dice_sct", "mean"),
        gt_hu=("gt_hu", "mean"),
        pred_hu=("pred_hu", "mean"),
        bias=("bias", "mean"),
        mae=("mae", "mean"),
        n_subj=("subj", "nunique"),
    ).reset_index()
    agg["gap"] = agg["dice_ceil"] - agg["dice_sct"]
    agg = agg.sort_values("dice_sct").set_index("name")
    agg.to_csv(os.path.join(RUN, "seg_label_table.csv"))

    # ---- overall coarse ----
    coarse = {}
    for nm in ["bone", "air", "soft"]:
        c = float(ps[f"dice_ceil_{nm}"].mean())
        s = float(ps[f"dice_sct_{nm}"].mean())
        coarse[nm] = {"ceiling": c, "sct": s, "gap": c - s}

    bone_hu = {
        "gt": float(ps.bone_gt_hu.mean()), "pred": float(ps.bone_pred_hu.mean()),
        "bias": float(ps.bone_bias.mean()), "mae": float(ps.bone_mae.mean()),
    }

    # ---- bone confusion (fractions of GT-bone voxels) ----
    cr = cf["conf_real"].astype(float)
    cs = cf["conf_sct"].astype(float)
    fr = cr / cr.sum()
    fs = cs / cs.sum()
    bone_frac_real = float(fr[BONE].sum())
    bone_frac_sct = float(fs[BONE].sum())
    # top non-bone labels the sCT relabels GT-bone voxels into
    nonbone = [l for l in range(len(names)) if l not in BONE]
    relabel = sorted(((names[l], float(fs[l])) for l in nonbone), key=lambda x: -x[1])[:6]

    g1 = coarse["bone"]["ceiling"]
    g2 = bone_hu["bias"]
    gates = {
        "G1_bone_ceiling_dice": {"val": g1, "pass": bool(g1 >= 0.80)},
        "G2_bone_hu_undershoot": {"val": g2, "pass": bool(g2 <= -40)},
    }
    stats = {
        "n_subjects": int(len(ps)),
        "coarse": coarse,
        "bone_hu": bone_hu,
        "bone_confusion": {
            "gt_bone_kept_as_bone_realCT": bone_frac_real,
            "gt_bone_kept_as_bone_sCT": bone_frac_sct,
            "sct_relabel_top": relabel,
        },
        "gates": gates,
        "all_pass": all(v["pass"] for v in gates.values()),
    }
    json.dump(stats, open(os.path.join(RUN, "seg_stats.json"), "w"), indent=2)

    print(f"[seg_aggregate] {stats['n_subjects']} subjects")
    print(f"  coarse Dice (ceiling -> sCT): "
          f"bone {coarse['bone']['ceiling']:.3f}->{coarse['bone']['sct']:.3f}  "
          f"air {coarse['air']['ceiling']:.3f}->{coarse['air']['sct']:.3f}  "
          f"soft {coarse['soft']['ceiling']:.3f}->{coarse['soft']['sct']:.3f}")
    print(f"  bone HU: GT {bone_hu['gt']:.0f} pred {bone_hu['pred']:.0f} "
          f"bias {bone_hu['bias']:.0f} mae {bone_hu['mae']:.0f}")
    print(f"  GT-bone kept as bone: realCT {bone_frac_real:.2f} -> sCT {bone_frac_sct:.2f}")
    print(f"  sCT relabels GT-bone into: {[(n, round(f,3)) for n,f in relabel[:4]]}")
    print(f"  GATES: {'ALL PASS' if stats['all_pass'] else 'SOME FAILED'} "
          f"(G1 ceiling {g1:.3f}, G2 bias {g2:.0f})")


if __name__ == "__main__":
    main()
