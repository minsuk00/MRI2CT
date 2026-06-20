"""Falsification tests for the central claim of report 09:
   "the U-Net localizes bone correctly; the downstream bone failure is a DENSITY
    (HU undershoot) problem, not a LOCALIZATION problem."

Two segmenter-free tests over all 207 subjects (the segmenter is HU-driven, so we
deliberately test the claim WITHOUT it):

TEST A -- ordering / AUC.  If the U-Net put bone in the right place but too low in
HU, then sCT HU should still RANK true-bone voxels above non-bone voxels almost as
well as the real CT does. AUC(sCT HU; GT-bone) ~ AUC(real CT HU; GT-bone) and both
high => spatial bone info is intact, only the scale is compressed. AUC near 0.5
would FALSIFY the claim (bone not separable => mislocalized).

TEST B -- where does true bone go, in HU?  Decompose GT-bone voxels by sCT HU:
  rendered_soft  : sCT < 50 HU   (looks like soft tissue -> a genuine localization miss)
  undershoot     : 50 <= sCT <150 (clearly elevated above soft, but below the bone
                   threshold -> recoverable by recalibration, i.e. density)
  correct        : sCT >= 150     (already reads as bone)
and sweep a single bone threshold on the sCT: if lowering it recovers the HU-bone
Dice toward the real-CT value, the structure was right and only the cut-point (scale)
was wrong. This quantifies the density-vs-localization split.

Writes verify_density.csv + prints the headline split. No GPU, no segmenter.
"""
import os
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
from sklearn.metrics import roc_auc_score

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
BONE = [7, 27, 28, 29, 30]
RNG = np.random.RandomState(0)
TS = np.arange(-100, 700, 10)


def get_region_key(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def dice(a, b):
    s = a.sum() + b.sum()
    return np.nan if s == 0 else float(2 * np.logical_and(a, b).sum() / s)


def process(s):
    try:
        gt = canon(os.path.join(DATA, s, "ct.nii"))
        sct = canon(os.path.join(EVAL, "volumes", "unet", s, "sample.nii.gz"))
        body = canon(os.path.join(DATA, s, "mask.nii")) > 0
        seg = canon(os.path.join(DATA, s, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
    except Exception as e:
        return None, f"{s}: {e}"
    gtb, scb = gt[body], sct[body]
    bone = np.isin(seg[body], BONE)
    if bone.sum() < 200 or (~bone).sum() < 200:
        return None, f"{s}: too few bone/non-bone"

    # TEST A: AUC of HU for bone vs non-bone (subsample for speed)
    bi = np.where(bone)[0]
    ni = np.where(~bone)[0]
    bi = RNG.choice(bi, min(40000, len(bi)), replace=False)
    ni = RNG.choice(ni, min(40000, len(ni)), replace=False)
    y = np.concatenate([np.ones(len(bi)), np.zeros(len(ni))])
    auc_sct = roc_auc_score(y, np.concatenate([scb[bi], scb[ni]]))
    auc_gt = roc_auc_score(y, np.concatenate([gtb[bi], gtb[ni]]))

    # TEST B: decomposition of GT-bone by sCT HU
    pb = scb[bone]
    frac_soft = float((pb < 50).mean())       # genuine miss (rendered as soft)
    frac_under = float(((pb >= 50) & (pb < 150)).mean())  # recoverable undershoot
    frac_ok = float((pb >= 150).mean())        # already bone

    # threshold sweep on sCT for HU-bone Dice vs GT bone; compare to real-CT@150
    d_real = dice(gtb > 150, bone)
    d_sct150 = dice(scb > 150, bone)
    dices = [dice(scb > t, bone) for t in TS]
    bi_best = int(np.nanargmax(dices))
    d_best = float(dices[bi_best])
    t_best = float(TS[bi_best])

    return {
        "subj": s, "region": get_region_key(s), "n_bone": int(bone.sum()),
        "auc_sct": auc_sct, "auc_gt": auc_gt,
        "frac_rendered_soft": frac_soft, "frac_undershoot": frac_under, "frac_correct": frac_ok,
        "dice_real_t150": d_real, "dice_sct_t150": d_sct150,
        "dice_sct_best": d_best, "t_best": t_best,
    }, None


def main():
    subs = sorted(p.name for p in os.scandir(os.path.join(EVAL, "volumes", "unet")) if p.is_dir())
    print(f"[verify_density] {len(subs)} subjects", flush=True)
    rows, errs = [], []
    with Pool(8) as pool:
        for r, e in pool.imap_unordered(process, subs):
            (errs if e else rows).append(e or r)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUN, "verify_density.csv"), index=False)

    print(f"\n[verify_density] {len(df)} subjects, {len([e for e in errs if e])} skipped")
    print("\n== TEST A: AUC of HU for bone-vs-nonbone (spatial separability) ==")
    print(f"  real CT  AUC {df.auc_gt.mean():.3f} +- {df.auc_gt.std():.3f}   (ceiling)")
    print(f"  sCT      AUC {df.auc_sct.mean():.3f} +- {df.auc_sct.std():.3f}")
    chance = (df.auc_sct.mean() - 0.5) / (df.auc_gt.mean() - 0.5) * 100  # 0.5 = no-info floor
    print(f"  -> sCT retains {chance:.1f}% of the real-CT ABOVE-CHANCE separability (chance floor 0.5)")

    print("\n== TEST B: where do GT-bone voxels land in sCT HU? ==")
    print("  (absolute HU buckets; an absolute cut cannot distinguish severe undershoot from")
    print("   a true miss -- TEST A and the recalibration oracle adjudicate that, not this split)")
    print(f"  reads as soft tissue (<50 HU)        : {df.frac_rendered_soft.mean()*100:.1f}%")
    print(f"  elevated but sub-threshold (50-150)  : {df.frac_undershoot.mean()*100:.1f}%")
    print(f"  already bone (>=150 HU)              : {df.frac_correct.mean()*100:.1f}%")

    print("\n== threshold-recovery (HU-bone Dice vs GT bone) ==")
    print(f"  real CT @150 HU        Dice {df.dice_real_t150.mean():.3f}  (ceiling)")
    print(f"  sCT @150 HU (fixed)    Dice {df.dice_sct_t150.mean():.3f}")
    print(f"  sCT @ best threshold   Dice {df.dice_sct_best.mean():.3f}  (mean best t = {df.t_best.mean():.0f} HU)")
    gap = df.dice_real_t150.mean() - df.dice_sct_t150.mean()
    rec2 = (df.dice_sct_best.mean() - df.dice_sct_t150.mean()) / gap if gap > 1e-6 else float("nan")
    print(f"  -> recalibrating only the threshold recovers {rec2*100:.0f}% of the sCT-vs-realCT Dice gap")


if __name__ == "__main__":
    main()
