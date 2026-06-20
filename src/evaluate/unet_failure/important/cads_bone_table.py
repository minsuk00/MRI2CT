"""Per-bone-label error contribution for the U-Net sCT, same columns as the
4-group table but broken out for the 5 CADS bone labels. Self-contained: reads
the raw volumes directly (GT CT, sCT, body mask, GT CADS seg), 207 center-wise
val subjects. Shares (%vox, %error) are relative to the WHOLE body, so they are
comparable to the group table.

  micro = pool all subjects' voxels, then average  (Sum|err| / Sum n)
  macro = MAE per subject, then average over subjects
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
BONE = {7: "Skull", 27: "Bone-other", 28: "Limb & girdle", 29: "Spine", 30: "Thoracic cage"}
LABELS = list(BONE)


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def process(s):
    """Per subject: whole-body n & Sum|err|, and per-bone-label n, Sum|err|, Sum(err)."""
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception:
        return None
    lab = seg[body]
    err = (sct - gt)[body]
    ae = np.abs(err)
    row = {"body_n": ae.size, "body_sabs": float(ae.sum())}
    for l in LABELS:
        m = lab == l
        row[f"n_{l}"] = int(m.sum())
        row[f"sabs_{l}"] = float(ae[m].sum())
        row[f"serr_{l}"] = float(err[m].sum())
    return row


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    R = [r for r in Pool(8).map(process, subs) if r is not None]
    df = pd.DataFrame(R)
    print(f"{len(df)} subjects")

    tot_n = df.body_n.sum()           # all body voxels over 207 subjects
    tot_abs = df.body_sabs.sum()      # all body |err| over 207 subjects

    rows = []
    for l in LABELS:
        n, sabs, serr = df[f"n_{l}"], df[f"sabs_{l}"], df[f"serr_{l}"]
        present = n > 0               # subjects that actually have this bone
        macro = (sabs[present] / n[present]).mean()
        rows.append({
            "bone label": BONE[l],
            "% body vox": 100 * n.sum() / tot_n,
            "micro MAE": sabs.sum() / n.sum(),
            "macro MAE": macro,
            "bias": serr.sum() / n.sum(),
            "% of body error": 100 * sabs.sum() / tot_abs,
        })
    # bone group total (all 5) and cortical-only (excl Bone-other)
    for name, ls in [("bone (all 5)", LABELS), ("cortical (excl Bone-other)", [7, 28, 29, 30])]:
        n = sum(df[f"n_{l}"] for l in ls)
        sabs = sum(df[f"sabs_{l}"] for l in ls)
        serr = sum(df[f"serr_{l}"] for l in ls)
        present = n > 0
        rows.append({
            "bone label": name,
            "% body vox": 100 * n.sum() / tot_n,
            "micro MAE": sabs.sum() / n.sum(),
            "macro MAE": (sabs[present] / n[present]).mean(),
            "bias": serr.sum() / n.sum(),
            "% of body error": 100 * sabs.sum() / tot_abs,
        })
    out = pd.DataFrame(rows)

    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(out.to_string(index=False))
    print(f"\nall 5 bone labels together: {out.iloc[5]['% of body error']:.1f}% of total body error "
          f"from {out.iloc[5]['% body vox']:.1f}% of voxels")
    out.to_csv("/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619/cads_bone_table.csv", index=False)


if __name__ == "__main__":
    main()
