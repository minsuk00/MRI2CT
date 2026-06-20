"""Within-bone error analysis: split all bone voxels (CADS labels 7,27,28,29,30)
by their GROUND-TRUTH HU into density bands (marrow/trabecular/cortical/dense/
above-cap) and report each band's per-voxel error and its share of total error.
This isolates genuinely dense (cortical) bone instead of just subtracting a label.

Self-contained: reads raw volumes (GT CT, sCT, body mask, GT CADS seg), 207
center-wise val subjects.
  micro MAE = Sum|err| / Sum vox ;  bias = Sum(err)/Sum vox  (pooled over subjects)
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from multiprocessing import Pool
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
BONE = [7, 27, 28, 29, 30]

EDGES = [-1024, 150, 300, 600, 1024, 4000]      # GT-HU band edges
BANDS = ["<150 (marrow/soft)", "150-300 (trabecular)", "300-600 (cortical)",
         "600-1024 (dense cortical)", ">1024 (above sigmoid cap)"]
NB = len(BANDS)


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def process(s):
    """Per subject: whole-body n & Sum|err|, and per-HU-band (within bone) n, Sum|err|, Sum(err)."""
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception:
        return None
    err = (sct - gt)[body]
    ae = np.abs(err)
    body_n, body_sabs = ae.size, float(ae.sum())

    bone = np.isin(seg[body], BONE)
    gtb, aeb, errb = gt[body][bone], ae[bone], err[bone]
    band = np.digitize(gtb, EDGES[1:-1])         # 0..NB-1 by GT HU
    n = np.bincount(band, minlength=NB)
    sabs = np.bincount(band, weights=aeb, minlength=NB)
    serr = np.bincount(band, weights=errb, minlength=NB)
    return np.concatenate([[body_n, body_sabs], n, sabs, serr])


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    R = np.array([r for r in Pool(8).map(process, subs) if r is not None])
    print(f"{len(R)} subjects")
    tot = R.sum(0)
    body_n, body_sabs = tot[0], tot[1]
    n = tot[2:2 + NB]
    sabs = tot[2 + NB:2 + 2 * NB]
    serr = tot[2 + 2 * NB:2 + 3 * NB]
    bone_n, bone_sabs = n.sum(), sabs.sum()

    out = pd.DataFrame({
        "GT-HU band": BANDS,
        "% body vox": 100 * n / body_n,
        "% bone vox": 100 * n / bone_n,
        "micro MAE": sabs / n,
        "bias": serr / n,
        "% of body error": 100 * sabs / body_sabs,
        "% of bone error": 100 * sabs / bone_sabs,
    })
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    print(out.to_string(index=False))
    print(f"\nbone total: {100*bone_n/body_n:.1f}% of body vox, {100*bone_sabs/body_sabs:.1f}% of body error")
    out.to_csv(os.path.join(RUN, "cads_bone_hu_split.csv"), index=False)

    # figure: contribution + severity by density band
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    x = range(NB)
    ax[0].bar(x, out["micro MAE"], color="#dc2626")
    ax[0].set_title("per-voxel MAE by GT-HU band")
    ax[1].bar(x, out["bias"], color="#dc2626"); ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_title("HU bias by GT-HU band (<0 undershoot)")
    ax[2].bar(x, out["% of body error"], color="#dc2626")
    ax[2].set_title("share of TOTAL body error by band (%)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(BANDS, rotation=30, ha="right", fontsize=8)
    fig.suptitle("Within-bone error by ground-truth density (all 5 CADS bone labels, 207 subjects)", y=1.03)
    fig.tight_layout()
    fig.savefig("/home/minsukc/MRI2CT/temp/bone_hu_split.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote /home/minsukc/MRI2CT/temp/bone_hu_split.png")


if __name__ == "__main__":
    main()
