"""Self-contained per-CADS-label MAE and HU-bias of the U-Net sCT (micro, all 35
labels incl. Background). Reads the raw volumes directly (GT CT, sCT, body mask,
GT CADS seg) for all 207 center-wise val subjects. No intermediate CSV.

  per label:  MAE = Sum|sCT-GT| / Sum vox ,  bias = Sum(sCT-GT) / Sum vox  (pooled)
  red = bone {7,27,28,29,30}   blue = air-organs {9 airway,13 lungs}
  purple = Background (label 0) grey = soft (everything else)
"""

import glob
import os
from multiprocessing import Pool

import matplotlib
import nibabel as nib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
OUT = "/home/minsukc/MRI2CT/temp/perlabel_mae_bias_standalone.png"
NL = 35
BONE, AIRORG = [7, 27, 28, 29, 30], [9, 13]
NAMES = [
    "Background",
    "Brain - other",
    "CSF",
    "Eyes & optic pathway",
    "Face & oral soft tissue",
    "Gray matter",
    "Head & neck glands",
    "Skull",
    "White matter",
    "Airway",
    "Breast",
    "Esophagus",
    "Heart",
    "Lungs",
    "Thoracic cavity",
    "Abdominal cavity",
    "Adrenals",
    "Bowel",
    "Gallbladder",
    "Kidneys",
    "Liver",
    "Pancreas",
    "Spleen",
    "Stomach",
    "Bladder",
    "Prostate & seminal vesicle",
    "Blood vessels",
    "Bone - other",
    "Limb & girdle bones",
    "Spine",
    "Thoracic cage",
    "Gland - other",
    "Muscle",
    "Spinal cord",
    "Subcutaneous tissue",
]


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def process(s):
    """Per-subject, per-label sums: n, Sum|err|, Sum(err) for labels 0..34."""
    try:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    except Exception:
        return None
    lab = seg[body]
    err = (sct - gt)[body]
    n = np.bincount(lab, minlength=NL)
    sabs = np.bincount(lab, weights=np.abs(err), minlength=NL)
    serr = np.bincount(lab, weights=err, minlength=NL)
    return np.stack([n, sabs, serr])  # (3, 35)


def color_of(label):
    if label in BONE:
        return "#dc2626"
    if label in AIRORG:
        return "#2563eb"
    if label == 0:
        return "#7c3aed"
    return "#9ca3af"


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    R = [r for r in Pool(8).map(process, subs) if r is not None]
    print(f"{len(R)} subjects")
    tot = np.sum(R, axis=0)  # (3, 35): summed n, sabs, serr over subjects
    n, sabs, serr = tot
    mae = sabs / n
    bias = serr / n
    color = [color_of(l) for l in range(NL)]
    name = np.array(NAMES)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))
    o = np.argsort(mae)
    ax[0].barh(range(NL), mae[o], color=[color[i] for i in o])
    ax[0].set_yticks(range(NL))
    ax[0].set_yticklabels(name[o], fontsize=8)
    ax[0].set_xlabel("mean absolute HU error (MAE)")
    ax[0].set_title("Per-CADS-label MAE")
    o = np.argsort(bias)
    ax[1].barh(range(NL), bias[o], color=[color[i] for i in o])
    ax[1].set_yticks(range(NL))
    ax[1].set_yticklabels(name[o], fontsize=8)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_xlabel("HU bias (sCT - GT);  <0 undershoot, >0 overshoot")
    ax[1].set_title("Per-CADS-label HU bias")

    legend = [
        Patch(color="#dc2626", label="bone (7,27,28,29,30)"),
        Patch(color="#2563eb", label="air-organs (airway, lungs)"),
        Patch(color="#7c3aed", label="Background (label 0)"),
        Patch(color="#9ca3af", label="soft (other)"),
    ]
    fig.suptitle("U-Net sCT error per CADS label (micro, 207 center-wise val subjects)", y=1.0, fontsize=13)
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    print("\n== bone labels (MAE / bias) ==")
    for l in BONE:
        print(f"  {NAMES[l]:<22} MAE {mae[l]:6.1f}   bias {bias[l]:+7.1f}")


if __name__ == "__main__":
    main()
