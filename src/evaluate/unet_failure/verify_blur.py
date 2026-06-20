"""Substantiate report 09's claim that the segmenter's bone loss is structural BLUR
(not HU magnitude). We histogram-match the sCT to the GT CT (removing the magnitude
difference), then measure the gradient magnitude on the GT-bone surface shell:
if the magnitude-matched sCT still has softer bone edges than the real CT, that is
genuine blur. Reported as the recal/real edge-gradient ratio (1.0 = as sharp as
real CT; <1 = blurrier), per region + overall, with a figure.
"""
import os
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from skimage.exposure import match_histograms
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
FIG = os.path.join(RUN, "figures")
BONE = [7, 27, 28, 29, 30]
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
PER_REGION = 5


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def gradmag(v):
    return np.sqrt(sum(x ** 2 for x in np.gradient(v)))


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    pick, seen = [], {}
    for s in subs:
        r = reg(s)
        if seen.get(r, 0) < PER_REGION:
            pick.append(s)
            seen[r] = seen.get(r, 0) + 1
    rows = []
    for s in pick:
        gt = canon(f"{DATA}/{s}/ct.nii")
        sct = canon(f"{EVAL}/volumes/unet/{s}/sample.nii.gz")
        body = canon(f"{DATA}/{s}/mask.nii") > 0
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
        recal = sct.copy()
        recal[body] = match_histograms(sct[body], gt[body])
        bone = np.isin(seg, BONE) & body
        shell = binary_dilation(bone) & ~binary_erosion(bone) & body
        gr, gc = gradmag(gt)[shell].mean(), gradmag(recal)[shell].mean()
        rows.append({"subj": s, "region": reg(s), "grad_real": gr, "grad_recal": gc, "ratio": gc / gr})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RUN, "verify_blur.csv"), index=False)

    rr = df.groupby("region")["ratio"].mean().reindex(REG)
    overall = df.ratio.mean()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.bar(range(len(rr)), rr.values, color="#dc2626")
    ax.axhline(1.0, color="#16a34a", ls="--", lw=1.2, label="as sharp as real CT")
    ax.set_xticks(range(len(rr)))
    ax.set_xticklabels(rr.index)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("bone-edge sharpness:\nmagnitude-matched sCT / real CT")
    ax.set_title("Bone-edge sharpness deficit (magnitude controlled): sCT bone is blurrier")
    ax.legend()
    for i, v in enumerate(rr.values):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    fig.savefig(os.path.join(FIG, "vf2_blur.png"), dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[verify_blur] {len(df)} subjects, overall recal/real edge-sharpness = {overall:.2f}")
    print(rr.round(2).to_string())


if __name__ == "__main__":
    main()
