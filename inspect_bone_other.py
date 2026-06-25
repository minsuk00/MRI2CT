"""Clean visualization of the CADS bone labels, bone-other in red, over the CT.
3 orthogonal views per subject at the slice with the most bone. Writes PNGs to
bone_other_overlay/ (not in the report)."""
import os
import glob
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

DATA = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
EVAL = "/home/minsukc/MRI2CT/evaluation_results/full_eval_20260617"
OUT = "/home/minsukc/MRI2CT/bone_other_overlay"
os.makedirs(OUT, exist_ok=True)
BONE = [27, 7, 29, 30, 28]
COL = {27: ("Bone-other", "#ef4444"), 7: ("Skull", "#06b6d4"),
       29: ("Spine", "#2563eb"), 30: ("Thoracic cage", "#22c55e"),
       28: ("Limb & girdle", "#f59e0b")}


def reg(s):
    m = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    return m.get(s[1:3].upper(), m.get(s[1:2].upper(), "abdomen"))


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def show(ax, ct, seg, axis, idx):
    sl = [slice(None)] * 3
    sl[axis] = idx
    c = np.rot90(ct[tuple(sl)])
    sg = np.rot90(seg[tuple(sl)])
    ax.imshow(c, cmap="gray", vmin=-200, vmax=600)         # CT clearly visible
    for lab in BONE:
        m = sg == lab
        if m.any():
            ax.imshow(np.ma.masked_where(~m, m), cmap=ListedColormap([COL[lab][1]]), alpha=0.35)
    ax.set_xticks([])
    ax.set_yticks([])


def main():
    subs = sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(EVAL + "/volumes/unet/*/sample.nii.gz"))
    picks = {}
    for s in subs:
        try:
            seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
        except Exception:
            continue
        n = int((seg == 27).sum())
        r = reg(s)
        if n > picks.get(r, (None, -1))[1]:
            picks[r] = (s, n)

    legend = [Patch(facecolor=c, label=nm) for nm, c in COL.values()]
    for r in ["thorax", "abdomen", "pelvis", "head_neck", "brain"]:
        if r not in picks:
            continue
        s = picks[r][0]
        ct = canon(f"{DATA}/{s}/ct.nii")
        seg = canon(f"{DATA}/{s}/cads_grouped_35_labels_seg.nii.gz", np.int16)
        bo = seg == 27   # slice selection on bone-other so the red is prominent
        ix = [bo.sum((1, 2)).argmax(), bo.sum((0, 2)).argmax(), bo.sum((0, 1)).argmax()]
        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        for ax, axis, nm in zip(axes, [2, 1, 0], ["axial", "coronal", "sagittal"]):
            show(ax, ct, seg, axis, ix[axis])
            ax.set_title(nm, fontsize=12)
        fig.legend(handles=legend, loc="lower center", ncol=5, fontsize=12, frameon=False)
        fig.suptitle(f"{r}  {s}  — CADS bone labels (bone-other = red)", fontsize=14)
        fig.tight_layout(rect=[0, 0.05, 1, 1])
        fig.savefig(f"{OUT}/bonelabels_{r}_{s}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[{r}] {s}")
    print("wrote to", OUT)


if __name__ == "__main__":
    main()
