"""Story figures for the CADS merge report (writes to cads_demo/).

Both figures show that the scary "out-of-FOV hallucination" cases are actually
REAL anatomy -- the region label simply doesn't bound the scan FOV.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from merge_demo import load_map, merge_case  # noqa: E402

DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
CMAP = plt.get_cmap("nipy_spectral", 35)


def sagittal_panel(subj, title, caption_path):
    m = load_map("grouped"); nc = int(m.paint_id.max()) + 1
    merged, ref, _ = merge_case(subj, m, nc)
    ct = np.asarray(nib.load(os.path.join(DATA, subj, "ct.nii")).dataobj)
    x = ct.shape[0] // 2                       # mid-sagittal: Z (cranio-caudal) vertical
    ct_sl = ct[x, :, :].T
    seg_sl = merged[x, :, :].T
    fig, ax = plt.subplots(1, 2, figsize=(9, 6))
    ax[0].imshow(ct_sl, cmap="gray", vmin=-1024, vmax=1024, origin="lower")
    ax[0].set_title("CT (mid-sagittal)"); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(ct_sl, cmap="gray", vmin=-1024, vmax=1024, origin="lower")
    masked = np.ma.masked_where(seg_sl == 0, seg_sl)
    ax[1].imshow(masked, cmap=CMAP, vmin=0, vmax=34, alpha=0.55, origin="lower", interpolation="nearest")
    ax[1].set_title("merged seg (all 9 tasks, priority)"); ax[1].set_xticks([]); ax[1].set_yticks([])
    fig.suptitle(title, fontsize=10)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(caption_path, dpi=110, bbox_inches="tight")
    print("saved", caption_path)


if __name__ == "__main__":
    sagittal_panel(
        "1THA293",
        "1THA293: labeled 'TH' but the FOV extends UP into the head\n"
        "(mandible + brain tissue present) -> task 557 'head' firing is REAL, not hallucination",
        "cads_demo/fig_realhead.png",
    )
    sagittal_panel(
        "1THB165",
        "1THB165: labeled 'TH' but is a 477mm WHOLE-TORSO scan\n"
        "(hip + sacrum at bottom, lungs at top) -> task 554 'hip' firing is REAL, not hallucination",
        "cads_demo/fig_fov.png",
    )
