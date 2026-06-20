"""Per-CADS-label MAE and HU-bias of the U-Net sCT (micro, all 35 labels incl.
Background). Shows that the bone *group* numbers are diluted by "Bone-other" (a
soft-HU skeleton filler); the 4 cortical bones are far more pronounced.

CSV-based: reads the per-(subject,label) sums in cads_per_label.csv.
  red  = bone {7,27,28,29,30}    blue = air-organs {9 airway, 13 lungs}
  purple = Background (label 0)  grey = soft (everything else)
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
OUT = os.path.join(RUN, "figures", "perlabel_mae_bias.png")
BONE = [7, 27, 28, 29, 30]
AIRORG = [9, 13]


def color_of(label):
    if label in BONE:
        return "#dc2626"            # bone (red)
    if label in AIRORG:
        return "#2563eb"            # air-organs (blue)
    if label == 0:
        return "#7c3aed"            # Background (purple)
    return "#9ca3af"                # soft (grey)


def main():
    df = pd.read_csv(os.path.join(RUN, "cads_per_label.csv"))
    g = df.groupby(["label", "name"]).agg(n=("n", "sum"), sabs=("sabs", "sum"), serr=("serr", "sum")).reset_index()
    g["mae"] = g.sabs / g.n               # micro per-label MAE
    g["bias"] = g.serr / g.n              # micro per-label signed bias
    g["color"] = g.label.map(color_of)
    g.to_csv(os.path.join(RUN, "cads_perlabel_mae_bias.csv"), index=False)

    fig, ax = plt.subplots(1, 2, figsize=(15, 10))

    d = g.sort_values("mae")
    ax[0].barh(range(len(d)), d.mae, color=d.color)
    ax[0].set_yticks(range(len(d)))
    ax[0].set_yticklabels(d.name, fontsize=8)
    ax[0].set_xlabel("mean absolute HU error (MAE)")
    ax[0].set_title("Per-CADS-label MAE")

    d = g.sort_values("bias")
    ax[1].barh(range(len(d)), d.bias, color=d.color)
    ax[1].set_yticks(range(len(d)))
    ax[1].set_yticklabels(d.name, fontsize=8)
    ax[1].axvline(0, color="k", lw=0.7)
    ax[1].set_xlabel("HU bias (sCT - GT);  <0 undershoot, >0 overshoot")
    ax[1].set_title("Per-CADS-label HU bias")

    legend = [Patch(color="#dc2626", label="bone (7,27,28,29,30)"),
              Patch(color="#2563eb", label="air-organs (airway, lungs)"),
              Patch(color="#7c3aed", label="Background (label 0)"),
              Patch(color="#9ca3af", label="soft (other)")]
    fig.suptitle("U-Net sCT error per CADS label (micro, 207 center-wise val subjects)", y=1.0, fontsize=13)
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT)
    print(g.sort_values("mae", ascending=False)[["label", "name", "mae", "bias"]].round(1).to_string(index=False))


if __name__ == "__main__":
    main()
