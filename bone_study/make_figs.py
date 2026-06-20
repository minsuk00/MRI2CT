"""Generate narrative figures from saved results (run after experiments)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from toy_data import make_dataset
from toy_core import train_regressor, predict

BS = "/home/minsukc/MRI2CT/bone_study"


def fig_baseline():
    """alpha=0, jitter=1.5: overlay CT, MR, regressor -> shows undershoot + blur."""
    mr_tr, ct_tr, m_tr, b_tr = make_dataset(1500, 0.0, 1.5, seed=1)
    mr_te, ct_te, m_te, b_te = make_dataset(400, 0.0, 1.5, seed=999)
    reg = train_regressor(mr_tr, ct_tr, loss="l1", epochs=400, seed=0)
    pred = predict(reg, mr_te)
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    for ax, s in zip(axes, [1, 5]):
        ax.plot(ct_te[s], "k", lw=2.2, label="CT (target)")
        ax.plot(mr_te[s], "0.6", lw=0.9, label="MR (input)")
        ax.plot(pred[s], "C0", lw=1.6, label="U-Net-style regressor (L1)")
        ax.set_ylim(-2, 7); ax.legend(fontsize=9, ncol=3)
    axes[0].set_title("Regressor undershoots bone peaks and blurs their edges (alpha=0, jitter=1.5)")
    plt.tight_layout(); plt.savefig(f"{BS}/figs/01_baseline_undershoot.png", dpi=95); plt.close()
    print("saved 01_baseline_undershoot.png")


def fig_real_bars():
    regions = [("head_neck", "Head & Neck (skull)"), ("thorax", "Thorax (ribs/spine)")]
    fig, axes = plt.subplots(1, len(regions), figsize=(6.2 * len(regions), 4.6), squeeze=False)
    for ax, (reg, title) in zip(axes[0], regions):
        d = np.load(f"{BS}/real1_{reg}.npy", allow_pickle=True).item()["out"]
        labels = ["no-info\nprior", "retrieval\n(kNN, MR)", "U-Net\nsCT", "info floor\n(neighbor std)"]
        bone = [d["bone"]["prior_mae"], d["bone"]["knn_mae"], d["bone"]["unet_mae"], d["bone"]["spread"]]
        cols = ["#bbbbbb", "#e8964e", "#4e79a7", "#59a14f"]
        x = np.arange(len(labels))
        ax.bar(x, bone, color=cols)
        for xi, v in zip(x, bone):
            ax.text(xi, v + 3, f"{v:.0f}", ha="center", fontsize=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("bone HU MAE"); ax.set_title(title)
        ax.set_ylim(0, max(bone) * 1.18)
    fig.suptitle("Real data: the trained U-Net already sits at the bone information floor; retrieval-by-MR is no better", fontsize=11)
    plt.tight_layout(); plt.savefig(f"{BS}/figs/06_real_bars.png", dpi=95); plt.close()
    print("saved 06_real_bars.png")


if __name__ == "__main__":
    import sys
    if "baseline" in sys.argv or len(sys.argv) == 1:
        fig_baseline()
    if "real" in sys.argv or len(sys.argv) == 1:
        fig_real_bars()
