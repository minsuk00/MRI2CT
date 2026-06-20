"""Verification figure for report 09: shows (A) bone HU separability is preserved
in the sCT (AUC), and (B) a pure intensity recalibration recovers most of an
HU-THRESHOLD bone task but NONE of the CNN-segmenter bone Dice -- proving the
segmenter's bone loss is not an HU-magnitude effect (the segmenter is intensity-
invariant), it is structural."""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_failure_20260619"
FIG = os.path.join(RUN, "figures")

vd = pd.read_csv(os.path.join(RUN, "verify_density.csv"))
vr = pd.read_csv(os.path.join(RUN, "verify_recalib.csv"))

auc_gt, auc_sct = vd.auc_gt.mean(), vd.auc_sct.mean()
# HU-threshold recovery (segmenter-free)
gap_thr = vd.dice_real_t150.mean() - vd.dice_sct_t150.mean()
rec_thr = (vd.dice_sct_best.mean() - vd.dice_sct_t150.mean()) / gap_thr * 100
# CNN recalibration oracle recovery
ce, sc, re_ = vr.dice_ceiling.mean(), vr.dice_sct.mean(), vr.dice_recal.mean()
rec_cnn = (re_ - sc) / (ce - sc) * 100

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))

ax[0].bar(["real CT\n(ceiling)", "sCT"], [auc_gt, auc_sct], color=["#93c5fd", "#1d4ed8"])
ax[0].axhline(0.5, color="#9ca3af", ls=":", lw=1)
ax[0].text(1.4, 0.51, "chance", color="#6b7280", fontsize=8)
for i, v in enumerate([auc_gt, auc_sct]):
    ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
ax[0].set_ylim(0.4, 1.0)
ax[0].set_ylabel("AUC: HU separates bone vs non-bone")
ax[0].set_title("Bone HU separability preserved in sCT\n(gross localization intact)")

bars = ax[1].bar(["HU-threshold task\n(retune cut-point)", "CNN segmenter\n(hist-match + re-seg)"],
                 [rec_thr, rec_cnn], color=["#16a34a", "#dc2626"])
ax[1].axhline(0, color="k", lw=0.8)
for b, v in zip(bars, [rec_thr, rec_cnn]):
    ax[1].text(b.get_x() + b.get_width() / 2, v + (2 if v >= 0 else -6),
               f"{v:.0f}%", ha="center", fontsize=11)
ax[1].set_ylabel("% of bone-Dice gap recovered\nby a pure intensity fix")
ax[1].set_title("Intensity recalibration fixes the threshold task,\nnot the CNN -> CNN loss is not HU magnitude")

fig.tight_layout()
fig.savefig(os.path.join(FIG, "vf1_verification.png"), dpi=120, bbox_inches="tight")
plt.close(fig)
print(f"[verify_figure] AUC {auc_gt:.3f}/{auc_sct:.3f}  thr-recovery {rec_thr:.0f}%  cnn-recovery {rec_cnn:.0f}%")
