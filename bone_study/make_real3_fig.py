"""Figures for the real retrieval/atlas experiment (real3_v2)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BS = "/home/minsukc/MRI2CT/bone_study"
d = np.load(f"{BS}/real3_v2_head_neck.npy", allow_pickle=True).item()
rows = d["rows"]


def col(key):
    return np.array([r[key] for r in rows], dtype=float)


fig, ax = plt.subplots(1, 2, figsize=(13, 5))

# --- left: bone MAE per predictor (mean bars + per-query dots) ---
labels = ["U-Net\n(ours)", "retrieval\nrealistic", "retrieval\noracle", "no-MR\nprior(median)"]
keys = ["unet_bmae", "realistic_fused_bmae", "oracle_fused_bmae", "prior_median_bmae"]
cols = ["#4e79a7", "#e8964e", "#9c6ade", "#bbbbbb"]
data = [col(k) for k in keys]
x = np.arange(len(labels))
ax[0].bar(x, [v.mean() for v in data], color=cols, alpha=0.85)
for xi, v in zip(x, data):
    ax[0].scatter(np.full(len(v), xi) + np.random.default_rng(0).uniform(-.12, .12, len(v)), v,
                  color="k", s=14, zorder=3, alpha=0.6)
    ax[0].text(xi, v.mean() + 6, f"{v.mean():.0f}", ha="center", fontweight="bold")
ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
ax[0].set_ylabel("bone HU MAE (lower=better)")
ax[0].set_title("Intensity: U-Net beats realistic AND oracle atlas on bone HU\n(dots = per-query, n=%d)" % len(rows))

# --- right: edge sharpness ---
ekeys = ["unet_edge", "realistic_fused_edge", "oracle_fused_edge"]
elab = ["U-Net", "retrieval\nrealistic", "retrieval\noracle"]
edata = [col(k) for k in ekeys]
x2 = np.arange(len(elab))
ax[1].bar(x2, [np.nanmean(v) for v in edata], color=["#4e79a7", "#e8964e", "#9c6ade"], alpha=0.85)
for xi, v in zip(x2, edata):
    ax[1].scatter(np.full(len(v), xi) + np.random.default_rng(1).uniform(-.12, .12, len(v)), v,
                  color="k", s=14, zorder=3, alpha=0.6)
    ax[1].text(xi, np.nanmean(v) + 0.01, f"{np.nanmean(v):.2f}", ha="center", fontweight="bold")
ax[1].axhline(1.0, color="k", ls=":", lw=1, label="real CT")
ax[1].set_xticks(x2); ax[1].set_xticklabels(elab); ax[1].set_ylabel("bone-edge sharpness (1=real CT)")
ax[1].set_title("Sharpness: the registered atlas is BLURRIER than the U-Net\nat the bone boundary (registration misplaces it)")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{BS}/figs/07_real_retrieval.png", dpi=95)
print("saved 07_real_retrieval.png")

# print summary for the report prose
print("\n=== per-predictor mean bone MAE ===")
for k in keys:
    print(f"  {k:26} {col(k).mean():.1f}")
print("win counts (unet < atlas):")
print(f"  unet<realistic: {(col('unet_bmae')<col('realistic_fused_bmae')).sum()}/{len(rows)}")
print(f"  unet<oracle   : {(col('unet_bmae')<col('oracle_fused_bmae')).sum()}/{len(rows)}")
print(f"  unet<prior_med: {(col('unet_bmae')<col('prior_median_bmae')).sum()}/{len(rows)}")
print("edge: unet sharper than oracle:", (col('unet_edge')>col('oracle_fused_edge')).sum(), "/", len(rows))
