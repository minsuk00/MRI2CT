"""Build the 16 figures for the U-Net failure-anatomy report. Reads the CSVs / npz /
json written by aggregate.py + extract.py; writes PNGs into RUN/figures.

Frame C = error vs GT clipped to [-1024,1024] (how validation was scored, primary).
Frame R = error vs raw GT. Sign: err = pred - gt; negative bias = undershoot.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
os.makedirs(FIG, exist_ok=True)
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
RLAB = ["brain", "head\nneck", "thorax", "abdomen", "pelvis"]
C_AIR, C_SOFT, C_BONE = "#9ca3af", "#2563eb", "#dc2626"
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 120})

s = pd.read_csv(os.path.join(RUN, "per_subject.csv"))
lab = pd.read_csv(os.path.join(RUN, "per_label.csv"))
rt = pd.read_csv(os.path.join(RUN, "region_tissue.csv"), index_col=0).reindex(REG)
rm = pd.read_csv(os.path.join(RUN, "region_mass.csv"), index_col=0).reindex(REG)
rb = pd.read_csv(os.path.join(RUN, "region_bonehu.csv"), index_col=0).reindex(REG)
pla = pd.read_csv(os.path.join(RUN, "per_label_agg.csv"), index_col=0)
bvn = pd.read_csv(os.path.join(RUN, "bone_vs_nonbone.csv"), index_col=0)
oracle = pd.read_csv(os.path.join(RUN, "oracle.csv"), index_col=0)
recon = pd.read_csv(os.path.join(RUN, "recon.csv"), index_col=0).reindex(REG)
worst = json.load(open(os.path.join(RUN, "region_worst.json")))
npz = np.load(os.path.join(RUN, "bone_hist.npz"))
H, GTE, PRE = npz["hist"], npz["gt_edges"], npz["pred_edges"]
GTC = (GTE[:-1] + GTE[1:]) / 2
PRC = (PRE[:-1] + PRE[1:]) / 2


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


def regmean(col):
    return [s[s.region == r][col].mean() for r in REG]


x = np.arange(len(REG))

# 1 — per-region MAE by tissue
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (nm, c) in enumerate([("air", C_AIR), ("soft", C_SOFT), ("bone", C_BONE)]):
    ax.bar(x + (i - 1) * 0.26, regmean(f"mae_{nm}_clip"), 0.26, label=nm, color=c)
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("MAE (HU), Frame C")
ax.set_title("Per-voxel MAE by GT tissue (clipped frame)"); ax.legend()
save(fig, "fig1_tissue_mae.png")

# 2 — per-region signed bias by tissue
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, (nm, c) in enumerate([("air", C_AIR), ("soft", C_SOFT), ("bone", C_BONE)]):
    ax.bar(x + (i - 1) * 0.26, regmean(f"bias_{nm}_clip"), 0.26, label=nm, color=c)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("signed bias (HU), Frame C")
ax.set_title("Signed bias by tissue (negative = undershoot)"); ax.legend()
save(fig, "fig2_tissue_bias.png")

# 3 — error-mass share by tissue per region + bone voxel-share marker
fig, ax = plt.subplots(figsize=(8, 4.2))
air, soft, bone = rm["air err-mass %"], rm["soft err-mass %"], rm["bone err-mass %"]
ax.bar(x, air, 0.6, label="air", color=C_AIR)
ax.bar(x, soft, 0.6, bottom=air, label="soft", color=C_SOFT)
ax.bar(x, bone, 0.6, bottom=air + soft, label="bone", color=C_BONE)
ax.scatter(x, rm["bone vox %"], marker="D", color="k", zorder=5, label="bone VOXEL share")
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("% of region abs-error mass")
ax.set_title("Error-mass share by tissue (bars) vs bone voxel share (◆)"); ax.legend(fontsize=8)
save(fig, "fig3_errormass_share.png")

# 4 — 2D joint hist pred vs raw GT (bone voxels)
fig, ax = plt.subplots(figsize=(6.6, 6))
im = ax.imshow(np.log10(H.T + 1), origin="lower", aspect="auto", cmap="magma",
               extent=[GTE[0], GTE[-1], PRE[0], PRE[-1]])
lim = [GTE[0], 1300]
ax.plot([GTE[0], PRE[-1]], [GTE[0], PRE[-1]], "w--", lw=1, label="identity")
pbm = s.pred_bone_max.mean()
ax.axhline(pbm, color="cyan", lw=1, ls=":", label=f"mean pred ceiling {pbm:.0f} HU")
ax.axvline(1024, color="lime", lw=1, ls="-", label="GT clip = 1024")
ax.axvspan(1024, GTE[-1], color="green", alpha=0.08)
ax.set_xlabel("GT HU (raw)"); ax.set_ylabel("pred HU")
ax.set_title("Bone voxels: pred vs GT HU (log count)\nGreen = clipped-away cortical region")
ax.legend(loc="lower right", fontsize=8)
fig.colorbar(im, ax=ax, label="log10(count+1)", shrink=0.8)
save(fig, "fig4_bone_joint_hist.png")

# 5 — 1D HU distributions in bone: GT raw vs pred
gt_marg = H.sum(axis=1); pred_marg = H.sum(axis=0)
fig, ax = plt.subplots(figsize=(8, 4.2))
_gw, _pw = GTE[1] - GTE[0], PRE[1] - PRE[0]  # true density: divide each by its own bin width
ax.plot(GTC, gt_marg / gt_marg.sum() / _gw, color=C_BONE, lw=1.8, label="GT (raw) in bone")
ax.plot(PRC, pred_marg / pred_marg.sum() / _pw, color="#111827", lw=1.8, label="pred in true bone")
ax.axvline(s.gt_bone_mean.mean(), color=C_BONE, ls=":", lw=1)
ax.axvline(s.pred_bone_mean.mean(), color="#111827", ls=":", lw=1)
ax.axvline(1024, color="lime", lw=1, label="GT clip = 1024")
ax.set_xlim(-200, 2600); ax.set_xlabel("HU"); ax.set_ylabel("density")
ax.set_title(f"HU inside true bone: GT mean {s.gt_bone_mean.mean():.0f} → pred mean "
             f"{s.pred_bone_mean.mean():.0f}; pred never exceeds ~{s.pred_bone_max.mean():.0f}")
ax.legend()
save(fig, "fig5_bone_hu_hist.png")

# 6 — midbone vs cortical, MAE & bias, both frames
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
cats = ["midbone\n200..1024", "cortical\n>1024"]
xc = np.arange(2)
mae_c = [s.mae_midbone_clip.mean(), s.mae_cortical_clip.mean()]
mae_r = [s.mae_midbone_raw.mean(), s.mae_cortical_raw.mean()]
axes[0].bar(xc - 0.2, mae_c, 0.4, label="Frame C (clipped)", color="#1d4ed8")
axes[0].bar(xc + 0.2, mae_r, 0.4, label="Frame R (raw)", color="#93c5fd")
axes[0].set_xticks(xc); axes[0].set_xticklabels(cats); axes[0].set_ylabel("MAE (HU)")
axes[0].set_title("Bone-magnitude MAE"); axes[0].legend(fontsize=8)
bias_c = [s.bias_midbone_clip.mean(), s.bias_cortical_clip.mean()]
bias_r = [s.bias_midbone_raw.mean(), s.bias_cortical_raw.mean()]
axes[1].bar(xc - 0.2, bias_c, 0.4, label="Frame C", color="#b91c1c")
axes[1].bar(xc + 0.2, bias_r, 0.4, label="Frame R", color="#fca5a5")
axes[1].axhline(0, color="k", lw=0.8)
axes[1].set_xticks(xc); axes[1].set_xticklabels(cats); axes[1].set_ylabel("signed bias (HU)")
axes[1].set_title("Bone-magnitude bias (undershoot)"); axes[1].legend(fontsize=8)
save(fig, "fig6_midcort.png")

# 7 — per-region cortical undershoot both frames
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(x - 0.2, regmean("bias_cortical_clip"), 0.4, label="Frame C (in-range failure)", color="#b91c1c")
ax.bar(x + 0.2, regmean("bias_cortical_raw"), 0.4, label="Frame R (incl. clipped target)", color="#fca5a5")
ax.axhline(0, color="k", lw=0.8)
for i, r in enumerate(REG):
    n = int(s[s.region == r].n_cortical.mean())
    ax.annotate(f"n≈{n}", (i, 30), ha="center", fontsize=7, color="#374151")
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("cortical signed bias (HU)")
ax.set_title("Cortical-bone undershoot per region"); ax.legend(fontsize=8)
save(fig, "fig7_region_cortical.png")

# 8 — cap pileup per region
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.bar(x - 0.2, [s[s.region == r].pred_near_ceiling_frac.mean() * 100 for r in REG], 0.4,
       label="pred ≥ 850 HU", color="#7c3aed")
ax.bar(x + 0.2, [s[s.region == r].pred_capped_frac.mean() * 100 for r in REG], 0.4,
       label="pred ≥ 1000 HU", color="#c4b5fd")
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("% of true-bone voxels")
ax.set_title("Prediction pile-up near the model's HU ceiling"); ax.legend(fontsize=8)
save(fig, "fig8_cap_pileup.png")

# 9 — ECDF pred vs GT in bone
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.plot(GTC, np.cumsum(gt_marg) / gt_marg.sum(), color=C_BONE, lw=1.8, label="GT (raw) in bone")
ax.plot(PRC, np.cumsum(pred_marg) / pred_marg.sum(), color="#111827", lw=1.8, label="pred in true bone")
ax.axvline(1024, color="lime", lw=1, label="GT clip = 1024")
ax.set_xlim(-200, 2600); ax.set_xlabel("HU"); ax.set_ylabel("cumulative fraction")
ax.set_title("ECDF of HU in true bone: pred saturates into a near-vertical wall"); ax.legend()
save(fig, "fig9_ecdf.png")

# 10 — per-label MAE barh (all labels, bone vs non-bone)
d = pla.sort_values("MAE (C)")
fig, ax = plt.subplots(figsize=(8, 9))
colors = [C_BONE if b else C_SOFT for b in d["is_bone"]]
ax.barh(range(len(d)), d["MAE (C)"], xerr=d["MAE std"].fillna(0), color=colors,
        error_kw={"elinewidth": 0.6, "ecolor": "#6b7280"})
ax.set_yticks(range(len(d)))
ax.set_yticklabels([f"{n}  (n={int(ns)})" for n, ns in zip(d.index, d["n subj"])], fontsize=8)
ax.set_xlabel("MAE (HU), Frame C")
ax.set_title("Per-CADS-label MAE (red = bone, blue = soft/organ; ±std across subjects)")
save(fig, "fig10_label_mae.png")

# 11 — per-label bias barh
d2 = pla.sort_values("bias (C)")
fig, ax = plt.subplots(figsize=(8, 9))
colors = [C_BONE if b else C_SOFT for b in d2["is_bone"]]
ax.barh(range(len(d2)), d2["bias (C)"], color=colors)
ax.axvline(0, color="k", lw=0.8)
ax.set_yticks(range(len(d2))); ax.set_yticklabels(d2.index, fontsize=8)
ax.set_xlabel("signed bias (HU), Frame C  (negative = undershoot)")
ax.set_title("Per-CADS-label signed bias (bone labels cluster strongly negative)")
save(fig, "fig11_label_bias.png")

# 12 — GT vs pred mean HU per label
fig, ax = plt.subplots(figsize=(6.8, 6.2))
for b, c, lb in [(True, C_BONE, "bone"), (False, C_SOFT, "soft/organ")]:
    sub = pla[pla["is_bone"] == b]
    ax.scatter(sub["GT HU"], sub["pred HU"], s=np.sqrt(sub["mean vox"]) / 3,
               color=c, alpha=0.7, label=lb, edgecolor="k", linewidth=0.3)
lo, hi = -1100, 1400
ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")
for n in ["skull", "spine", "thoracic_cage", "limb_girdle"]:
    if n in pla.index:
        ax.annotate(n, (pla.loc[n, "GT HU"], pla.loc[n, "pred HU"]), fontsize=7,
                    xytext=(4, -8), textcoords="offset points")
ax.set_xlim(lo, hi); ax.set_ylim(lo, 1100)
ax.set_xlabel("GT mean HU (raw)"); ax.set_ylabel("pred mean HU")
ax.set_title("Per-label GT vs pred mean HU\n(bone points fall below identity = undershoot)")
ax.legend()
save(fig, "fig12_scatter.png")

# 13 — per-region worst-5 labels
fig, axes = plt.subplots(1, 5, figsize=(15, 3.8), sharey=False)
for ax, r in zip(axes, REG):
    recs = worst[r][::-1]
    names = [x["name"] for x in recs]
    vals = [x["MAE"] for x in recs]
    cols = [C_BONE if x["is_bone"] else C_SOFT for x in recs]
    ax.barh(range(len(recs)), vals, color=cols)
    ax.set_yticks(range(len(recs))); ax.set_yticklabels(names, fontsize=8)
    ax.set_title(r); ax.set_xlabel("MAE (HU)")
fig.suptitle("Top-5 worst-predicted CADS labels per region (red = bone)", y=1.03)
save(fig, "fig13_region_worst.png")

# 14 — per-region contribution to total error mass, stacked by tissue
fig, ax = plt.subplots(figsize=(8, 4.2))
grand = s.total_aerr_sum_clip.sum()
air_m = [s[s.region == r].aerr_sum_air_clip.sum() / grand * 100 for r in REG]
soft_m = [s[s.region == r].aerr_sum_soft_clip.sum() / grand * 100 for r in REG]
bone_m = [s[s.region == r].aerr_sum_bone_clip.sum() / grand * 100 for r in REG]
ax.bar(x, air_m, 0.6, label="air", color=C_AIR)
ax.bar(x, soft_m, 0.6, bottom=air_m, label="soft", color=C_SOFT)
ax.bar(x, bone_m, 0.6, bottom=np.array(air_m) + np.array(soft_m), label="bone", color=C_BONE)
ax.set_xticks(x); ax.set_xticklabels(RLAB); ax.set_ylabel("% of TOTAL dataset error mass")
ax.set_title("Where the dataset's total error mass lives (region × tissue)"); ax.legend(fontsize=8)
save(fig, "fig14_region_massshare.png")

# 15 — reconciliation
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].bar(x - 0.2, recon["recomp MAE (R)"], 0.4, label="recomputed", color="#0ea5e9")
axes[0].bar(x + 0.2, recon["released synthrad_mae"], 0.4, label="released", color="#a7f3d0")
axes[0].set_xticks(x); axes[0].set_xticklabels(RLAB); axes[0].set_ylabel("MAE (HU)")
axes[0].set_title(f"Frame R vs synthrad_mae (max Δ={recon['Δ R'].max():.3f})"); axes[0].legend(fontsize=8)
axes[1].bar(x - 0.2, recon["recomp body MAE (C)"], 0.4, label="recomputed", color="#0ea5e9")
axes[1].bar(x + 0.2, recon["released body_mae_hu"], 0.4, label="released", color="#a7f3d0")
axes[1].set_xticks(x); axes[1].set_xticklabels(RLAB); axes[1].set_ylabel("MAE (HU)")
axes[1].set_title(f"Frame C vs body_mae_hu (max Δ={recon['Δ C'].max():.3f})"); axes[1].legend(fontsize=8)
save(fig, "fig15_recon.png")

# 16 — oracle: body MAE as-is vs bone-fixed
order = REG + ["OVERALL"]
od = oracle.reindex(order)
xo = np.arange(len(order))
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(xo - 0.2, od["body MAE (C)"], 0.4, label="as-is (Frame C)", color="#374151")
ax.bar(xo + 0.2, od["bone-fixed (C)"], 0.4, label="bone predicted perfectly", color="#10b981")
for i, r in enumerate(order):
    ax.annotate(f"-{od.loc[r,'drop % (C)']:.0f}%", (i, od.loc[r, "body MAE (C)"] + 1),
                ha="center", fontsize=8, color="#065f46")
ax.set_xticks(xo); ax.set_xticklabels([r.replace("_", "\n") for r in order])
ax.set_ylabel("body-voxel MAE (HU), Frame C")
ax.set_title("Oracle: body MAE if bone were predicted perfectly (the scored metric)")
ax.legend()
save(fig, "fig16_oracle.png")

print("[figures] done ->", FIG)
