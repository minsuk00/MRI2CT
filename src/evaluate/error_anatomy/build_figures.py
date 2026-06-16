"""Build analytical figures (PNG) + key-stats JSON for the UNet error-anatomy report.
Reads summary.csv / structures.csv / oracle_fix.csv produced by extract.py + oracle_fix.py.
"""
import os, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_error_analysis_20260616"
FIG = os.path.join(RUN, "figures"); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "font.size": 11, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.axisbelow": True})
REGION_ORDER = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
MCOL = {"unet": "#2563eb", "amix": "#7c3aed", "maisi": "#dc2626", "mcddpm": "#ea580c"}

s = pd.read_csv(os.path.join(RUN, "summary.csv"))
st = pd.read_csv(os.path.join(RUN, "structures.csv"))
orc = pd.read_csv(os.path.join(RUN, "oracle_fix.csv"))
u = s[s.model == "unet"]
stats = {}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(); fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print("  wrote", name)


# ---- Fig 1: per-voxel tissue MAE (UNet), overall + per region ----
fig, ax = plt.subplots(figsize=(7.5, 4))
tis = ["mae_air", "mae_soft", "mae_bone"]
lab = ["air/gas", "soft tissue", "bone"]
x = np.arange(len(REGION_ORDER)); w = 0.26
for i, (t, l) in enumerate(zip(tis, lab)):
    vals = [u[u.region == r][t].mean() for r in REGION_ORDER]
    ax.bar(x + (i - 1) * w, vals, w, label=l, color=["#94a3b8", "#22c55e", "#1e3a8a"][i])
ax.set_xticks(x); ax.set_xticklabels(REGION_ORDER); ax.set_ylabel("MAE (HU)")
ax.set_title("UNet per-voxel MAE by tissue — bone is ~5× soft everywhere")
ax.legend()
save(fig, "fig1_tissue_mae.png")
stats["tissue_mae_all"] = {l: float(u[t].mean()) for t, l in zip(tis, lab)}

# ---- Fig 2: per-bone-structure MAE + signed bias (UNet) ----
bones = st[(st.is_bone) & (st.model == "unet")]
g = bones.groupby("name").agg(mae=("mae", "mean"), bias=("bias", "mean")).sort_values("mae")
fig, ax = plt.subplots(figsize=(7.5, 4))
y = np.arange(len(g))
ax.barh(y, g["mae"], color="#1e3a8a", label="MAE")
ax.barh(y, g["bias"], color="#ef4444", alpha=0.85, label="signed bias (neg = undershoot)")
ax.set_yticks(y); ax.set_yticklabels(g.index)
ax.set_xlabel("HU"); ax.axvline(0, color="k", lw=0.8)
ax.set_title("UNet bone error by structure — every bone is UNDERSHOT (red < 0)")
ax.legend()
save(fig, "fig2_bone_structures.png")
stats["skull_mae"] = float(g.loc["skull", "mae"]); stats["skull_bias"] = float(g.loc["skull", "bias"])

# ---- Fig 3: diffusion test — predicted bone HU ceiling vs GT ----
gtmean = float(u["gt_bone_mean"].mean())
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4))
mods = ["unet", "amix", "maisi", "mcddpm"]
pm = [s[s.model == m]["pred_bone_max"].mean() for m in mods]
pmean = [s[s.model == m]["pred_bone_mean"].mean() for m in mods]
a1.bar(mods, pm, color=[MCOL[m] for m in mods])
a1.axhline(2976, color="k", ls="--", label="real cortical bone ≈ 2976 HU")
a1.axhline(1024, color="gray", ls=":", label="sigmoid cap (unet/amix)")
a1.set_ylabel("mean predicted bone MAX (HU)")
a1.set_title("Even uncapped MC-DDPM tops out ~1430 HU"); a1.legend(fontsize=8)
cb = [s[s.model == m]["bias_cortical"].mean() for m in mods]
a2.bar(mods, cb, color=[MCOL[m] for m in mods])
a2.set_ylabel("cortical (>1024) signed bias (HU)")
a2.set_title("All models undershoot cortical bone by ~730–900 HU")
save(fig, "fig3_diffusion_test.png")
stats["pred_bone_max"] = dict(zip(mods, [round(v, 0) for v in pm]))
stats["cortical_bias"] = dict(zip(mods, [round(v, 0) for v in cb]))

# ---- Fig 4: ORACLE FIX -> reported PSNR (the headline) ----
fixes = ["base", "fix_air", "fix_soft", "fix_bone", "fix_cortical", "fix_skull"]
flab = ["baseline", "fix air", "fix soft", "fix bone", "fix cortical\n(>1024)", "fix skull"]
ou = orc[orc.model == "unet"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.2))
allv = [ou[f"{f}_psnr"].mean() for f in fixes]
brv = [ou[ou.region == "brain"][f"{f}_psnr"].mean() for f in fixes]
cols = ["#475569", "#94a3b8", "#22c55e", "#1e3a8a", "#1e3a8a", "#1e3a8a"]
for ax, vals, ttl in [(a1, allv, "all regions"), (a2, brv, "brain")]:
    b = ax.bar(flab, vals, color=cols)
    ax.axhline(vals[0], color="#475569", ls="--", lw=1)
    ax.set_ylabel("reported body PSNR (dB)"); ax.set_title(f"Oracle 'perfect tissue' PSNR — {ttl}")
    ax.set_ylim(min(vals) - 0.5, max(vals) + 0.6)
    for rect, v in zip(b, vals):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.05, f"+{v-vals[0]:.2f}" if v != vals[0] else "base",
                ha="center", fontsize=8)
fig.suptitle("Fixing AIR beats fixing BONE on reported PSNR — the metric clips bone away", y=1.02)
save(fig, "fig4_oracle_psnr.png")
stats["oracle_psnr_all"] = {f: float(ou[f"{f}_psnr"].mean()) for f in fixes}
stats["oracle_psnr_brain"] = {f: float(ou[ou.region == "brain"][f"{f}_psnr"].mean()) for f in fixes}

# ---- Fig 5: ORACLE FIX -> full-HU MAE (Track B) ----
fixes2 = ["base", "fix_air", "fix_soft", "fix_bone", "fix_cortical"]
fig, ax = plt.subplots(figsize=(8.5, 4.2))
x = np.arange(len(REGION_ORDER)); w = 0.16
for i, f in enumerate(fixes2):
    vals = [ou[ou.region == r][f"{f}_smae"].mean() for r in REGION_ORDER]
    ax.bar(x + (i - 2) * w, vals, w, label=f.replace("fix_", "fix "),
           color=["#475569", "#94a3b8", "#22c55e", "#1e3a8a", "#3b82f6"][i])
ax.set_xticks(x); ax.set_xticklabels(REGION_ORDER); ax.set_ylabel("full-HU body MAE (HU, lower=better)")
ax.set_title("Full-HU MAE (unclipped): fixing bone helps, esp. brain/HN"); ax.legend(fontsize=8, ncol=5)
save(fig, "fig5_oracle_smae.png")

# ---- Fig 6: error-mass composition (stacked) ----
uu = u.copy()
uu["tot"] = uu.mae_raw * uu.n_body
uu["airm"] = uu.mae_air * uu.n_air
uu["softm"] = uu.mae_soft * uu.n_soft
comp = {}
for r in REGION_ORDER:
    gg = uu[uu.region == r]; tot = gg.tot.sum()
    comp[r] = [100 * gg.airm.sum() / tot, 100 * gg.softm.sum() / tot,
               100 * (gg.bone_err_sum.sum() - gg.cortical_err_sum.sum()) / tot,
               100 * gg.cortical_err_sum.sum() / tot]
fig, ax = plt.subplots(figsize=(8, 4))
arr = np.array([comp[r] for r in REGION_ORDER])
bottom = np.zeros(len(REGION_ORDER))
for i, (l, c) in enumerate(zip(["air/gas", "soft", "bone 200–1024", "cortical >1024"],
                               ["#94a3b8", "#22c55e", "#3b82f6", "#1e3a8a"])):
    ax.bar(REGION_ORDER, arr[:, i], bottom=bottom, label=l, color=c); bottom += arr[:, i]
ax.set_ylabel("% of total body abs-error mass"); ax.legend(fontsize=9)
ax.set_title("Error-mass composition — brain/HN carry a large bone share; others air+soft")
save(fig, "fig6_errormass.png")
stats["errormass"] = {r: dict(zip(["air", "soft", "midbone", "cortical"], [round(v, 1) for v in comp[r]])) for r in REGION_ORDER}

with open(os.path.join(RUN, "key_stats.json"), "w") as f:
    json.dump(stats, f, indent=2)
print("[figures] done; key_stats.json written")
