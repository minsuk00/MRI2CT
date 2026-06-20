"""Figures for the U-Net bone deep-dive (report 06). Reads bone_subject.csv,
bone_oracle.csv, mrct_hist.npz, bone_stats.json + raw volumes for qualitative slices.
"""
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL = os.path.join(REPO, "evaluation_results/full_eval_20260617/volumes/unet")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
os.makedirs(FIG, exist_ok=True)
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
RLAB = ["brain", "head\nneck", "thorax", "abdomen", "pelvis"]
SCEN = ["air", "soft", "bone", "cortical", "skull"]
SC_COL = {"air": "#9ca3af", "soft": "#2563eb", "bone": "#dc2626", "cortical": "#7f1d1d", "skull": "#f59e0b"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 120})

b = pd.read_csv(os.path.join(RUN, "bone_subject.csv"))
oracle = pd.read_csv(os.path.join(RUN, "bone_oracle.csv"), index_col=0).reindex(SCEN)
st = json.load(open(os.path.join(RUN, "bone_stats.json")))
npz = np.load(os.path.join(RUN, "mrct_hist.npz"))


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, name), bbox_inches="tight")
    plt.close(fig)
    print("  ", name)


# F1 — comparative oracle (3 metrics)
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
specs = [("dPSNR", "PSNR gain (dB)", "higher = bigger improvement"),
         ("dBodyMAE", "body-MAE reduction (HU)", "clipped, the reported metric"),
         ("dFullHU_MAE", "full-HU MAE reduction (HU)", "raw, dose-relevant")]
for ax, (col, ylab, sub) in zip(axes, specs):
    vals = oracle[col]
    ax.bar(range(len(SCEN)), vals, color=[SC_COL[s] for s in SCEN])
    win = vals.idxmax()
    ax.set_xticks(range(len(SCEN))); ax.set_xticklabels(SCEN, rotation=30)
    ax.set_ylabel(ylab); ax.set_title(f"{ylab}\n(winner: {win})", fontsize=10)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.2f}" if col == "dPSNR" else f"{v:.0f}", (i, v), ha="center",
                    va="bottom" if v >= 0 else "top", fontsize=8)
fig.suptitle("Oracle: gain from making ONE tissue perfect. Air wins the pixel metrics; bone is mid-pack.", y=1.04)
save(fig, "bf1_oracle.png")

# F2 — per-voxel severity vs aggregate leverage
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
sev = st["severity"]
axes[0].bar(["soft", "bone"], [sev["mae_soft"], sev["mae_bone"]], color=[SC_COL["soft"], SC_COL["bone"]])
axes[0].set_ylabel("per-voxel MAE (HU)")
axes[0].set_title(f"Per-voxel severity: bone {sev['bone_per_voxel_ratio']:.1f}x soft")
for i, v in enumerate([sev["mae_soft"], sev["mae_bone"]]):
    axes[0].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=9)
# aggregate leverage = body-MAE reduction from oracle
lev = oracle["dBodyMAE"].reindex(["air", "soft", "bone"])
axes[1].bar(range(3), lev, color=[SC_COL[s] for s in ["air", "soft", "bone"]])
axes[1].set_xticks(range(3)); axes[1].set_xticklabels(["air", "soft", "bone"])
axes[1].set_ylabel("body-MAE reduction if fixed (HU)")
axes[1].set_title("Aggregate metric leverage: air & soft beat bone")
for i, v in enumerate(lev):
    axes[1].annotate(f"{v:.1f}", (i, v), ha="center", va="bottom", fontsize=9)
fig.suptitle("The disconnect: bone is worst PER VOXEL, but moves the aggregate metric least (it is rare).", y=1.03)
save(fig, "bf2_severity_vs_leverage.png")

# F3 — universality of undershoot
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(b.bias_bone_clip, bins=30, color=SC_COL["bone"], alpha=0.7, label="bone bias (clip)")
axes[0].hist(b.bias_cortical_clip, bins=30, color=SC_COL["cortical"], alpha=0.6, label="cortical bias (clip)")
axes[0].axvline(0, color="k", lw=1)
axes[0].set_xlabel("signed bias (HU), negative = undershoot"); axes[0].set_ylabel("# subjects")
uni = st["universality"]
axes[0].set_title(f"{uni['pct_bone_under_clip']:.0f}% of subjects undershoot bone; "
                  f"{uni['pct_cort_under_clip']:.0f}% undershoot cortical")
axes[0].legend(fontsize=8)
# per-region frac of bone voxels undershot
data = [b[b.region == r].frac_bone_under * 100 for r in REG]
axes[1].boxplot(data, labels=RLAB, showmeans=True)
axes[1].set_ylabel("% of bone voxels undershot per subject")
axes[1].set_title(f"Per-subject: mean {uni['mean_frac_bone_under']:.0f}% of bone voxels undershot (min {uni['min_frac_bone_under']:.0f}%)")
save(fig, "bf3_universality.png")

# F4 — MR-rank vs CT-HU 2D hist, bone vs soft
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
me, he = npz["mr_edges"], npz["hu_edges"]
for ax, key, ttl in [(axes[0], "bone", "BONE voxels"), (axes[1], "soft", "SOFT voxels")]:
    h = npz[key]
    hn = h / np.clip(h.sum(axis=1, keepdims=True), 1, None)  # normalize per MR-rank column
    im = ax.imshow(hn.T, origin="lower", aspect="auto", cmap="viridis",
                   extent=[0, 1, he[0], he[-1]])
    ax.set_xlabel("MR intensity percentile rank (within subject)")
    ax.set_ylabel("CT HU")
    red = st["mr"][f"mr_reduction_{key}"]
    ax.set_title(f"{ttl}: P(CT HU | MR rank)   MR narrows spread only {red:.0f}%")
    fig.colorbar(im, ax=ax, shrink=0.8)
fig.suptitle("Bone CT-HU is wide and barely depends on MR rank (broad vertical smear); soft CT-HU is a narrow band near 0.", y=1.04)
save(fig, "bf4_mrct_hist.png")

# F5 — intrinsic CT-HU spread + how little MR reduces it
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
mr = st["mr"]
axes[0].bar(["bone", "soft"], [mr["ctstd_bone_mean"], mr["ctstd_soft_mean"]],
            color=[SC_COL["bone"], SC_COL["soft"]])
for i, v in enumerate([mr["ctstd_bone_mean"], mr["ctstd_soft_mean"]]):
    axes[0].annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=10)
axes[0].set_ylabel("intrinsic CT-HU std (HU)")
axes[0].set_title(f"Bone HU is ~{mr['ctstd_bone_mean']/mr['ctstd_soft_mean']:.0f}x wider than soft")
axes[1].bar(["bone", "soft"], [mr["mr_reduction_bone"], mr["mr_reduction_soft"]],
            color=[SC_COL["bone"], SC_COL["soft"]])
for i, v in enumerate([mr["mr_reduction_bone"], mr["mr_reduction_soft"]]):
    axes[1].annotate(f"{v:.0f}%", (i, v), ha="center", va="bottom", fontsize=10)
axes[1].set_ylabel("% of CT-HU spread removed by knowing MR")
axes[1].set_title(f"MR resolves almost none of bone HU\n(rho^2 bone {mr['rho2_bone']:.2f}, soft {mr['rho2_soft']:.2f})")
fig.suptitle("Bone HU is intrinsically wide AND MR-unresolvable -> L1 regresses to the median -> undershoot.", y=1.03)
save(fig, "bf5_spread.png")

# F6 — localization vs magnitude
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
lm = st["locmag"]
axes[0].bar(range(len(REG)), [lm[r]["shape_dice"] for r in REG], color="#0ea5e9")
axes[0].set_xticks(range(len(REG))); axes[0].set_xticklabels(RLAB); axes[0].set_ylabel("bone shape Dice (pred>200 vs GT>200)")
axes[0].set_title("Localization: where is bone?")
axes[1].bar(range(len(REG)), [lm[r]["missed_frac"] * 100 for r in REG], color="#f59e0b")
axes[1].set_xticks(range(len(REG))); axes[1].set_xticklabels(RLAB); axes[1].set_ylabel("% GT bone missed (pred<200)")
axes[1].set_title("Under-detection of thin bone")
w = 0.38
xr = np.arange(len(REG))
axes[2].bar(xr - w / 2, [lm[r]["mae_bone_interior"] for r in REG], w, label="interior", color="#b91c1c")
axes[2].bar(xr + w / 2, [lm[r]["mae_bone_boundary"] for r in REG], w, label="boundary", color="#fca5a5")
axes[2].set_xticks(xr); axes[2].set_xticklabels(RLAB); axes[2].set_ylabel("MAE (HU)")
axes[2].set_title("Magnitude: interior worse than edge"); axes[2].legend(fontsize=8)
fig.suptitle("It is a density-MAGNITUDE failure (interior error high), plus thin-bone under-detection in body regions.", y=1.03)
save(fig, "bf6_locmag.png")

# F7 — loss imbalance
fig, ax = plt.subplots(figsize=(6, 4.2))
loss = st["loss"]
bone_err_share = st["tissue"]["bone"]["err_share_pct"]   # clipped frame (the frame the L1 optimizes)
ax.bar(["voxel share", "error share"], [loss["bone_vox_pct"], bone_err_share],
       color=["#94a3b8", SC_COL["bone"]])
for i, v in enumerate([loss["bone_vox_pct"], bone_err_share]):
    ax.annotate(f"{v:.1f}%", (i, v), ha="center", va="bottom", fontsize=10)
ax.set_ylabel("% of body (clipped frame)")
ax.set_title(f"Bone is {loss['bone_vox_pct']:.0f}% of voxels: tiny weight in the uniform L1 loss\n"
             f"(it is {bone_err_share:.0f}% of the error, {bone_err_share/loss['bone_vox_pct']:.1f}x its voxel share)")
save(fig, "bf7_loss_imbalance.png")

# F8 — ceiling vs regression-to-mean (pooled pred-bone HU dist via bone marginal of 05 npz)
fig, ax = plt.subplots(figsize=(8, 4.2))
bh = np.load(os.path.join(RUN, "cads_bone_calib.npz"))  # GT x pred over CADS-bone voxels
H, PRE = bh["hist"], bh["pred_edges"]
GTE = bh["gt_edges"]
prc = (PRE[:-1] + PRE[1:]) / 2
gtc = (GTE[:-1] + GTE[1:]) / 2
pred_marg = H.sum(axis=0); gt_marg = H.sum(axis=1)
_gw, _pw = GTE[1] - GTE[0], PRE[1] - PRE[0]  # true density: divide each by its own bin width
ax.plot(gtc, gt_marg / gt_marg.sum() / _gw, color=SC_COL["bone"], lw=1.6, label="GT bone HU")
ax.plot(prc, pred_marg / pred_marg.sum() / _pw, color="#111827", lw=1.6, label="pred in true bone")
ax.axvline(1024, color="lime", lw=1.4, label="sigmoid cap = 1024 HU")
ax.axvline(sev := st["severity"]["pred_bone_max_mean"], color="purple", ls="--", lw=1.2,
           label=f"mean pred max = {sev:.0f} HU")
ax.axvline(st["severity"]["pred_bone_mean"], color="#111827", ls=":", lw=1)
ax.set_xlim(-200, 2600); ax.set_xlabel("HU"); ax.set_ylabel("density")
ax.set_title("The cap is NOT the binding limit: predictions stop ~951 HU, well BELOW the 1024 cap\n"
             "(regression to the conditional mean, not clipping)")
ax.legend(fontsize=8)
save(fig, "bf8_ceiling.png")


# F9 — qualitative example slices (reuse error_anatomy/examples.py renderer)
def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def best_bone_slice(gt, body):
    bone = (gt > 300) & body
    return int(np.argmax(bone.sum(axis=(0, 1))))


def render(sid, region, bone_win=(-200, 1500)):
    gt = canon(os.path.join(DATA, sid, "ct.nii"))
    mr = canon(os.path.join(DATA, sid, "moved_mr.nii"))
    body = canon(os.path.join(DATA, sid, "mask.nii")) > 0
    pr = canon(os.path.join(VOL, sid, "sample.nii.gz"))
    z = best_bone_slice(gt, body)
    G, P, M, B = gt[:, :, z].T, pr[:, :, z].T, mr[:, :, z].T, body[:, :, z].T
    err = (P - G); err[~B] = 0
    errc = (np.clip(P, -1024, 1024) - np.clip(G, -1024, 1024)); errc[~B] = 0
    hidden = np.abs(err) - np.abs(errc); hidden[~B] = 0
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    mrv = np.percentile(M[M > 0], 99) if (M > 0).any() else 1
    panels = [
        (M, dict(cmap="gray", vmin=0, vmax=mrv), "input MR", False),
        (G, dict(cmap="gray", vmin=bone_win[0], vmax=bone_win[1]), "GT CT (bone window)", False),
        (P, dict(cmap="gray", vmin=bone_win[0], vmax=bone_win[1]), "U-Net sCT (same window)", False),
        (err, dict(cmap="seismic", vmin=-700, vmax=700), "true error (pred-GT, full HU)", True),
        (hidden, dict(cmap="hot", vmin=0, vmax=700), "error hidden by the +/-1024 clip", True),
    ]
    for ax, (im, kw, ttl, cbar) in zip(axes, panels):
        him = ax.imshow(np.flipud(im), origin="lower", **kw)
        ax.set_title(ttl, fontsize=9); ax.axis("off")
        if cbar:
            fig.colorbar(him, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{sid} ({region}): the predicted skull/bone is visibly grey (undershot) in panel 3; "
                 f"panel 5 is the dense-bone error the clipped metric never sees", y=1.05, fontsize=10)
    for ax in axes:
        ax.grid(False)
    save(fig, f"bf9_example_{region}.png")


def pick_median(region):
    sr = b[b.region == region].sort_values("mae_bone")
    return sr.iloc[len(sr) // 2].subj_id


for r in REG:
    try:
        render(pick_median(r), r)
    except Exception as e:
        print("  example fail", r, e)

# ---- explanatory proof figures (q1 air paradox, q2 calibration, q3 MR conflation) ----
TC = {"air": "#9ca3af", "soft": "#2563eb", "bone": "#dc2626", "cort": "#7f1d1d"}

# q1 — air paradox: commonness x per-voxel error -> total error
s05 = pd.read_csv(os.path.join(RUN, "per_subject.csv"))
tv, te = s05.n_body.sum(), s05.total_aerr_sum_clip.sum()
ts = ["air", "soft", "bone"]
vox = [100 * s05[f"n_{t}"].sum() / tv for t in ts]
pv = [s05[f"mae_{t}_clip"].mean() for t in ts]
sh = [100 * s05[f"aerr_sum_{t}_clip"].sum() / te for t in ts]
fig, ax = plt.subplots(1, 3, figsize=(13, 4))
for a, (vals, lab, ttl) in zip(ax, [(vox, "% of body voxels", "How common (count)"),
                                    (pv, "per-voxel MAE (HU)", "How wrong each voxel is"),
                                    (sh, "% of total error = oracle gain", "Total error contributed")]):
    a.bar(ts, vals, color=[TC[t] for t in ts])
    for i, v in enumerate(vals):
        a.annotate(f"{v:.0f}", (i, v), ha="center", va="bottom", fontsize=10)
    a.set_ylabel(lab); a.set_title(ttl)
fig.suptitle("Air carries the most error not because it is hard, but because it is common (27%) AND moderately wrong (123 HU).\n"
             "Bone is worst per-voxel (241 HU) but rare (5%). All voxels are INSIDE the body (lung/gas/sinus), not background.", y=1.08, fontsize=10)
save(fig, "q1_air_paradox.png")

# q2 — regression-to-mean calibration (true bone HU -> mean predicted HU), CADS bone
z = np.load(os.path.join(RUN, "cads_bone_calib.npz"))
H, GTE, PRE = z["hist"], z["gt_edges"], z["pred_edges"]
gtc = (GTE[:-1] + GTE[1:]) / 2
prc = (PRE[:-1] + PRE[1:]) / 2
edges = [200, 400, 700, 1000, 1500, 2000, 2900]
xs, ys = [], []
for lo, hi in zip(edges[:-1], edges[1:]):
    w = H[(gtc >= lo) & (gtc < hi)].sum(0)
    if w.sum() < 1:
        continue
    xs.append((lo + hi) / 2)
    ys.append((prc * w).sum() / w.sum())
fig, ax = plt.subplots(figsize=(7, 5.5))
ax.plot([0, 2900], [0, 2900], "k--", lw=1, label="perfect (identity)")
ax.plot(xs, ys, "o-", color=TC["bone"], lw=2, ms=7, label="what the U-Net predicts")
ax.axhline(np.average(gtc, weights=H.sum(1)), color="gray", ls=":", label="GT bone MEAN (L1's 'safe' answer)")
for x, y in zip(xs, ys):
    ax.annotate(f"-{x - y:.0f}", (x, y), fontsize=8, xytext=(3, -12), textcoords="offset points")
ax.set_xlabel("TRUE bone HU"); ax.set_ylabel("mean predicted HU")
ax.set_title("Regression to the mean: the denser the true bone,\nthe further the prediction flattens below the identity line")
ax.legend()
save(fig, "q2_calibration.png")

# q3 — MR conflation: MR-rank distribution per tissue
mp = np.load(os.path.join(RUN, "mr_tissue_pool.npz"))
mst = json.load(open(os.path.join(RUN, "mr_tissue_stats.json")))
bins = np.linspace(0, 1, 41)
fig, ax = plt.subplots(figsize=(8, 4.6))
for k, lab in [("air", "air (lung/gas)"), ("soft", "soft tissue"), ("bone", "bone (all)"), ("cort", "cortical bone")]:
    ax.hist(mp[k], bins=bins, density=True, histtype="step", lw=2, color=TC[k],
            label=f"{lab}  (median {mst['median'][k]:.2f})")
ax.set_xlabel("MR intensity percentile rank within subject  (0 = darkest, 1 = brightest)")
ax.set_ylabel("density")
ax.set_title(f"MR cannot distinguish bone density: bone overlaps SOFT ({mst['overlap_bone_soft']:.2f}) "
             f"and cortical overlaps AIR ({mst['overlap_cort_air']:.2f}).\n"
             "One MR brightness maps to many CT densities.")
ax.legend(fontsize=9)
save(fig, "q3_mr_conflation.png")

print("[bone_figures] done ->", FIG)
