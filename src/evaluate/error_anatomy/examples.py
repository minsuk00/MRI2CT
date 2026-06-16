"""Qualitative image examples + pooled bone-HU histogram for the report.
- fig7_bone_hist: pooled predicted-vs-GT HU distribution inside bone (undershoot + clip).
- example_<subj>.png: axial MR | GT CT | UNet pred | TRUE error | error-as-metric-sees-it (clipped),
  proving the reported metric is blind to cortical-bone error.
"""
import os, numpy as np, nibabel as nib, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
VOL = os.path.join(REPO, "evaluation_results/full_eval_20260609/volumes/unet")
RUN = os.path.join(REPO, "evaluation_results/unet_error_analysis_20260616")
FIG = os.path.join(RUN, "figures")
plt.rcParams.update({"figure.dpi": 130, "font.size": 10})


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


# ---------- pooled bone-HU histogram ----------
s = pd.read_csv(os.path.join(RUN, "summary.csv"))
subs = s[s.model == "unet"][["subj_id", "region"]].drop_duplicates()
# sample across regions for a representative pool
pool = subs.groupby("region").head(8).subj_id.tolist()
gt_bone, pred_bone = [], []
for sid in pool:
    try:
        gt = canon(os.path.join(DATA, sid, "ct.nii"))
        body = canon(os.path.join(DATA, sid, "mask.nii")) > 0
        pr = canon(os.path.join(VOL, sid, "sample.nii.gz"))
    except Exception:
        continue
    m = body & (gt > 200)
    if m.sum() > 5000:
        idx = np.random.RandomState(0).choice(int(m.sum()), 60000, replace=True)
        gt_bone.append(gt[m][idx]); pred_bone.append(pr[m][idx])
gt_bone = np.concatenate(gt_bone); pred_bone = np.concatenate(pred_bone)
fig, ax = plt.subplots(figsize=(8, 4))
bins = np.linspace(-200, 2800, 120)
ax.hist(gt_bone, bins=bins, alpha=0.6, color="#1e3a8a", label=f"GT bone HU (mean {gt_bone.mean():.0f})", density=True)
ax.hist(pred_bone, bins=bins, alpha=0.6, color="#ef4444", label=f"UNet predicted HU (mean {pred_bone.mean():.0f})", density=True)
ax.axvline(1024, color="gray", ls=":", label="+1024 sigmoid cap")
ax.set_xlabel("HU"); ax.set_ylabel("density"); ax.legend()
ax.set_title("Inside true bone, the UNet collapses to the mean — it never predicts dense cortical HU")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig7_bone_hist.png"), bbox_inches="tight"); plt.close(fig)
print("wrote fig7_bone_hist.png  GTmean=%.0f predMean=%.0f" % (gt_bone.mean(), pred_bone.mean()))


# ---------- spatial examples ----------
def best_bone_slice(gt, body, axis=2):
    bone = (gt > 300) & body
    counts = bone.sum(axis=(0, 1)) if axis == 2 else bone.sum(axis=(1, 2))
    return int(np.argmax(counts))


def render(sid, region, bone_win=(-200, 1500)):
    gt = canon(os.path.join(DATA, sid, "ct.nii"))
    mr = canon(os.path.join(DATA, sid, "moved_mr.nii"))
    body = canon(os.path.join(DATA, sid, "mask.nii")) > 0
    pr = canon(os.path.join(VOL, sid, "sample.nii.gz"))
    z = best_bone_slice(gt, body)
    G, P, M, B = gt[:, :, z].T, pr[:, :, z].T, mr[:, :, z].T, body[:, :, z].T
    err = (P - G); err[~B] = 0
    errc = (np.clip(P, -1024, 1024) - np.clip(G, -1024, 1024)); errc[~B] = 0
    hidden = np.abs(err) - np.abs(errc); hidden[~B] = 0  # error the ±1024 clip erases
    fig, axes = plt.subplots(1, 5, figsize=(17, 3.8))
    mrv = np.percentile(M[M > 0], 99) if (M > 0).any() else 1
    panels = [
        (M, dict(cmap="gray", vmin=0, vmax=mrv), "input MR", False),
        (G, dict(cmap="gray", vmin=bone_win[0], vmax=bone_win[1]), "GT CT (bone window)", False),
        (P, dict(cmap="gray", vmin=bone_win[0], vmax=bone_win[1]), "UNet sCT (same window)", False),
        (err, dict(cmap="seismic", vmin=-700, vmax=700), "TRUE error (full HU)", True),
        (hidden, dict(cmap="hot", vmin=0, vmax=700), "error HIDDEN by the ±1024 clip", True),
    ]
    for ax, (im, kw, ttl, cbar) in zip(axes, panels):
        m = ax.imshow(np.flipud(im), origin="lower", **kw); ax.set_title(ttl, fontsize=9); ax.axis("off")
        if cbar:
            fig.colorbar(m, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"{sid} ({region}) — pred skull is visibly grey/undershot (panel 3); the right panel is the "
                 f"cortical-bone error the reported metric never sees", y=1.04, fontsize=11)
    fig.tight_layout()
    out = os.path.join(FIG, f"example_{sid}.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.basename(out), "slice", z)


# pick a representative brain (median brain bone-MAE) + one HN + one pelvis
ub = s[s.model == "unet"]
def pick(region):
    g = ub[ub.region == region].sort_values("mae_bone")
    return g.iloc[len(g) // 2].subj_id  # median
for reg in ["brain", "head_neck", "pelvis"]:
    try:
        render(pick(reg), reg)
    except Exception as e:
        print("ex fail", reg, e)
print("[examples] done")
