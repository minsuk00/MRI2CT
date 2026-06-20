"""Figures for report 10 (CADS-label error decomposition). Reads cads_*.csv/npz +
loads volumes for the GT-CT vs sCT qualitative panels. Writes c*.png to RUN/figures."""
import os
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_erosion

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
BONE = [7, 27, 28, 29, 30]


def save(fig, n):
    fig.savefig(os.path.join(FIG, n), dpi=120, bbox_inches="tight")
    plt.close(fig)


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def fig_groups(gr):
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    gr = gr.set_index("group")
    order = ["soft (other CADS)", "unlabeled (CADS=0)", "bone (5 labels)", "air-organs (airway+lung)"]
    gr = gr.reindex([o for o in order if o in gr.index])
    gcol = {"soft (other CADS)": "#9ca3af", "unlabeled (CADS=0)": "#7c3aed",
            "bone (5 labels)": "#dc2626", "air-organs (airway+lung)": "#2563eb"}
    cols = [gcol[g] for g in gr.index]
    ax[0].bar(range(len(gr)), gr.voxshare_pct, color=cols)
    ax[0].set_title("voxel share of body (%)")
    ax[1].bar(range(len(gr)), gr.mae, color=cols)
    ax[1].set_title("per-voxel MAE (HU)")
    ax[2].bar(range(len(gr)), gr.errmass_pct, color=cols)
    ax[2].set_title("share of total body error (%)")
    for a in ax:
        a.set_xticks(range(len(gr)))
        a.set_xticklabels([g.split(" (")[0] for g in gr.index], rotation=20, ha="right", fontsize=9)
    fig.suptitle("CADS group decomposition (micro, additive) — severity (MAE) vs leverage (error share)", y=1.02)
    save(fig, "c_groups.png")


AIRORG = {9, 13}   # air-organs group = airway + lungs (matches the group decomposition)


def _cols(df):
    out = []
    for l, b in zip(df.label, df.is_bone):
        if b:
            out.append("#dc2626")            # bone
        elif l in AIRORG:
            out.append("#2563eb")            # air-organs (airway+lungs)
        elif l == 0:
            out.append("#7c3aed")            # unlabeled (Background/air)
        else:
            out.append("#9ca3af")            # soft (incl. bowel)
    return out


def fig_perlabel(lab):
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    panels = [("errmass_pct", "share of total body error (%)", True),
              ("mae", "per-voxel MAE (HU)", True),
              ("bias", "HU bias (sCT - GT)", False)]
    for a, (col, title, desc) in zip(ax, panels):
        d = lab.sort_values(col, ascending=not desc if col != "bias" else True)
        a.barh(range(len(d)), d[col], color=_cols(d))
        a.set_yticks(range(len(d)))
        a.set_yticklabels(d.name, fontsize=7)
        a.set_xlabel(title)
        if col == "bias":
            a.axvline(0, color="k", lw=0.6)
    ax[0].set_title("error contribution per CADS label")
    ax[1].set_title("MAE per CADS label")
    ax[2].set_title("HU bias per CADS label")
    fig.suptitle("Per CADS label — red = bone, blue = air-organs (airway+lungs), purple = unlabeled (background/air), "
                 "grey = soft (incl. bowel)", y=1.01)
    save(fig, "c_perlabel.png")


def fig_calib(npz):
    d = np.load(npz)
    ge, pe = d["gt_edges"], d["pred_edges"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for i, (k, t) in enumerate([("unlabeled", "unlabeled (CADS=0)"), ("soft", "soft"),
                                ("airorg", "air-organs"), ("bone", "bone")]):
        H = d[k]
        ax[i].imshow(np.log1p(H.T), origin="lower", cmap="magma", aspect="auto",
                     extent=[ge[0], ge[-1], pe[0], pe[-1]])
        ax[i].plot([pe[0], pe[-1]], [pe[0], pe[-1]], "w--", lw=0.8)
        ax[i].set_title(t)
        ax[i].set_xlabel("GT HU")
        if i == 0:
            ax[i].set_ylabel("sCT HU")
    fig.suptitle("GT vs sCT HU calibration within each CADS group (white dashed = perfect)", y=1.03)
    save(fig, "c_calib.png")


def fig_lab0_hist(npz, audit):
    d = np.load(npz)
    ge = d["gt_edges"]
    gtmarg = d["unlabeled"].sum(1)
    cent = 0.5 * (ge[:-1] + ge[1:])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.fill_between(cent, gtmarg / gtmarg.sum(), color="#7c3aed", alpha=0.6)
    ax.axvline(-300, color="k", ls=":", lw=1)
    ax.text(-290, ax.get_ylim()[1] * 0.7, "air thr -300", fontsize=8)
    ax.set_xlabel("GT HU of unlabeled (CADS=0) in-body voxels")
    ax.set_ylabel("density")
    ax.set_title(f"What the unlabeled {audit['pct_body_unlabeled']:.0f}% of body is, by GT HU "
                 f"({audit['pct_body_lab0_air']:.0f}% of body is unlabeled air)")
    save(fig, "c_lab0_hist.png")


def fig_maskaudit(su):
    # tight-body split (proper): external loose-mask air vs internal unlabeled gas
    ms = pd.read_csv(os.path.join(RUN, "cads_mask_split.csv"))
    tot_body = ms.n_body.sum()
    tot_abs = ms.sabs_body.sum()
    pct_ext = 100 * ms.n_ext.sum() / tot_body
    pct_int = 100 * ms.n_int.sum() / tot_body
    em_ext = 100 * ms.sabs_ext.sum() / tot_abs
    em_int = 100 * ms.sabs_int.sum() / tot_abs
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].bar(["external\n(loose mask)", "internal\n(unlabeled gas)"], [pct_ext, pct_int], color=["#ef4444", "#06b6d4"])
    ax[0].set_ylabel("% of body voxels")
    ax[0].set_title("label-0 air: external vs internal (% of body)", fontsize=11)
    for i, v in enumerate([pct_ext, pct_int]):
        ax[0].text(i, v + 0.3, f"{v:.1f}%", ha="center")
    ax[1].bar(["external air", "internal gas"], [em_ext, em_int], color=["#ef4444", "#06b6d4"])
    ax[1].set_ylabel("% of total body error")
    ax[1].set_title("its error contribution (% of body error)", fontsize=11)
    for i, v in enumerate([em_ext, em_int]):
        ax[1].text(i, v + 0.3, f"{v:.1f}%", ha="center")
    fig.suptitle(f"CADS label-0 is {pct_ext:.0f}% external loose-mask air (only {pct_int:.1f}% internal gas)", y=1.04)
    fig.tight_layout()
    save(fig, "c_maskaudit.png")


def fig_qual(su):
    # GT CADS bone CONTOUR overlaid on BOTH GT CT and sCT -> localization (same place) vs density (sCT greyer)
    fig, axes = plt.subplots(len(REG), 3, figsize=(11, 2.9 * len(REG)))
    cols = ["GT CT + GT-bone outline", "sCT + GT-bone outline", "error (sCT - GT)"]
    for ri, r in enumerate(REG):
        sub = su[su.region == r]
        if not len(sub):
            continue
        med = sub.body_mae.median()
        S = sub.iloc[(sub.body_mae - med).abs().argmin()].subj
        try:
            gt = canon(f"{DATA}/{S}/ct.nii")
            sct = canon(f"{EVAL}/volumes/unet/{S}/sample.nii.gz")
            seg = canon(f"{DATA}/{S}/cads_grouped_35_labels_seg.nii.gz", np.int16)
        except Exception:
            continue
        bone = np.isin(seg, BONE)
        z = bone.sum((0, 1)).argmax()
        sl = lambda v: np.rot90(v[:, :, z])
        gts, scs, bs = sl(gt), sl(sct), sl(bone).astype(float)
        for ci, im in enumerate([gts, scs]):
            axes[ri, ci].imshow(im, cmap="gray", vmin=-200, vmax=1200)
            axes[ri, ci].contour(bs, levels=[0.5], colors="#ef4444", linewidths=0.6)
        axes[ri, 2].imshow(scs - gts, cmap="seismic", vmin=-700, vmax=700)
        for ci in range(3):
            axes[ri, ci].set_xticks([])
            axes[ri, ci].set_yticks([])
            if ri == 0:
                axes[ri, ci].set_title(cols[ci], fontsize=10)
        axes[ri, 0].set_ylabel(f"{r}\n{S}", fontsize=9)
    fig.suptitle("Same GT-CADS-bone outline on GT CT vs sCT: bone is in the same place (localized),\n"
                 "but inside the outline the sCT is greyer (undershoot) and softer-edged (blur); blue in error map = undershoot",
                 y=1.0, fontsize=11)
    save(fig, "c_qualitative.png")


def fig_zoom(su):
    # zoom into a bone region for one thorax subject: GT vs sCT with bone outline -> see undershoot + blur
    sub = su[su.region == "thorax"]
    med = sub.body_mae.median()
    S = sub.iloc[(sub.body_mae - med).abs().argmin()].subj
    gt = canon(f"{DATA}/{S}/ct.nii")
    sct = canon(f"{EVAL}/volumes/unet/{S}/sample.nii.gz")
    seg = canon(f"{DATA}/{S}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    bone = np.isin(seg, BONE)
    z = bone.sum((0, 1)).argmax()
    gts, scs, bs = np.rot90(gt[:, :, z]), np.rot90(sct[:, :, z]), np.rot90(bone[:, :, z]).astype(float)
    ys, xs = np.where(bs > 0.5)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2
    h = max(y1 - y0, x1 - x0) // 3
    sy, sx = slice(max(cy - h, 0), cy + h), slice(max(cx - h, 0), cx + h)
    fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    for a, im, t in [(ax[0], gts, "GT CT"), (ax[1], scs, "sCT")]:
        a.imshow(im[sy, sx], cmap="gray", vmin=-200, vmax=1400)
        a.contour(bs[sy, sx], levels=[0.5], colors="#22d3ee", linewidths=1.0)
        a.set_title(t, fontsize=12)
        a.set_xticks([]); a.set_yticks([])
    ax[2].imshow((scs - gts)[sy, sx], cmap="seismic", vmin=-900, vmax=900)
    ax[2].set_title("error (sCT - GT); blue = undershoot", fontsize=12)
    ax[2].set_xticks([]); ax[2].set_yticks([])
    fig.suptitle(f"Zoom on bone ({S}): GT cortical rim is bright & sharp; sCT is dimmer (undershoot) and smeared (blur)", y=1.02)
    save(fig, "c_zoom.png")


def fig_locdens():
    # localization-vs-density, all CADS-bone-based (no babyseg): AUC + edge-sharpness
    import pandas as pd
    vd = pd.read_csv(os.path.join(RUN, "verify_density.csv"))
    vb = pd.read_csv(os.path.join(RUN, "verify_blur.csv"))
    aucg, aucs = vd.auc_gt.mean(), vd.auc_sct.mean()
    blur = vb.ratio.mean()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].bar(["real CT", "sCT"], [aucg, aucs], color=["#93c5fd", "#1d4ed8"])
    ax[0].axhline(0.5, color="#9ca3af", ls=":")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_ylabel("AUC: HU separates GT-CADS-bone vs rest")
    ax[0].set_title(f"Localization (GT CADS bone): intact\n{aucs:.2f} vs {aucg:.2f} ceiling")
    for i, v in enumerate([aucg, aucs]):
        ax[0].text(i, v + 0.01, f"{v:.3f}", ha="center")
    ax[1].bar(["bone-edge sharpness\n(magnitude-matched)"], [blur], color="#dc2626")
    ax[1].axhline(1.0, color="#16a34a", ls="--", label="= real CT")
    ax[1].set_ylim(0, 1.1)
    ax[1].set_title(f"Density-edges: sCT bone edges {blur:.2f}x as sharp as real CT")
    ax[1].text(0, blur + 0.02, f"{blur:.2f}", ha="center")
    ax[1].legend()
    fig.suptitle("Localization vs density (CADS-bone, no segmenter): located correctly, undershot & blurred", y=1.03)
    save(fig, "c_locdens.png")


def main():
    import json
    lab = pd.read_csv(os.path.join(RUN, "cads_label_micro.csv"))
    gr = pd.read_csv(os.path.join(RUN, "cads_groups.csv"))
    su = pd.read_csv(os.path.join(RUN, "cads_subject.csv"))
    audit = json.load(open(os.path.join(RUN, "cads_audit.json")))
    npz = os.path.join(RUN, "cads_calib.npz")
    fig_groups(gr)
    fig_perlabel(lab)
    fig_calib(npz)
    fig_lab0_hist(npz, audit)
    fig_maskaudit(su)
    fig_qual(su)
    fig_zoom(su)
    fig_locdens()
    print("[cads_figures] done")


if __name__ == "__main__":
    main()
