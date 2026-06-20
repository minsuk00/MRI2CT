"""Report 11 aggregation + figures.

Reads every model's per-model outputs (mm_extract / mm_mr), derives the report-10
style tables per model (cads_analyze logic: label_micro, groups, audit, mask-split),
then writes:
  - per-model derived files into OUTROOT/<model>/  (so detail figs + report can use them)
  - cross-model comparison CSVs + figures into OUTROOT/_cross/
  - per-model detail figures into OUTROOT/<model>/figures/

  python mm_cross.py
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import mm_common as C  # noqa: E402

GROUP_ORDER = ["bone (5 labels)", "air-organs (airway+lung)", "soft (other CADS)", "unlabeled (CADS=0)"]
GCOL = {"bone (5 labels)": "#dc2626", "air-organs (airway+lung)": "#2563eb",
        "soft (other CADS)": "#9ca3af", "unlabeled (CADS=0)": "#7c3aed"}


# ----------------------------------------------------------------------------- derive per model
def analyze(model):
    """Replicates cads_analyze.py + cads_mask_split.py rollups for one model."""
    RUN = C.run_dir(model)
    pl = pd.read_csv(os.path.join(RUN, "cads_per_label.csv"))
    su = pd.read_csv(os.path.join(RUN, "cads_subject.csv"))

    g = pl.groupby(["label", "name", "is_bone"]).agg(
        n=("n", "sum"), sabs=("sabs", "sum"), serr=("serr", "sum"),
        sgt=("sgt", "sum"), spred=("spred", "sum"), n_gt1024=("n_gt1024", "sum"),
        n_subj=("subj", "nunique")).reset_index()
    g["mae"] = g.sabs / g.n
    g["bias"] = g.serr / g.n
    g["gt_hu"] = g.sgt / g.n
    g["pred_hu"] = g.spred / g.n
    tot_abs, tot_n = g.sabs.sum(), g.n.sum()
    g["errmass_pct"] = 100 * g.sabs / tot_abs
    g["voxshare_pct"] = 100 * g.n / tot_n
    g = g.sort_values("errmass_pct", ascending=False)
    g.to_csv(os.path.join(RUN, "cads_label_micro.csv"), index=False)

    def grp(mask, name):
        sub = g[mask]
        return {"group": name, "voxshare_pct": float(sub.voxshare_pct.sum()),
                "mae": float(sub.sabs.sum() / sub.n.sum()),
                "errmass_pct": float(sub.errmass_pct.sum()),
                "bias": float(sub.serr.sum() / sub.n.sum())}
    groups = pd.DataFrame([
        grp(g.label.isin(C.BONE), "bone (5 labels)"),
        grp(g.label.isin(C.AIRORG), "air-organs (airway+lung)"),
        grp((~g.label.isin(C.BONE + C.AIRORG)) & (g.label != 0), "soft (other CADS)"),
        grp(g.label == 0, "unlabeled (CADS=0)"),
    ])
    groups.to_csv(os.path.join(RUN, "cads_groups.csv"), index=False)

    body_mae_micro = tot_abs / su.n_body.sum()
    body_mae_macro = su.body_mae.mean()
    gate_ok = abs(tot_n - su.n_body.sum()) < 1
    audit = {
        "n_subj": int(len(su)),
        "body_mae_micro": float(body_mae_micro), "body_mae_macro": float(body_mae_macro),
        "pct_body_unlabeled": float(100 * su.n_lab0.sum() / su.n_body.sum()),
        "errmass_lab0_pct": float(100 * su.sabs_lab0.sum() / tot_abs),
        "gate_micro_reconstructs_body_mae": bool(gate_ok),
    }
    json.dump(audit, open(os.path.join(RUN, "cads_audit.json"), "w"), indent=2)

    # tight-body external/internal (cads_mask_split)
    tb_body = su.n_body.sum()
    ms = {
        "n_subj": len(su),
        "pct_body_lab0": float(100 * su.n_lab0.sum() / tb_body),
        "pct_body_lab0_external": float(100 * su.n_ext.sum() / tb_body),
        "pct_body_lab0_internal": float(100 * su.n_int.sum() / tb_body),
        "external_share_of_lab0": float(100 * su.n_ext.sum() / su.n_lab0.sum()),
        "errmass_external_pct": float(100 * su.sabs_ext.sum() / su.sabs_body.sum()),
        "errmass_internal_pct": float(100 * su.sabs_int.sum() / su.sabs_body.sum()),
        "mae_external": float(su.sabs_ext.sum() / max(su.n_ext.sum(), 1)),
        "mae_internal": float(su.sabs_int.sum() / max(su.n_int.sum(), 1)),
    }
    json.dump(ms, open(os.path.join(RUN, "cads_mask_split.json"), "w"), indent=2)
    return g, groups, audit, ms


def calib_highbone(model):
    """mean predicted HU for GT-bone voxels whose GT HU is in each band (cap evidence)."""
    d = np.load(os.path.join(C.run_dir(model), "cads_calib.npz"))
    H = d["bone"]                       # [GT bins, pred bins]
    ge, pe = d["gt_edges"], d["pred_edges"]
    gc = 0.5 * (ge[:-1] + ge[1:])
    pc = 0.5 * (pe[:-1] + pe[1:])
    out = {}
    for lo, hi, key in [(1024, 4000, "gt_over1024"), (600, 1024, "gt_600_1024"), (300, 600, "gt_300_600")]:
        sel = (gc >= lo) & (gc < hi)
        sub = H[sel].sum(0)             # pred marginal for these GT bins
        out[key] = float((pc * sub).sum() / sub.sum()) if sub.sum() > 0 else float("nan")
    # global max predicted HU (highest pred bin with any bone mass)
    predmarg = H.sum(0)
    nz = np.where(predmarg > 0)[0]
    out["pred_bone_max"] = float(pc[nz[-1]]) if len(nz) else float("nan")
    return out


# ----------------------------------------------------------------------------- cross-model figures
def save(fig, name):
    fig.savefig(os.path.join(C.cross_dir(), "figures", name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def grouped_bar(ax, data, models, title, ylab, hline=None):
    """data: dict model -> list of per-category values; x = categories."""
    cats = len(next(iter(data.values())))
    x = np.arange(cats)
    w = 0.8 / len(models)
    for i, m in enumerate(models):
        ax.bar(x + i * w, data[m], w, label=C.MODEL_LABEL[m], color=C.MODEL_COLOR[m])
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_title(title)
    ax.set_ylabel(ylab)
    if hline is not None:
        ax.axhline(hline, color="k", lw=0.7, ls=":")
    return x


def cross_figs(D):
    models = C.MODELS
    G = {m: D[m]["groups"].set_index("group").reindex(GROUP_ORDER) for m in models}
    catlab = [g.split(" (")[0] for g in GROUP_ORDER]

    # group error-mass share
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = grouped_bar(ax, {m: G[m].errmass_pct.values for m in models}, models,
                    "Share of total body error by CADS group", "% of body error")
    ax.set_xticklabels(catlab)
    ax.legend(ncol=3, fontsize=9)
    save(fig, "cf_group_errmass.png")

    # group per-voxel MAE
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = grouped_bar(ax, {m: G[m].mae.values for m in models}, models,
                    "Per-voxel MAE by CADS group", "MAE (HU)")
    ax.set_xticklabels(catlab)
    ax.legend(ncol=3, fontsize=9)
    save(fig, "cf_group_mae.png")

    # group bias (regression to the mean)
    fig, ax = plt.subplots(figsize=(11, 4.6))
    x = grouped_bar(ax, {m: G[m].bias.values for m in models}, models,
                    "HU bias by CADS group (<0 undershoot, >0 overshoot)", "bias (HU)", hline=0)
    ax.set_xticklabels(catlab)
    ax.legend(ncol=3, fontsize=9)
    save(fig, "cf_group_bias.png")

    # whole-body micro/macro MAE
    fig, ax = plt.subplots(figsize=(9, 4.4))
    micro = [D[m]["audit"]["body_mae_micro"] for m in models]
    macro = [D[m]["audit"]["body_mae_macro"] for m in models]
    xi = np.arange(len(models))
    ax.bar(xi - 0.2, micro, 0.4, label="micro (pooled)", color="#9ca3af")
    ax.bar(xi + 0.2, macro, 0.4, label="macro (= synthrad_mae)", color="#1f2937")
    ax.set_xticks(xi)
    ax.set_xticklabels([C.MODEL_LABEL[m] for m in models])
    ax.set_ylabel("body MAE (HU)")
    ax.set_title("Whole-body MAE per model (full-range HU, in body mask)")
    for i, v in enumerate(macro):
        ax.text(i + 0.2, v + 0.5, f"{v:.1f}", ha="center", fontsize=8)
    ax.legend()
    save(fig, "cf_wholebody_mae.png")

    # bone HU-band bias lines (the cap story)
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        bt = D[m]["bonehu"]
        ax.plot(range(C.NHU), bt.bias.values, "-o", color=C.MODEL_COLOR[m],
                label=f"{C.MODEL_LABEL[m]} ({C.MODEL_CAP[m]})")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xticks(range(C.NHU))
    ax.set_xticklabels(C.HU_BANDS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("HU bias (sCT - GT)")
    ax.set_title("Within-bone HU bias by ground-truth density band: where each model's ceiling bites")
    ax.legend(fontsize=8)
    save(fig, "cf_bone_hu_bias.png")

    # bone HU-band MAE lines
    fig, ax = plt.subplots(figsize=(10, 5))
    for m in models:
        bt = D[m]["bonehu"]
        ax.plot(range(C.NHU), bt.micro_mae.values, "-o", color=C.MODEL_COLOR[m], label=C.MODEL_LABEL[m])
    ax.set_xticks(range(C.NHU))
    ax.set_xticklabels(C.HU_BANDS, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("micro MAE (HU)")
    ax.set_title("Within-bone per-voxel MAE by ground-truth density band")
    ax.legend(fontsize=8)
    save(fig, "cf_bone_hu_mae.png")

    # mean predicted HU where GT bone > 1024 (cap evidence)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    vals = [D[m]["highbone"]["gt_over1024"] for m in models]
    ax.bar(range(len(models)), vals, color=[C.MODEL_COLOR[m] for m in models])
    ax.axhline(1024, color="#dc2626", ls="--", lw=1.2, label="1024 HU ceiling")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([f"{C.MODEL_LABEL[m]}\n{C.MODEL_CAP[m]}" for m in models], fontsize=8)
    ax.set_ylabel("mean sCT HU")
    ax.set_title("Mean predicted HU where ground-truth bone exceeds 1024 HU")
    for i, v in enumerate(vals):
        ax.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=9)
    ax.legend()
    save(fig, "cf_cap_over1024.png")

    # external loose-mask air error share
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ext = [D[m]["ms"]["errmass_external_pct"] for m in models]
    maex = [D[m]["ms"]["mae_external"] for m in models]
    ax[0].bar(range(len(models)), ext, color=[C.MODEL_COLOR[m] for m in models])
    ax[0].set_xticks(range(len(models))); ax[0].set_xticklabels([C.MODEL_LABEL[m] for m in models], rotation=15)
    ax[0].set_ylabel("% of total body error"); ax[0].set_title("External loose-mask air: share of body error")
    for i, v in enumerate(ext):
        ax[0].text(i, v + 0.3, f"{v:.0f}%", ha="center", fontsize=9)
    ax[1].bar(range(len(models)), maex, color=[C.MODEL_COLOR[m] for m in models])
    ax[1].set_xticks(range(len(models))); ax[1].set_xticklabels([C.MODEL_LABEL[m] for m in models], rotation=15)
    ax[1].set_ylabel("MAE (HU)"); ax[1].set_title("External loose-mask air: per-voxel MAE")
    fig.suptitle("External out-of-patient air (loose body mask), tight-body split", y=1.02)
    save(fig, "cf_external.png")

    # localization AUC + blur
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    aucg = [D[m]["vd"].auc_gt.mean() for m in models]
    aucs = [D[m]["vd"].auc_sct.mean() for m in models]
    xi = np.arange(len(models))
    ax[0].bar(xi - 0.2, aucg, 0.4, label="real CT (ceiling)", color="#93c5fd")
    ax[0].bar(xi + 0.2, aucs, 0.4, label="sCT", color="#1d4ed8")
    ax[0].axhline(0.5, color="#9ca3af", ls=":")
    ax[0].set_ylim(0.4, 1.0)
    ax[0].set_xticks(xi); ax[0].set_xticklabels([C.MODEL_LABEL[m] for m in models], rotation=15)
    ax[0].set_ylabel("AUC: HU separates GT-bone vs rest"); ax[0].set_title("Bone localization (intact if near ceiling)")
    ax[0].legend(fontsize=8)
    blur = [D[m]["vb"].ratio.mean() for m in models]
    ax[1].bar(xi, blur, color=[C.MODEL_COLOR[m] for m in models])
    ax[1].axhline(1.0, color="#16a34a", ls="--", label="= real CT")
    ax[1].set_ylim(0, 1.2)
    ax[1].set_xticks(xi); ax[1].set_xticklabels([C.MODEL_LABEL[m] for m in models], rotation=15)
    ax[1].set_ylabel("magnitude-matched sCT / real CT"); ax[1].set_title("Bone-edge sharpness (1.0 = as sharp as real CT)")
    for i, v in enumerate(blur):
        ax[1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
    ax[1].legend(fontsize=8)
    save(fig, "cf_localization.png")

    # bone calibration 2D-hist grid (visual ceiling)
    fig, axes = plt.subplots(2, 3, figsize=(15, 8.6))
    for ax_, m in zip(axes.ravel(), models):
        d = np.load(os.path.join(C.run_dir(m), "cads_calib.npz"))
        ge, pe = d["gt_edges"], d["pred_edges"]
        ax_.imshow(np.log1p(d["bone"].T), origin="lower", cmap="magma", aspect="auto",
                   extent=[ge[0], ge[-1], pe[0], pe[-1]])
        ax_.plot([pe[0], pe[-1]], [pe[0], pe[-1]], "w--", lw=0.7)
        ax_.axhline(1024, color="#22d3ee", ls=":", lw=0.9)
        ax_.set_xlim(-200, 2200)
        ax_.set_title(f"{C.MODEL_LABEL[m]}  ({C.MODEL_CAP[m]})", fontsize=11)
        ax_.set_xlabel("GT HU"); ax_.set_ylabel("sCT HU")
    fig.suptitle("GT vs sCT HU within GT-bone, per model (white = perfect, cyan = 1024). "
                 "Capped models flatten below the diagonal at high GT HU.", y=1.0)
    save(fig, "cf_bone_calib_grid.png")


def cross_qualitative(D):
    """Same subject/slice: GT + each model sCT with GT-bone outline, + error row."""
    su = pd.read_csv(os.path.join(C.run_dir("unet"), "cads_subject.csv"))
    sub = su[su.region == "thorax"]
    med = sub.body_mae.median()
    S = sub.iloc[(sub.body_mae - med).abs().argmin()].subj
    gt = C.canon(f"{C.DATA}/{S}/ct.nii")
    seg = C.canon(f"{C.DATA}/{S}/cads_grouped_35_labels_seg.nii.gz", np.int16)
    bone = np.isin(seg, C.BONE)
    z = bone.sum((0, 1)).argmax()
    sl = lambda v: np.rot90(v[:, :, z])
    gts, bs = sl(gt), sl(bone).astype(float)

    scts = {m: sl(C.canon(C.sct_path(m, S))) for m in C.MODELS}
    ncol = 1 + len(C.MODELS)
    fig, axes = plt.subplots(2, ncol, figsize=(2.5 * ncol, 5.6))
    # row 0: images with bone outline
    axes[0, 0].imshow(gts, cmap="gray", vmin=-200, vmax=1400)
    axes[0, 0].contour(bs, levels=[0.5], colors="#ef4444", linewidths=0.5)
    axes[0, 0].set_title("GT CT", fontsize=10)
    axes[1, 0].axis("off")
    for j, m in enumerate(C.MODELS, start=1):
        axes[0, j].imshow(scts[m], cmap="gray", vmin=-200, vmax=1400)
        axes[0, j].contour(bs, levels=[0.5], colors="#ef4444", linewidths=0.5)
        axes[0, j].set_title(f"{C.MODEL_LABEL[m]}\n{C.MODEL_CAP[m]}", fontsize=9)
        axes[1, j].imshow(scts[m] - gts, cmap="seismic", vmin=-700, vmax=700)
        axes[1, j].set_title("error", fontsize=9)
    for a in axes.ravel():
        a.set_xticks([]); a.set_yticks([])
    axes[1, 0].text(0.5, 0.5, "error\n(sCT - GT)\nblue = undershoot", ha="center", va="center", fontsize=9)
    fig.suptitle(f"Same slice ({S}, thorax): GT-bone outline on each sCT. "
                 "Inside the outline the capped models read greyer (undershoot).", y=1.0, fontsize=11)
    save(fig, "cf_qualitative.png")


# ----------------------------------------------------------------------------- per-model detail figs
def detail_figs(model, g, groups, audit, ms):
    FIG = C.fig_dir(model)

    def msave(fig, n):
        fig.savefig(os.path.join(FIG, n), dpi=120, bbox_inches="tight")
        plt.close(fig)

    # groups
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    gg = groups.set_index("group").reindex(GROUP_ORDER)
    cols = [GCOL[x] for x in gg.index]
    ax[0].bar(range(len(gg)), gg.voxshare_pct, color=cols); ax[0].set_title("voxel share of body (%)")
    ax[1].bar(range(len(gg)), gg.mae, color=cols); ax[1].set_title("per-voxel MAE (HU)")
    ax[2].bar(range(len(gg)), gg.errmass_pct, color=cols); ax[2].set_title("share of total body error (%)")
    for a in ax:
        a.set_xticks(range(len(gg)))
        a.set_xticklabels([x.split(" (")[0] for x in gg.index], rotation=20, ha="right", fontsize=9)
    fig.suptitle(f"{C.MODEL_LABEL[model]}: CADS group decomposition (severity vs leverage)", y=1.02)
    msave(fig, "c_groups.png")

    # per-label
    def lcol(df):
        out = []
        for l, b in zip(df.label, df.is_bone):
            out.append("#dc2626" if b else "#2563eb" if l in C.AIRORG else "#7c3aed" if l == 0 else "#9ca3af")
        return out
    fig, ax = plt.subplots(1, 3, figsize=(16, 9))
    for a, (col, title, asc) in zip(ax, [("errmass_pct", "error contribution (%)", False),
                                         ("mae", "MAE (HU)", False), ("bias", "HU bias", None)]):
        d = g.sort_values(col, ascending=True if col == "bias" else not asc)
        a.barh(range(len(d)), d[col], color=lcol(d))
        a.set_yticks(range(len(d))); a.set_yticklabels(d.name, fontsize=7)
        a.set_xlabel(title)
        if col == "bias":
            a.axvline(0, color="k", lw=0.6)
    fig.suptitle(f"{C.MODEL_LABEL[model]}: per CADS label (red=bone, blue=air-organs, purple=unlabeled, grey=soft)", y=1.01)
    msave(fig, "c_perlabel.png")

    # calib
    d = np.load(os.path.join(C.run_dir(model), "cads_calib.npz"))
    ge, pe = d["gt_edges"], d["pred_edges"]
    fig, ax = plt.subplots(1, 4, figsize=(18, 4.6))
    for i, (k, t) in enumerate([("unlabeled", "unlabeled"), ("soft", "soft"), ("airorg", "air-organs"), ("bone", "bone")]):
        ax[i].imshow(np.log1p(d[k].T), origin="lower", cmap="magma", aspect="auto", extent=[ge[0], ge[-1], pe[0], pe[-1]])
        ax[i].plot([pe[0], pe[-1]], [pe[0], pe[-1]], "w--", lw=0.8)
        ax[i].set_title(t); ax[i].set_xlabel("GT HU")
        if i == 0:
            ax[i].set_ylabel("sCT HU")
    fig.suptitle(f"{C.MODEL_LABEL[model]}: GT vs sCT HU calibration per CADS group (white dashed = perfect)", y=1.03)
    msave(fig, "c_calib.png")

    # bone HU split
    bt = D[model]["bonehu"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    x = range(C.NHU)
    ax[0].bar(x, bt.micro_mae, color=C.MODEL_COLOR[model]); ax[0].set_title("per-voxel MAE by GT-HU band")
    ax[1].bar(x, bt.bias, color=C.MODEL_COLOR[model]); ax[1].axhline(0, color="k", lw=0.7)
    ax[1].set_title("HU bias by GT-HU band (<0 undershoot)")
    ax[2].bar(x, bt.pct_body_error, color=C.MODEL_COLOR[model]); ax[2].set_title("share of total body error (%)")
    for a in ax:
        a.set_xticks(x); a.set_xticklabels(C.HU_BANDS, rotation=30, ha="right", fontsize=8)
    fig.suptitle(f"{C.MODEL_LABEL[model]}: within-bone error by GT density (all 5 bone labels)", y=1.03)
    msave(fig, "c_bone_hu.png")


# ----------------------------------------------------------------------------- main
D = {}


def main():
    global D
    for m in C.MODELS:
        C.ensure(m)
        g, groups, audit, ms = analyze(m)
        D[m] = {
            "g": g, "groups": groups, "audit": audit, "ms": ms,
            "bonehu": pd.read_csv(os.path.join(C.run_dir(m), "cads_bone_hu_split.csv")),
            "bonemeta": json.load(open(os.path.join(C.run_dir(m), "cads_bone_hu_meta.json"))),
            "vd": pd.read_csv(os.path.join(C.run_dir(m), "verify_density.csv")),
            "vb": pd.read_csv(os.path.join(C.run_dir(m), "verify_blur.csv")),
            "loose": json.load(open(os.path.join(C.run_dir(m), "loose_stats.json"))),
            "highbone": calib_highbone(m),
        }
        print(f"[mm_cross] analyzed {m}: macro MAE {audit['body_mae_macro']:.1f}")

    # cross-model summary csv
    rows = []
    for m in C.MODELS:
        gg = D[m]["groups"].set_index("group")
        bt = D[m]["bonehu"].set_index("GT-HU band")
        rows.append({
            "model": m, "cap": C.MODEL_CAP[m],
            "body_mae_micro": D[m]["audit"]["body_mae_micro"],
            "body_mae_macro": D[m]["audit"]["body_mae_macro"],
            "bone_errmass_pct": gg.loc["bone (5 labels)", "errmass_pct"],
            "bone_mae": gg.loc["bone (5 labels)", "mae"],
            "bone_bias": gg.loc["bone (5 labels)", "bias"],
            "bone_voxshare_pct": gg.loc["bone (5 labels)", "voxshare_pct"],
            "airorg_bias": gg.loc["air-organs (airway+lung)", "bias"],
            "soft_bias": gg.loc["soft (other CADS)", "bias"],
            "over1024_bias": bt.loc[">1024 (above 1024)", "bias"],
            "over1024_mae": bt.loc[">1024 (above 1024)", "micro_mae"],
            "pred_HU_at_gt_over1024": D[m]["highbone"]["gt_over1024"],
            "pred_bone_max": D[m]["highbone"]["pred_bone_max"],
            "external_errmass_pct": D[m]["ms"]["errmass_external_pct"],
            "mae_external": D[m]["ms"]["mae_external"],
            "auc_gt": D[m]["vd"].auc_gt.mean(),
            "auc_sct": D[m]["vd"].auc_sct.mean(),
            "blur_ratio": D[m]["vb"].ratio.mean(),
            "loose_r_mr_sct": D[m]["loose"]["pooled_r_mr_sct"],
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(C.cross_dir(), "summary.csv"), index=False)
    summary.to_json(os.path.join(C.cross_dir(), "summary.json"), orient="records", indent=2)
    print(summary.round(1).to_string(index=False))

    cross_figs(D)
    cross_qualitative(D)
    for m in C.MODELS:
        detail_figs(m, D[m]["g"], D[m]["groups"], D[m]["audit"], D[m]["ms"])
    print("[mm_cross] done")


if __name__ == "__main__":
    main()
