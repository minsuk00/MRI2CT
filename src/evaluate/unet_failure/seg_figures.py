"""Figures for the seg-downstream U-Net failure analysis (report 09).
Reads the aggregate CSVs/json + seg volumes; writes PNGs to RUN/figures/.
F1 per-label Dice ceiling vs sCT | F2 Dice gap | F3 per-label HU bias |
F4 localization-vs-density scatter | F5 bone confusion | F6 per-region bone Dice |
F7 qualitative (MR | GT CT | sCT | GT-bone | realseg-bone | sctseg-bone).
"""
import os
import json
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = "/home/minsukc/MRI2CT"
DATA = os.path.join(REPO, "dataset/1.5mm_registered_flat_masked")
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
BONE = [7, 27, 28, 29, 30]
RCOL = {"brain": "#6366f1", "head_neck": "#0891b2", "thorax": "#16a34a",
        "abdomen": "#d97706", "pelvis": "#dc2626"}


def save(fig, name):
    fig.savefig(os.path.join(FIG, name), dpi=120, bbox_inches="tight")
    plt.close(fig)


def canon(p, dt=np.float32):
    return np.asarray(nib.as_closest_canonical(nib.load(p)).dataobj, dtype=dt)


def f1_f2_f3(lab):
    # F1: ceiling vs sCT, sorted by sCT
    d = lab.sort_values("dice_sct")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(y + 0.2, d.dice_ceil, height=0.4, color="#93c5fd", label="ceiling: babyseg(real CT)")
    ax.barh(y - 0.2, d.dice_sct, height=0.4, color="#1d4ed8", label="babyseg(sCT)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"$\\bf{{{n}}}$" if b else n for n, b in zip(d.index, d.is_bone)], fontsize=8)
    ax.set_xlabel("Dice vs GT CADS")
    ax.set_title("Per-CADS-label segmentability: real-CT ceiling vs synthetic CT\n(bold = bone label)")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    save(fig, "f1_perlabel_dice.png")

    # F2: gap sorted, bone colored
    d = lab.sort_values("gap", ascending=False)
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#dc2626" if b else "#9ca3af" for b in d.is_bone]
    ax.barh(np.arange(len(d))[::-1], d.gap, color=colors)
    ax.set_yticks(np.arange(len(d))[::-1])
    ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("Dice drop (ceiling - sCT)")
    ax.set_title("Dice drop (ceiling - sCT) per CADS label  (red = bone)")
    ax.axvline(0, color="k", lw=0.6)
    save(fig, "f2_dice_gap.png")

    # F3: per-label HU bias
    d = lab.sort_values("bias")
    fig, ax = plt.subplots(figsize=(8, 10))
    colors = ["#dc2626" if b else "#9ca3af" for b in d.is_bone]
    ax.barh(np.arange(len(d)), d.bias, color=colors)
    ax.set_yticks(np.arange(len(d)))
    ax.set_yticklabels(d.index, fontsize=8)
    ax.set_xlabel("HU bias inside GT ROI  (pred - GT;  negative = undershoot)")
    ax.set_title("Mean HU bias inside GT ROI per CADS label  (red = bone)")
    ax.axvline(0, color="k", lw=0.6)
    save(fig, "f3_hu_bias.png")


def f4_scatter(pl):
    # per region x bone-label: mean gap vs mean bias
    b = pl[pl.is_bone].copy()
    g = b.groupby(["region", "name"]).agg(
        gap=("dice_ceil", "mean"), sct=("dice_sct", "mean"), bias=("bias", "mean")).reset_index()
    g["gap"] = g["gap"] - g["sct"]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for r in REG:
        sub = g[g.region == r]
        if len(sub):
            ax.scatter(sub.bias, sub.gap, s=70, color=RCOL[r], label=r, edgecolor="k", lw=0.4, alpha=0.85)
    ax.axhline(0, color="#9ca3af", lw=0.6)
    ax.axvline(0, color="#9ca3af", lw=0.6)
    ax.set_xlabel("density error: mean HU bias in GT bone ROI (pred - GT)")
    ax.set_ylabel("localization loss: Dice drop (ceiling - sCT)")
    ax.set_title("Bone Dice drop vs HU bias, per region x bone label")
    ax.legend(fontsize=9)
    save(fig, "f4_loc_vs_density.png")


def f5_confusion(st):
    names = json.load(open(os.path.join(RUN, "seg_stats.json")))  # ensure latest
    bc = st["bone_confusion"]
    kept_r = bc["gt_bone_kept_as_bone_realCT"]
    kept_s = bc["gt_bone_kept_as_bone_sCT"]
    rel = bc["sct_relabel_top"]
    rel = [(n, f) for n, f in rel if f > 0.005][:5]
    labels = ["bone"] + [n for n, _ in rel] + ["other"]
    # realCT row
    real_rest = 1 - kept_r
    sct_rest = [f for _, f in rel]
    sct_other = max(0.0, 1 - kept_s - sum(sct_rest))
    real_vals = [kept_r] + [0] * len(rel) + [real_rest]
    sct_vals = [kept_s] + sct_rest + [sct_other]
    fig, ax = plt.subplots(figsize=(8.5, 3.2))
    palette = ["#dc2626"] + ["#3b82f6", "#60a5fa", "#93c5fd", "#bfdbfe", "#dbeafe"][:len(rel)] + ["#d1d5db"]
    left_r = left_s = 0
    for i, lab in enumerate(labels):
        ax.barh(1, real_vals[i], left=left_r, color=palette[i], edgecolor="w")
        ax.barh(0, sct_vals[i], left=left_s, color=palette[i], edgecolor="w",
                label=lab if i < len(labels) else None)
        left_r += real_vals[i]
        left_s += sct_vals[i]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["babyseg(sCT)", "babyseg(real CT)"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of true (GT) bone voxels assigned to each tissue")
    ax.set_title("What the segmenter calls true-bone voxels: real CT vs synthetic CT")
    ax.legend(ncol=len(labels), fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.35))
    save(fig, "f5_bone_confusion.png")


def f6_region(reg):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(reg))
    ax.bar(x - 0.2, reg.dice_ceil_bone, 0.4, color="#93c5fd", label="ceiling: babyseg(real CT)")
    ax.bar(x + 0.2, reg.dice_sct_bone, 0.4, color="#1d4ed8", label="babyseg(sCT)")
    for i, (c, s) in enumerate(zip(reg.dice_ceil_bone, reg.dice_sct_bone)):
        ax.text(i, max(c, s) + 0.02, f"-{(c - s):.2f}", ha="center", fontsize=9, color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels(reg.index)
    ax.set_ylabel("bone-union Dice vs GT CADS")
    ax.set_ylim(0, 1)
    ax.set_title("Bone localization per region: real-CT ceiling vs synthetic CT (red = drop)")
    ax.legend()
    save(fig, "f6_region_bone_dice.png")


def f7_qualitative(ps):
    fig, axes = plt.subplots(len(REG), 6, figsize=(15, 2.6 * len(REG)))
    cols = ["MR", "GT CT (bone win)", "sCT (bone win)", "GT CADS bone",
            "babyseg(real CT) bone", "babyseg(sCT) bone"]
    for ri, r in enumerate(REG):
        sub = ps[(ps.region == r) & ps.dice_sct_bone.notna()]
        if not len(sub):
            continue
        # representative: subject with median sCT bone Dice
        med = sub.dice_sct_bone.median()
        S = sub.iloc[(sub.dice_sct_bone - med).abs().argmin()].subj
        try:
            mr = canon(os.path.join(DATA, S, "moved_mr.nii"))
            gct = canon(os.path.join(DATA, S, "ct.nii"))
            sct = canon(os.path.join(EVAL, "volumes", "unet", S, "sample.nii.gz"))
            gseg = canon(os.path.join(DATA, S, "cads_grouped_35_labels_seg.nii.gz"), np.int16)
            rseg = canon(os.path.join(EVAL, "seg", "realct", S, "seg.nii.gz"), np.int16)
            sseg = canon(os.path.join(EVAL, "seg", "unet", S, "seg.nii.gz"), np.int16)
        except Exception:
            continue
        gbone = np.isin(gseg, BONE)
        z = gbone.sum((0, 1)).argmax()  # axial slice with most bone
        sl = lambda v: np.rot90(v[:, :, z])
        rb, sb = np.isin(rseg, BONE), np.isin(sseg, BONE)
        panels = [
            (sl(mr), "gray", None), (sl(gct), "gray", (-200, 1500)), (sl(sct), "gray", (-200, 1500)),
            (sl(gct), "gray", (-200, 1500), sl(gbone)),
            (sl(gct), "gray", (-200, 1500), sl(rb)),
            (sl(sct), "gray", (-200, 1500), sl(sb)),
        ]
        for ci, p in enumerate(panels):
            ax = axes[ri, ci]
            base, cmap = p[0], p[1]
            vlim = p[2]
            if vlim:
                ax.imshow(base, cmap=cmap, vmin=vlim[0], vmax=vlim[1])
            else:
                ax.imshow(base, cmap=cmap)
            if len(p) == 4:
                ov = np.ma.masked_where(~p[3], p[3])
                ax.imshow(ov, cmap="autumn", alpha=0.55)
            ax.set_xticks([])
            ax.set_yticks([])
            if ci == 0:
                ax.set_ylabel(f"{r}\n{S}", fontsize=9)
            if ri == 0:
                ax.set_title(cols[ci], fontsize=10)
    fig.suptitle("Bone localization, representative (median-bone-Dice) subject per region", y=1.0)
    save(fig, "f7_qualitative.png")


def f8_f9_distributions(ps):
    # F8: per-subject bone Dice distribution, ceiling vs sCT (systematic, not an average)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bins = np.linspace(0, 1, 41)
    ax.hist(ps.dice_ceil_bone.dropna(), bins=bins, color="#93c5fd", alpha=0.8,
            label=f"ceiling: real CT (median {ps.dice_ceil_bone.median():.2f})")
    ax.hist(ps.dice_sct_bone.dropna(), bins=bins, color="#1d4ed8", alpha=0.6,
            label=f"sCT (median {ps.dice_sct_bone.median():.2f})")
    ax.set_xlabel("per-subject bone-union Dice vs GT CADS")
    ax.set_ylabel("# subjects")
    ax.set_title("Per-subject bone-union Dice: real CT ceiling vs sCT")
    ax.legend()
    save(fig, "f8_dice_dist.png")

    # F9: per-subject bone HU bias distribution
    b = ps.bone_bias.dropna()
    pct_under = 100 * (b < 0).mean()
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.hist(b, bins=30, color="#dc2626", alpha=0.8)
    ax.axvline(0, color="k", lw=1)
    ax.axvline(b.mean(), color="#111827", ls="--", lw=1.2, label=f"mean {b.mean():.0f} HU")
    ax.set_xlabel("per-subject bone HU bias inside GT bone ROI (pred - GT)")
    ax.set_ylabel("# subjects")
    ax.set_title(f"Per-subject bone HU bias  ({pct_under:.0f}% of subjects undershoot)")
    ax.legend()
    save(fig, "f9_bias_dist.png")


def main():
    os.makedirs(FIG, exist_ok=True)
    lab = pd.read_csv(os.path.join(RUN, "seg_label_table.csv"), index_col="name")
    reg = pd.read_csv(os.path.join(RUN, "seg_region.csv"), index_col=0).reindex(REG)
    pl = pd.read_csv(os.path.join(RUN, "seg_per_label.csv"))
    ps = pd.read_csv(os.path.join(RUN, "seg_per_subject.csv"))
    st = json.load(open(os.path.join(RUN, "seg_stats.json")))

    f1_f2_f3(lab)
    f4_scatter(pl)
    f5_confusion(st)
    f6_region(reg)
    f7_qualitative(ps)
    f8_f9_distributions(ps)
    print("[seg_figures] wrote figures to", FIG)


if __name__ == "__main__":
    main()
