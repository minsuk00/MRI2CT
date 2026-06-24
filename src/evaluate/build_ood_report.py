"""Self-contained HTML report for the OOD distribution analysis.

Reads the sCT NIfTIs + stats.json written by ood_distribution_gen.py (and, if present,
koalAI sCTs), builds qualitative panels / HU-histogram overlays / bar charts, and emits
one portable HTML with base64-embedded figures.

Usage:
    python src/evaluate/build_ood_report.py --out _html/13_ood_distribution_analysis.html
"""
import argparse
import base64
import io
import json
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
OUT_ROOT = os.path.join(GPFS, "external_inference", "ood_distribution")

# display order + human labels (models discovered from files are filtered to this order)
MODEL_ORDER = [
    ("unet_centerwise_new", "U-Net center-wise (nbn71048)"),
    ("unet_centerwise_old", "U-Net center-wise old (9xmodnhn)"),
    ("unet_fulldata", "U-Net full-data (krdhs2k0)"),
    ("mcddpm", "MC-IDDPM diffusion (a3g28rez)"),
    ("koalai", "koalAI / nnsyn (SynthRAD'25 winner)"),
]
DATASET_LABEL = {
    "cfb_gbm": "CFB-GBM (brain, glioblastoma)",
    "gold_atlas": "Gold Atlas (male pelvis)",
    "learn2reg": "Learn2Reg (abdomen)",
}
DATASET_ORDER = ["cfb_gbm", "gold_atlas", "learn2reg"]
# CT display window per region (lo, hi) HU
CT_WIN = {"cfb_gbm": (-1000, 1500), "gold_atlas": (-1024, 1024), "learn2reg": (-1024, 1024)}


def b64fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def load_vol(path):
    return np.asarray(nib.load(path).dataobj).astype(np.float32) if os.path.exists(path) else None


def discover(dataset, subject):
    """Return dict model_key -> {seq -> path} for sCTs present for this subject."""
    d = os.path.join(OUT_ROOT, dataset, subject)
    found = defaultdict(dict)
    if not os.path.isdir(d):
        return found
    for f in os.listdir(d):
        if f.startswith("sct_") and f.endswith(".nii.gz"):
            rest = f[len("sct_"):-len(".nii.gz")]
            for mkey, _ in MODEL_ORDER:
                if rest.startswith(mkey + "_"):
                    seq = rest[len(mkey) + 1:]
                    found[mkey][seq] = os.path.join(d, f)
                    break
    return found


def rot(a):
    return np.rot90(a)


# ─── qualitative panels ───────────────────────────────────────────────────────
def panel_figure(dataset, subjects, models_present):
    """One figure per dataset: rows = (subject, seq), cols = MR | each model | gtCT."""
    lo, hi = CT_WIN[dataset]
    rows = []
    for subj in subjects:
        d = os.path.join(OUT_ROOT, dataset, subj)
        found = discover(dataset, subj)
        if not found:
            continue
        # choose one representative sequence (prefer t1gd / T2 / MR)
        seqs = set()
        for m in found:
            seqs.update(found[m].keys())
        for pref in ("t1gd", "T2", "MR", "T1"):
            if pref in seqs:
                seq = pref
                break
        else:
            seq = sorted(seqs)[0]
        mr = load_vol(os.path.join(d, f"mr_{seq}.nii.gz"))
        gt = load_vol(os.path.join(d, "gtCT.nii.gz"))
        body = load_vol(os.path.join(d, "body.nii.gz"))
        if mr is None:
            continue
        # representative axial slice: densest body slice
        zsel = mr.shape[2] // 2
        if body is not None:
            zsel = int(np.argmax(body.sum(axis=(0, 1))))
        rows.append((subj, seq, mr, gt, found, zsel))

    if not rows:
        return None
    cols = ["Input MR"] + [lbl for k, lbl in MODEL_ORDER if k in models_present] + ["Ground-truth CT"]
    ncol = len(cols)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.5 * ncol, 2.8 * len(rows)))
    if len(rows) == 1:
        axes = axes.reshape(1, -1)
    for r, (subj, seq, mr, gt, found, z) in enumerate(rows):
        # MR display: percentile stretch within body
        mrsl = rot(mr[:, :, z])
        v = mr[mr > mr.min()]
        mlo, mhi = (np.percentile(v, 1), np.percentile(v, 99)) if v.size else (0, 1)
        axes[r, 0].imshow(mrsl, cmap="gray", vmin=mlo, vmax=mhi)
        axes[r, 0].set_ylabel(f"{subj}\n{seq}", fontsize=9)
        ci = 1
        for mkey, _ in MODEL_ORDER:
            if mkey not in models_present:
                continue
            ax = axes[r, ci]
            sp = found.get(mkey, {}).get(seq) or (next(iter(found[mkey].values())) if found.get(mkey) else None)
            sct = load_vol(sp) if sp else None
            if sct is not None:
                ax.imshow(rot(sct[:, :, z]), cmap="gray", vmin=lo, vmax=hi)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center", transform=ax.transAxes)
            ci += 1
        if gt is not None:
            axes[r, ci].imshow(rot(gt[:, :, z]), cmap="gray", vmin=lo, vmax=hi)
        else:
            axes[r, ci].text(0.5, 0.5, "n/a", ha="center", va="center", transform=axes[r, ci].transAxes)
    for c, title in enumerate(cols):
        axes[0, c].set_title(title, fontsize=9)
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"{DATASET_LABEL[dataset]} — sCT comparison (axial, CT window [{lo},{hi}] HU)",
                 fontsize=12, y=1.005)
    fig.tight_layout()
    return b64fig(fig)


# ─── HU histograms (pooled over subjects, body voxels) ───────────────────────
def hist_figure(dataset, subjects, models_present):
    lo, hi = -1000, 1600
    bins = np.linspace(lo, hi, 120)
    pooled = defaultdict(list)  # key -> list of HU arrays
    for subj in subjects:
        d = os.path.join(OUT_ROOT, dataset, subj)
        body = load_vol(os.path.join(d, "body.nii.gz"))
        if body is None:
            continue
        bm = body > 0
        gt = load_vol(os.path.join(d, "gtCT.nii.gz"))
        if gt is not None and dataset != "learn2reg":  # gt body for l2r differs; still show
            pooled["gt"].append(gt[bm])
        elif gt is not None:
            pooled["gt"].append(gt[gt > -1000])
        found = discover(dataset, subj)
        for mkey in models_present:
            for sp in found.get(mkey, {}).values():
                sct = load_vol(sp)
                if sct is not None:
                    pooled[mkey].append(sct[bm])
                break
    if not pooled:
        return None
    fig, ax = plt.subplots(figsize=(8, 4.2))
    order = ["gt"] + [k for k, _ in MODEL_ORDER if k in models_present]
    labels = {"gt": "Ground-truth CT", **{k: lbl for k, lbl in MODEL_ORDER}}
    for key in order:
        if key not in pooled:
            continue
        arr = np.concatenate(pooled[key])
        h, _ = np.histogram(arr, bins=bins, density=True)
        c = bins[:-1] + np.diff(bins) / 2
        ax.plot(c, h, label=labels[key], lw=2.2 if key == "gt" else 1.4,
                color="k" if key == "gt" else None,
                ls="-" if key == "gt" else "-")
    ax.axvline(300, color="gray", ls=":", lw=1)
    ax.text(310, ax.get_ylim()[1] * 0.9, "bone (>300 HU)", fontsize=8, color="gray")
    ax.set_xlabel("HU"); ax.set_ylabel("density (body voxels)")
    ax.set_title(f"{DATASET_LABEL[dataset]} — body-voxel HU distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return b64fig(fig)


# ─── bar charts ───────────────────────────────────────────────────────────────
def bar_figure(stats, models_present):
    """Two-panel: bone fraction and MAE, grouped by dataset, bars per model (+ gt for bone)."""
    # aggregate mean over subjects/seqs
    agg = defaultdict(lambda: defaultdict(list))   # dataset -> model -> [bone_frac]
    mae = defaultdict(lambda: defaultdict(list))
    gtbone = defaultdict(list)
    for r in stats:
        agg[r["dataset"]][r["model"]].append(r["bone_frac"])
        if r.get("mae_hu") is not None:
            mae[r["dataset"]][r["model"]].append(r["mae_hu"])
        if "gt_bone_frac" in r:
            gtbone[r["dataset"]].append(r["gt_bone_frac"])

    mkeys = [k for k, _ in MODEL_ORDER if k in models_present]
    labels = {k: lbl for k, lbl in MODEL_ORDER}
    ds = [d for d in DATASET_ORDER if d in agg]
    x = np.arange(len(ds))
    n = len(mkeys) + 1  # +1 for gt
    w = 0.8 / n

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.3))
    # bone fraction
    for i, mk in enumerate(mkeys):
        vals = [np.mean(agg[d][mk]) if agg[d][mk] else 0 for d in ds]
        a1.bar(x + (i - n / 2) * w + w / 2, vals, w, label=labels[mk])
    gtv = [np.mean(gtbone[d]) if gtbone[d] else 0 for d in ds]
    a1.bar(x + (len(mkeys) - n / 2) * w + w / 2, gtv, w, label="Ground-truth CT", color="k")
    a1.set_xticks(x); a1.set_xticklabels([DATASET_LABEL[d].split(" (")[0] for d in ds], fontsize=9)
    a1.set_ylabel("bone fraction (frac body HU > 300)")
    a1.set_title("Bone fraction vs ground truth")
    a1.legend(fontsize=7)

    # MAE (only aligned datasets)
    ds_mae = [d for d in ds if any(mae[d].values())]
    xm = np.arange(len(ds_mae))
    nm = len(mkeys)
    wm = 0.8 / max(nm, 1)
    for i, mk in enumerate(mkeys):
        vals = [np.mean(mae[d][mk]) if mae[d].get(mk) else 0 for d in ds_mae]
        a2.bar(xm + (i - nm / 2) * wm + wm / 2, vals, wm, label=labels[mk])
    a2.set_xticks(xm); a2.set_xticklabels([DATASET_LABEL[d].split(" (")[0] for d in ds_mae], fontsize=9)
    a2.set_ylabel("MAE vs gtCT (HU)")
    a2.set_title("Body MAE (aligned datasets only)")
    a2.legend(fontsize=7)
    fig.tight_layout()
    return b64fig(fig)


# ─── summary table ────────────────────────────────────────────────────────────
def summary_table(stats, models_present):
    agg = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    gt = defaultdict(lambda: defaultdict(list))
    for r in stats:
        a = agg[r["dataset"]][r["model"]]
        a["body_mean"].append(r["body_mean_hu"])
        a["bone_frac"].append(r["bone_frac"])
        if r.get("mae_hu") is not None:
            a["mae"].append(r["mae_hu"])
        gt[r["dataset"]]["body_mean"].append(r.get("gt_body_mean_hu", np.nan))
        gt[r["dataset"]]["bone_frac"].append(r.get("gt_bone_frac", np.nan))
    labels = {k: lbl for k, lbl in MODEL_ORDER}
    rows_html = []
    for d in DATASET_ORDER:
        if d not in agg:
            continue
        gbm = np.nanmean(gt[d]["body_mean"]); gbf = np.nanmean(gt[d]["bone_frac"])
        rows_html.append(
            f'<tr class="gtrow"><td><b>{DATASET_LABEL[d]}</b></td><td>Ground-truth CT</td>'
            f'<td>{gbm:.0f}</td><td>{gbf:.3f}</td><td>—</td></tr>')
        for mk in [k for k, _ in MODEL_ORDER if k in models_present]:
            if mk not in agg[d]:
                continue
            a = agg[d][mk]
            mae = f'{np.mean(a["mae"]):.0f}' if a["mae"] else "n/a*"
            rows_html.append(
                f'<tr><td></td><td>{labels[mk]}</td>'
                f'<td>{np.mean(a["body_mean"]):.0f}</td>'
                f'<td>{np.mean(a["bone_frac"]):.3f}</td><td>{mae}</td></tr>')
    return ("<table><tr><th>Dataset</th><th>Model</th><th>body-mean HU</th>"
            "<th>bone frac (&gt;300 HU)</th><th>MAE (HU)</th></tr>"
            + "".join(rows_html) + "</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="_html/13_ood_distribution_analysis.html")
    args = ap.parse_args()

    stats = json.load(open(os.path.join(OUT_ROOT, "stats.json")))
    models_present = sorted({r["model"] for r in stats}, key=lambda m: [k for k, _ in MODEL_ORDER].index(m)
                            if m in [k for k, _ in MODEL_ORDER] else 99)

    # build figures
    panels = {d: panel_figure(d, sorted({r["subject"] for r in stats if r["dataset"] == d}), models_present)
              for d in DATASET_ORDER}
    hists = {d: hist_figure(d, sorted({r["subject"] for r in stats if r["dataset"] == d}), models_present)
             for d in DATASET_ORDER}
    bars = bar_figure(stats, models_present)
    table = summary_table(stats, models_present)

    nmodels = len(models_present)
    nsct = sum(1 for r in stats)
    cov = defaultdict(lambda: defaultdict(int))
    for r in stats:
        cov[r["model"]][r["dataset"]] += 1
    model_list = "".join(
        f"<li><b>{lbl}</b> — split: {MODELS_SPLIT.get(k,'?')} · volumes: "
        + ", ".join(f"{DATASET_LABEL[d].split(' (')[0]} {cov[k].get(d,0)}" for d in DATASET_ORDER)
        + "</li>"
        for k, lbl in MODEL_ORDER if k in models_present)

    def img(b):
        return f'<img src="data:image/png;base64,{b}"/>' if b else "<p><i>(no data)</i></p>"

    sections = ""
    for d in DATASET_ORDER:
        if d not in panels or panels[d] is None:
            continue
        sections += f"""
        <h3>{DATASET_LABEL[d]}</h3>
        <div class="note">{DATASET_NOTE[d]}</div>
        {img(panels[d])}
        {img(hists[d])}
        """

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>OOD distribution analysis — MR→CT models</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
        max-width: 1180px; margin: 24px auto; padding: 0 18px; color: #1a1a1a; line-height: 1.5; }}
 h1 {{ font-size: 25px; border-bottom: 2px solid #333; padding-bottom: 6px; }}
 h2 {{ font-size: 20px; margin-top: 34px; border-bottom: 1px solid #ccc; padding-bottom: 4px; }}
 h3 {{ font-size: 16px; margin-top: 24px; }}
 img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 4px; margin: 8px 0; }}
 table {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
 th, td {{ border: 1px solid #ccc; padding: 4px 9px; text-align: right; }}
 th, td:first-child, td:nth-child(2) {{ text-align: left; }}
 th {{ background: #f0f0f0; }}
 tr.gtrow td {{ background: #f7f7f2; border-top: 2px solid #999; }}
 .note {{ background: #f5f8fb; border-left: 3px solid #4a78b5; padding: 7px 12px; font-size: 13px; margin: 8px 0; }}
 .key {{ background: #fff8e6; border-left: 3px solid #d9a400; padding: 10px 14px; margin: 14px 0; }}
 .sub {{ color: #666; font-size: 13px; }}
 code {{ background: #f0f0f0; padding: 1px 4px; border-radius: 3px; font-size: 12px; }}
</style></head><body>

<h1>Out-of-distribution generalization of MR&rarr;CT synthesis models</h1>
<p class="sub">Generated {nsct} sCT volumes across {nmodels} models &times; 3 external public datasets.
Outputs under <code>external_inference/ood_distribution/</code>.</p>

<div class="key">
<b>Setup.</b> Every model was trained on <b>SynthRAD2025</b> (5 anatomies: abdomen, brain, head&amp;neck,
pelvis, thorax). We test on three <b>entirely different public datasets</b> of the same anatomical regions.
Because brain, pelvis and abdomen are <b>all present in the SynthRAD training set</b>, the distribution shift
here is <b>dataset shift</b> (different scanners, MR sequences, cohorts, FOV), <b>not anatomical-region novelty</b>.
</div>

<h2>Models</h2>
<ul>{model_list}</ul>
<p class="sub">center-wise split = trained on center A of each region, held-out centers for val/test;
full-data = all centers; koalAI fold 0 = its center-wise (OOD) design. MR normalization: per-volume
min&ndash;max within the body mask (matching the flat-masked training regime). All sCTs masked to the
CT-derived body.</p>

<h2>Summary statistics</h2>
{table}
<p class="sub">Means over subjects&times;sequences per (dataset, model). bone fraction = fraction of body
voxels with HU&gt;300 (cortical bone proxy). <i>*MAE n/a for Learn2Reg: MR and CT are co-gridded but
deformably mis-aligned (the registration challenge), so pixelwise MAE is not meaningful.</i></p>

{img(bars)}

<h2>Qualitative comparison &amp; HU distributions</h2>
{sections}

<h2>Analysis</h2>
{ANALYSIS}

</body></html>"""

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        f.write(html)
    print(f"wrote {args.out}  ({len(html)//1024} KB, models={models_present})")


MODELS_SPLIT = {
    "unet_centerwise_new": "center-wise (nbn71048, ep799)",
    "unet_centerwise_old": "center-wise (9xmodnhn, ep799)",
    "unet_fulldata": "all-data (krdhs2k0, ep999)",
    "mcddpm": "center-wise (a3g28rez, ep3003)",
    "koalai": "per-region fold 0 / center-wise",
}
DATASET_NOTE = {
    "cfb_gbm": "CFB-GBM glioblastoma brain (TCIA). MR rigidly registered to T1Gd; CT co-registered. "
               "MAE-vs-gtCT is meaningful. Input MR shown: T1Gd.",
    "gold_atlas": "Gold Atlas male pelvis. CT deformably registered (B-spline) to the T2 MR grid; "
                  "MAE-vs-gtCT is meaningful (approximate). Input MR shown: T2.",
    "learn2reg": "Learn2Reg AbdomenMRCT. MR and CT are co-gridded but NOT voxel-aligned (deformable "
                 "registration is the unsolved challenge task) &rarr; compare sCT anatomy to the MR, "
                 "not pixelwise to the CT panel.",
}
ANALYSIS = """
<p>Every model here was trained on SynthRAD2025, whose training set already contains
brain, pelvis and abdomen. So none of these three datasets is a <i>new anatomical region</i>:
the shift is in scanner, MR pulse sequence, cohort and field-of-view. The three datasets land
in three very different places on the robustness spectrum.</p>

<h3>1. Brain (CFB-GBM) &mdash; severe failure, driven by MR-appearance shift, not missing training data</h3>
<p>On CFB-GBM every U-Net and the diffusion model collapses to a near-uniform soft-tissue blob with
<b>no skull</b>: body-mean &asymp; &minus;575 HU, bone fraction &asymp; 0.01 against a ground-truth
0.19&ndash;0.23, and body MAE &asymp; 770&ndash;790 HU. The qualitative panels make it unmistakable &mdash;
the bright cortical-bone rim that dominates the ground-truth CT is simply absent in the predictions.
Crucially, the <b>full-data U-Net (krdhs2k0), which was trained on every SynthRAD brain center, fails just
as hard as the center-wise models</b>. That rules out &ldquo;not enough brain in training&rdquo; as the cause:
the failure is a distribution shift in the <i>MR input itself</i> (CFB-GBM is post-contrast glioblastoma
imaging on different scanners/sequences), which the body-mask min&ndash;max normalization cannot absorb.
The skull is the specific casualty, consistent with our earlier finding that cortical-bone HU is the
information-limited part of MR&rarr;CT.</p>
<p>The one model that is clearly more robust is <b>koalAI</b> (the SynthRAD2025 winner: a ResEnc-UNet-L
region specialist trained with the anatomical MAP loss). It roughly halves brain MAE (&asymp; 450&ndash;530 HU)
and produces 3&ndash;4&times; more bone (frac &asymp; 0.03&ndash;0.04). It still falls far short of the true skull,
but architecture + loss + region specialization buy a meaningful margin under shift that the generic
U-Nets and the diffusion model do not.</p>

<h3>2. Pelvis (Gold Atlas) &mdash; near-in-distribution, all models succeed</h3>
<p>Gold Atlas is the positive control. Every model reproduces pelvic bone correctly (bone fraction
&asymp; 0.03 matching the ground truth) with body MAE in the 40&ndash;110 HU range &mdash; the same regime as
in-distribution SynthRAD validation. Pelvic T1/T2 MR from a different center is close enough to the
SynthRAD pelvis distribution that synthesis transfers cleanly. This confirms the brain result is a
genuine, region-specific distribution effect and not a pipeline artifact.</p>

<h3>3. Abdomen (Learn2Reg) &mdash; plausible soft tissue, under-produced bone</h3>
<p>All models yield anatomically plausible abdominal sCT but consistently under-produce bone
(frac &asymp; 0.003 vs ground-truth 0.01&ndash;0.02). Pixelwise MAE is intentionally <b>not</b> reported here:
Learn2Reg's MR and CT are co-gridded but deformably mis-aligned (alignment is the unsolved challenge task),
so a voxelwise comparison would measure registration error, not synthesis error. The bone deficit is
consistent with an MR intensity-distribution shift (Learn2Reg MR behaves fat-suppressed/T2-SPIR-like with
extreme bright outliers) that squashes the training-matched normalization.</p>

<h3>Cross-model summary</h3>
<ul>
<li><b>OOD robustness tracks MR-appearance proximity far more than anatomical-region coverage.</b>
Pelvis (familiar MR) is easy; brain (unfamiliar post-contrast glioblastoma MR) is catastrophic &mdash;
even with the region fully represented in training.</li>
<li><b>The three U-Nets are nearly interchangeable</b> on every dataset; the full-data model is only
marginally better on pelvis and gives no brain benefit. <b>Diffusion (MC-IDDPM) is no more robust than a
plain U-Net.</b></li>
<li><b>The SynthRAD-winner (koalAI) is the most shift-robust</b>, most visibly on brain, but it does not
solve severe shift &mdash; the skull is still largely missing.</li>
<li><b>Bone is the universal failure mode.</b> Soft tissue transfers; cortical bone is where every model
degrades under shift, echoing the information-ceiling result that bone HU is the hard, information-limited
target in MR&rarr;CT.</li>
</ul>
"""

if __name__ == "__main__":
    main()
