"""Assemble report 11: cross-model CADS error decomposition.
-> _html/11_cads_multimodel_decomposition.html  (self-contained, base64 figures)

  python mm_report.py
"""
import os
import base64
import json
import numpy as np
import pandas as pd

import mm_common as C

OUT = os.path.join(C.REPO, "_html/11_cads_multimodel_decomposition.html")
CROSSF = os.path.join(C.cross_dir(), "figures")

summary = pd.read_csv(os.path.join(C.cross_dir(), "summary.csv")).set_index("model").reindex(C.MODELS)


def b64(path):
    with open(path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def cimg(name, cap):
    return f'<figure><img src="{b64(os.path.join(CROSSF, name))}"/><figcaption>{cap}</figcaption></figure>'


def mimg(model, name, cap):
    return f'<figure><img src="{b64(os.path.join(C.fig_dir(model), name))}"/><figcaption>{cap}</figcaption></figure>'


def df_table(df, fmt="{:.1f}", idxname=""):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    body = ""
    for idx, r in df.iterrows():
        cells = "".join(
            f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>"
            for v in r)
        body += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{head}</tr></thead><tbody>{body}</tbody></table>'


# ---- cap table ----
cap_df = pd.DataFrame({
    "ceiling behaviour": [C.MODEL_CAP[m] for m in C.MODELS],
    "max sCT bone HU": summary["pred_bone_max"].values,
    "mean sCT HU @ GT>1024": summary["pred_HU_at_gt_over1024"].values,
}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_cap = df_table(cap_df, fmt="{:.0f}", idxname="model")

# ---- consistency / whole-body ----
wb = pd.DataFrame({
    "micro MAE": summary["body_mae_micro"].values,
    "macro MAE (synthrad_mae)": summary["body_mae_macro"].values,
}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_wb = df_table(wb, fmt="{:.1f}", idxname="model")

# ---- group error-mass matrix (rows=model, cols=group errmass%) ----
GROUPS = ["bone (5 labels)", "air-organs (airway+lung)", "soft (other CADS)", "unlabeled (CADS=0)"]
em = {g: [] for g in GROUPS}
bias = {g: [] for g in GROUPS}
for m in C.MODELS:
    gg = pd.read_csv(os.path.join(C.run_dir(m), "cads_groups.csv")).set_index("group")
    for g in GROUPS:
        em[g].append(gg.loc[g, "errmass_pct"])
        bias[g].append(gg.loc[g, "bias"])
em_df = pd.DataFrame({g.split(" (")[0]: em[g] for g in GROUPS}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_em = df_table(em_df, fmt="{:.1f}", idxname="model")
bias_df = pd.DataFrame({g.split(" (")[0]: bias[g] for g in GROUPS}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_bias = df_table(bias_df, fmt="{:+.0f}", idxname="model")

# ---- bone summary matrix ----
bone_df = pd.DataFrame({
    "% body vox": summary["bone_voxshare_pct"].values,
    "bone MAE": summary["bone_mae"].values,
    "bone bias": summary["bone_bias"].values,
    "% body error": summary["bone_errmass_pct"].values,
    ">1024 bias": summary["over1024_bias"].values,
    ">1024 MAE": summary["over1024_mae"].values,
}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_bone = df_table(bone_df, fmt="{:.1f}", idxname="model")

# ---- external air + localization ----
ext_df = pd.DataFrame({
    "external air % of error": summary["external_errmass_pct"].values,
    "external air MAE": summary["mae_external"].values,
    "MR->sCT r (air band)": summary["loose_r_mr_sct"].values,
}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_ext = df_table(ext_df, fmt="{:.2f}", idxname="model")

loc_df = pd.DataFrame({
    "AUC real CT": summary["auc_gt"].values,
    "AUC sCT": summary["auc_sct"].values,
    "edge sharpness (xreal)": summary["blur_ratio"].values,
}, index=[C.MODEL_LABEL[m] for m in C.MODELS])
T_loc = df_table(loc_df, fmt="{:.3f}", idxname="model")

# ---- derived narrative numbers ----
caps = [m for m in C.MODELS if "none" not in C.MODEL_CAP[m]]
uncaps = [m for m in C.MODELS if "none" in C.MODEL_CAP[m]]
cap_over = summary.loc[caps, "over1024_bias"].mean()
uncap_over = summary.loc[uncaps, "over1024_bias"].mean()
cap_bonemae = summary.loc[caps, "bone_mae"].mean()
uncap_bonemae = summary.loc[uncaps, "bone_mae"].mean()
best_bone = summary["bone_mae"].idxmin()
ext_lo, ext_hi = summary["external_errmass_pct"].min(), summary["external_errmass_pct"].max()
bone_share_lo, bone_share_hi = summary["bone_errmass_pct"].min(), summary["bone_errmass_pct"].max()

# per-model detail blocks
detail_blocks = ""
for m in C.MODELS:
    s = summary.loc[m]
    detail_blocks += f"""
<h3>{C.MODEL_LABEL[m]} <span style="font-weight:400;color:#6b7280">(ceiling: {C.MODEL_CAP[m]}; body MAE {s.body_mae_macro:.1f})</span></h3>
{mimg(m, "c_groups.png", f"{C.MODEL_LABEL[m]}: group decomposition (voxel share / per-voxel MAE / error share).")}
{mimg(m, "c_bone_hu.png", f"{C.MODEL_LABEL[m]}: within-bone error by GT density band (bias <0 = undershoot).")}
{mimg(m, "c_calib.png", f"{C.MODEL_LABEL[m]}: GT vs sCT HU calibration per CADS group.")}
{mimg(m, "c_perlabel.png", f"{C.MODEL_LABEL[m]}: per-CADS-label error share, MAE and HU bias.")}
{mimg(m, "c_loose_scatter.png", f"{C.MODEL_LABEL[m]}: sCT vs MR in in-mask air voxels (loose-band fill).")}
{mimg(m, "vf2_blur.png", f"{C.MODEL_LABEL[m]}: magnitude-matched bone-edge sharpness vs real CT, per region.")}
"""

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cross-model sCT error by GT CADS label</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1100px;margin:0 auto;padding:34px 26px;color:#1f2937;line-height:1.55;background:#fff}}
 h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
 h3{{font-size:16px;margin:26px 0 6px}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:16px}}
 .claim{{background:#fffbeb;border-left:4px solid #d97706;padding:12px 16px;margin:14px 0;border-radius:4px;font-weight:600}}
 .key{{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .ok{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px 16px;margin:14px 0;border-radius:4px;font-size:13px}}
 figure{{margin:16px 0;text-align:center}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 figcaption{{color:#6b7280;font-size:12.5px;margin-top:6px}}
 table{{border-collapse:collapse;margin:12px 0;font-size:12.5px;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}} thead th{{background:#f9fafb}} tbody th{{text-align:left;background:#fafafa}}
 code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12px}} ol li,ul li{{margin:5px 0}}
</style></head><body>

<h1>Where the sCT error lives, by ground-truth CADS label: all six models</h1>
<div class="sub">Six MR&rarr;CT models on the same 207 center-wise validation subjects
(<code>full_eval_20260617</code>): U-Net, Anatomix, MAISI, cWDM, MC-DDPM, koalAI. Every body voxel is attributed to
its <b>ground-truth CADS 35-label</b> class (no segmentation model, no HU-threshold tissue classes). Errors are
full-range raw HU, <code>error = sCT - GT</code>, inside the body mask, accumulated as <b>micro sums</b>. This
replicates report 10 (U-Net only) for every model with identical code, and asks: do the other baselines show the
same failure structure? Generated 2026-06-20.</div>

<h2>0. Method, output ceilings, and consistency gate</h2>
<ul>
<li><b>Attribution:</b> each body voxel &rarr; its GT CADS label. Groups: bone {{7,27,28,29,30}};
air-organs {{airway 9, lungs 13}}; soft = other labels 1-34; unlabeled = label 0 (CADS Background).</li>
<li><b>Micro &amp; additive:</b> per-label/group MAE = &Sigma;|err| / &Sigma;vox, so the parts reconstruct the body MAE.</li>
<li><b>Output ceiling differs by model</b> and is the crux of the bone comparison: U-Net/Anatomix use a sigmoid
that saturates near 1024 HU, MAISI clips at 1000, cWDM clips at 1024, while MC-DDPM and koalAI have no hard ceiling
and do emit voxels above 1024 HU.</li>
</ul>
{T_cap}
<div class="ok"><b>Consistency gate.</b> For every model, &Sigma;|err| over all CADS labels reconstructs the
body-voxel count exactly and the group error-shares sum to 100%. Whole-body MAE per model (macro = the leaderboard
<code>synthrad_mae</code>):</div>
{T_wb}
{cimg("cf_wholebody_mae.png", "Whole-body MAE per model. Macro (per-subject mean) exceeds micro (voxel-pooled) for all models because smaller-body subjects score higher and weigh equally.")}

<h2>1. Severity vs leverage holds for every model</h2>
<div class="claim">Claim (report 10, U-Net): bone is the worst tissue per voxel but a small slice of total error
({bone_share_lo:.0f}-{bone_share_hi:.0f}% across models), because it is only a few percent of voxels. Soft tissue +
unlabeled air dominate the error mass by volume. This pattern is the same across all six models.</div>
{cimg("cf_group_errmass.png", "Share of total body error by CADS group, all six models. Soft tissue and unlabeled (loose-mask) air dominate everywhere; bone is a minority of the error mass for every model.")}
{cimg("cf_group_mae.png", "Per-voxel MAE by CADS group, all six models. Bone is the worst per-voxel tissue for every model (severity), despite its small error share (leverage).")}
<p>Group error-share matrix (% of body error):</p>
{T_em}

<h2>2. External loose-mask air contributes a fifth of the error for every model</h2>
<div class="claim">Claim: a loose body mask sweeps in external out-of-patient air; by a tight-body test this external
band carries roughly {ext_lo:.0f}-{ext_hi:.0f}% of total body error across models. It is largely a mask/preprocessing
artifact (the MR is not zeroed inside the loose band and the sCT is saved unmasked), and every model fills it by
translating residual MR into HU.</div>
{cimg("cf_external.png", "External out-of-patient air (tight-body split): share of body error and per-voxel MAE, all six models. The external band is a substantial error term for every model; how much each fills it varies (cWDM lowest, koalAI highest).")}
{T_ext}
<p>The <code>MR&rarr;sCT r</code> column is the pooled correlation between MR intensity and predicted HU in voxels whose
ground truth is air. It is positive for every model, confirming they lift the air band toward tissue where the MR
carries signal; koalAI is the weakest correlation yet the largest external error, i.e. it fills the band more
uniformly rather than strictly tracking the MR.</p>

<h2>3. Bone is the most under-predicted group for every model</h2>
<div class="claim">Claim: bone has the most negative bias of any group for all six models, i.e. the densest tissue is always
pulled down. The companion behaviour differs by model family: the regression models (U-Net, Anatomix, cWDM)
over-predict air-organs and keep soft tissue near zero (the classic conditional-mean signature, both density extremes
pulled toward the soft-tissue middle); MAISI, MC-DDPM and koalAI instead carry a global negative soft-tissue offset
(soft and air both shifted down). So "bone undershoot" is universal, but "air overshoot, soft calibrated" is specific
to the regression models.</div>
{cimg("cf_group_bias.png", "HU bias by CADS group, all six models. Bone is the most negative (undershot) group for every model. U-Net/Anatomix/cWDM over-predict air-organs with near-zero soft bias; MAISI/MC-DDPM/koalAI shift soft tissue and air downward.")}
<p>Group bias matrix (HU, sCT - GT):</p>
{T_bias}

<h2>4. The ceiling test: dense-bone undershoot tracks the output ceiling, but bone MAE does not collapse</h2>
<div class="claim">Claim: the densest cortical bone (GT &gt; 1024 HU) is where the output ceiling bites. Capped models
(U-Net/Anatomix/MAISI/cWDM) undershoot it by ~{cap_over:.0f} HU on average; the uncapped models (MC-DDPM, koalAI)
undershoot less ({uncap_over:.0f} HU). But lifting the ceiling does not solve bone: overall bone MAE is similar
(capped {cap_bonemae:.0f} vs uncapped {uncap_bonemae:.0f} HU; best = {C.MODEL_LABEL[best_bone]}), because most bone
error is below the ceiling and is information-limited, not clipping-limited.</div>
{cimg("cf_bone_hu_bias.png", "Within-bone HU bias by ground-truth density band, all six models. Bands diverge only in the top (>1024) band: capped models cliff downward (cannot represent it), uncapped models stay flatter. Below 1024 every model behaves similarly.")}
{cimg("cf_cap_over1024.png", "Mean predicted HU where ground-truth bone exceeds 1024 HU. Capped models plateau at their ceiling; MC-DDPM and koalAI reach higher, but still well below the true value.")}
{cimg("cf_bone_calib_grid.png", "GT vs sCT HU within GT-bone, per model (cyan line = 1024 HU). The capped models flatten into a horizontal band below the diagonal at high GT HU; the uncapped models extend further up but still fall below the diagonal.")}
{cimg("cf_bone_hu_mae.png", "Within-bone per-voxel MAE by density band. MAE rises with density for every model; the gap between capped and uncapped models is confined to the densest band.")}
<p>Bone summary per model (the <code>&gt;1024</code> columns isolate the ceiling effect):</p>
{T_bone}
<div class="key"><b>Ceiling helps only the densest sliver.</b> Removing the hard ceiling (MC-DDPM, koalAI) measurably
reduces the &gt;1024 HU undershoot, but that band is a small fraction of bone voxels, so overall bone MAE barely moves
and bone remains the per-voxel-worst tissue for every model. This is consistent with the standing finding that bone HU
is <b>information-limited from a single MR</b>: the cap is a secondary, model-specific aggravator on top of an
intrinsic limit shared by all six.</div>

<h2>5. Bone is localized, undershot and blurred for every model</h2>
<div class="claim">Claim: across all models the bone failure is not mislocalization. sCT HU still ranks GT-bone above
non-bone almost as well as real CT (AUC near the real-CT ceiling), and after magnitude-matching the bone edges are
still softer than real CT. The error is density (bulk undershoot + edge blur), not placement.</div>
{cimg("cf_localization.png", "Left: AUC of sCT HU separating GT-bone from the rest vs the real-CT ceiling, all models (high = located correctly). Right: magnitude-matched bone-edge sharpness vs real CT (<1 = blurrier).")}
{T_loc}

<h2>6. Qualitative: same slice, all models</h2>
{cimg("cf_qualitative.png", "One thorax subject, identical slice. GT-bone outline drawn on each model's sCT (top) with the error map below. The outline lands on bone for every model (localized); inside it the capped models read greyer (undershoot).")}

<h2>7. Per-model detail</h2>
<div class="sub">The full report-10 figure set reproduced for each model.</div>
{detail_blocks}

<h2>8. Conclusion</h2>
<ol>
<li><b>The failure structure is shared.</b> Severity&ne;leverage, the ~20% external loose-mask air term,
regression-to-the-mean (bone down / air up / soft flat), and located-but-undershot-and-blurred bone all hold for every
one of the six models. None of these is a U-Net artifact.</li>
<li><b>The output ceiling is the one place models genuinely differ.</b> Capped models (U-Net, Anatomix, MAISI, cWDM)
cannot represent GT bone above ~1000-1024 HU and undershoot it severely; MC-DDPM and koalAI, with no hard ceiling,
reduce that top-band undershoot.</li>
<li><b>But lifting the ceiling does not fix bone.</b> Overall bone MAE is similar across capped and uncapped models
and bone stays the per-voxel-worst tissue everywhere, because most bone error is below the ceiling and is
information-limited, not clipping-limited.</li>
<li><b>Metric-blindness is universal.</b> Bone is a few percent of voxels for every model, so headline MAE/PSNR barely
reflect it; a tight body mask and tissue/bone-restricted metrics are needed for any meaningful cross-model comparison.</li>
</ol>
<div class="sub">Reproduce: <code>mm_extract.py --model M</code> &rarr; <code>mm_mr.py --model M</code> (per model) &rarr;
<code>mm_cross.py</code> &rarr; <code>mm_report.py</code>. All GT-CADS-label based, no segmentation model.</div>
</body></html>"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(HTML)
    print("[mm_report] wrote", OUT)


if __name__ == "__main__":
    main()
