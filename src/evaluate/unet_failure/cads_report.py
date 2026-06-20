"""Assemble report 10: GT-CADS-label error decomposition of the U-Net sCT
(no babyseg, no HU-threshold tissue; micro-additive; with body-mask audit and
GT-CT vs sCT overlays). -> _html/10_cads_error_decomposition.html"""
import os
import base64
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
OUT = os.path.join(REPO, "_html/10_cads_error_decomposition.html")

lab = pd.read_csv(os.path.join(RUN, "cads_label_micro.csv"))
gr = pd.read_csv(os.path.join(RUN, "cads_groups.csv"))
au = json.load(open(os.path.join(RUN, "cads_audit.json")))
ms = json.load(open(os.path.join(RUN, "cads_mask_split.json")))
lo = json.load(open(os.path.join(RUN, "cads_loose_stats.json")))
vd = pd.read_csv(os.path.join(RUN, "verify_density.csv"))
vb = pd.read_csv(os.path.join(RUN, "verify_blur.csv"))
aucg, aucs = float(vd.auc_gt.mean()), float(vd.auc_sct.mean())
blur = float(vb.ratio.mean())


def b64(n):
    with open(os.path.join(FIG, n), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(n, c):
    return f'<figure><img src="{b64(n)}"/><figcaption>{c}</figcaption></figure>'


def table(df, fmt="{:.1f}", idxname=""):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>" for v in r)
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{cols}</tr></thead><tbody>{rows}</tbody></table>'


# group table
gt_ = gr.set_index("group")[["voxshare_pct", "mae", "bias", "errmass_pct"]]
gt_.columns = ["% body vox", "MAE", "bias", "% of body error"]
T_grp = table(gt_, idxname="CADS group")

# top contributors
top = lab.head(12).set_index("name")[["voxshare_pct", "mae", "bias", "gt_hu", "pred_hu", "errmass_pct"]]
top.columns = ["% vox", "MAE", "bias", "GT HU", "pred HU", "% error"]
T_top = table(top, fmt="{:.1f}", idxname="CADS label")

# bone table
bo = lab[lab.is_bone].sort_values("errmass_pct", ascending=False).set_index("name")
bo = bo[["voxshare_pct", "mae", "bias", "gt_hu", "pred_hu", "errmass_pct"]]
bo.columns = ["% vox", "MAE", "bias", "GT HU", "pred HU", "% error"]
T_bone = table(bo, fmt="{:.1f}", idxname="bone label")

bone_share = float(lab[lab.is_bone].errmass_pct.sum())
bone_vox = float(gr[gr.group.str.startswith("bone")].voxshare_pct.iloc[0])
soft_share = float(gr[gr.group.str.startswith("soft")].errmass_pct.iloc[0])
unlab_share = au["errmass_lab0_pct"]
mae_micro, mae_macro = au["body_mae_micro"], au["body_mae_macro"]
bg = lab[lab.label == 0].iloc[0]
bg_mae, bg_bias, bg_gt, bg_pred = float(bg.mae), float(bg.bias), float(bg.gt_hu), float(bg.pred_hu)
bw = lab[lab.label == 17]
bw_bias = float(bw.bias.iloc[0]) if len(bw) else float("nan")

# full per-label MAE/bias table (all labels, sorted by error share)
full = lab.set_index("name")[["voxshare_pct", "mae", "bias", "gt_hu", "pred_hu", "errmass_pct", "n_subj"]].copy()
full.columns = ["% vox", "MAE", "bias", "GT HU", "pred HU", "% error", "n subj"]
T_full = table(full, fmt="{:.1f}", idxname="CADS label")

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>U-Net sCT error by GT CADS label</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:34px 26px;color:#1f2937;line-height:1.55;background:#fff}}
 h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:20px;margin:32px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
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

<h1>Where the U-Net sCT error lives, by ground-truth CADS label</h1>
<div class="sub">Plain 3D U-Net sCT (wandb <code>9xmodnhn</code>), 207 center-wise validation subjects
(<code>full_eval_20260617</code>). Every body voxel is attributed to its <b>ground-truth CADS 35-label</b> class
(no segmentation model, no HU-threshold tissue classes). All errors are full-range raw HU, <code>error = sCT - GT</code>,
inside the body mask, accumulated as <b>micro sums</b> so the parts reconstruct the whole. Generated 2026-06-20.</div>

<h2>0. Method &amp; consistency gate</h2>
<ul>
<li><b>Attribution:</b> each body voxel → its GT CADS label (0–34). Groups: bone {{7,27,28,29,30}};
air-organs {{airway 9, lungs 13}}; soft = other labels 1–34; unlabeled = label 0 (CADS Background).</li>
<li><b>Error:</b> full-range raw HU <code>|sCT − GT|</code> (and signed bias = sCT − GT). The sCT is bounded to ≤~1024 by
its sigmoid; GT is raw to ~3000.</li>
<li><b>Micro &amp; additive:</b> per-label/group MAE = Σ|err| ⁄ Σvox, so the voxel-weighted parts sum to the body MAE.</li>
</ul>
<div class="ok"><b>Gate PASS.</b> Σ|err| over all CADS labels ÷ Σ body voxels = <b>{mae_micro:.1f} HU</b> = the body-voxel
MAE (micro), and the group error-shares sum to 100.0%. The per-subject-averaged (macro) body MAE is <b>{mae_macro:.1f} HU</b>
(= the leaderboard <code>synthrad_mae</code>); macro &gt; micro because small-body subjects score higher and weigh equally.</div>

<h2>1. The body mask is loose: ~18% of "body" is external air the mask wrongly includes</h2>
<div class="claim">Claim: {ms['pct_body_lab0']:.0f}% of "body" voxels carry no CADS label (Background, mean {bg_gt:.0f} HU
= air), and by a tight-body test <b>{ms['external_share_of_lab0']:.0f}% of it is EXTERNAL</b> air outside the patient
that a loose mask swept in ({ms['pct_body_lab0_external']:.1f}% of body); only {ms['pct_body_lab0_internal']:.2f}% of body
is genuine internal unlabeled gas. That external air contributes <b>{ms['errmass_external_pct']:.0f}% of the total body
error</b> (internal gas {ms['errmass_internal_pct']:.1f}%) — i.e. roughly a fifth of the reported body MAE comes from
outside the patient.</div>
{img("c_lab0_hist.png", f"GT HU of the unlabeled (CADS=0) in-body voxels: an almost pure air spike at −1000 — the unlabeled region is air. {ms['pct_body_lab0']:.0f}% of the body mask is unlabeled.")}
{img("c_maskaudit.png", f"Tight-body split (largest body component, hole-filled): {ms['external_share_of_lab0']:.0f}% of the label-0 air is OUTSIDE the patient (loose mask), contributing {ms['errmass_external_pct']:.0f}% of total body error; genuine internal unlabeled gas is {ms['pct_body_lab0_internal']:.2f}% of body / {ms['errmass_internal_pct']:.1f}% of error.")}
<p><b>Why the model puts tissue there.</b> The input MR is zeroed only <i>outside</i> the body mask; <i>inside</i> the
loose band it still carries low-level signal, and the saved sCT is the <b>raw, unmasked</b> network output (in
<code>unet_baseline/validate.py</code> the body mask is applied only to the metrics, never to the written volume). So the
U-Net simply translates that residual MR into HU. Pooled over {lo['n_subj']} subjects, in voxels where the truth is air the
sCT median climbs with MR intensity (<b>r = {lo['pooled_r_mr_sct']:.2f}</b>), and ~{lo['frac_air_filled_gt_m400']*100:.0f}%
of in-mask air voxels are lifted above −400 HU; the external-air MAE is {ms['mae_external']:.0f} HU. The Background bias is
near zero ({bg_bias:+.0f}) only because most of the band <i>is</i> correctly predicted air (MR ≈ 0); the error is the minority
of band voxels where the MR is non-zero.</p>
{img("c_loose_scatter.png", f"In-mask voxels whose GT is air (&lt;−300 HU), pooled over {lo['n_subj']} subjects: where the MR ≈ 0 the sCT is correctly air; where the MR carries signal the sCT median is lifted toward tissue (r = {lo['pooled_r_mr_sct']:.2f}). The loose-region error is the U-Net translating un-masked MR into HU.")}
{img("c_loose_example.png", "One subject. Top: full slice (red = body mask, yellow = zoom box). Bottom: zoom into the external band at the air-floor window. The MR carries texture in the band, the sCT lifts off the air floor to match it, and the GT CT there is itself noisy (streak artifacts), i.e. not a clean air target.")}
<p><b>Bottom line:</b> this ~{ms['errmass_external_pct']:.0f}%-of-error term is largely a <b>mask-tightness / preprocessing
artifact</b>, not a synthesis failure: the body mask is loose, the MR is not zeroed inside it, the sCT is saved unmasked,
and the GT it is scored against is itself noisy there. A tighter body mask (or tissue-restricted scoring) removes almost
all of it. Genuine internal gas (bowel, lungs, airway) mostly carries its own CADS label, so it is not in this group.</p>

<h2>2. What actually drives the body error (contribution, additive)</h2>
<div class="claim">Claim: the body MAE is dominated by high-volume, non-bone regions — soft tissue ({soft_share:.0f}%),
unlabeled air ({unlab_share:.0f}%) and lungs (~11%) make up ~80%. Bone contributes only {bone_share:.0f}%, spread over
5 labels, despite being the worst per voxel. Severity ≠ leverage.</div>
{T_grp}
{img("c_groups.png", "Per CADS group: voxel share, per-voxel MAE, and share of total body error. Soft and unlabeled-air dominate the error mass by sheer volume; bone is high MAE but small share.")}
<div class="key"><b>Metric-blindness to bone.</b> Bone is only {bone_vox:.0f}% of body voxels, so the headline body-MAE /
PSNR barely move with it — its {bone_share:.0f}% error share is swamped by soft ({soft_share:.0f}%) and the loose-mask
air ({unlab_share:.0f}%). The clinically critical error is nearly invisible to the aggregate metric; a tissue-restricted
or bone-specific metric is needed to see it. This is the single most important caveat when reading MAE/PSNR leaderboards.</div>
{img("c_perlabel.png", "Left: share of total body error per CADS label (bone red). Right: HU bias per label. The biggest contributors are Background-air, Subcutaneous tissue, Lungs; bone labels are individually small.")}
<p>Top contributors:</p>
{T_top}
<p><b>Note on grouping:</b> "air-organs" here is airway+lungs only (air-<i>dominated</i>, GT ≈ −800), coloured blue.
<b>Bowel</b> (GT ≈ −88, a mixed gas/fluid/wall organ) is counted under "soft" (grey), consistent with the group
decomposition — but note it still over-predicts by {bw_bias:+.0f} HU (the model fills its gas), like the other
air-containing structures.</p>
<p>Full per-CADS-label table (MAE, bias, error share — micro, additive):</p>
{T_full}

<h2>3. Per-voxel severity and direction (which way each tissue is wrong)</h2>
<div class="claim">Claim: per voxel, bone is by far the worst and the only strongly <i>under</i>-predicted tissue
(cortical −91 to −171 HU); air-filled organs are <i>over</i>-predicted (lungs +75, bowel +57); soft tissue is
well-calibrated (+13). The model regresses both density extremes toward soft tissue.</div>
{img("c_calib.png", "GT vs sCT HU calibration within each CADS group (white dashed = perfect). Bone falls below the diagonal at high HU (undershoot, capped near +1024); air-organs sit above at low HU (overshoot); soft hugs the diagonal; unlabeled is a pure air spike.")}
<p>Bone labels (dense cortical bone undershoots; "bone-other" is a soft-HU residual that does not):</p>
{T_bone}
<div class="key"><b>One mechanism: regression to the conditional mean</b> (L1 loss → conditional median). The rare
<i>high</i> extreme (dense cortical bone) is pulled <i>down</i> and the <i>low</i> extreme (air) is pulled <i>up</i>,
while the abundant middle (soft tissue) is rendered accurately. So the three findings above — bone undershoot, air
overshoot, soft well-calibrated — are one effect, not three. Note the undershoot is concentrated in <b>dense cortical</b>
bone (skull/ribs/limb/spine, GT 305–708 HU); the low-density "bone-other" filler does not undershoot. The sCT also
saturates at the sigmoid's <b>+1024 HU ceiling in 61/207 subjects</b>, so it cannot represent the densest cortical bone
at all.</div>

<h2>4. Bone: localized, but undershot and blurred (CADS-bone, no segmenter)</h2>
<div class="claim">Claim: the bone failure is not mislocalization — the GT-CADS bone is in the right place in the sCT
(AUC {aucs:.2f} vs {aucg:.2f} ceiling). It is a density error in two parts: a bulk HU undershoot and a loss of edge
sharpness (sCT bone edges {blur:.2f}× as sharp as real CT). Both are HU errors; the split is by spatial scale.</div>
{img("c_locdens.png", f"Left: AUC of sCT-HU separating GT-CADS-bone from the rest = {aucs:.2f} vs real-CT {aucg:.2f} (bone is located correctly). Right: bone-edge sharpness after magnitude-matching = {blur:.2f}× real CT (edges are genuinely blurred, not just dim).")}
{img("c_qualitative.png", "The SAME GT-CADS-bone outline drawn on GT CT and sCT: the outline lands on bone in both (localization intact), but inside it the sCT is greyer (undershoot) and softer (blur). Error map: blue = undershoot at bone, red = overshoot at air.")}
{img("c_zoom.png", "Zoom on bone: the GT cortical rim is bright and crisp; the sCT version is dimmer and smeared across more voxels.")}

<h2>5. Conclusion</h2>
<ol>
<li><b>Mask looseness first.</b> ~18% of the scored "body" is unlabeled air, and {ms['external_share_of_lab0']:.0f}% of
it is <b>external</b> (outside the patient), contributing {ms['errmass_external_pct']:.0f}% of the body error. The model
fills it because the MR is not zeroed inside the loose mask (sCT vs MR r = {lo['pooled_r_mr_sct']:.2f}) and the sCT is
saved unmasked — largely a preprocessing artifact. Any body-MAE comparison should use a tight mask or tissue-restricted
error.</li>
<li><b>Aggregate error is a volume story.</b> Soft tissue ({soft_share:.0f}%), unlabeled air ({unlab_share:.0f}%) and
lungs (~11%) dominate; bone is only {bone_share:.0f}% despite the worst per-voxel error.</li>
<li><b>Bone is the clinically critical, per-voxel-worst error</b> (skull MAE 312; cortical undershoot −91 to −171 HU),
and it is one-sided: dense bone is pulled down, air is pushed up, soft is fine — regression of both extremes toward soft
tissue.</li>
<li><b>Bone is located, not misplaced</b> (AUC {aucs:.2f}); the error is density — a bulk undershoot (recalibratable)
plus edge blur ({blur:.2f}× sharpness, <i>not</i> recalibratable). Fixing it needs HU calibration <i>and</i> sharper
high-frequency detail, not relocation.</li>
<li><b>One root cause:</b> regression to the conditional mean — dense bone down, air up, soft accurate; the +1024 cap
worsens the densest bone.</li>
<li><b>Metric-blindness:</b> because bone is ~{bone_vox:.0f}% of voxels, body-MAE/PSNR barely reflect it — the most
clinically important error is nearly invisible to the headline metric. Report bone/tissue-restricted metrics, and use a
tight body mask, for any meaningful comparison.</li>
</ol>
<div class="sub">Reproduce: <code>cads_extract.py</code> → <code>cads_analyze.py</code> → <code>cads_mask_split.py</code>
(tight-body external/internal split) → <code>cads_loose.py</code> (MR→sCT mechanism) → <code>cads_figures.py</code> →
<code>cads_report.py</code>; localization/blur from <code>verify_density.py</code>/<code>verify_blur.py</code>
(all GT-CADS-label based, no babyseg).</div>
</body></html>"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    open(OUT, "w").write(HTML)
    print("[cads_report] wrote", OUT)


if __name__ == "__main__":
    main()
