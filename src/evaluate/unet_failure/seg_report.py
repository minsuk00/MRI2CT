"""Assemble the self-contained seg-downstream U-Net failure report (report 09):
_html/09_unet_seg_downstream.html. Base64-embeds figures from RUN/figures."""
import os
import base64
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
OUT = os.path.join(REPO, "_html/09_unet_seg_downstream.html")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]

st = json.load(open(os.path.join(RUN, "seg_stats.json")))
lab = pd.read_csv(os.path.join(RUN, "seg_label_table.csv"), index_col="name")
reg = pd.read_csv(os.path.join(RUN, "seg_region.csv"), index_col=0).reindex(REG)
hu = json.load(open(os.path.join(RUN, "hu_stats.json")))
vd = pd.read_csv(os.path.join(RUN, "verify_density.csv"))
vr = pd.read_csv(os.path.join(RUN, "verify_recalib.csv"))
auc_gt, auc_sct = float(vd.auc_gt.mean()), float(vd.auc_sct.mean())
auc_ret = (auc_sct - 0.5) / (auc_gt - 0.5) * 100
thr_real, thr_sct, thr_best = float(vd.dice_real_t150.mean()), float(vd.dice_sct_t150.mean()), float(vd.dice_sct_best.mean())
rec_thr = (thr_best - thr_sct) / (thr_real - thr_sct) * 100
rec_ce, rec_sc, rec_re = float(vr.dice_ceiling.mean()), float(vr.dice_sct.mean()), float(vr.dice_recal.mean())
rec_cnn = (rec_re - rec_sc) / (rec_ce - rec_sc) * 100
vb = pd.read_csv(os.path.join(RUN, "verify_blur.csv"))
blur_overall = float(vb.ratio.mean())
blur_pelvis = float(vb[vb.region == "pelvis"].ratio.mean())


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(name, cap):
    return f'<figure><img src="{b64(name)}"/><figcaption>{cap}</figcaption></figure>'


def table(df, fmt="{:.3f}", idxname=""):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(
            f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>"
            for v in r)
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{cols}</tr></thead><tbody>{rows}</tbody></table>'


c = st["coarse"]
bh = st["bone_hu"]
bc = st["bone_confusion"]
g = st["gates"]
gate_pass = "PASS" if st["all_pass"] else "SOME FAILED"
gate_li = "".join(f"<li><b>{'✓' if v['pass'] else '✗'}</b> <code>{k}</code> = {v['val']:.3f}</li>" for k, v in g.items())

# coarse table
coarse_df = pd.DataFrame({
    "ceiling (real CT)": {k: c[k]["ceiling"] for k in ["bone", "soft"]},
    "synthetic CT": {k: c[k]["sct"] for k in ["bone", "soft"]},
    "Dice drop": {k: c[k]["gap"] for k in ["bone", "soft"]},
})
T_coarse = table(coarse_df, idxname="tissue group")

# region table
rt = reg[["dice_ceil_bone", "dice_sct_bone", "gap_bone", "bone_bias", "bone_mae"]].copy()
rt.columns = ["bone ceiling", "bone sCT", "bone Dice drop", "bone HU bias", "bone MAE"]
T_region = table(rt, fmt="{:.2f}", idxname="region")

# per-label table (show all, key columns)
lt = lab[["is_bone", "dice_ceil", "dice_sct", "gap", "gt_hu", "pred_hu", "bias", "mae", "n_subj"]].copy()
lt["is_bone"] = lt["is_bone"].map({True: "bone", False: ""})
lt.columns = ["", "Dice ceil", "Dice sCT", "Dice drop", "GT HU", "pred HU", "HU bias", "HU MAE", "n subj"]


def fmt_label_table(df):
    head = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = ""
        for cn, v in zip(df.columns, r):
            if isinstance(v, str):
                cells += f"<td>{v}</td>"
            elif pd.isna(v):
                cells += "<td>-</td>"
            elif cn in ("Dice ceil", "Dice sCT", "Dice drop"):
                cells += f"<td>{v:.2f}</td>"
            else:
                cells += f"<td>{v:.0f}</td>"
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>CADS label</th>{head}</tr></thead><tbody>{rows}</tbody></table>'


T_label = fmt_label_table(lt)

kept_r = bc["gt_bone_kept_as_bone_realCT"]
kept_s = bc["gt_bone_kept_as_bone_sCT"]
relabel_str = ", ".join(f"{n} {f*100:.1f}%" for n, f in bc["sct_relabel_top"][:3] if f > 0.005)

# HU-tissue table (segmenter-free, by HU value)
ho = hu["overall"]
hu_df = pd.DataFrame({
    "GT HU": {t: ho[t]["gt_hu"] for t in ["air", "soft", "bone"]},
    "pred HU": {t: ho[t]["pred_hu"] for t in ["air", "soft", "bone"]},
    "HU bias": {t: ho[t]["bias"] for t in ["air", "soft", "bone"]},
    "HU MAE": {t: ho[t]["mae"] for t in ["air", "soft", "bone"]},
    "vox %": {t: ho[t]["frac"] * 100 for t in ["air", "soft", "bone"]},
})
T_hu = table(hu_df, fmt="{:.0f}", idxname="HU tissue")
air_keep = hu["air_kept_as_air"] * 100
air_soft = hu["air_to_soft"] * 100
hb_keep = hu["bone_kept_as_bone"] * 100
hb_soft = hu["bone_to_soft"] * 100

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>U-Net MR→CT seg-downstream failure: localization vs density</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;
      padding:34px 26px;color:#1f2937;line-height:1.55;background:#fff}}
 h1{{font-size:26px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
 h3{{font-size:15px;margin:20px 0 6px;color:#111827}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
 .key{{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .warn{{background:#fef2f2;border-left:4px solid #dc2626;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .ok{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px 16px;margin:14px 0;border-radius:4px;font-size:13px}}
 figure{{margin:18px 0;text-align:center}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 figcaption{{color:#6b7280;font-size:12.5px;margin-top:6px}}
 table{{border-collapse:collapse;margin:12px 0;font-size:12.5px;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}} thead th{{background:#f9fafb}}
 tbody th{{text-align:left;background:#fafafa}}
 code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12px}}
 ol li,ul li{{margin:5px 0}} ul.gates{{columns:2;font-size:12.5px}}
</style></head><body>

<h1>U-Net MR→CT failure, seen through a CT segmenter: localization vs density</h1>
<div class="sub">Plain 3D U-Net baseline sCT (wandb <code>9xmodnhn</code>, center-wise split), 207 center-wise validation
subjects (<code>full_eval_20260617</code>). We run a fixed CADS 35-label segmenter — BabyUNet <code>di54npq3</code>,
trained on the same center-wise split (epoch 419) — on the <b>real CT</b> and on the <b>synthetic CT</b>, and score
both against the ground-truth CADS segmentation. This separates two questions: does the sCT keep structures
<i>findable</i> (localization, via Dice), and does it carry the right <i>HU</i> (density, via per-ROI bias)?
Generated 2026-06-19. <b>Sign convention:</b> bias = pred − GT, so negative = undershoot.</div>

<div class="key"><b>Bottom line (all on your data; this version corrects an earlier mechanistic overclaim — see §6 verification).</b>
Bone is the failure; soft tissue and air are fine. But "bone" fails in <b>two distinct ways</b> that must not be
conflated:
<ol style="margin:8px 0">
<li><b>Density (HU magnitude).</b> Inside the true bone ROI the sCT undershoots HU by <b>{bh['bias']:.0f} HU</b>
(GT {bh['gt']:.0f} → pred {bh['pred']:.0f}); by raw HU value the undershoot is {ho['bone']['bias']:+.0f} HU. Gross
<i>localization is intact</i>: sCT HU still separates bone from non-bone at AUC <b>{auc_sct:.2f}</b> vs the real-CT
ceiling {auc_gt:.2f} ({auc_ret:.0f}% of the above-chance separability). For an HU-threshold task, simply retuning the
threshold recovers <b>{rec_thr:.0f}%</b> of the bone-Dice gap. So this is a calibration/scale error, and it is the
clinically dominant one (dose/HU).</li>
<li><b>Fine bone structure (what a CNN segmenter sees).</b> The segmenter finds less bone in the sCT (bone-union Dice
{c['bone']['ceiling']:.2f}→{c['bone']['sct']:.2f}). This is <b>NOT</b> the HU undershoot: the segmenter uses instance
norm and is invariant to absolute HU, and a monotonic intensity recalibration (histogram-match sCT→GT, then
re-segment) recovers <b>{rec_cnn:.0f}%</b> of that gap — nothing. So the segmenter drop reflects structural
degradation — measured to be blur: magnitude-matched sCT bone edges are <b>{blur_overall:.2f}×</b> as sharp as real CT
(§6 Test D) — a separate problem a recalibration cannot fix.</li>
</ol>
So the model puts bone in roughly the right place; it both undershoots its HU and blurs its fine structure, and those
two errors are independent (one fixable by recalibration, one not).</div>

<h2>0. Method and correctness gates</h2>
<p>How every number here is produced:</p>
<ul>
<li><b>Segmenter.</b> BabyUNet CADS model <code>di54npq3</code> (center-wise split, epoch 419, <code>best.pth</code>),
35-output-channel U-Net. Loaded with <code>strict=True</code>; <code>argmax</code> gives labels 0–34, aligned 1:1 with
the GT CADS labels (verified).</li>
<li><b>Input normalization.</b> CT clipped to [−1024, 1024] HU, linearly mapped to [0, 1]. Verified equivalent to the
training-time per-patch min-max stretch (Dice within 0.001; the network's instance-norm cancels the difference).</li>
<li><b>Inference.</b> MONAI <code>sliding_window_inference</code>, 128³ ROI, overlap 0.5, bf16. Real-CT input = dataset
<code>ct.nii</code> (raw HU); sCT input = eval <code>sample.nii.gz</code>.</li>
<li><b>Dice.</b> 2|A∩B| / (|A|+|B|), every mask intersected with the body mask. <i>Ceiling</i> = Dice(babyseg(real CT),
GT CADS); <i>sCT</i> = Dice(babyseg(sCT), GT CADS). A label absent in a subject is excluded (NaN), not scored 0.</li>
<li><b>HU bias / MAE (density).</b> Computed inside the GT CADS ROI (GT==label & body) on raw HU, segmenter-free.</li>
<li><b>Coarse groups.</b> bone = {{skull, spine, thoracic cage, limb/girdle, bone-other}}; air/lung = {{airway, lungs}};
soft = all other in-body labels.</li>
<li><b>Ceiling-correction.</b> The segmenter is imperfect even on real CT (it is a small distilled student of the full
CADS pipeline), so we never expect Dice 1.0. We report its real-CT score as the reachable ceiling and attribute only
the gap below it to the synthetic CT.</li>
</ul>
<div class="ok"><b>Correctness gates: {gate_pass}.</b><ul class="gates">{gate_li}</ul>
G1 confirms the segmenter finds bone reliably on real CT (so a low sCT bone Dice is sCT-attributable).
G2 confirms the bone HU undershoot is real.</div>

<h2>1. Is each structure still findable in the synthetic CT?</h2>
<p>Coarse tissue groups the segmenter handles robustly (bone = the 5 bone labels; soft = the named soft-tissue
organs). Air is not a CADS class and is covered separately by HU in §5.</p>
{T_coarse}
{img("f1_perlabel_dice.png", "Per-CADS-label Dice: real-CT ceiling (light) vs synthetic CT (dark), sorted by sCT Dice. Bone labels in bold. Small/thin organs sit low even on real CT (the segmenter's own ceiling); the sCT-vs-ceiling gap is the sCT-attributable part.")}
{img("f2_dice_gap.png", "Dice drop (ceiling − sCT) per label. Bone labels (red) are among the largest drops.")}
<p>And it is systematic, not an averaging artifact: the whole per-subject distribution shifts down.</p>
{img("f8_dice_dist.png", "Per-subject bone-union Dice across all subjects: real-CT ceiling (light) vs synthetic CT (dark). The entire distribution moves down on the sCT.")}

<h2>2. Bone: localization or density?</h2>
<p>The density error inside true ROIs is segmenter-independent (GT CADS ROI only) and shows bone undershoot directly.
Each value is the per-subject mean HU error, averaged across subjects (macro; not voxel-weighted).</p>
{img("f3_hu_bias.png", "Mean signed HU bias inside each GT CADS ROI (pred − GT). Bone (red) undershoots; soft tissue is near zero.")}
{img("f3b_hu_mae.png", "Mean absolute HU error |pred − GT| inside each GT CADS ROI. Bone (red) has the largest absolute errors.")}
<p>Full per-label table (signed bias and absolute MAE, with segmentability for reference):</p>
{T_label}
<p><b>Note on "bone-other".</b> The CADS bone union is {{skull, spine, thoracic cage, limb/girdle, bone-other}}.
"Bone-other" is a residual class with mean GT HU ≈ 34 (soft-tissue range, present in all 207 subjects), so it barely
undershoots (bias +8) but has large two-sided error (MAE 212). The undershoot is concentrated in <i>dense cortical</i>
bone — skull −154, thoracic cage −180, limb/girdle −173, spine −105 HU (GT 307–646) — and the bone-union mean (−106)
is diluted by bone-other.</p>
<p>Across regions, bone Dice drop and HU bias are correlated, but correlation is not the mechanism — §6 shows the
segmenter drop is not actually caused by the HU magnitude.</p>
{img("f4_loc_vs_density.png", "Per region × bone label: segmenter Dice drop (ceiling − sCT) vs HU bias inside the bone ROI. The two co-vary across regions, but the verification in §6 shows they are driven by different, separable defects.")}
<p>The segmenter also relabels true-bone voxels more often on the sCT:</p>
{img("f5_bone_confusion.png", f"Of true (GT) bone voxels, what the segmenter assigns: real CT keeps {kept_r*100:.0f}% as bone, the sCT only {kept_s*100:.0f}%, reassigning the rest mostly to soft tissue/muscle.")}
<p>And the HU undershoot (separate, segmenter-free measurement) is near-universal across subjects:</p>
{img("f9_bias_dist.png", "Per-subject mean bone HU bias (pred − GT inside the GT bone ROI). Almost every subject undershoots (mass left of zero).")}
<div class="warn"><b>Two effects, do not conflate them.</b> The HU undershoot (Fig f3, f9) is real and segmenter-free.
The segmenter bone-Dice drop and relabeling (Fig f1, f5) are also real — but, as §6 proves, they are <i>not</i> caused
by the undershoot (the segmenter is intensity-invariant). An earlier version of this report claimed the undershoot
"pushes bone below the segmenter's bone appearance, so it relabels"; that causal link is <b>refuted</b> in §6. The
segmenter drop instead reflects degraded fine bone structure (blur / lost cortical detail).</div>

<h2>3. Per region</h2>
{T_region}
{img("f6_region_bone_dice.png", "Bone-union Dice per region: real-CT ceiling vs synthetic CT (red = drop).")}

<h2>4. Do we see it?</h2>
{img("f7_qualitative.png", "Representative (median sCT-bone-Dice) subject per region. Columns: MR, GT CT, sCT (both bone-windowed), GT CADS bone, babyseg(real CT) bone, babyseg(sCT) bone. The sCT bone mask is thinner/patchier than the real-CT one despite the bone being present in the GT.")}

<h2>5. Air / soft / bone by HU value (segmenter-free, density view)</h2>
<p>The CADS segmenter has no "air" class, so true air (gas, ≈ −1000 HU) is defined by HU, not anatomy. Here we
classify every body voxel by HU value alone — <b>air</b> &lt; −300, <b>soft</b> −300…150, <b>bone</b> &gt; 150 —
and compare GT CT vs sCT directly (no segmentation). This complements the bone localization above with a pure density
picture, and it is the only way to see <i>air</i>.</p>
{T_hu}
<p>The clearest single view is the calibration map: where predicted HU sits versus true HU for every body voxel.</p>
{img("hu3_calibration.png", "GT vs predicted HU over all body voxels (207 subjects). On the diagonal = perfect. The low-HU end bends ABOVE the diagonal (air predicted too dense) and the high-HU end falls BELOW it (bone undershot, capped near +1024).")}
{img("hu4_hist.png", "HU distribution within each true tissue class, GT (grey) vs sCT (red), dashed lines = means. Air is shifted up (filled denser), bone is shifted/compressed down (undershot), soft is well matched.")}
{img("hu1_bias.png", "Per-tissue HU bias by region (bars) with the overall mean (black ticks). Air overshoots (positive), bone undershoots (negative), soft is near zero.")}
{img("hu2_confusion.png", f"HU-tissue confusion (row = true GT class, col = what the sCT renders it as). True air is kept as air only {air_keep:.0f}% of the time, becoming soft {air_soft:.0f}%; true bone is kept as bone {hb_keep:.0f}%, becoming soft {hb_soft:.0f}%.")}
<div class="key"><b>By HU, the failure is bone, not air.</b> Air is handled well: inside true-air voxels the sCT predicts
{ho['air']['pred_hu']:.0f} HU vs GT {ho['air']['gt_hu']:.0f} HU (only a mild {ho['air']['bias']:+.0f} HU overshoot),
and {air_keep:.0f}% of true-air voxels still read as air on the sCT ({air_soft:.0f}% leak to soft). Soft tissue is
also well calibrated (bias {ho['soft']['bias']:+.0f} HU). <b>Bone is the outlier</b>: HU-defined bone (&gt; 150 HU)
undershoots by {ho['bone']['bias']:+.0f} HU (GT {ho['bone']['gt_hu']:.0f} → pred {ho['bone']['pred_hu']:.0f}), so
{hb_soft:.0f}% of true-bone voxels drop below 150 HU and read as soft tissue, with a visible pile-up at the
sigmoid's +1024 HU cap (Fig hu3, hu4). So the model regresses the dense extreme toward soft-tissue density while
leaving air and soft essentially correct.</div>

<h2>6. Verification: is it really density, or localization? (four falsification tests)</h2>
<p>The segmenter is HU-driven, so we deliberately test the localization-vs-density question with experiments designed
to <i>break</i> the claim, including two that do not use the segmenter at all.</p>
{img("vf1_verification.png", "Left: AUC of HU separating bone vs non-bone, real CT vs sCT (chance = 0.5). Right: fraction of the bone-Dice gap recovered by a pure intensity recalibration — for an HU-threshold task (segmenter-free) vs for the CNN segmenter.")}
<h3>Test A — is bone still spatially separable by HU? (segmenter-free, 207 subjects)</h3>
<p>If the model put bone in the right place but too low in value, sCT HU should still rank true-bone above non-bone
nearly as well as the real CT. It does: AUC <b>{auc_sct:.3f}</b> (sCT) vs <b>{auc_gt:.3f}</b> (real CT) =
<b>{auc_ret:.0f}%</b> of the above-chance separability retained. <b>Gross bone localization is intact.</b></p>
<h3>Test B — does a pure recalibration fix an HU-threshold bone task? (segmenter-free)</h3>
<p>Thresholding HU at 150 gives bone Dice {thr_sct:.2f} on the sCT vs {thr_real:.2f} on real CT; retuning the single
threshold (best ≈ {vd.t_best.mean():.0f} HU) lifts the sCT to {thr_best:.2f}, recovering <b>{rec_thr:.0f}%</b> of the
gap. For a threshold/HU task the error is largely a recalibratable scale problem.</p>
<h3>Test C — causal oracle: does fixing the intensity recover the CNN segmenter? (GPU, 30 subjects)</h3>
<p>We apply a monotonic histogram-match of the sCT to the GT CT — which cannot move anything spatially, only remaps
values (verified to actually lift bone HU, e.g. 128→161 on one case) — and re-run the segmenter. Bone Dice goes
{rec_sc:.3f} → {rec_re:.3f} (ceiling {rec_ce:.3f}): recovery <b>{rec_cnn:.0f}%</b>, i.e. essentially none. The reason
is established independently in §0: the segmenter uses instance norm and is invariant to absolute HU (identical Dice
under different intensity scalings). <b>So the segmenter's bone-Dice drop is not caused by the HU undershoot</b> — it is
structural, which a recalibration cannot repair.</p>
<h3>Test D — is the structural defect actually blur? (measured, 25 subjects)</h3>
<p>To name the structural defect rather than infer it, we histogram-match the sCT to the GT (removing the magnitude
difference) and measure gradient magnitude on the GT-bone surface. If the magnitude-matched sCT still has softer bone
edges, that is genuine blur. It does: bone-edge sharpness is <b>{blur_overall:.2f}×</b> the real CT's overall. The
deficit tracks the Dice drop region-wise — <b>pelvis ≈ {blur_pelvis:.2f}</b> (no blur, and the smallest Dice drop,
0.04) while thorax/abdomen/brain sit at 0.75–0.76 (blur, larger drops). So the segmenter loss is lost high-frequency
bone detail (cortical edges), confirmed not inferred.</p>
{img("vf2_blur.png", "Bone-edge sharpness of the magnitude-matched sCT relative to real CT (1.0 = as sharp; <1 = blurrier). Pelvis is sharp (and barely drops in Dice); other regions lose ~20-25% edge sharpness.")}
<div class="warn"><b>Correction.</b> An earlier version concluded "bone fails purely by density; the segmenter relabels
because HU is low." Test C refutes the causal link: a perfect intensity fix recovers {rec_cnn:.0f}% of the segmenter
gap. The corrected reading is two independent defects — an HU undershoot (density, recalibratable, dominant for
dose/threshold tasks) and a fine-structure/texture degradation (what the CNN sees, not recalibratable).</div>

<h2>7. Conclusion</h2>
<ol>
<li><b>Soft tissue and air are fine</b> (soft Dice drop {c['soft']['gap']:.2f}; air kept {air_keep:.0f}% as air,
{ho['air']['bias']:+.0f} HU; soft {ho['soft']['bias']:+.0f} HU).</li>
<li><b>Gross bone localization is intact.</b> sCT HU separates bone from non-bone at AUC {auc_sct:.2f} vs {auc_gt:.2f}
ceiling ({auc_ret:.0f}% retained) — the model puts bone in roughly the right place.</li>
<li><b>Bone fails in two independent ways.</b> (a) <b>Density:</b> HU undershoot of {bh['bias']:.0f} HU (bone ROI) /
{ho['bone']['bias']:+.0f} HU (HU&gt;150); recalibrating a threshold recovers {rec_thr:.0f}% of an HU-task's gap, so it
is a scale/calibration error and the clinically dominant one. (b) <b>Fine structure:</b> the CNN segmenter loses bone
(Dice {c['bone']['ceiling']:.2f}→{c['bone']['sct']:.2f}); a pure intensity fix recovers only {rec_cnn:.0f}%, and
magnitude-matched bone edges are {blur_overall:.2f}× as sharp as real CT, so this is a sharpness/blur degradation, not
an HU effect.</li>
<li><b>Implication.</b> For dose/HU accuracy, fix the bone <i>calibration</i> (magnitude). For a segmenter or any
edge/texture-dependent downstream use, calibration alone will not help — the sCT's bone <i>structure</i> needs to be
sharper. The two require different interventions.</li>
</ol>
<div class="sub">Reproduce: <code>seg_infer.py</code> → <code>seg_extract.py</code> → <code>seg_aggregate.py</code> →
<code>seg_figures.py</code>; HU view <code>hu_tissue.py</code>; verification <code>verify_density.py</code>,
<code>verify_recalib_reseg.py</code>, <code>verify_blur.py</code>, <code>verify_figure.py</code>; assemble <code>seg_report.py</code>
(in <code>src/evaluate/unet_failure/</code>).</div>
</body></html>"""


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        f.write(HTML)
    print("[seg_report] wrote", OUT)


if __name__ == "__main__":
    main()
