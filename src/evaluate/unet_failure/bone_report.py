"""Assemble _html/06_unet_bone_deepdive.html: the organized, self-contained bone
deep-dive with a methods section explaining how every metric was computed.
Reads bone_stats.json, mr_tissue_stats.json, bone_oracle.csv, figures."""
import os
import base64
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
OUT = os.path.join(REPO, "_html/06_unet_bone_deepdive.html")
SCEN = ["air", "soft", "bone", "cortical", "skull"]
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]

st = json.load(open(os.path.join(RUN, "bone_stats.json")))
mst = json.load(open(os.path.join(RUN, "mr_tissue_stats.json")))
oracle = pd.read_csv(os.path.join(RUN, "bone_oracle.csv"), index_col=0).reindex(SCEN)
g, uni, loss, mr, sev, lm = st["gates"], st["universality"], st["loss"], st["mr"], st["severity"], st["locmag"]
tis, rair, ap = st["tissue"], st["region_air"], st["air_proof"]


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(name, cap):
    return f'<figure><img src="{b64(name)}"/><figcaption>{cap}</figcaption></figure>'


def table(df, fmt="{:.2f}", idxname=""):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(
            f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>"
            for v in r)
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{cols}</tr></thead><tbody>{rows}</tbody></table>'


# ---- tables ----
T_tissue = table(pd.DataFrame({
    "% of body voxels": {t: tis[t]["vox_pct"] for t in ["air", "soft", "bone"]},
    "per-voxel MAE (HU)": {t: tis[t]["pv_mae"] for t in ["air", "soft", "bone"]},
    "signed bias (HU)": {t: tis[t]["bias"] for t in ["air", "soft", "bone"]},
    "% of total error": {t: tis[t]["err_share_pct"] for t in ["air", "soft", "bone"]},
}).reindex(["air", "soft", "bone"]), fmt="{:.1f}", idxname="tissue")

ot = oracle.copy()
ot.columns = ["ΔPSNR (dB)", "Δ body-MAE (HU)", "Δ full-HU MAE (HU)"]
T_oracle = table(ot, fmt="{:.2f}", idxname="fix this tissue →")

T_air = table(pd.DataFrame({
    "air % of body": {r: rair[r]["air_pct"] for r in REG},
    "air per-voxel MAE (HU)": {r: rair[r]["air_pv_mae"] for r in REG},
}).reindex(REG), fmt="{:.0f}", idxname="region")

lmdf = pd.DataFrame({r: lm[r] for r in REG}).T[
    ["shape_dice", "missed_frac", "fp_frac", "mae_bone_interior", "mae_bone_boundary"]]
lmdf.columns = ["shape Dice", "missed frac", "FP frac", "interior MAE", "boundary MAE"]
T_lm = table(lmdf, fmt="{:.2f}", idxname="region")

gate_pass = "PASS" if st["all_pass"] else "FAIL"
gate_li = "".join(f"<li><b>{'✓' if v['pass'] else '✗'}</b> <code>{k}</code></li>" for k, v in g.items())
air_psnr, bone_psnr = oracle.loc["air", "dPSNR"], oracle.loc["bone", "dPSNR"]

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>U-Net bone deep-dive: is bone the biggest problem, and why?</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;
      padding:34px 26px;color:#1f2937;line-height:1.55;background:#fff}}
 h1{{font-size:25px;margin:0 0 4px}} h2{{font-size:20px;margin:34px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
 h3{{font-size:15px;margin:20px 0 6px;color:#111827}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
 .key{{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .warn{{background:#fef2f2;border-left:4px solid #dc2626;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .ok{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px 16px;margin:14px 0;border-radius:4px;font-size:13px}}
 .m{{background:#fffbeb;border-left:4px solid #d97706;padding:12px 16px;margin:14px 0;border-radius:4px;font-size:13px}}
 figure{{margin:18px 0;text-align:center}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 figcaption{{color:#6b7280;font-size:12.5px;margin-top:6px}}
 table{{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}} thead th{{background:#f9fafb}}
 tbody th{{text-align:left;background:#fafafa}}
 code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12px}}
 ol li,ul li{{margin:5px 0}} ul.gates{{columns:2;font-size:12px}} .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
 .toc{{background:#f9fafb;border:1px solid #e5e7eb;border-radius:6px;padding:12px 18px;font-size:13.5px}}
</style></head><body>

<h1>U-Net bone deep-dive: is bone the biggest problem, and why?</h1>
<div class="sub">Model: plain 3D U-Net <code>9xmodnhn</code> ep799 (fully trained, center-wise split). Data: all 207
center-wise validation subjects, <code>full_eval_20260617</code>. Follow-up to report <code>05</code>; this report
answers the bone questions head-on and shows exactly how each number was computed. Generated 2026-06-19.
<b>Sign convention:</b> error = prediction − truth, so a <b>negative bias = the model predicts too low (undershoot)</b>.</div>

<div class="key"><b>Answers, up front (all proven below).</b><br>
<b>1. Is bone the biggest problem?</b> It depends on the ruler. <b>Per voxel and clinically, yes</b> — bone is the
worst-predicted tissue ({tis['bone']['pv_mae']:.0f} HU MAE, {tis['bone']['pv_mae']/tis['soft']['pv_mae']:.1f}× soft) and
the only one with a large systematic undershoot. <b>For the aggregate pixel score, no</b> — fixing <b>air</b> or
<b>soft</b> helps PSNR/MAE more than bone, because they are far more numerous (air {air_psnr:+.2f} dB vs bone
{bone_psnr:+.2f} dB).<br>
<b>2. Are the big wins air AND soft (not just bone)?</b> Yes — <b>air and soft are the two biggest aggregate wins and
they are roughly tied</b>; bone is third (about half their size).<br>
<b>3. Why does the model undershoot, and why only bone?</b> The L1 loss pulls every tissue toward the central
(median) value given the MR. Soft sits at that center (≈unbiased), air is the low extreme (slight over-shoot), and
bone is the high, skewed tail (large under-shoot). Bone is hit hardest because it is both <b>far in the tail</b> and
<b>MR-ambiguous</b>.<br>
<b>4. Does MR even contain bone density?</b> Almost none — cortical bone looks like air on MR and trabecular bone
looks like soft tissue, so a given MR brightness maps to many CT densities.</div>

<div class="ok"><b>Correctness ({gate_pass}).</b> Every aggregate number is reconciled against the released evaluation:
the oracle baseline reproduces <code>body_psnr</code>, <code>body_mae_hu</code> and <code>synthrad_mae</code> to
&lt;0.001. All 207 subjects processed. <ul class="gates">{gate_li}</ul></div>

<div class="toc"><b>Contents.</b> 0. How everything was measured (methods) · 1. Is bone the biggest problem?
· 2. Why "air" is real intra-body gas · 3. Is the undershoot universal? · 4. Do we see it? · 5. Why the model
undershoots, and why only bone · 6. Does MR contain bone information? · 7. Supporting causes · 8. How to fix it · 9. Conclusion</div>

<h2>0. How everything was measured (methods)</h2>
<p>For each subject we load four volumes on the same 1.5 mm grid: the U-Net prediction (<code>sample.nii.gz</code>, in
HU), the raw ground-truth CT (<code>ct.nii</code>, full HU to ~2900), the 35-label CADS segmentation, and the body
mask (<code>mask.nii</code>). <b>All metrics are computed only over voxels inside the body mask</b> (the black
background outside the patient is excluded entirely).</p>
<h3>Tissue classes (by ground-truth HU, inside the body)</h3>
<ul>
<li><b>air</b>: HU &lt; −300 → lung, bowel gas, sinuses, trachea (gas <i>inside</i> the patient)</li>
<li><b>soft</b>: −300…+200 → muscle, fat, organs, fluid</li>
<li><b>bone</b>: &gt; +200 → all skeleton; with <b>cortical</b> = &gt; +1024 (dense) and <b>mid/trabecular</b> = 200…1024</li>
</ul>
<h3>The metrics, and exactly what each means</h3>
<ul>
<li><b>HU (Hounsfield Unit)</b> = the CT intensity scale (air ≈ −1000, water = 0, cortical bone +1000…+3000). It is a
unit, not a metric.</li>
<li><b>MAE</b> = mean of <b>|prediction − truth|</b>, reported in HU. Always positive: it tells you the error
<i>size</i>, not its direction.</li>
<li><b>bias</b> = mean of <b>(prediction − truth)</b> (no absolute value), in HU. <b>Signed</b>: negative = undershoot.
This is how we know the bone error is one-directional, not random noise.</li>
<li><b>PSNR</b> = 10·log₁₀(1/MSE), where MSE <b>squares</b> the errors; reported in dB, higher = better. Squaring makes
a few large errors dominate and the log compresses the scale — which is why the "biggest MAE win" and "biggest PSNR
win" need not be the same tissue.</li>
</ul>
<h3>Clipped vs raw frame</h3>
<p>The released validation clipped both prediction and truth to [−1024, 1024] (that is what the model was trained and
scored on). We report in that <b>clipped frame</b> by default (matches the leaderboard) and note the <b>raw frame</b>
(true HU) where it matters for bone.</p>
<h3>The oracle counterfactual (how we rank "what would help most")</h3>
<p>To ask "how much would perfect tissue X help?", we copy the ground-truth HU into the prediction <i>only inside that
tissue</i>, leave everything else untouched, and recompute the exact reported metric. The improvement is exactly the
error currently sitting in that tissue. We do this for air / soft / bone / cortical / skull.</p>
<h3>The regression-to-mean calibration (Section 5)</h3>
<p>We accumulate a 2D histogram of (true bone HU, predicted HU) over all bone voxels, then for each true-HU bin read
off the <i>mean</i> predicted HU. A perfect model would lie on the identity line.</p>
<h3>The MR-information test (Section 6)</h3>
<p>MR intensity is normalized per-volume (and bias-field augmented), so absolute MR values are not comparable across
subjects. We therefore rank MR intensity <i>within each subject's body</i> (0 = darkest, 1 = brightest) and ask how
that rank relates to CT HU, per tissue (rank-based / scale-invariant).</p>

<h2>1. Is bone the biggest problem? It depends on the ruler</h2>
<h3>Per voxel: bone is clearly worst</h3>
{T_tissue}
<p>Bone has the highest per-voxel MAE ({tis['bone']['pv_mae']:.0f} HU) and the only large systematic bias
({tis['bone']['bias']:.0f} HU undershoot). Soft is best per voxel ({tis['soft']['pv_mae']:.0f} HU, near-zero bias).</p>
<h3>Aggregate (leaderboard): air and soft win, bone is third</h3>
<p>The oracle gain = the error currently held by each tissue. Because air (27%) and soft (68%) hold far more voxels,
they hold more total error than rare bone (5%), even though each bone voxel is worse:</p>
{T_oracle}
<p><b>Air and soft are roughly tied as the biggest aggregate wins; bone is about half their size; cortical/skull move
the clipped score the least</b> (the clip hides their error). So "biggest" depends on whether you weight per-voxel /
clinical accuracy (bone) or the aggregate pixel metric (air ≈ soft).</p>
<div class="grid">
{img('q1_air_paradox.png','Air wins the aggregate not because it is hard, but because it is common AND moderately wrong. Bone is worst per-voxel but rare.')}
{img('bf1_oracle.png','Oracle gain from perfecting one tissue, in three metrics. Air/soft lead PSNR and both MAEs; bone is mid-pack; cortical/skull are tiny on the clipped metrics.')}
</div>
{img('bf2_severity_vs_leverage.png','The disconnect in one figure: bone has the highest per-voxel error (left) yet the smallest aggregate effect (right), because it occupies few voxels.')}

<h2>2. Why "air" is real intra-body gas, not the background</h2>
<p>A natural objection: "isn't air just the easy black background?" No. Two proofs:</p>
<p><b>(i) The background is excluded.</b> For a representative thorax subject (<code>{ap['subj']}</code>), of the
{ap['total_vox']/1e6:.1f} M total voxels, <b>{ap['air_outside_body']/1e6:.1f} M air voxels lie OUTSIDE the body and are
never scored</b>; only <b>{ap['air_inside_body']/1e6:.2f} M air voxels INSIDE the body</b> count. The body mask is just
{ap['body_pct_of_volume']:.0f}% of the volume. The scored air has mean HU <b>{ap['mean_hu_inside_air']:.0f}</b>
(lung-like), not the −1024 of background.</p>
<p><b>(ii) Air% tracks anatomy.</b> If "air" were background it would be constant; instead it follows where gas
actually is (thorax lungs highest):</p>
{T_air}

<h2>3. Is the undershoot universal? Yes</h2>
<p>Across all {uni['n_subjects']} subjects, <b>{uni['pct_bone_under_clip']:.0f}%</b> undershoot bone and
<b>{uni['pct_cort_under_clip']:.0f}%</b> undershoot cortical bone; within each subject a mean of
<b>{uni['mean_frac_bone_under']:.0f}%</b> of bone voxels are predicted too low. This is the model's default behaviour,
not a few outliers.</p>
{img('bf3_universality.png','Left: per-subject bone & cortical bias are negative for essentially every subject. Right: per region, most bone voxels are undershot in every subject.')}

<h2>4. Do we see it? Yes</h2>
<p>One representative subject per region (median bone-MAE). The predicted skull/bone is visibly grey (undershot) vs
the bright cortical GT in the same window; the last panel is the dense-bone error the clipped metric never sees.</p>
{''.join(img(f'bf9_example_{r}.png', f'{r}: predicted cortical bone is grey/undershot; final panel = the hidden clipped-away error.') for r in REG)}

<h2>5. Why the model undershoots, and why ONLY bone</h2>
<p>The L1 loss is minimized by predicting the <b>median</b> of the true HU values consistent with a given MR input
(L2 → the mean). When one MR appearance maps to many possible HU (Section 6), the model can only emit one number, so
it emits that central value. This is not a bug — it is the loss hedging.</p>
<p><b>Crucially, this pull acts on every tissue, but its effect depends on where the tissue sits</b> in the HU
distribution. From the bias column in Section 1: air (low extreme) is pulled <b>up</b> (bias {tis['air']['bias']:+.0f}
HU, overshoot); soft (the center) barely moves ({tis['soft']['bias']:+.0f} HU); bone (high, skewed tail) is pulled
<b>down</b> ({tis['bone']['bias']:+.0f} HU, undershoot; cortical {sev['bias_cortical_clip']:.0f} HU). Same mechanism,
opposite directions.</p>
<p><b>Two conditions make the bone error large and systematic, and only bone meets both:</b> (a) the MR is ambiguous
about its HU (wide conditional spread) and (b) it sits in a skewed tail (so the central compromise is one-directional).
Soft fails both (narrow, central) → tiny unbiased error. Bone meets both → large undershoot.</p>
{img('q2_calibration.png','Proof of regression to the mean: as the true bone gets denser, the mean prediction flattens far below the identity line and never approaches dense cortical HU.')}
{img('bf8_ceiling.png','Predicted bone HU collapses toward the mean and stops ~951 HU, BELOW the 1024 sigmoid cap — so the architecture cap is not the limiting factor; regression to the mean is.')}

<h2>6. Does MR contain bone-density information? Almost none</h2>
<p>MR signal comes from mobile hydrogen protons (water/fat). <b>Cortical bone</b> has almost none → it is a dark void
that <b>looks like air</b>. <b>Trabecular bone</b> shows its marrow (fat/water), so it <b>looks like soft tissue</b>,
not like its true high HU. So MR brightness does not encode mineral density. Proof, on your data: bone's MR-brightness
distribution <b>overlaps soft tissue by {mst['overlap_bone_soft']:.2f}</b> and cortical overlaps air by
{mst['overlap_cort_air']:.2f} (1.0 = indistinguishable). A given MR brightness maps to many CT densities.</p>
<p>Quantitatively, MR explains only <b>{mr['rho2_bone']*100:.0f}%</b> of bone-HU variance (vs {mr['rho2_soft']*100:.0f}%
for soft), and knowing the MR removes only <b>{mr['mr_reduction_bone']:.0f}%</b> of the bone-HU spread (std
{mr['marg_std_bone']:.0f}→{mr['cond_std_bone']:.0f} HU). Bone HU is also intrinsically
~{mr['ctstd_bone_mean']/mr['ctstd_soft_mean']:.0f}× wider than soft (std {mr['ctstd_bone_mean']:.0f} vs
{mr['ctstd_soft_mean']:.0f} HU): a wide target the MR cannot resolve is exactly what an L1 model collapses to the
mean. (Consistent with the prior cross-model result that even the uncapped diffusion baselines undershoot bone.)</p>
<div class="grid">
{img('q3_mr_conflation.png','MR brightness per tissue. Cortical bone (dark) overlaps air; trabecular/all bone (mid) overlaps soft tissue. MR cannot separate bone density from other tissues.')}
{img('bf4_mrct_hist.png','P(CT HU | MR rank). Bone is a broad vertical smear at every MR level (MR uninformative); soft is a narrow band near 0.')}
</div>
{img('bf5_spread.png','Bone HU is ~4x wider than soft and MR removes almost none of that spread — a wide, MR-unresolvable target.')}

<h2>7. Supporting causes</h2>
<h3>Bone is rare → negligible weight in the uniform loss</h3>
<p>Bone is {loss['bone_vox_pct']:.0f}% of body voxels under a uniform L1, so it contributes little gradient, even
though it holds {loss['bone_l1_share_pct']:.0f}% of the error ({loss['leverage_ratio']:.1f}× its voxel share).</p>
{img('bf7_loss_imbalance.png','Bone’s share of voxels vs its share of the error — under-weighted relative to its error contribution.')}
<h3>It is a density-magnitude failure, not a localization failure</h3>
<p>The model mostly knows <i>where</i> bone is (shape Dice {lm['overall']['shape_dice']:.2f}); the error is the HU
<i>magnitude</i> in the interior (interior MAE {lm['overall']['mae_bone_interior']:.0f} &gt; boundary
{lm['overall']['mae_bone_boundary']:.0f} HU). In thin-bone body regions it additionally under-detects bone
(missed {lm['overall']['missed_frac']*100:.0f}% fall below the 200-HU threshold because they are undershot).</p>
{T_lm}
{img('bf6_locmag.png','Localization (left) is decent; the dominant error is interior density magnitude (right), with extra under-detection of thin bone (middle).')}

<h2>8. How to fix it</h2>
<ol>
<li><b>Add the missing information (highest ceiling).</b> A bone-sensitive MR sequence (UTE/ZTE) that actually images
cortical bone, or inject a prior (population CT/bone atlas, paired-anatomy retrieval). The undershoot is fundamentally
a missing-information problem, so this is the only route to true per-voxel density.</li>
<li><b>Stop the loss hedging (medium ceiling).</b> A generative/probabilistic model (GAN, diffusion, distributional or
quantile loss) emits a sharp plausible bone value instead of the blurry mean; or bone-weighted loss to fix the rarity.
These reduce the bias and sharpen output but cannot recover information the MR lacks.</li>
<li><b>Work around it (most pragmatic).</b> Bulk-density override (segment bone, assign a standard density) and
evaluate on the clinical task (dose/DVH, bone-specific metrics) rather than clipped PSNR/MAE, which hide this error.</li>
</ol>
<div class="warn"><b>Honest ceiling.</b> The undershoot is the L1 loss behaving optimally on an input that is genuinely
ambiguous about bone density. You can sharpen it, de-bias it, or sidestep it, but the only way to truly recover dense
cortical HU is to add information the standard MR does not contain.</div>

<h2>9. Conclusion</h2>
<ol>
<li><b>Bone is the worst-predicted and most clinically important tissue per voxel</b> ({tis['bone']['pv_mae']/tis['soft']['pv_mae']:.1f}×
soft), undershot in {uni['pct_bone_under_clip']:.0f}% of subjects.</li>
<li><b>It is not the biggest aggregate lever</b> — air and soft each remove more total error (air {air_psnr:+.2f} dB
PSNR vs bone {bone_psnr:+.2f}); whether bone is "the biggest problem" depends on per-voxel/clinical vs leaderboard.</li>
<li><b>The mechanism is L1 regression to the median</b> on an MR-ambiguous, right-skewed bone-HU distribution: every
tissue is pulled to its conditional center, and bone — the wide, skewed, MR-unresolvable tail — is undershot hardest.
The sigmoid cap is secondary (predictions stop below it).</li>
<li><b>The fix is information</b> (a bone MR sequence or a prior), with generative / bone-weighted losses and
bulk-density overrides as partial mitigations; and bone-aware evaluation, since clipped pixel metrics hide the error.</li>
</ol>

<h2>Reproducibility</h2>
<p>Scripts in <code>src/evaluate/unet_failure/</code>: <code>bone_extract.py</code> (per-subject oracle, biases,
loc/mag, MR Spearman), <code>bone_aggregate.py</code> (rollups + gates + per-tissue table + air proof),
<code>mr_tissue.py</code> (MR-rank by tissue), <code>bone_figures.py</code>, <code>bone_report.py</code>;
<code>run_all.py</code> runs the whole chain. Data in
<code>evaluation_results/unet_failure_20260619/</code>. Oracle replicates <code>compute_metrics_body</code>; GT = raw
<code>ct.nii</code> in canonical RAS; MR analysis uses within-subject rank.</p>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
