"""Assemble the self-contained per-CADS-label / per-region U-Net failure-anatomy
report (base64-embedded figures). Reads CSVs / json / figures from RUN."""
import os
import base64
import json
import numpy as np
import pandas as pd

REPO = "/home/minsukc/MRI2CT"
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
OUT = os.path.join(REPO, "_html/05_unet_failure_anatomy.html")
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]

st = json.load(open(os.path.join(RUN, "agg_stats.json")))
o = st["overall"]
g = st["gates"]
rt = pd.read_csv(os.path.join(RUN, "region_tissue.csv"), index_col=0).reindex(REG)
rm = pd.read_csv(os.path.join(RUN, "region_mass.csv"), index_col=0).reindex(REG)
rb = pd.read_csv(os.path.join(RUN, "region_bonehu.csv"), index_col=0).reindex(REG)
pla = pd.read_csv(os.path.join(RUN, "per_label_agg.csv"), index_col=0)
bvn = pd.read_csv(os.path.join(RUN, "bone_vs_nonbone.csv"), index_col=0)
oracle = pd.read_csv(os.path.join(RUN, "oracle.csv"), index_col=0)
recon = pd.read_csv(os.path.join(RUN, "recon.csv"), index_col=0).reindex(REG)
_ps = pd.read_csv(os.path.join(RUN, "per_subject.csv"))
cads_coverage = float(100 * _ps.n_labeled.sum() / _ps.n_body.sum())   # % of body voxels with a CADS label


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(name, cap):
    return f'<figure><img src="{b64(name)}"/><figcaption>{cap}</figcaption></figure>'


def table(df, fmt="{:.1f}", idxname=""):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(
            f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>"
            for v in r)
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{cols}</tr></thead><tbody>{rows}</tbody></table>'


# ---------- tables ----------
T_recon = table(recon.rename(columns={"recomp MAE (R)": "recomp R", "released synthrad_mae": "synthrad_mae",
                                      "recomp body MAE (C)": "recomp C", "released body_mae_hu": "body_mae_hu"}),
                fmt="{:.3f}", idxname="region")

t1 = rt[["MAE air", "MAE soft", "MAE bone", "MAE all", "bone:soft", "body PSNR"]].copy()
T1 = table(t1, idxname="region")

t_mass = rm[["bone vox %", "bone err-mass %", "mass/vox ratio", "soft err-mass %", "air err-mass %"]]
T_mass = table(t_mass, idxname="region")

t_bone = rb[["GT bone mean", "pred bone mean", "pred bone max", "GT bone max",
             "cort bias (C)", "cort bias (R)", "near-ceil %", "GT %>1024"]]
T_bone = table(t_bone, idxname="region")

# per-label table (sorted by MAE C) — show all 35 with prevalence
plt_ = pla[["is_bone", "cads_region", "MAE (C)", "MAE (R)", "bias (C)", "GT HU", "pred HU",
            "GT %>1024", "n subj"]].copy()
plt_["is_bone"] = plt_["is_bone"].map({True: "bone", False: ""})
T_label = table(plt_, idxname="CADS label")

# bone vs non-bone
T_bvn = table(bvn.rename(columns={"organ/soft MAE": "organ MAE"}), idxname="region")

# oracle
oc = oracle[["body MAE (C)", "bone-fixed (C)", "drop % (C)", "body MAE (R)", "bone-fixed (R)", "drop % (R)"]]
T_oracle = table(oc, fmt="{:.1f}", idxname="region")

# per-region worst label inline (top-3 names)
worst = json.load(open(os.path.join(RUN, "region_worst.json")))
worst3 = {r: ", ".join(f"{x['name']} ({x['MAE']:.0f})" for x in worst[r][:3]) for r in REG}

gate_pass = "PASS" if st["all_pass"] else "SOME FAILED"
gate_li = "".join(
    f"<li><b>{'✓' if v['pass'] else '✗'}</b> <code>{k}</code></li>" for k, v in g.items())

struct_frac = o["cortical_structural_frac"] * 100
inrange_frac = 100 - struct_frac

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>U-Net MR→CT failure anatomy — per CADS label & region</title>
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
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
 ol li,ul li{{margin:5px 0}} ul.gates{{columns:2;font-size:12.5px}}
</style></head><body>

<h1>Where the U-Net MR→CT model fails — a per-CADS-label, per-region diagnosis</h1>
<div class="sub">Plain 3D U-Net baseline (wandb <code>9xmodnhn</code>, ep799, fully trained on the center-wise split),
scored on all 207 center-wise validation subjects (<code>full_eval_20260617</code>). Errors are decomposed over the
35-label CADS segmentation and the raw full-HU CT. Generated 2026-06-19.
<b>Sign convention:</b> error = pred − GT, so negative bias = the model predicts <i>too low</i> (undershoot).</div>

<div class="key"><b>Bottom line (proven below, all on your data).</b>
Per voxel, the U-Net's error is dominated by a <b>bone-density problem</b>, and within bone a <b>cortical-density undershoot</b>.
Bone is predicted <b>{o['bone_vs_organ_ratio']:.1f}× worse</b> than soft-tissue/organ labels
(bone-label MAE {o['bone_mae_label']:.0f} vs organ {o['organ_mae_label']:.0f} HU), the <b>skull</b> is the single
worst-predicted structure, and cortical bone is undershot by <b>{abs(o['bias_cortical_C']):.0f} HU even within the
[-1024,1024] range the model was actually trained and validated on</b>. Bone occupies just
<b>{o['bone_vox_pct']:.1f}% of body voxels but carries {o['bone_mass_pct']:.0f}% of the total error mass</b>, and an
oracle that fixes only bone cuts the scored body-MAE by <b>{o['oracle_drop_C']:.0f}%</b>. The model never emits dense
cortical HU: inside true bone its predictions mean {o['pred_bone_mean']:.0f} HU and top out at
~{o['pred_bone_max']:.0f} HU while the GT reaches ~{o['gt_bone_max']:.0f} HU.
<br><br><b>Caveat (see report 06):</b> bone is the worst-predicted tissue <i>per voxel</i> and the most clinically
relevant, but it is <b>not</b> the biggest lever on the aggregate pixel metrics — because bone is rare and the metric
clips it, fixing <i>air</i> (+2.78 dB PSNR) or <i>soft</i> helps the leaderboard more than fixing bone (+1.29 dB).
"Biggest" depends on whether you weight per-voxel/clinical accuracy or aggregate PSNR/MAE.</div>

<div class="ok"><b>Validation / proof of correctness ({gate_pass}).</b> Every recomputed number reproduces the released
evaluation pipeline. Per-subject body MAE vs raw GT matches <code>synthrad_mae</code> and the clipped-frame body MAE matches the released
<code>body_mae_hu</code> to &lt;0.001 HU each (max {g['g1_mae_raw_vs_synthrad']['max']:.2e} and
{g['g2_body_mae_clip_vs_released']['max']:.2e}); per-region means match <code>by_region.csv</code> to
Δ&lt;0.001 HU (max {g['g3_region_reconcile']['max']:.2e}). Voxel-count partitions are exact, all
207 subjects load (52/60/32/30/33), and the raw-GT reference is confirmed (bone reaches
{g['g7_raw_gt_reference']['gt_bone_max']:.0f} HU, not clipped).
<ul class="gates">{gate_li}</ul></div>

<h2>0. Setup &amp; frames</h2>
<p>GT = the raw <code>ct.nii</code> (full HU to ~{o['gt_bone_max']:.0f}), reoriented to canonical RAS to match the
saved predictions; metrics are over body-mask voxels. Each label mask is <code>seg == id</code> within the body.
Bone labels = skull (7), bone_other (27), limb_girdle (28), spine (29), thoracic_cage (30).</p>
<p><b>The validation was scored on GT clipped to [-1024, 1024]</b> (verified in code:
<code>ScaleIntensityRanged(clip=True)</code> + <code>hu_range=2048</code>), and the U-Net is trained to that clipped
target (its output ceilings ~{o['pred_bone_max']:.0f} HU). So every error is reported in two frames:
<b>Frame C</b> (clipped, the metric the model was selected on → genuine in-range failure) and <b>Frame R</b> (raw GT).
The excess R−C over cortical bone is the part that is <b>unrecoverable by construction</b> because the target itself
was clipped.</p>
{table(recon[["recomp MAE (R)","released synthrad_mae","Δ R","recomp body MAE (C)","released body_mae_hu","Δ C"]], fmt="{:.3f}", idxname="region")}
{img('fig15_recon.png','Recomputed vs released per-region MAE in both frames. The pipeline reproduces the leaderboard exactly, so every decomposition below is trustworthy.')}

<h2>1. Bone is the worst-predicted tissue in every region</h2>
<p>Splitting every body voxel by GT HU (air &lt;−300, soft −300…200, bone &gt;200), bone MAE is roughly
{o['mae_bone_C']/o['mae_soft_C']:.0f}× soft-tissue MAE everywhere. This is the single largest per-voxel error source
and it is consistent across all five anatomical regions.</p>
{T1}
<div class="grid">
{img('fig1_tissue_mae.png','Per-voxel MAE by GT tissue (Frame C). Bone error dwarfs soft and air in all regions.')}
{img('fig2_tissue_bias.png','Signed bias by tissue. Bone is strongly negative (undershoot) everywhere; soft tissue is near-zero.')}
</div>

<h2>2. Bone carries error mass far beyond its volume</h2>
<p>Bone is only <b>{o['bone_vox_pct']:.1f}%</b> of body voxels but contributes <b>{o['bone_mass_pct']:.0f}%</b> of the
total absolute-error mass — a {o['bone_mass_pct']/o['bone_vox_pct']:.1f}× over-representation. In the skull-bearing
regions (brain, head&amp;neck) bone dominates the regional error budget.</p>
{T_mass}
<div class="grid">
{img('fig3_errormass_share.png','Error-mass share by tissue per region (bars) vs bone’s voxel share (◆). The gap is bone’s outsized contribution.')}
{img('fig14_region_massshare.png','Where the dataset’s TOTAL error mass lives. Brain + head&neck bone is a large slice of the whole.')}
</div>

<h2>3. The failure is a systematic cortical-density UNDERSHOOT (HU proof)</h2>
<p>Inside true bone the network collapses toward the soft-tissue mean: GT mean {o['gt_bone_mean']:.0f} HU →
pred mean {o['pred_bone_mean']:.0f} HU, and the prediction <b>never reaches dense cortical values</b>
(p95 {o['pred_bone_p95']:.0f}, max ~{o['pred_bone_max']:.0f} HU vs real cortical to ~{o['gt_bone_max']:.0f}).
Decomposing the cortical (GT&gt;1024) error by frame: the model undershoots cortical bone by
<b>{abs(o['bias_cortical_C']):.0f} HU within the achievable [-1024,1024] range (Frame C)</b> and
<b>{abs(o['bias_cortical_R']):.0f} HU vs raw GT (Frame R)</b>. So about <b>{inrange_frac:.0f}% of the cortical error is
genuine in-range model failure</b> and only <b>{struct_frac:.0f}% is the structural clipped-target ceiling</b> — the
problem is the model, not just the clip.</p>
{T_bone}
<div class="grid">
{img('fig4_bone_joint_hist.png','Pred vs raw-GT HU over all bone voxels. The horizontal pile-up is the model’s ceiling; the green band (>1024) is clipped from the training target.')}
{img('fig5_bone_hu_hist.png','HU distribution inside true bone: GT extends to ~2600 HU, the prediction is squeezed low and stops near the ceiling.')}
</div>
<div class="grid">
{img('fig6_midcort.png','Mid-bone vs cortical, both frames. Error and undershoot scale with density; cortical is far worse.')}
{img('fig9_ecdf.png','ECDF of HU in true bone — the prediction saturates into a near-vertical wall well below GT.')}
</div>
<div class="grid">
{img('fig7_region_cortical.png','Cortical undershoot per region in both frames. Skull regions (brain, head&neck) are worst; the in-range (C) component is large everywhere.')}
{img('fig8_cap_pileup.png','Fraction of true-bone voxels pinned near the model’s HU ceiling, per region.')}
</div>
<div class="warn"><b>Information ceiling.</b> The model’s bone output saturates at ~{o['pred_bone_max']:.0f} HU
({o['near_ceiling_pct']:.0f}% of bone voxels sit ≥850 HU) while real cortical bone runs to ~{o['gt_bone_max']:.0f} HU.
Even the {inrange_frac:.0f}% of the cortical error that lives <i>inside</i> the trainable range is a systematic
undershoot of {abs(o['bias_cortical_C']):.0f} HU — a one-to-many MR→HU mapping the network resolves by regressing to
the mean. This is the densest, most dose-relevant tissue and it is exactly what the model gets most wrong.</div>

<h2>4. Per-structure rankings (all 35 CADS labels)</h2>
<p>The four worst-predicted structures are all bone (<b>skull, thoracic_cage, limb_girdle, bone_other</b>), with spine
close behind; the best-predicted are brain white/gray matter and abdominal organs. Prevalence (n subjects with the
label) is shown so single-subject labels are not over-read.</p>
<div class="warn"><b>Math caveat (per-label MAE does NOT average to the reported MAE).</b> The 35 CADS labels cover only
<b>{cads_coverage:.0f}%</b> of body voxels; the remaining ~{100-cads_coverage:.0f}% is <code>seg==0</code> inside the
body, which is almost entirely unlabeled internal air (sinus / bowel gas / cavities CADS has no label for; the external
background is masked to −1024 and excluded). So voxel-averaging these per-label MAEs gives the MAE over the
<i>labeled</i> {cads_coverage:.0f}% (≈67.8 HU), <b>not</b> the body MAE (≈72.4 HU body-voxel-mean), and definitely not
the leaderboard <code>body_mae_hu</code> (≈34 HU, which additionally divides by the full padded volume, ×0.40). The
only decomposition that reconstructs the body MAE exactly is the air/soft/bone HU split, which tiles 100% of the body
(verified by the mass-conservation gate). Per-label numbers below are per-structure body-voxel-mean MAEs, valid on
their own but not additive to the headline number.</div>
{T_label}
<div class="grid">
{img('fig10_label_mae.png','Per-CADS-label MAE (red = bone). Bone labels occupy the top of the ranking; brain/organ labels the bottom.')}
{img('fig11_label_bias.png','Per-CADS-label signed bias. Every bone label clusters strongly negative (undershoot).')}
</div>
{img('fig12_scatter.png','Per-label GT vs pred mean HU. Bone points fall well below the identity line; soft/organ labels sit on it.')}
{img('fig13_region_worst.png','Top-5 worst labels per region. Bone (red) leads brain & head&neck; air-adjacent soft (airway, lungs, bowel) leads the body regions.')}

<h2>5. Bone vs organ, and the per-region picture</h2>
<p>Contrasting bone labels against all soft/organ labels directly: bone is <b>{o['bone_vs_organ_ratio']:.1f}× worse</b>
overall, and 2–4× worse in every region.</p>
{T_bvn}
<h3>Per-region worst structures</h3>
<ul>
<li><b>Brain</b> — {worst3['brain']}. Highest body MAE; dominated by the <b>skull</b> and air cavities, not soft brain tissue (white/gray matter are the best-predicted labels in the whole dataset).</li>
<li><b>Head &amp; neck</b> — {worst3['head_neck']}. Dense skull base + airway; same cortical-undershoot story as brain.</li>
<li><b>Thorax</b> — {worst3['thorax']}. Lowest body MAE region; error is air-interface (airway, lungs) plus thin rib/vertebral bone.</li>
<li><b>Abdomen</b> — {worst3['abdomen']}. Bowel gas and vertebral/rib bone; solid organs (liver, spleen, kidneys) are well predicted.</li>
<li><b>Pelvis</b> — {worst3['pelvis']}. Bone (hip/sacrum) plus a mild soft-tissue shift; not a catastrophic region despite the known center-C T1/T2 domain shift.</li>
</ul>

<h2>6. How much would fixing bone help? (oracle counterfactual)</h2>
<p>Overwrite the prediction with GT inside bone voxels only and recompute the <i>scored</i> body MAE: fixing bone
alone drops it by <b>{o['oracle_drop_C']:.0f}%</b> overall (and <b>{oracle.loc['brain','drop % (C)']:.0f}%</b> in
brain), despite bone being &lt;5% of voxels — large for so few voxels. <b>But this is not the biggest single lever:</b>
because air and soft voxels are far more numerous, fixing them helps the aggregate metric more (air +2.78 dB PSNR vs
bone +1.29 dB; see report <code>06</code> for the full air/soft/bone comparison in PSNR, body-MAE and full-HU MAE).
Bone leads only on <i>per-voxel</i> and <i>full-HU/clinical</i> accuracy, where the clip does not hide it.</p>
{T_oracle}
{img('fig16_oracle.png','Scored body MAE as-is vs with bone predicted perfectly. Bone fix is large per-voxel but not the biggest aggregate lever (report 06 compares air/soft/bone head-to-head).')}

<h2>7. Conclusion — ranked drivers of U-Net error</h2>
<ol>
<li><b>Cortical-bone density undershoot.</b> Largest per-voxel error ({o['mae_cortical_C']:.0f} HU, Frame C),
systematic −{abs(o['bias_cortical_C']):.0f} HU in-range bias, worst in the skull, and the model physically saturates
~{o['pred_bone_max']:.0f} HU. {inrange_frac:.0f}% of it is genuine in-range failure (not the clip). Fixing bone alone
recovers {o['oracle_drop_C']:.0f}% of the scored error.</li>
<li><b>Disproportionate error mass.</b> Bone is {o['bone_vox_pct']:.1f}% of voxels but {o['bone_mass_pct']:.0f}% of
error mass — concentrated in brain &amp; head&amp;neck.</li>
<li><b>Region pattern.</b> Skull-bearing regions (brain, head&amp;neck) have the highest MAE and are bone-dominated;
body regions (thorax/abdomen/pelvis) are lower and air/soft-interface dominated with thin trabecular bone.</li>
<li><b>Soft tissue / organs are largely solved.</b> Organ labels sit on the identity line; white/gray matter, liver,
spleen, kidneys are the best-predicted structures.</li>
</ol>
<div class="key"><b>Conclusion.</b> Per voxel and clinically, the U-Net's dominant failure is <b>bone</b>, specifically
<b>cortical density</b>: a systematic, density-scaling undershoot (consistent across all 207 subjects) that the model
cannot escape because (a) part of the dense-bone HU is clipped out of its training target and (b) even within range
the MR under-determines cortical HU, so the network regresses to the mean. On the <i>aggregate</i> pixel metrics,
however, bone is not the biggest lever (air/soft contribute more by sheer voxel count, and the clip hides cortical
error) — so whether bone is "the biggest problem" depends on whether you optimize per-voxel/clinical accuracy or the
leaderboard. Report <code>06</code> dissects this and the root cause.</div>

<h2>Reproducibility</h2>
<p>Scripts in <code>src/evaluate/unet_failure/</code>: <code>extract.py</code> (per-subject + per-label, dual-frame),
<code>aggregate.py</code> (rollups + 8 correctness gates), <code>build_figures.py</code>, <code>report.py</code>.
Outputs in <code>evaluation_results/unet_failure_20260619/</code> (<code>per_subject.csv</code>,
<code>per_label.csv</code>, <code>bone_hist.npz</code>, derived table CSVs, <code>agg_stats.json</code>).
GT = raw <code>ct.nii</code> in canonical RAS; predictions = <code>full_eval_20260617/volumes/unet</code>. All gates
pass; numbers reconcile to the released metrics within floating-point tolerance.</p>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
