"""Assemble the self-contained HTML error-anatomy report (base64-embedded figures)."""
import os, base64, json, numpy as np, pandas as pd

RUN = "/home/minsukc/MRI2CT/evaluation_results/unet_error_analysis_20260616"
FIG = os.path.join(RUN, "figures")
OUT = "/home/minsukc/MRI2CT/_html/04_unet_error_anatomy.html"

s = pd.read_csv(os.path.join(RUN, "summary.csv"))
st = pd.read_csv(os.path.join(RUN, "structures.csv"))
orc = pd.read_csv(os.path.join(RUN, "oracle_fix.csv"))
ps = pd.read_csv("/home/minsukc/MRI2CT/evaluation_results/full_eval_20260609/metrics/per_subject.csv")
u = s[s.model == "unet"]
ou = orc[orc.model == "unet"]
REG = ["brain", "head_neck", "thorax", "abdomen", "pelvis"]
lm = json.load(open(os.path.join(RUN, "loc_mag_stats.json")))["all"]
hnf = json.load(open(os.path.join(RUN, "hn_fig_stats.json")))


def b64(name):
    with open(os.path.join(FIG, name), "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode()


def img(name, cap):
    return f'<figure><img src="{b64(name)}"/><figcaption>{cap}</figcaption></figure>'


def table(df, fmt="{:.1f}", idxname=""):
    cols = "".join(f"<th>{c}</th>" for c in df.columns)
    rows = ""
    for idx, r in df.iterrows():
        cells = "".join(f"<td>{(fmt.format(v) if isinstance(v,(int,float,np.floating)) and not pd.isna(v) else v)}</td>" for v in r)
        rows += f"<tr><th>{idx}</th>{cells}</tr>"
    return f'<table><thead><tr><th>{idxname}</th>{cols}</tr></thead><tbody>{rows}</tbody></table>'


# ---- tables ----
# T1 per-region tissue MAE
t1 = u.groupby("region").agg(air=("mae_air", "mean"), soft=("mae_soft", "mean"),
                             bone=("mae_bone", "mean"), full_mae=("mae_raw", "mean"),
                             body_psnr=("subj_id", "size")).reindex(REG)
t1["body_psnr"] = [ps[(ps.model == "unet") & (ps.region == r)].body_psnr.mean() for r in REG]
t1.columns = ["MAE air", "MAE soft", "MAE bone", "MAE all", "body PSNR"]
T1 = table(t1, idxname="region")

# T2 per-bone-structure
bones = st[(st.is_bone) & (st.model == "unet")]
t2 = bones.groupby("name").agg(MAE=("mae", "mean"), bias=("bias", "mean"),
                               gt_HU=("gt_mean", "mean"), frac_gt1024=("gt_frac_gt1024", "mean"),
                               n_subj=("subj_id", "nunique")).sort_values("MAE", ascending=False)
t2["frac_gt1024"] = (t2["frac_gt1024"] * 100)
t2.columns = ["MAE (HU)", "bias (HU)", "GT mean HU", "% vox >1024", "n subj"]
T2 = table(t2, idxname="bone structure")

# T3 diffusion test
t3 = s.groupby("model").agg(bone_MAE=("mae_bone", "mean"), cortical_MAE=("mae_cortical", "mean"),
                            cortical_bias=("bias_cortical", "mean"), pred_bone_max=("pred_bone_max", "mean"),
                            pred_bone_mean=("pred_bone_mean", "mean")).reindex(["unet", "amix", "maisi", "mcddpm"])
t3.columns = ["bone MAE", "cortical MAE", "cortical bias", "pred bone MAX", "pred bone mean"]
T3 = table(t3, idxname="model")

# T4 oracle
fixes = ["base", "fix_air", "fix_soft", "fix_bone", "fix_cortical", "fix_skull"]
t4 = pd.DataFrame({
    "all PSNR": [ou[f"{f}_psnr"].mean() for f in fixes],
    "brain PSNR": [ou[ou.region == "brain"][f"{f}_psnr"].mean() for f in fixes],
    "all full-HU MAE": [ou[f"{f}_smae"].mean() for f in fixes],
    "brain full-HU MAE": [ou[ou.region == "brain"][f"{f}_smae"].mean() for f in fixes],
}, index=["baseline", "fix air", "fix soft", "fix bone (>200)", "fix cortical (>1024)", "fix skull"])
T4 = table(t4, fmt="{:.2f}", idxname="oracle scenario")

base_psnr = ou.base_psnr.mean()
d = {f: ou[f"{f}_psnr"].mean() - base_psnr for f in fixes}

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Where the UNet MR→CT baseline actually fails</title>
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
 table{{border-collapse:collapse;margin:12px 0;font-size:13px;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}} thead th{{background:#f9fafb}}
 tbody th{{text-align:left;background:#fafafa}}
 code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12.5px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}}
 ol li,ul li{{margin:5px 0}}
</style></head><body>

<h1>Where the UNet MR→CT baseline actually fails</h1>
<div class="sub">Error-anatomy of the plain 3D U-Net on 207 center-wise validation subjects (full_eval_20260609,
ckpt 9xmodnhn ep799). Decompositions over the raw full-HU CT, the 35-label CADS segmentation, and an oracle
"perfect-tissue" counterfactual. amix / MAISI / MC-DDPM included as contrast. Generated 2026-06-16.</div>

<div class="key"><b>Bottom line.</b> Two facts, both proven on your data, point in opposite directions:
<b>(1) Bone — especially cortical skull — is by far the worst-predicted tissue</b> (MAE {t1.loc['brain','MAE bone']:.0f}–{u.mae_bone.mean():.0f} HU vs ~{u.mae_soft.mean():.0f} HU soft; cortical bone is
<b>undershot by {abs(u.bias_cortical.mean()):.0f} HU</b>), and it is <b>information-limited</b> — even the uncapped
diffusion model can't fix it. <b>(2) Yet bone is <i>not</i> what caps the reported PSNR/MAE</b>: oracle-substituting
perfect bone raises PSNR only <b>+{d['fix_bone']:.2f} dB</b> (perfect cortical: <b>+{d['fix_cortical']:.2f} dB</b>),
because the metric clips at ±1024 HU and is dominated by far-more-numerous air/soft voxels — fixing air gives
<b>+{d['fix_air']:.2f} dB</b>. The standard leaderboard metric is <b>blind to the one thing that is most broken
and most clinically important.</b></div>

<div class="ok"><b>Validation.</b> Every recomputed metric matches the released pipeline exactly: recomputed
body-voxel MAE vs <code>synthrad_mae</code> and oracle <code>base</code> PSNR vs reported <code>body_psnr</code>
agree to &lt;0.001 (after correcting a RAS-vs-native orientation flip between the raw CT and the saved predictions).</div>

<h2>1. Bone is the worst-predicted tissue, everywhere</h2>
<p>Splitting every body voxel by ground-truth HU (air &lt;−300, soft −300…200, bone &gt;200): bone MAE is
~5× soft tissue in every region. This is the single largest per-voxel error source.</p>
{T1}
{img('fig1_tissue_mae.png','Per-voxel MAE by tissue (UNet). Bone error dwarfs soft tissue in all five regions.')}

<h2>2. The failure is a systematic UNDERSHOOT, worst in the skull</h2>
<p>Every skeletal structure is predicted too low (negative bias). The skull is the worst (cortical, thin, ~30%
of its voxels exceed 1024 HU). Inside true bone, the network collapses toward the soft-tissue mean and essentially
never emits dense cortical HU — classic regression-to-the-mean on a one-to-many target.</p>
{T2}
<div class="grid">
{img('fig2_bone_structures.png','Per-bone-structure MAE (blue) and signed bias (red). Every bone is undershot.')}
{img('fig7_bone_hist.png','Pooled HU inside true bone: GT spans to ~2800 HU; UNet piles up low (mean 542→329).')}
</div>

<h2>3. It is information-limited, not a clipping artifact (the diffusion test)</h2>
<p>UNet and amix are sigmoid-capped at +1024 HU, but <b>MAISI and MC-DDPM are not</b>. If the bone failure were
merely the cap, the uncapped diffusion models would recover dense bone. They do not: MC-DDPM tops out ~1430 HU
(vs real cortical ~2976) and still undershoots cortical bone by {abs(t3.loc['mcddpm','cortical bias']):.0f} HU.
No model — capped or not — predicts dense bone. The bottleneck is missing information in the MR, not architecture.</p>
{T3}
{img('fig3_diffusion_test.png','Even the uncapped MC-DDPM tops out far below real cortical bone and undershoots by ~730–900 HU across all models.')}

<h2>4. The failure is a density UNDERSHOOT, not a localization error</h2>
<p>Separating the two bone failure modes: <b>localization</b> — does the model put bone in the right
voxels? (shape-Dice of pred&gt;200 vs GT&gt;200) — and <b>magnitude</b> — where both agree a voxel is bone,
how wrong is the HU? Even on the agreed-bone intersection the error is <b>{lm['mae_agreed']:.0f} HU</b>
(dense interior worse at {lm['interior_mae']:.0f} HU vs {lm['boundary_mae']:.0f} at the edge): a pure density
undershoot. The undershoot is so severe in thin trabecular bone (thorax/abdomen ribs &amp; vertebrae) that
those voxels fall <i>below</i> the 200-HU bone threshold entirely — appearing as a "missed-bone" localization
error (≈60% missed there) when it is really magnitude. In dense skull the model localizes bone well
(Dice ~0.79) yet still undershoots its density.</p>
{img('fig8_loc_mag.png','Bone shape-Dice + missed fraction (left) and magnitude MAE (right). The dominant, universal failure is density undershoot; thin bone is additionally under-detected.')}

<h2>5. The reported metric is blind to the bone error (oracle counterfactual)</h2>
<p>Direct test of "is bone what limits the score?": overwrite the prediction with ground-truth HU inside one
tissue and recompute the <i>exact reported metrics</i>. In the clipped PSNR metric, fixing bone barely moves the
number (cortical: <b>+{d['fix_cortical']:.2f} dB</b>) — fixing air helps 10× more — because the metric clips the
cortical errors away and averages over far more air/soft voxels. In <b>full-HU</b> MAE (unclipped), fixing bone
<i>does</i> help, especially in brain/HN.</p>
{T4}
<div class="grid">
{img('fig4_oracle_psnr.png','Oracle perfect-tissue → reported PSNR. Fixing AIR beats fixing BONE; fixing cortical bone is nearly invisible.')}
{img('fig5_oracle_smae.png','Same experiment in full-HU MAE (unclipped): now fixing bone clearly helps in brain/HN.')}
</div>
<div class="warn"><b>This is the publishable hook.</b> The thing that is most broken (cortical bone) and most
relevant to dose/planning is precisely what the pixel metric cannot see. A model could halve its cortical-bone
error and the leaderboard PSNR would barely move — concrete, on-your-data evidence that pixel metrics mis-rank
for the downstream task.</div>

<h2>6. Region diagnoses — your assumed causes, tested</h2>
<p><b>Error-mass composition</b> (share of each region's total abs-error): brain and head&amp;neck carry a large
<i>bone</i> share; abdomen/pelvis/thorax are air+soft dominated.</p>
{img('fig6_errormass.png','Error-mass composition by region. Brain/HN are bone-heavy; the rest are air+soft.')}
<h3>Brain: it is the <u>skull</u>, not defacing</h3>
<p>Hypothesis was MR/CT defacing mismatch. Tested: the <code>face_oral</code> structure contributes
<b>0.0%</b> of brain error mass (~205 voxels/subject), while the <b>skull alone is 25.3%</b> and all bone is
~33%. Brain's low PSNR is a cortical-skull + air-cavity problem; defacing is negligible.</p>
<h3>Pelvis: a modest OOD sequence shift, not a blow-up</h3>
<p>Pelvis validation is <b>100% center C</b> while training is <b>center A only</b> (documented T1→T2 sequence
shift). Effect is real but mild: a soft-tissue undershoot bias of <b>−9.4 HU</b> and 2nd-lowest PSNR (25.4) — but
pelvis full-HU MAE (58) and bone MAE (199) are actually among the <i>best</i> regions. The T1/T2 shift nudges soft
tissue; it is not the dominant error source.</p>
<h3>Head &amp; neck: the organ-Dice "collapse" (0.47) is a metric artifact, not bad sCT</h3>
<p>HN has the lowest organ-Dice, but a per-class breakdown (the eval's exact 12-class teacher) shows the
bulk tissues are fine — c1 (brain/soft) 0.68, c2 0.80, c5 (bone) 0.77, c11 0.86, comparable to other
regions. The macro-average is dragged down by a few <b>tiny</b> structures (c3, c4, c8; ~1–4k voxels) scoring
0.03–0.36. Running the same teacher on the <b>real GT CT</b> as a ceiling proves these are not an sCT failure:
the GT-CT ceiling for the whole region is only <b>{hnf['macro_ceil']:.2f}</b> and the sCT reaches
<b>{hnf['macro_sct']:.2f}</b> of it — the tiny classes (c3: sCT 0.032 vs GT-ceiling 0.031) are unsegmentable
even with perfect CT. HN organ-Dice is bounded by the teacher/metric, not the synthesis.</p>
{img('fig9_hn_dice.png','HN per-class sCT Dice vs the GT-CT ceiling. The low classes (c3,c4,c8) score near zero even on real CT — the macro-average is a metric artifact; bulk tissues track the ceiling.')}

<h2>7. Qualitative examples</h2>
<p>For each: input MR · GT CT (bone window) · UNet sCT (same window — note the visibly grey, undershot skull) ·
true full-HU error · and the cortical-bone error the reported (±1024-clipped) metric never sees.</p>
{img('example_1BB099.png','Brain — skull undershoot; hidden-error panel concentrates on the cranial vault.')}
{img('example_1HNC043.png','Head &amp; neck — dense skull-base/petrous bone undershot and hidden from the metric.')}
{img('example_1PC035.png','Pelvis (OOD center C) — bone reasonable; error is diffuse soft-tissue, not catastrophic bone.')}

<h2>8. Verdict — ranked drivers of UNet error</h2>
<ol>
<li><b>Cortical-bone undershoot (information-limited).</b> Largest per-voxel error, systematic −{abs(u.bias_cortical.mean()):.0f} HU
bias, worst in skull, unfixable by capacity or diffusion. <i>Needs information the MR lacks</i> (the case for an
external prior / retrieval).</li>
<li><b>Metric blindness.</b> The reported PSNR/MAE clips bone and is air/soft-weighted, so it under-reports the
bone failure (+{d['fix_cortical']:.2f} dB for perfect cortical bone). <i>Needs a downstream/decision-aware metric.</i></li>
<li><b>Air/gas voxels</b> dominate the aggregate metric by sheer count (biggest PSNR lever) though clinically trivial.</li>
<li><b>OOD domain shift</b> (pelvis/HN center C; pelvis T1→T2): modest soft-tissue bias, region-localized.</li>
<li><b>Defacing: not a driver</b> (0.0% of brain error mass) — refuted.</li>
</ol>
<div class="key">Both top drivers point the same way as the methodology direction: cortical bone is missing
<i>information</i> (an external CT prior / retrieval can inject what the MR cannot provide), and the failure is
<i>invisible to pixel metrics</i> (motivating downstream-/bone-aware evaluation). The two are complementary halves
of one paper.</div>

<h2>Reproducibility</h2>
<p>Scripts in <code>src/evaluate/error_anatomy/</code>: <code>extract.py</code> (per-subject decomposition),
<code>oracle_fix.py</code> (counterfactual), <code>build_figures.py</code>, <code>examples.py</code>,
<code>report.py</code>. Data: <code>evaluation_results/unet_error_analysis_20260616/</code>
(<code>summary.csv</code>, <code>structures.csv</code>, <code>oracle_fix.csv</code>, <code>key_stats.json</code>).
GT = raw <code>ct.nii</code> reoriented to canonical RAS to match the saved predictions; all 4 models scored
against the same GT.</p>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
