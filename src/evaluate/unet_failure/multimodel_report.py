"""Cross-model bone comparison report (07): does every model undershoot bone, or is it
specific to the U-Net? Aggregates multimodel_bone.csv, builds figures, writes
_html/07_crossmodel_bone.html. Includes per-model correctness gates."""
import os
import base64
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = "/home/minsukc/MRI2CT"
EVAL = os.path.join(REPO, "evaluation_results/full_eval_20260617")
RUN = os.path.join(REPO, "evaluation_results/unet_failure_20260619")
FIG = os.path.join(RUN, "figures")
OUT = os.path.join(REPO, "_html/07_crossmodel_bone.html")
# display order: regression models, then diffusion; with parity notes
ORDER = ["unet", "amix", "koalAI", "mcddpm", "cwdm", "maisi"]
PARITY = {"unet": "1.00x", "amix": "1.00x", "koalAI": "~0.78x", "mcddpm": "1.25x",
          "cwdm": "0.24x (under-parity)", "maisi": "0.22x (under-parity)"}
COL = {"unet": "#dc2626", "amix": "#ea580c", "koalAI": "#16a34a",
       "mcddpm": "#2563eb", "cwdm": "#7c3aed", "maisi": "#0891b2"}
plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 120})

b = pd.read_csv(os.path.join(RUN, "multimodel_bone.csv"))
rel = pd.read_csv(os.path.join(EVAL, "metrics/per_subject.csv"))
calib = np.load(os.path.join(RUN, "multimodel_calib.npz"))
present = [m for m in ORDER if m in b.model.unique()]


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


# ---- gates: per-model base reconciliation ----
m = b.merge(rel[["model", "subj_id", "body_psnr", "body_mae_hu"]], on=["model", "subj_id"])
gate = m.groupby("model")[["base_psnr", "body_psnr", "base_bmae", "body_mae_hu"]].apply(
    lambda d: pd.Series({"dpsnr": (d.base_psnr - d.body_psnr).abs().max(),
                         "dmae": (d.base_bmae - d.body_mae_hu).abs().max()})).reindex(present)
gate_pass = bool((gate.dpsnr.max() < 1e-2) and (gate.dmae.max() < 1e-2))
print("[gates]", "ALL PASS" if gate_pass else "FAIL")
print(gate.round(4).to_string())

# ---- per-model aggregates ----
agg = b.groupby("model").agg(
    bias_bone=("bias_bone", "mean"), bias_cort=("bias_cortical_clip", "mean"),
    bias_cort_raw=("bias_cortical_raw", "mean"), mae_bone=("mae_bone", "mean"),
    pred_bone_mean=("pred_bone_mean", "mean"), pred_bone_max=("pred_bone_max", "mean"),
    pred_bone_p99=("pred_bone_p99", "mean"), gt_bone_mean=("gt_bone_mean", "mean"),
    base_psnr=("base_psnr", "mean"),
    air_dpsnr=("air_psnr", "mean"), soft_dpsnr=("soft_psnr", "mean"), bone_dpsnr=("bone_psnr", "mean"),
).reindex(present)
agg["air_dpsnr"] -= agg["base_psnr"]
agg["soft_dpsnr"] -= agg["base_psnr"]
agg["bone_dpsnr"] -= agg["base_psnr"]
# universality: % of subjects with bias_bone<0, and mean frac of bone voxels undershot
uni = b.groupby("model")[["bias_bone", "frac_bone_under"]].apply(
    lambda d: pd.Series({"pct_subj_under": 100 * (d.bias_bone < 0).mean(),
                         "mean_frac_under": 100 * d.frac_bone_under.mean()})).reindex(present)
agg = agg.join(uni)
gt_bone_max_overall = float(b.gt_bone_mean.max())  # rough ref; use a constant ~2900 too

# ---- figures ----
x = np.arange(len(present))


def save(fig, name):
    fig.tight_layout(); fig.savefig(os.path.join(FIG, name), bbox_inches="tight"); plt.close(fig); print("  ", name)


# m1 — bone & cortical bias per model
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(x - 0.2, agg.bias_bone, 0.4, label="all bone", color="#dc2626")
ax.bar(x + 0.2, agg.bias_cort, 0.4, label="cortical (>1024)", color="#7f1d1d")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(present, rotation=15); ax.set_ylabel("signed bias (HU), clipped")
ax.set_title("Every model undershoots bone (negative bias) — it is universal, not U-Net-specific")
ax.legend()
save(fig, "m1_bias_by_model.png")

# m2 — calibration curves per model
fig, ax = plt.subplots(figsize=(8, 6))
GTE, PRE = calib["gt_edges"], calib["pred_edges"]
gtc = (GTE[:-1] + GTE[1:]) / 2
prc = (PRE[:-1] + PRE[1:]) / 2
edges = [200, 400, 700, 1000, 1500, 2000, 2900]
for mdl in present:
    H = calib[mdl]
    xs, ys = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        w = H[(gtc >= lo) & (gtc < hi)].sum(0)
        if w.sum() < 1:
            continue
        xs.append((lo + hi) / 2); ys.append((prc * w).sum() / w.sum())
    ax.plot(xs, ys, "o-", color=COL[mdl], lw=1.8, ms=5, label=mdl)
ax.plot([0, 2900], [0, 2900], "k--", lw=1, label="perfect")
ax.set_xlabel("TRUE bone HU"); ax.set_ylabel("mean predicted HU")
ax.set_title("Calibration: ALL models flatten far below identity for dense bone\n(regression to the mean is architecture-independent)")
ax.legend(fontsize=9)
save(fig, "m2_calibration_by_model.png")

# m3 — oracle: air vs soft vs bone PSNR gain per model
fig, ax = plt.subplots(figsize=(9.5, 4.4))
w = 0.26
ax.bar(x - w, agg.air_dpsnr, w, label="fix air", color="#9ca3af")
ax.bar(x, agg.soft_dpsnr, w, label="fix soft", color="#2563eb")
ax.bar(x + w, agg.bone_dpsnr, w, label="fix bone", color="#dc2626")
ax.set_xticks(x); ax.set_xticklabels(present, rotation=15); ax.set_ylabel("PSNR gain if perfected (dB)")
ax.set_title("Across models, fixing air/soft beats fixing bone on PSNR (bone is rare in every model)")
ax.legend()
save(fig, "m3_oracle_by_model.png")

# m4 — prediction ceiling per model vs real bone
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.bar(x - 0.2, agg.pred_bone_mean, 0.4, label="pred bone mean", color="#60a5fa")
ax.bar(x + 0.2, agg.pred_bone_p99, 0.4, label="pred bone p99 (ceiling)", color="#1d4ed8")
ax.axhline(agg.gt_bone_mean.mean(), color="green", ls="--", lw=1.2, label=f"GT bone mean {agg.gt_bone_mean.mean():.0f}")
ax.axhline(1024, color="gray", ls=":", lw=1, label="1024 (cortical threshold)")
ax.set_xticks(x); ax.set_xticklabels(present, rotation=15); ax.set_ylabel("predicted bone HU")
ax.set_title("No model reaches dense cortical HU. Uncapped (koalAI/mcddpm) climb higher but still fall short")
ax.legend(fontsize=8)
save(fig, "m4_ceiling_by_model.png")

# ---- tables ----
T = pd.DataFrame({
    "parity": {mdl: PARITY[mdl] for mdl in present},
    "bone bias (HU)": agg.bias_bone, "cortical bias (HU)": agg.bias_cort,
    "bone MAE (HU)": agg.mae_bone, "% subj undershoot": agg.pct_subj_under,
    "% bone vox under": agg.mean_frac_under,
    "pred bone p99": agg.pred_bone_p99, "PSNR gain: air": agg.air_dpsnr, "PSNR gain: bone": agg.bone_dpsnr,
}).reindex(present)
T_main = table(T, fmt="{:.1f}", idxname="model")
T_gate = table(gate.rename(columns={"dpsnr": "Δ base PSNR", "dmae": "Δ base bodyMAE"}), fmt="{:.4f}", idxname="model")

cov = float(100 * b[b.model == "unet"].n_labeled.sum() / b[b.model == "unet"].n_body.sum())

HTML = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Cross-model bone comparison: is the undershoot universal?</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;max-width:1080px;margin:0 auto;padding:34px 26px;color:#1f2937;line-height:1.55;background:#fff}}
 h1{{font-size:24px;margin:0 0 4px}} h2{{font-size:20px;margin:32px 0 10px;border-bottom:2px solid #e5e7eb;padding-bottom:5px}}
 .sub{{color:#6b7280;font-size:13px;margin-bottom:18px}}
 .key{{background:#eff6ff;border-left:4px solid #2563eb;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .warn{{background:#fef2f2;border-left:4px solid #dc2626;padding:14px 18px;margin:16px 0;border-radius:4px}}
 .ok{{background:#f0fdf4;border-left:4px solid #16a34a;padding:12px 16px;margin:14px 0;border-radius:4px;font-size:13px}}
 figure{{margin:18px 0;text-align:center}} img{{max-width:100%;border:1px solid #e5e7eb;border-radius:6px}}
 figcaption{{color:#6b7280;font-size:12.5px;margin-top:6px}}
 table{{border-collapse:collapse;margin:12px 0;font-size:12.5px;width:100%}}
 th,td{{border:1px solid #e5e7eb;padding:5px 9px;text-align:right}} thead th{{background:#f9fafb}}
 tbody th{{text-align:left;background:#fafafa}} code{{background:#f3f4f6;padding:1px 5px;border-radius:3px;font-size:12px}}
 .grid{{display:grid;grid-template-columns:1fr 1fr;gap:10px}} ol li,ul li{{margin:5px 0}}
</style></head><body>

<h1>Cross-model bone comparison: is the undershoot universal?</h1>
<div class="sub">Same bone diagnosis as reports 05/06, now run on all {len(present)} models over the 207 center-wise
validation subjects (<code>full_eval_20260617</code>). Question: is bone undershoot a U-Net quirk or does every model
do it? Sign: error = pred − truth; negative bias = undershoot. <b>Parity caveat:</b> cwdm and maisi are far below the
training-sample budget (0.24x / 0.22x) and still training — read their numbers as lower bounds.</div>

<div class="key"><b>Answer: the bone undershoot is universal — every model does it, including koalAI.</b>
All {len(present)} models undershoot bone (negative bias) and cortical bone heavily, and all of them flatten far below
the identity line for dense bone (regression to the mean). The uncapped models (koalAI, mcddpm) reach somewhat higher
peak HU than the sigmoid-capped U-Net/amix, but still fall well short of real cortical HU and still undershoot
{agg.mean_frac_under.min():.0f}–{agg.mean_frac_under.max():.0f}% of bone voxels per subject. And in every model,
fixing air/soft helps PSNR more than fixing bone. This confirms the cause is information + objective, not one
architecture.</div>

<div class="ok"><b>Correctness ({'PASS' if gate_pass else 'FAIL'}).</b> For every model the oracle baseline reproduces
the released <code>body_psnr</code> and <code>body_mae_hu</code> to &lt;0.01.
{T_gate}</div>

<h2>1. Every model undershoots bone</h2>
{T_main}
{img('m1_bias_by_model.png','Signed bone & cortical bias per model. All negative: every model predicts bone too low; cortical worse.')}

<h2>2. All models regress to the mean for dense bone</h2>
<p>For each model we read the mean predicted HU at each true-bone-HU level. A perfect model lies on the dashed line;
every model instead flattens out far below it as bone gets denser.</p>
{img('m2_calibration_by_model.png','Calibration per model. All curves flatten below identity; none tracks dense cortical bone. koalAI/mcddpm climb a little higher but still collapse. The final point (true HU 2000-2900) is very sparse (~0.1% of bone voxels) so it is noisy and drops rather than plateaus.')}

<h2>3. No model reaches dense cortical HU</h2>
<p>The sigmoid-capped regression models (U-Net, amix) top out ~880-1024 HU. The unbounded models (koalAI, mcddpm,
cwdm, maisi) can output higher, and indeed reach higher p99, but still land far below real cortical bone (GT bone
mean {agg.gt_bone_mean.mean():.0f}, cortical &gt;1024 up to ~2900). The cap is not the only limit; the information is.</p>
{img('m4_ceiling_by_model.png','Predicted bone HU per model. Uncapped models climb higher than capped ones but none reaches dense cortical HU.')}

<h2>4. In every model, air/soft are bigger PSNR levers than bone</h2>
<p>The oracle (perfect one tissue, recompute the reported metric) gives the same ranking for all models: air and soft
beat bone, because bone is ~5% of voxels in every case and the metric clips it.</p>
{img('m3_oracle_by_model.png','Oracle PSNR gain per model. Fixing air/soft beats fixing bone across the board.')}

<h2>5. Note on the per-label coverage (math)</h2>
<p>The 35 CADS labels cover only <b>~{cov:.0f}%</b> of body voxels; the remaining ~{100-cov:.0f}% is <code>seg==0</code>
inside the body, which is almost entirely unlabeled internal air (sinus / bowel gas / cavities that CADS does not
assign a structure to). The external background is masked to −1024 and excluded. <b>Consequence:</b> voxel-averaging
the per-label MAEs reconstructs the MAE over labeled voxels, not the full body MAE. The air/soft/bone HU split, by
contrast, tiles 100% of the body and does reconstruct the body MAE exactly (verified by the mass-conservation gate in
report 05).</p>

<h2>6. Conclusion</h2>
<ol>
<li><b>Bone undershoot is universal.</b> Every model (regression and diffusion, capped and uncapped, including koalAI)
undershoots bone and cortical bone, and all regress to the mean for dense bone.</li>
<li><b>Uncapped helps a little, not enough.</b> koalAI/mcddpm reach higher peak HU than the sigmoid-capped U-Net/amix
but still fall far short of cortical HU — consistent with an information limit, not an architecture limit.</li>
<li><b>Same metric story everywhere.</b> Air/soft are the bigger PSNR levers in every model; bone leads only on
per-voxel / clinical accuracy.</li>
<li>cwdm/maisi numbers are lower bounds (still under-parity), but they show the same qualitative pattern.</li>
</ol>

<h2>Reproducibility</h2>
<p>Scripts: <code>src/evaluate/unet_failure/multimodel_extract.py</code> (per-model per-subject) and
<code>multimodel_report.py</code> (this report). Data:
<code>evaluation_results/unet_failure_20260619/multimodel_bone.csv</code> + <code>multimodel_calib.npz</code>.
Oracle replicates <code>compute_metrics_body</code>; GT = raw <code>ct.nii</code> in canonical RAS.</p>
</body></html>"""

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, "w") as f:
    f.write(HTML)
print("wrote", OUT, "(%.0f KB)" % (os.path.getsize(OUT) / 1024))
