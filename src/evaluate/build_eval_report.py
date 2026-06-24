"""Build the self-contained full_eval HTML report.

Reads metrics/{overall,by_region,inference_time}.csv + figures/*.png + CHOICES.md
from the eval root and emits a single portable report.html with base64-embedded
figures, styled like _reports/training_budget.html.

Usage:
    python src/evaluate/build_eval_report.py --eval_root /gpfs/.../full_eval_20260601 \
        --repo_copy _reports/full_eval_20260601.html
"""
import argparse
import base64
import csv
import html
import json
import os

# Verified checkpoint / sample-count provenance (from CHOICES.md). Edit here if
# the still-training models' snapshots change.
CKPT_TABLE = [
    # model, checkpoint, samples_seen, vs3.2M, note
    ("amix", "6hjye9gp/checkpoint_last.pt (ep799)", "3,200,000", "1.00×", "reference"),
    ("unet", "9xmodnhn/checkpoint_last.pt (ep799)", "3,200,000", "1.00×", "reference"),
    ("MC-DDPM", "a3g28rez/mcddpm_epoch01000.pt", "4,004,000", "1.25×", "intermediate (no ep800 saved)"),
    ("MAISI", "5hprtpwl/checkpoint_last.pt", "~840,000", "0.26×", "still training — under-parity"),
    ("cWDM", "smg8thkr/synthrad_last.pt (step ~349k)", "~349,000", "0.11×", "still training — under-parity"),
    ("koalAI", "per-region fold_0 checkpoint_final.pth", "~500,000/region", "~0.78×", "per-region; pre-generated"),
]
MODEL_ORDER = ["amix", "unet", "maisi", "mcddpm", "cwdm", "koalAI"]
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]

# Defaults for the original 6-model report. An optional <eval_root>/report_meta.json
# can override any of: "title", "heading", "subtitle", "parity_caveat" (HTML strings)
# and "ckpt_table" (list of [model, checkpoint, samples, vs3.2M, note] rows). When the
# file is absent, these defaults are used so the original report is unchanged.
DEFAULT_TITLE = "MRI→CT Full Model Evaluation — full_eval_20260601"
DEFAULT_HEADING = "MRI→CT Synthesis — Six-Model Evaluation"
DEFAULT_PARITY_CAVEAT = (
    '<span class="pill warn">parity caveat</span> Only <b>amix/unet</b> (1.00×) and <b>koalAI</b> (~0.78×)\n'
    'are near equal training budget. <b>MC-DDPM</b> is 1.25× (slightly over); <b>MAISI</b> (0.26×) and\n'
    '<b>cWDM</b> (0.11×) are still training and under-parity — read their numbers as lower bounds, not\n'
    'converged performance.')

CSS = """
:root{--bg:#0f1117;--panel:#171a23;--ink:#e6e9ef;--muted:#9aa3b2;--good:#37d399;
--bad:#ff6b6b;--warn:#ffc857;--accent:#5aa9ff;--line:#262b38}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.5 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:32px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:34px 0 12px;
padding-bottom:6px;border-bottom:1px solid var(--line)}
.sub{color:var(--muted);margin:0 0 18px}
table{width:100%;border-collapse:collapse;margin:10px 0;background:var(--panel);
border-radius:8px;overflow:hidden;font-size:14px}
th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line)}
th{color:var(--muted);font-weight:600;font-size:12px;text-transform:uppercase;letter-spacing:.04em}
td.num,th.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.best{color:var(--good);font-weight:700}
.bad{color:var(--bad);font-weight:600}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:12px;font-weight:600}
.pill.good{background:rgba(55,211,153,.15);color:var(--good)}
.pill.bad{background:rgba(255,107,107,.15);color:var(--bad)}
.pill.warn{background:rgba(255,200,87,.15);color:var(--warn)}
.card{background:var(--panel);border-radius:10px;padding:16px 18px;margin:14px 0}
.card.tldr{border:1px solid rgba(55,211,153,.4);
background:linear-gradient(180deg,rgba(55,211,153,.06),rgba(23,26,35,1))}
.fig{margin:18px 0}.fig img{width:100%;border-radius:8px;border:1px solid var(--line)}
.fig .cap{color:var(--muted);font-size:13px;margin-top:4px}
pre{background:var(--panel);border-radius:8px;padding:14px;overflow:auto;font-size:13px;color:var(--ink)}
small{color:var(--muted)}
"""


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def b64img(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def fmt(v, nd=2):
    try:
        return f"{float(v):.{nd}f}"
    except (ValueError, TypeError):
        return "—"


def fmt_ms(mean, std, nd=2):
    """Render 'mean ± std'; falls back to mean-only if std missing/non-numeric."""
    m = fmt(mean, nd)
    if m == "—":
        return "—"
    s = fmt(std, nd)
    return f"{m} ± {s}" if s != "—" else m


def per_subject_region_std(eval_root):
    """{(model, region, col): std} computed from per_subject.csv. by_region.csv
    stores only means, so per-region error bars in tables come from here."""
    import statistics
    p = os.path.join(eval_root, "metrics", "per_subject.csv")
    rows = read_csv(p)
    buckets = {}
    for r in rows:
        for col, v in r.items():
            if col in ("model", "subj_id", "region"):
                continue
            try:
                buckets.setdefault((r["model"], r["region"], col), []).append(float(v))
            except (ValueError, TypeError):
                pass
    return {k: (statistics.pstdev(vs) if len(vs) > 1 else 0.0) for k, vs in buckets.items()}


def metric_table(overall, cols, labels, lower_better, nd=2):
    """Build an HTML table of overall(micro) means; bold the best per column."""
    micro = {r["model"]: r for r in overall if r["agg"] == "micro"}
    # best per column
    best = {}
    for c in cols:
        vals = []
        for m in MODEL_ORDER:
            if m in micro and micro[m].get(f"{c}_mean"):
                try:
                    x = float(micro[m][f"{c}_mean"])
                    if x == x:  # skip NaN
                        vals.append((x, m))
                except ValueError:
                    pass
        if vals:
            best[c] = (min if lower_better[c] else max)(vals)[1]
    head = "<tr><th>Model</th>" + "".join(f'<th class="num">{html.escape(l)}</th>' for l in labels) + "</tr>"
    body = ""
    for m in MODEL_ORDER:
        if m not in micro:
            continue
        cells = ""
        for c in cols:
            v = micro[m].get(f"{c}_mean")
            sd = micro[m].get(f"{c}_std")
            cls = "num best" if best.get(c) == m else "num"
            cells += f'<td class="{cls}">{fmt_ms(v, sd, nd)}</td>'
        body += f"<tr><td>{m}</td>{cells}</tr>"
    return f"<table>{head}{body}</table>"


def region_table(by_region, col, lower_better, nd=1, std_lookup=None):
    rows = {(r["model"], r["region"]): r for r in by_region}
    present = [m for m in MODEL_ORDER if any(mdl == m for mdl, _ in rows)]
    head = "<tr><th>Model</th>" + "".join(f'<th class="num">{r}</th>' for r in REGIONS) + "</tr>"
    # best per region
    best = {}
    for reg in REGIONS:
        vals = []
        for m in MODEL_ORDER:
            r = rows.get((m, reg))
            if r and r.get(f"{col}_mean"):
                try:
                    x = float(r[f"{col}_mean"])
                    if x == x:  # skip NaN
                        vals.append((x, m))
                except ValueError:
                    pass
        if vals:
            best[reg] = (min if lower_better else max)(vals)[1]
    body = ""
    for m in present:
        cells = ""
        for reg in REGIONS:
            r = rows.get((m, reg))
            v = r.get(f"{col}_mean") if r else None
            sd = std_lookup.get((m, reg, col)) if std_lookup else None
            cls = "num best" if best.get(reg) == m else "num"
            cells += f'<td class="{cls}">{fmt_ms(v, sd, nd)}</td>'
        body += f"<tr><td>{m}</td>{cells}</tr>"
    return f"<table>{head}{body}</table>"


def paired_significance(eval_root, metric="body_mae_hu"):
    """Pairwise paired Wilcoxon on `metric` over subjects shared by both models.
    Returns HTML (table of p-values) or a note if unavailable."""
    p = os.path.join(eval_root, "metrics", "per_subject.csv")
    if not os.path.exists(p):
        return ""
    try:
        from scipy.stats import wilcoxon
    except Exception:
        return "<p class='sub'>(scipy unavailable — significance test skipped)</p>"
    by_model = {}
    for r in read_csv(p):
        v = r.get(metric)
        if v in (None, ""):
            continue
        try:
            by_model.setdefault(r["model"], {})[r["subj_id"]] = float(v)
        except ValueError:
            pass
    models = [m for m in MODEL_ORDER if m in by_model]
    if len(models) < 2:
        return ""
    head = "<tr><th>vs</th>" + "".join(f"<th class='num'>{m}</th>" for m in models) + "</tr>"
    body = ""
    for a in models:
        cells = ""
        for b in models:
            if a == b:
                cells += '<td class="num">—</td>'
                continue
            common = sorted(set(by_model[a]) & set(by_model[b]))
            xa = [by_model[a][s] for s in common]
            xb = [by_model[b][s] for s in common]
            try:
                if xa == xb:
                    cells += '<td class="num">—</td>'
                else:
                    _, pv = wilcoxon(xa, xb)
                    sig = "good" if pv < 0.05 else "warn"
                    cells += f'<td class="num"><span class="pill {sig}">p={pv:.1e}</span></td>'
            except Exception:
                cells += '<td class="num">n/a</td>'
        body += f"<tr><td>{a}</td>{cells}</tr>"
    return (f"<p class='sub'>Paired Wilcoxon on per-subject <b>{metric}</b> (shared subjects). "
            f"<span class='pill good'>p&lt;0.05</span> = the two models differ significantly; "
            f"<span class='pill warn'>p≥0.05</span> = gap is within noise (treat as a tie).</p>"
            f"<table>{head}{body}</table>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--repo_copy", default=None)
    args = ap.parse_args()

    M = os.path.join(args.eval_root, "metrics")
    overall = read_csv(os.path.join(M, "overall.csv"))
    by_region = read_csv(os.path.join(M, "by_region.csv"))
    itime = {r["model"]: r for r in read_csv(os.path.join(M, "inference_time.csv"))}
    micro = {r["model"]: r for r in overall if r["agg"] == "micro"}
    macro = {r["model"]: r for r in overall if r["agg"] == "macro"}

    # ---- optional per-eval-root metadata overrides ----
    meta_path = os.path.join(args.eval_root, "report_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    # An eval root with extra model columns (e.g. old + new side by side) can set
    # "model_order" in report_meta.json to control row order across every table.
    if meta.get("model_order"):
        global MODEL_ORDER
        MODEL_ORDER = meta["model_order"]

    # headline ranking by Track-A body MAE, MACRO (equal region weight — avoids the
    # larger regions dominating; regions differ a lot in size + OOD difficulty).
    def _rank(agg):
        r = sorted([(float(agg[m]["body_mae_hu_mean"]), m) for m in MODEL_ORDER
                    if m in agg and agg[m].get("body_mae_hu_mean")], key=lambda x: x[0])
        return " &lt; ".join(f"<b>{m}</b> ({v:.0f}HU)" for v, m in r) if r else "(pending)"
    rank_str = _rank(macro)
    rank_str_micro = _rank(micro)

    # ---- checkpoint table ----
    ck_rows = "".join(
        f"<tr><td>{m}</td><td><small>{html.escape(c)}</small></td>"
        f'<td class="num">{s}</td><td class="num">{v}</td><td><small>{html.escape(n)}</small></td></tr>'
        for m, c, s, v, n in (meta.get("ckpt_table") or CKPT_TABLE))
    ck_table = ("<table><tr><th>Model</th><th>Checkpoint</th><th class='num'>Samples</th>"
                "<th class='num'>vs 3.2M</th><th>Note</th></tr>" + ck_rows + "</table>")

    # ---- Track A / B tables ----
    trackA = metric_table(
        overall,
        ["body_mae_hu", "body_psnr", "body_ssim", "mae_hu", "psnr", "ssim", "dice_score_bone", "dice_score_all"],
        ["MAE(HU)↓ body", "PSNR↑ body", "SSIM↑ body", "MAE(HU)↓ full", "PSNR↑ full", "SSIM↑ full", "Hard Bone Dice↑", "Hard Dice↑"],
        {"body_mae_hu": True, "body_psnr": False, "body_ssim": False, "mae_hu": True,
         "psnr": False, "ssim": False, "dice_score_bone": False, "dice_score_all": False},
        nd=3)
    trackB = metric_table(
        overall, ["synthrad_mae", "synthrad_psnr", "synthrad_ms_ssim"],
        ["MAE(HU)↓", "PSNR↑", "MS-SSIM↑"],
        {"synthrad_mae": True, "synthrad_psnr": False, "synthrad_ms_ssim": False}, nd=3)

    # ---- inference time ----
    it_rows = ""
    for m in MODEL_ORDER:
        if m not in micro:
            continue
        r = itime.get(m)
        t = r["mean_time_sec_per_volume"] if r else ""
        it_rows += f'<tr><td>{m}</td><td class="num">{fmt(t,1) if t else "n/a (pre-generated)"}</td></tr>'
    it_table = f"<table><tr><th>Model</th><th class='num'>Mean inference time / volume (s)</th></tr>{it_rows}</table>"

    # ---- region tables for primary metrics ----
    region_std = per_subject_region_std(args.eval_root)
    reg_mae = region_table(by_region, "body_mae_hu", lower_better=True, nd=1, std_lookup=region_std)
    reg_psnr = region_table(by_region, "body_psnr", lower_better=False, nd=2, std_lookup=region_std)
    reg_dice = region_table(by_region, "dice_score_bone", lower_better=False, nd=3, std_lookup=region_std)
    reg_dice_all = region_table(by_region, "dice_score_all", lower_better=False, nd=3, std_lookup=region_std)
    sig_html = paired_significance(args.eval_root, "body_mae_hu")

    # ---- figures ----
    figs = ""
    fdir = os.path.join(args.eval_root, "figures")
    for reg in REGIONS:
        p = os.path.join(fdir, f"{reg}.png")
        if os.path.exists(p):
            figs += (f'<div class="fig"><img src="data:image/png;base64,{b64img(p)}"/>'
                     f'<div class="cap">{reg}: MRI · GT CT · all models (representative subject, median amix-PSNR).</div></div>')

    # ---- error-bar figures (mean ± std over subjects) ----
    def embed(name, cap=""):
        p = os.path.join(fdir, name)
        if not os.path.exists(p):
            return ""
        c = f'<div class="cap">{cap}</div>' if cap else ""
        return f'<div class="fig"><img src="data:image/png;base64,{b64img(p)}"/>{c}</div>'
    barsA = embed("trackA_bars.png", "Mean ± std over 207 val subjects.")
    barsB = embed("trackB_bars.png", "Mean ± std over 207 val subjects.")
    bars_region = (embed("region_mae_bars.png") + embed("region_psnr_bars.png")
                   + embed("region_bonedice_bars.png") + embed("region_dice_bars.png"))

    # ---- training-config comparison (paper/original vs ours) ----
    tcp = os.path.join(args.eval_root, "training_config.html")
    train_cfg = open(tcp).read() if os.path.exists(tcp) else ""

    # ---- choices ----
    cp = os.path.join(args.eval_root, "CHOICES.md")
    choices = html.escape(open(cp).read()) if os.path.exists(cp) else "(CHOICES.md not found)"

    n_subj = max((int(micro[m]["n"]) for m in micro), default=0)
    title = meta.get("title", DEFAULT_TITLE)
    heading = meta.get("heading", DEFAULT_HEADING)
    subtitle = meta.get(
        "subtitle",
        f"center-wise validation split · {n_subj} subjects · 5 regions · generated 2026-06-01")
    parity_caveat = meta.get("parity_caveat", DEFAULT_PARITY_CAVEAT)
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{CSS}</style></head>
<body><div class="wrap">
<h1>{heading}</h1>
<p class="sub">{subtitle}</p>

<div class="card tldr"><b>TL;DR —</b> Ranking by body-masked MAE (amix-clip), <b>macro</b> (equal region weight): {rank_str}.
<br><small>(micro / subject-weighted: {rank_str_micro})</small>
{parity_caveat}</div>

<h2>Models &amp; checkpoints (training-sample parity)</h2>
{ck_table}

<h2>Track A — amix-clip [-1024, 1024] (hu_range 2048)</h2>
<p class="sub"><b>Body-masked metrics (first 3 columns) are the primary, fair comparison</b> — voxels outside
the body mask zeroed, metric over the volume. <b>Hard Bone Dice</b> and <b>Hard Dice</b> (mean over all 11 foreground
organ classes) via the Baby-UNet teacher, computed on <i>argmax labels</i> following the standard evaluation
convention (<a href="https://github.com/ancestor-mithril/dice-score-3d">dice-score-3d</a>): for each class,
pred &amp; GT both absent → 1.0, exactly one absent → 0.0, else 2|A∩B|/(|A|+|B|).
<span class="pill warn">note</span> The <i>full-volume</i> MAE/PSNR/SSIM columns are secondary and
penalize models that flatten the background: koalAI sets everything outside the body to a constant
−1000 HU, so its full-volume SSIM looks low (~0.5) even though its body-masked SSIM (~0.9) matches the
others. Compare models on the body-masked columns. Best per column in <span class="best">green</span>.</p>
{barsA}
{trackA}

<h2>Track B — SynthRAD-native [-1024, 3000] (official ImageMetrics)</h2>
<p class="sub">Official challenge metric vs the <b>raw full-HU CT</b> (bone up to ~3000): MAE body-masked on raw HU
(÷mask.sum); PSNR &amp; MS-SSIM clip to [-1024,3000], data_range 4024. <span class="pill warn">read carefully</span>
amix / unet / cWDM cap predictions at +1024 HU and MAISI at +1000 (their training clipped CT), so Track B
<b>penalizes them on dense cortical bone they structurally cannot represent</b> — this reflects a preprocessing
choice, not raw model quality. koalAI was trained on full-range CT and is judged fairly here. Use Track A to
compare models within the [-1024,1024] range they were built for, and Track B for who delivers true bone HU
(e.g. for dose calculation).</p>
{barsB}
{trackB}

<h2>Per-region — body MAE (HU) &amp; Hard Bone Dice (Track A)</h2>
<p class="sub">Error bars = mean ± std over each region's val subjects. Hard Dice exists only in Track A
(computed in the amix-clip space). <span class="pill warn">read per-region Dice via Bone Dice</span>
<b>The per-region all-class Hard Dice is region-confounded — trust Hard Bone Dice, not the all-class mean.</b>
The all-class Hard Dice averages over all 11 teacher organ classes, but a limited FOV (e.g. head/neck)
anatomically contains only ~6 of them. The GT seg (<code>ct_seg.nii</code>) additionally carries <b>spurious
labels</b> for the absent torso organs — a head/neck case is tagged with ~3,000 voxels of "breast implant"
plus stray abdominal-cavity / pericardium / mediastinum — which the synthesis (correctly) does not reproduce,
so those classes score ~0 and drag the all-class mean down (head/neck amix all-class <b>0.49</b> vs Bone
<b>0.79</b>). This is a metric + GT-label artifact, <b>not</b> a synthesis deficit: the classes that ARE
present score well (head/neck Brain 0.96, Bones 0.81, Muscle 0.77) and the effect hits every model equally.
Hard Bone Dice (a class present in every region) is the trustworthy per-region Dice signal.</p>
<p class="sub">Worked example — head/neck subject <code>1HNC007</code> (amix prediction), per-class Hard Dice.
The five <span class="bad">spurious / out-of-FOV classes</span> (breast implant, abdominal cavity,
pericardium, thoracic cavity, mediastinum) score ~0 and pull the 11-class mean down to 0.394, while every
class that genuinely belongs in a head/neck scan scores well:</p>
<table style="max-width:560px">
<tr><th>class</th><th class="num">GT vox</th><th class="num">pred vox</th><th class="num">Hard Dice</th></tr>
<tr><td>Brain</td><td class="num">300,196</td><td class="num">313,412</td><td class="num">0.963</td></tr>
<tr><td>Bones</td><td class="num">212,082</td><td class="num">226,037</td><td class="num">0.810</td></tr>
<tr><td>Muscle</td><td class="num">147,819</td><td class="num">163,604</td><td class="num">0.773</td></tr>
<tr><td>Subcutaneous</td><td class="num">131,843</td><td class="num">125,728</td><td class="num">0.765</td></tr>
<tr><td>Spinal cord</td><td class="num">1,792</td><td class="num">667</td><td class="num">0.501</td></tr>
<tr><td>Gland</td><td class="num">4,345</td><td class="num">5,895</td><td class="num">0.447</td></tr>
<tr><td class="bad">Abdominal cav</td><td class="num">1,300</td><td class="num">1,272</td><td class="num">0.000</td></tr>
<tr><td class="bad">Breast implant</td><td class="num">2,975</td><td class="num">116</td><td class="num">0.043</td></tr>
<tr><td class="bad">Thoracic cav</td><td class="num">396</td><td class="num">25</td><td class="num">0.033</td></tr>
<tr><td class="bad">Pericardium</td><td class="num">1,915</td><td class="num">0</td><td class="num">0.000</td></tr>
<tr><td class="bad">Mediastinum</td><td class="num">300</td><td class="num">0</td><td class="num">0.000</td></tr>
<tr><td><b>mean (= Hard Dice)</b></td><td class="num">—</td><td class="num">—</td><td class="num"><b>0.394</b></td></tr>
</table>
{bars_region}
<h3>Per-region body MAE (HU) ↓</h3>
{reg_mae}
<h3>Per-region body PSNR ↑</h3>
{reg_psnr}
<h3>Per-region Hard Bone Dice ↑</h3>
{reg_dice}
<h3>Per-region Hard Dice ↑</h3>
{reg_dice_all}

<h2>Significance — paired Wilcoxon (body MAE)</h2>
{sig_html}

<h2>Inference time</h2>
<p class="sub">Per-volume inference. <b>amix / unet / maisi re-measured with CUDA synchronization</b> on 10
representative subjects (2 per region, spanning small brain → large thorax). An earlier wall-clock without
sync under-counted these fast models (measured here: unet <b>~11×</b>, amix <b>~2.9×</b>, maisi ~1.6×). The
<b>diffusion models are sync-invariant</b> (GPU time is dominated by the multi-step sampling loop), so
<b>cWDM and MC-DDPM use their full-207 generation means as-is</b>. <b>Times scale strongly with volume
size</b> — treat as order-of-magnitude. <b>koalAI</b> = end-to-end nnU-Net predict over all 207 (incl.
per-region model load), a different inference paradigm. amix/unet/maisi = sliding-window forward only (model
load excluded); cWDM = DDIM-100, MC-DDPM = 50-step DDIM over 3D slabs (both full-207 means).</p>
{it_table}

<h2>Qualitative comparison</h2>
{figs or '<p class="sub">(figures pending)</p>'}

<h2>Training configuration — original (paper/code) vs ours</h2>
{train_cfg or '<p class="sub">(training_config.html not found)</p>'}

<h2>Methods &amp; choices</h2>
<pre>{choices}</pre>
</div></body></html>"""

    out = os.path.join(args.eval_root, "report.html")
    with open(out, "w") as f:
        f.write(doc)
    print(f"[report] wrote {out} ({len(doc)//1024} KB)")
    if args.repo_copy:
        os.makedirs(os.path.dirname(args.repo_copy) or ".", exist_ok=True)
        with open(args.repo_copy, "w") as f:
            f.write(doc)
        print(f"[report] copied to {args.repo_copy}")


if __name__ == "__main__":
    main()
