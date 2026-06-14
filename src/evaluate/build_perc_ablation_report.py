"""Standalone HTML report for the UNet perceptual-loss ablation.

Reads the metrics CSVs written by score_perc_ablation.py and emits a single
self-contained HTML: variant legend, overall (micro + macro) Track-A + Hard-Dice
table, per-region body-MAE and Bone-Dice tables, and a paired-Wilcoxon matrix on
per-subject body MAE. Deliberately separate from full_eval_20260601.html.

Usage:
    python src/evaluate/build_perc_ablation_report.py \
        --eval_root /gpfs/.../perc_ablation_20260603 \
        --repo_copy _reports/perc_ablation_20260603.html
"""
import argparse
import csv
import html
import os
from collections import defaultdict

import numpy as np
from scipy.stats import wilcoxon

# Known runs in this ablation: tag -> (human label, perceptual_w, dice_w).
VARIANTS = {
    "9xmodnhn_ep400": ("no-perc (baseline)", 0.0, 0.1),
    "06e850ny_ep400": ("perceptual", 0.5, 0.1),
    "ye820cq0_ep400": ("perceptual, no-dice", 0.5, 0.0),
    # 100k-step (epoch 200) contrast for run 91hdk0ka: clean single-axis
    # perceptual on/off, identical dice 0.1. 9xmodnhn is the no-perc sibling.
    "9xmodnhn_ep200": ("no-perc (baseline)", 0.0, 0.1),
    "91hdk0ka_ep200": ("perceptual (ncc)", 0.1, 0.1),
}
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]
# (column, label, higher_is_better)
HEADLINE = [
    ("body_mae_hu", "body MAE (HU)", False),
    ("body_psnr", "body PSNR", True),
    ("body_ssim", "body SSIM", True),
    ("dice_score_bone", "Hard Bone Dice", True),
    ("dice_score_all", "Hard Dice (all)", True),
]


def read_per_subject(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rec = {"model": r["model"], "subj_id": r["subj_id"], "region": r["region"]}
            for k, v in r.items():
                if k not in rec:
                    rec[k] = float(v) if v not in ("", None) else np.nan
            rows.append(rec)
    return rows


def mean_std(rows, col):
    vals = [r[col] for r in rows if not np.isnan(r.get(col, np.nan))]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)


def macro_mean(rows, col):
    rmeans = [mean_std([r for r in rows if r["region"] == reg], col)[0] for reg in REGIONS]
    rmeans = [x for x in rmeans if not np.isnan(x)]
    return np.mean(rmeans) if rmeans else np.nan


def fmt(v, prec=3):
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{prec}f}"


def best_tag(per_tag_val, higher):
    items = [(t, v) for t, v in per_tag_val.items() if not np.isnan(v)]
    if not items:
        return None
    return (max if higher else min)(items, key=lambda kv: kv[1])[0]


def cell(v, is_best, prec, std=None):
    style = ' style="background:#d8f5d8;font-weight:600"' if is_best else ""
    txt = fmt(v, prec)
    if std is not None and not (isinstance(std, float) and np.isnan(std)):
        txt += f" <small>± {fmt(std, prec)}</small>"
    return f"<td{style}>{txt}</td>"


def macro_mean_std(rows, col):
    """Mean and dispersion across the 5 per-region means (equal region weight)."""
    rmeans = [mean_std([r for r in rows if r["region"] == reg], col)[0] for reg in REGIONS]
    rmeans = [x for x in rmeans if not np.isnan(x)]
    return (np.mean(rmeans), np.std(rmeans)) if rmeans else (np.nan, np.nan)


def table_block(title, tags, rows, cols, agg="micro"):
    out = [f"<h3>{html.escape(title)}</h3>", "<table>", "<tr><th>variant</th>"]
    for _, lbl, _ in cols:
        out.append(f"<th>{html.escape(lbl)}</th>")
    out.append("</tr>")
    col_vals = {}
    for col, _, hib in cols:
        per_tag, per_std = {}, {}
        for t in tags:
            sub = [r for r in rows if r["model"] == t]
            if agg == "micro":
                m, s, _ = mean_std(sub, col)
            else:
                m, s = macro_mean_std(sub, col)
            per_tag[t], per_std[t] = m, s
        col_vals[col] = (per_tag, per_std, best_tag(per_tag, hib))
    for t in tags:
        lbl = VARIANTS.get(t, (t, None, None))[0]
        out.append(f"<tr><td style='text-align:left'>{html.escape(lbl)}<br><small>{html.escape(t)}</small></td>")
        for col, _, _ in cols:
            per_tag, per_std, bt = col_vals[col]
            prec = 1 if "mae" in col else (2 if "psnr" in col else 3)
            out.append(cell(per_tag[t], t == bt, prec, std=per_std[t]))
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)


def per_region_table(title, tags, rows, col, higher, prec):
    out = [f"<h3>{html.escape(title)}</h3>", "<table>", "<tr><th>variant</th>"]
    for reg in REGIONS:
        out.append(f"<th>{reg}</th>")
    out.append("</tr>")
    best = {}
    for reg in REGIONS:
        per_tag = {t: mean_std([r for r in rows if r["model"] == t and r["region"] == reg], col)[0] for t in tags}
        best[reg] = best_tag(per_tag, higher)
    for t in tags:
        lbl = VARIANTS.get(t, (t, None, None))[0]
        out.append(f"<tr><td style='text-align:left'>{html.escape(lbl)}</td>")
        for reg in REGIONS:
            v = mean_std([r for r in rows if r["model"] == t and r["region"] == reg], col)[0]
            out.append(cell(v, best[reg] == t, prec))
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)


def wilcoxon_matrix(tags, rows, col="body_mae_hu"):
    by = defaultdict(dict)
    for r in rows:
        if not np.isnan(r.get(col, np.nan)):
            by[r["model"]][r["subj_id"]] = r[col]
    out = ["<h3>Paired Wilcoxon — per-subject body MAE</h3>",
           "<p><small>Shared subjects only. p&lt;0.05 ⇒ the two variants differ significantly; "
           "p≥0.05 ⇒ within noise.</small></p>", "<table>", "<tr><th></th>"]
    for t in tags:
        out.append(f"<th>{html.escape(VARIANTS.get(t, (t,))[0])}</th>")
    out.append("</tr>")
    for a in tags:
        out.append(f"<tr><td style='text-align:left'>{html.escape(VARIANTS.get(a, (a,))[0])}</td>")
        for b in tags:
            if a == b:
                out.append("<td>—</td>")
                continue
            shared = sorted(set(by[a]) & set(by[b]))
            xa = [by[a][s] for s in shared]
            xb = [by[b][s] for s in shared]
            if len(shared) < 5 or np.allclose(xa, xb):
                out.append("<td>n/a</td>")
                continue
            p = wilcoxon(xa, xb).pvalue
            sig = "background:#d8f5d8" if p < 0.05 else "background:#f5e8d8"
            out.append(f"<td style='{sig}'>p={p:.1e}</td>")
        out.append("</tr>")
    out.append("</table>")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--metrics_dir", default=None, help="default <eval_root>/metrics")
    ap.add_argument("--repo_copy", default=None)
    ap.add_argument("--step_label", default="200k-step (epoch 400)",
                    help="parity-checkpoint description shown in the intro line")
    ap.add_argument("--figures_dir", default=None,
                    help="dir of per-region PNGs to embed (default <eval_root>/figures); "
                         "pass 'none' to skip")
    args = ap.parse_args()

    mdir = args.metrics_dir or os.path.join(args.eval_root, "metrics")
    rows = read_per_subject(os.path.join(mdir, "per_subject.csv"))
    present = [t for t in VARIANTS if any(r["model"] == t for r in rows)]
    extra = sorted({r["model"] for r in rows} - set(VARIANTS))
    tags = present + extra
    n = {t: len({r["subj_id"] for r in rows if r["model"] == t}) for t in tags}

    legend = "".join(
        f"<li><b>{html.escape(VARIANTS.get(t, (t,))[0])}</b> "
        f"(<code>{html.escape(t)}</code>): perceptual_w={VARIANTS.get(t,(None,'?','?'))[1]}, "
        f"dice_w={VARIANTS.get(t,(None,'?','?'))[2]}, n={n[t]}</li>"
        for t in tags)

    # The 3-run ep400 ablation varied perceptual_w AND dice_w, so it needs a
    # confound caveat. A clean single-axis contrast (no dice_w==0 variant) does not.
    has_no_dice = any(VARIANTS.get(t, (None, None, None))[2] == 0.0 for t in tags)
    caveat = ("""<div class="note"><b>Confound caveat.</b> These runs vary on <i>two</i> axes (perceptual_w and dice_w).
The clean perceptual contrast is <b>no-perc (perc 0, dice 0.1)</b> vs <b>perceptual (perc 0.5, dice 0.1)</b>
— identical dice, differing only in perceptual loss. The <i>perceptual, no-dice</i> run isolates the dice
contribution. Do not read the Hard-Dice gap to the no-dice run as a perceptual effect.</div>""" if has_no_dice
              else """<div class="note"><b>Single-axis contrast.</b> The two variants share identical config
(dice_w=0.1, ssim_w=0.1, l1_w=1.0, center-wise split) and differ in the perceptual loss term only. They are
nonetheless <i>separate training runs</i> (different seed/trajectory), so a small part of any gap is run-to-run
noise — but the per-subject Wilcoxon (below) tests whether the body-MAE difference is consistent across subjects.</div>""")

    # Embed per-region qualitative PNGs (base64, self-contained) if present.
    fig_dir = args.figures_dir if args.figures_dir is not None else os.path.join(args.eval_root, "figures")
    figures_html = ""
    if fig_dir.lower() != "none":
        import base64
        imgs = []
        for reg in REGIONS:
            p = os.path.join(fig_dir, f"{reg}.png")
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    b64 = base64.b64encode(fh.read()).decode()
                imgs.append(f'<h4>{reg}</h4><img style="max-width:100%;border:1px solid #ddd" '
                            f'src="data:image/png;base64,{b64}">')
        if imgs:
            figures_html = ("<h3>Qualitative slices — one median-body-MAE subject per region</h3>"
                            "<p><small>Columns: MRI | GT CT | each variant. Per-variant captions show "
                            "body MAE / SSIM / Hard Bone Dice. brain &amp; head_neck use a narrow "
                            "[-100,100] HU window; others [-1024,1024].</small></p>" + "\n".join(imgs))

    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>UNet perceptual-loss ablation — {html.escape(os.path.basename(args.eval_root))}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:1000px;color:#222}}
 table{{border-collapse:collapse;margin:0.5rem 0 1.5rem}}
 th,td{{border:1px solid #ccc;padding:4px 10px;text-align:right;font-variant-numeric:tabular-nums}}
 th{{background:#f0f0f0}} td:first-child{{text-align:left}}
 small{{color:#666}} code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}
 .note{{background:#fff8e6;border-left:4px solid #e8b000;padding:8px 14px;margin:1rem 0}}
</style></head><body>
<h1>U-Net perceptual-loss ablation</h1>
<p>center-wise validation split · {n.get(tags[0], '?')} subjects · all at the {html.escape(args.step_label)} parity
checkpoint · Track-A metrics + Hard Dice, identical definitions to full_eval_20260601.</p>
<h3>Variants</h3><ul>{legend}</ul>
{caveat}
{table_block("Overall — micro (subject-weighted)", tags, rows, HEADLINE, "micro")}
{table_block("Overall — macro (equal region weight)", tags, rows, HEADLINE, "macro")}
<p><small>± in the micro table is the std across subjects; ± in the macro table is the std across the 5 per-region means.</small></p>
{per_region_table("Per-region — body MAE (HU) ↓", tags, rows, "body_mae_hu", False, 1)}
{per_region_table("Per-region — Hard Bone Dice ↑", tags, rows, "dice_score_bone", True, 3)}
{wilcoxon_matrix(tags, rows)}
{figures_html}
<p><small>Best per column highlighted green. Generated by build_perc_ablation_report.py.</small></p>
</body></html>"""

    out_path = os.path.join(args.eval_root, "report_perc_ablation.html")
    with open(out_path, "w") as f:
        f.write(body)
    print(f"[report] wrote {out_path}")
    if args.repo_copy:
        os.makedirs(os.path.dirname(args.repo_copy), exist_ok=True)
        with open(args.repo_copy, "w") as f:
            f.write(body)
        print(f"[report] repo copy {args.repo_copy}")


if __name__ == "__main__":
    main()
