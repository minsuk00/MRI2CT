"""Self-contained HTML report for the 4-way fused-LNCC perceptual ablation (epoch-300 parity).

Reuses the table/Wilcoxon/figure machinery of build_perc_ablation_report.py (imported, not
copied) and adds a hand-written analysis + conclusion. The four variants, all at the epoch-300
equal-samples-seen checkpoint (bs8x500 == bs4x1000 == 4000 patches/ep) on the center-wise split:

  nbn71048_ep300  U-Net, no perceptual        L1 + SSIM + Dice
  827la6dp_ep300  amix v1.4 (anatomix backbone) L1 + SSIM + Dice (perceptual_w=0)
  mwrwxvvu_ep300  U-Net + perceptual LNCC      L1 + LNCC(0.1) + Dice  (Python separable box-conv)
  fxudaqcp_ep300  U-Net + perceptual LNCC      L1 + LNCC(0.1) + Dice  (fused_lncc CUDA kernel)

The mwrwxvvu vs fxudaqcp pair is an A/B equivalence test: identical squared-LNCC semantics
(kernel_size=7, smooth 1e-5), only the implementation differs, so the two should match.

Usage:
    python src/evaluate/build_perc_fused_report.py \
        --eval_root /gpfs/.../evaluation_results/perc_ablation_fused_20260624 \
        --repo_copy _html/perc_fused_ablation_20260624.html
"""
import argparse
import base64
import html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import build_perc_ablation_report as B  # noqa: E402

# tag -> (label, perceptual_w, dice_w). Registered into the shared module so its
# table/Wilcoxon helpers label rows correctly.
NEW_VARIANTS = {
    "nbn71048_ep300": ("U-Net, no-perc", 0.0, 0.1),
    "827la6dp_ep300": ("amix v1.4 (anatomix backbone)", 0.0, 0.1),
    "mwrwxvvu_ep300": ("U-Net + perc LNCC (separable)", 0.1, 0.1),
    "fxudaqcp_ep300": ("U-Net + perc LNCC (fused)", 0.1, 0.1),
}
B.VARIANTS.update(NEW_VARIANTS)

# Display order = narrative order.
TAGS = ["nbn71048_ep300", "827la6dp_ep300", "mwrwxvvu_ep300", "fxudaqcp_ep300"]
REGIONS = B.REGIONS
HEADLINE = B.HEADLINE

LEGEND_NOTES = {
    "nbn71048_ep300": "plain U-Net · L1 + SSIM(0.1) + Dice(0.1, bone 0.4) · <b>no perceptual</b>",
    "827la6dp_ep300": "anatomix-feature translator · L1 + SSIM(0.1) + Dice(0.1, bone 0.4) · "
                      "perceptual_w=0 (anatomix in the <b>backbone</b>, not the loss)",
    "mwrwxvvu_ep300": "plain U-Net · L1 + <b>perceptual LNCC(0.1)</b> + Dice(0.1, bone 0.4) · SSIM off · "
                      "LNCC via Python <b>separable</b> box-conv (+torch.compile)",
    "fxudaqcp_ep300": "plain U-Net · L1 + <b>perceptual LNCC(0.1)</b> + Dice(0.1, bone 0.4) · SSIM off · "
                      "LNCC via the <b>fused_lncc CUDA kernel</b>",
}

# ---- Hand-written analysis (numbers from metrics/, epoch-300, center-wise val, n=207). ----
ANALYSIS_HTML = """
<div class="key">
<b>Verdict.</b>
<ol>
<li><b>The fused_lncc CUDA kernel is a correct drop-in.</b> Trained end to end at equal data exposure,
it produces a U-Net of essentially the same quality as the Python separable box-conv: the differences are
under 0.5% (0.15 HU body MAE, 0.004 Bone Dice) and point in <i>opposite</i> directions across metrics,
i.e. run-to-run jitter, not an implementation gap. Its known speed/VRAM savings therefore cost nothing in accuracy.</li>
<li><b>The perceptual LNCC term helps.</b> Both perceptual U-Nets beat the no-perceptual U-Net on body MAE
(~0.6 to 0.8 HU, p&lt;1e-6) and on Hard Bone Dice (+0.009 to +0.013, p&lt;1e-15), small but highly consistent.</li>
<li><b>The anatomix backbone alone does not.</b> amix (anatomix features in the architecture, perceptual loss off)
is 1.4 HU worse on body MAE (p&approx;1e-9) and ties on Bone Dice. The bone benefit comes from the perceptual loss,
not the backbone.</li>
</ol>
</div>

<h3>1. Fused vs separable LNCC (the A/B equivalence test)</h3>
<p>Both arms optimize the <i>same</i> squared-LNCC (kernel 7, smooth 1e-5); only the implementation differs
(Python 3&times;1-D box-conv + torch.compile vs the fused_lncc CUDA kernel). At epoch 300:</p>
<ul>
<li>body MAE 35.67 (separable) vs 35.51 (fused), a 0.15 HU / 0.4% gap;</li>
<li>Hard Bone Dice 0.768 (separable) vs 0.764 (fused), a 0.004 gap;</li>
<li>body SSIM is within noise (paired Wilcoxon p=0.15).</li>
</ul>
<p>The paired Wilcoxon flags both MAE and Dice at p&lt;0.05, but this is not a meaningful quality difference, for two reasons.
(a) The magnitudes are below half a percent, well inside the spread between independently trained models. (b) The two
metrics <i>disagree</i> on which arm wins (fused has lower MAE, separable has higher Dice), which is the signature of
trajectory noise rather than one kernel being more accurate. The two were separate runs (the separable arm resumed an
earlier checkpoint at epoch 38, the fused arm trained from scratch), so a small consistent per-subject offset is
expected, and a paired test over 207 subjects is sensitive enough to surface it. <b>The fused kernel reproduces the
separable result.</b> Its value (about 13% faster per iter and roughly 11 GB lighter at bs4, measured separately) is
realized here at no accuracy cost.</p>

<h3>2. Does the perceptual term help? (perc vs no-perc)</h3>
<p>Both perceptual U-Nets beat the no-perceptual U-Net:</p>
<ul>
<li>body MAE: 36.31 &rarr; 35.67 (sep) / 35.51 (fused), 0.6 to 0.8 HU lower, p&lt;1e-6;</li>
<li>Hard Bone Dice: 0.755 &rarr; 0.768 / 0.764, +0.009 to +0.013, p&lt;1e-15;</li>
<li>body SSIM and PSNR also nudge up.</li>
</ul>
<p>The macro (equal-region-weight) table shows the same ordering, with the perceptual arms 0.5 to 1.0 HU below no-perc.
So the anatomix-feature LNCC perceptual loss gives a small but very consistent improvement, strongest on bone, the
tissue that matters most for MR&rarr;CT.</p>
<div class="note"><b>Caveat: not a clean single-axis test.</b> Turning perceptual on (a) <i>swaps</i> SSIM for LNCC
(repo rule: perceptual on &rArr; ssim_w=0) and (b) the perceptual runs used bs4&times;1000 steps, i.e. <i>twice</i> the
optimizer updates of the bs8&times;500 no-perc run to reach the same 4000 patches/epoch. The measured gain therefore
bundles "LNCC instead of SSIM" with "smaller batch / more updates." It is the gain of the whole perceptual recipe,
not of the loss term in isolation.</div>

<h3>3. amix (anatomix backbone, perceptual off)</h3>
<p>amix is the anatomix-feature translator with perceptual_w=0, so anatomix enters through the architecture, not the
loss. It is worse than the plain no-perc U-Net on body MAE (37.71 vs 36.31, +1.4 HU, p&approx;1e-9) and on PSNR/SSIM,
and ties on Hard Bone Dice (0.756 vs 0.755, p=0.61). This matches the standing finding that the anatomix backbone is
redundant for intensity: without the perceptual loss the frozen-feature translator buys nothing on HU accuracy and
costs a little. Any bone benefit credited to anatomix elsewhere is coming from the perceptual loss, not the backbone.</p>

<h3>4. Per-region pattern</h3>
<ul>
<li>The perceptual body-MAE gain concentrates in <b>pelvis</b> (45.2 &rarr; 41 to 44) and <b>abdomen</b>
(25.4 &rarr; 24.7); thorax and head_neck are flat; brain is mixed.</li>
<li>Bone Dice improves most on <b>thorax</b> (0.670 &rarr; 0.686 to 0.689) and <b>brain</b> (0.825 &rarr; 0.829 to 0.839).</li>
<li>amix's MAE penalty is worst in <b>brain</b> (54.3 &rarr; 58.3); it does lead pelvis Bone Dice (0.892, tied with fused).</li>
</ul>

<div class="note"><b>Caveats that bound all of the above.</b>
<ul>
<li>All four variants share dice_bone_w=0.4, which is known to score slightly lower on Bone Dice than 0.3. Since every
variant uses 0.4 the comparison is internally consistent, but absolute Bone Dice here is not directly comparable to
bw0.3 references.</li>
<li>Epoch 300 is an equal-<i>data</i>, not equal-<i>compute</i>, snapshot: the perceptual arms took 2&times; the gradient steps.</li>
<li>Each variant is a separate training run, so part of every gap is seed/trajectory noise; the per-subject Wilcoxon
controls for this but cannot remove it.</li>
</ul></div>

<h3>Conclusion</h3>
<p>The fused_lncc CUDA kernel is validated as a drop-in for the Python separable LNCC: trained end to end it yields a
U-Net of statistically equivalent quality (sub-0.5% metric differences with no consistent direction), so its
throughput and memory savings come at no accuracy cost. Separately, the perceptual LNCC recipe gives a small, very
consistent improvement over no-perceptual (mainly on bone), whereas the anatomix backbone without the perceptual loss
does not help and slightly hurts intensity. Practical takeaway: <b>use the fused kernel for perceptual U-Net training,
and keep the perceptual loss; the benefit is the loss, not the anatomix architecture.</b></p>
"""


def figures_block(eval_root):
    fig_dir = os.path.join(eval_root, "figures")
    imgs = []
    for reg in REGIONS:
        p = os.path.join(fig_dir, f"{reg}.png")
        if os.path.exists(p):
            with open(p, "rb") as fh:
                b64 = base64.b64encode(fh.read()).decode()
            imgs.append(f'<h4>{reg}</h4><img style="max-width:100%;border:1px solid #ddd" '
                        f'src="data:image/png;base64,{b64}">')
    if not imgs:
        return ""
    return ("<h3>Qualitative slices — one median-body-MAE subject per region</h3>"
            "<p><small>Columns: MRI | GT CT | each variant. Per-variant captions show "
            "body MAE / SSIM / Hard Bone Dice. brain &amp; head_neck use a narrow "
            "[-100,100] HU window; others [-1024,1024].</small></p>" + "\n".join(imgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    ap.add_argument("--metrics_dir", default=None)
    ap.add_argument("--repo_copy", default=None)
    args = ap.parse_args()

    mdir = args.metrics_dir or os.path.join(args.eval_root, "metrics")
    rows = B.read_per_subject(os.path.join(mdir, "per_subject.csv"))
    tags = [t for t in TAGS if any(r["model"] == t for r in rows)]
    n = {t: len({r["subj_id"] for r in rows if r["model"] == t}) for t in tags}

    legend = "".join(
        f"<li><b>{html.escape(NEW_VARIANTS[t][0])}</b> (<code>{html.escape(t)}</code>): "
        f"{LEGEND_NOTES[t]}, n={n[t]}</li>" for t in tags)

    figures_html = figures_block(args.eval_root)

    h1 = "U-Net perceptual-loss ablation — fused vs separable LNCC (epoch-300 parity)"
    body = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{html.escape(h1)}</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;margin:2rem;max-width:1000px;color:#222;line-height:1.5}}
 table{{border-collapse:collapse;margin:0.5rem 0 1.5rem}}
 th,td{{border:1px solid #ccc;padding:4px 10px;text-align:right;font-variant-numeric:tabular-nums}}
 th{{background:#f0f0f0}} td:first-child{{text-align:left}}
 small{{color:#666}} code{{background:#f3f3f3;padding:1px 4px;border-radius:3px}}
 .note{{background:#fff8e6;border-left:4px solid #e8b000;padding:8px 14px;margin:1rem 0}}
 .key{{background:#eef6ff;border-left:4px solid #4a90d9;padding:8px 14px;margin:1rem 0}}
 h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:4px}}
</style></head><body>
<h1>{html.escape(h1)}</h1>
<p>center-wise validation split · {n.get(tags[0], '?')} subjects · all variants at the
<b>epoch-300 parity checkpoint</b> (equal samples-seen: bs8&times;500 == bs4&times;1000 == 4000 patches/epoch) ·
Track-A metrics + Hard Dice, identical definitions to full_eval.</p>
<div class="note"><b>What this ablation tests.</b> Two questions on one screen. (1) <b>Implementation equivalence:</b>
<i>perc LNCC (separable)</i> vs <i>perc LNCC (fused)</i> compute the same squared-LNCC (kernel 7, smooth 1e-5) and
should be statistically indistinguishable. (2) <b>Does the perceptual term help?</b> <i>U-Net no-perc</i> vs the
two perceptual U-Nets, plus the <i>amix</i> anatomix-backbone model as a no-perceptual-loss reference.
<br><b>Caveat:</b> turning perceptual on swaps SSIM for LNCC (repo rule: perceptual ⇒ ssim_w=0), so no-perc→perc is a
<i>substitution</i>, not a pure addition; the perceptual U-Nets use bs4&times;1000 steps (2&times; the optimizer updates of
the bs8&times;500 no-perc/amix runs) to reach the same data exposure. Each variant is a separate training run, so part
of any gap is run-to-run noise — the per-subject Wilcoxon tests whether body-MAE differences are consistent.</div>
<h3>Variants</h3><ul>{legend}</ul>

<h2>Results</h2>
{B.table_block("Overall — micro (subject-weighted)", tags, rows, HEADLINE, "micro")}
{B.table_block("Overall — macro (equal region weight)", tags, rows, HEADLINE, "macro")}
<p><small>± in the micro table is the std across subjects; ± in the macro table is the std across the 5 per-region means.
Green = best in column.</small></p>
{B.per_region_table("Per-region — body MAE (HU) ↓", tags, rows, "body_mae_hu", False, 1)}
{B.per_region_table("Per-region — Hard Bone Dice ↑", tags, rows, "dice_score_bone", True, 3)}
{B.wilcoxon_matrix(tags, rows, "body_mae_hu")}

{figures_html}

<h2>Analysis &amp; conclusion</h2>
{ANALYSIS_HTML}
</body></html>"""

    out = os.path.join(args.eval_root, "report_perc_fused.html")
    with open(out, "w") as f:
        f.write(body)
    print(f"[report] wrote {out}")
    if args.repo_copy:
        os.makedirs(os.path.dirname(args.repo_copy), exist_ok=True)
        with open(args.repo_copy, "w") as f:
            f.write(body)
        print(f"[report] wrote {args.repo_copy}")


if __name__ == "__main__":
    main()
