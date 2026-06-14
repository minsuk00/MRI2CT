"""Error-bar (mean ± std over subjects) figures for full_eval.

Reads metrics/per_subject.csv and writes to figures/:
  trackA_bars.png       body MAE / body PSNR / body SSIM / Bone Dice / Organ Dice
  trackB_bars.png       SynthRAD MAE / PSNR / MS-SSIM
  region_mae_bars.png   Track-A body MAE per region (grouped by model)
  region_bonedice_bars.png  Track-A Bone Dice per region (grouped by model)

Usage: python src/evaluate/plot_metric_bars.py --eval_root /gpfs/.../full_eval_20260601
"""
import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

MODELS = ["amix", "unet", "maisi", "mcddpm", "cwdm", "koalAI"]
COLORS = {"amix": "#5aa9ff", "unet": "#37d399", "maisi": "#ffc857",
          "mcddpm": "#c792ea", "cwdm": "#ff8a65", "koalAI": "#ff6b6b"}
REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]


def load_rows(eval_root):
    p = os.path.join(eval_root, "metrics", "per_subject.csv")
    rows = []
    for r in csv.DictReader(open(p)):
        d = {"model": r["model"], "region": r["region"]}
        for k, v in r.items():
            if k in ("model", "subj_id", "region"):
                continue
            try:
                d[k] = float(v)
            except (ValueError, TypeError):
                d[k] = np.nan
        rows.append(d)
    return rows


def stat(rows, model, col, region=None):
    vals = [r[col] for r in rows if r["model"] == model and col in r
            and not np.isnan(r[col]) and (region is None or r["region"] == region)]
    return (np.mean(vals), np.std(vals), len(vals)) if vals else (np.nan, np.nan, 0)


def metric_panel(rows, metrics, labels, lower, out, suptitle):
    """One subplot per metric; bar=mean over subjects, errorbar=std; x=models."""
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(2.7 * n, 4.2))
    if n == 1:
        axes = [axes]
    x = np.arange(len(MODELS))
    for ax, met, lab in zip(axes, metrics, labels):
        means = [stat(rows, m, met)[0] for m in MODELS]
        stds = [stat(rows, m, met)[1] for m in MODELS]
        ax.bar(x, means, yerr=stds, capsize=4, color=[COLORS[m] for m in MODELS],
               edgecolor="#222", linewidth=0.5, error_kw=dict(lw=1, ecolor="#444"))
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
        arrow = "↓" if lower.get(met) else "↑"
        ax.set_title(f"{lab} {arrow}", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
        # mark best (min if lower else max)
        best = int(np.nanargmin(means) if lower.get(met) else np.nanargmax(means))
        ax.get_children()  # noop
        ax.text(best, means[best], "★", ha="center", va="bottom", fontsize=11, color="#d4af37")
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[bars] wrote {out}")


def region_panel(rows, col, lower, out, title):
    """Grouped bars: x = regions, one bar per model, mean ± std within region."""
    fig, ax = plt.subplots(figsize=(11, 4.6))
    nb = len(MODELS)
    w = 0.8 / nb
    xr = np.arange(len(REGIONS))
    for i, m in enumerate(MODELS):
        means = [stat(rows, m, col, region=rg)[0] for rg in REGIONS]
        stds = [stat(rows, m, col, region=rg)[1] for rg in REGIONS]
        ax.bar(xr + i * w - 0.4 + w / 2, means, w, yerr=stds, capsize=2, label=m,
               color=COLORS[m], edgecolor="#222", linewidth=0.4,
               error_kw=dict(lw=0.7, ecolor="#666"))
    ax.set_xticks(xr)
    ax.set_xticklabels(REGIONS)
    arrow = "↓" if lower else "↑"
    ax.set_title(f"{title} {arrow}  (per region, mean ± std)", fontsize=12)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[bars] wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval_root", required=True)
    args = ap.parse_args()
    rows = load_rows(args.eval_root)
    # Restrict to models actually present in the data so a subset eval (e.g. just
    # unet + koalAI) doesn't render empty ghost bars for absent models.
    global MODELS
    present = {r["model"] for r in rows}
    MODELS = [m for m in MODELS if m in present]
    figs = os.path.join(args.eval_root, "figures")
    os.makedirs(figs, exist_ok=True)

    metric_panel(
        rows,
        ["body_mae_hu", "body_psnr", "body_ssim", "dice_score_bone", "dice_score_all"],
        ["body MAE (HU)", "body PSNR", "body SSIM", "Hard Bone Dice", "Hard Dice"],
        {"body_mae_hu": True},
        os.path.join(figs, "trackA_bars.png"),
        "Track A — amix-clip [-1024,1024], body-masked (mean ± std over 207 val subjects)",
    )
    metric_panel(
        rows,
        ["synthrad_mae", "synthrad_psnr", "synthrad_ms_ssim"],
        ["MAE (HU)", "PSNR", "MS-SSIM"],
        {"synthrad_mae": True},
        os.path.join(figs, "trackB_bars.png"),
        "Track B — SynthRAD-native [-1024,3000] (mean ± std over 207 val subjects)",
    )
    region_panel(rows, "body_mae_hu", True, os.path.join(figs, "region_mae_bars.png"),
                 "body MAE (HU)")
    region_panel(rows, "body_psnr", False, os.path.join(figs, "region_psnr_bars.png"),
                 "body PSNR")
    region_panel(rows, "dice_score_bone", False, os.path.join(figs, "region_bonedice_bars.png"),
                 "Hard Bone Dice")
    region_panel(rows, "dice_score_all", False, os.path.join(figs, "region_dice_bars.png"),
                 "Hard Dice")


if __name__ == "__main__":
    main()
