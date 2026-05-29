"""Compare two trained runs head-to-head from already-saved validation artifacts.

No model loading / no GPU: reuses each run's `val_rankings.txt` (per-subject metrics,
computed by the training validation loop) and `predictions/<snapshot>/pred_<subj>.nii.gz`
volumes, plus the GT CT / input MRI from the data root.

Defaults are wired for the two brain UNet-baseline runs:
  Model A (center-wise / OOD): 6y9h1g7v
  Model B (random      / iid): 6lqy7zz7

Outputs (under --out_dir):
  <prefix>_metrics_summary.txt   human-readable aggregate table
  <prefix>_metrics_summary.csv   per-subject metrics, both runs
  <prefix>_error_chart.png       mean +/- std bars over each run's own val set
  <prefix>_paired_overlap.png    paired slope plot over shared val subjects
  <prefix>_overlap_slices.png    MRI | GT CT | Model A | Model B, middle slice
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.data import get_split_subjects, get_subject_paths  # noqa: E402

GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
DEFAULT_RUN_A = f"{GPFS}/wandb_logs/runs/20260521_2153_6y9h1g7v"
DEFAULT_RUN_B = f"{GPFS}/wandb_logs/runs/20260521_2153_6lqy7zz7"
DEFAULT_ROOT = f"{GPFS}/SynthRAD/1.5mm_registered_flat_masked"

# val_rankings.txt columns after `rank subject_id`
METRIC_KEYS = ["mae_hu", "ssim", "psnr", "dice_score_all", "dice_score_bone"]
LOWER_BETTER = {"mae_hu"}


def parse_val_rankings(path):
    """Parse `val_rankings.txt` -> (step, {subj_id: {metric: float}})."""
    step = None
    metrics = {}
    with open(path) as f:
        lines = f.readlines()
    if lines and lines[0].startswith("#") and "Step" in lines[0]:
        step = lines[0].strip().split("Step")[-1].strip().split()[0]
    for line in lines:
        parts = line.split()
        # data rows: rank(int) subj_id mae ssim psnr dice_all dice_bone
        if len(parts) < 2 + len(METRIC_KEYS):
            continue
        if not parts[0].isdigit():
            continue
        subj = parts[1]
        vals = parts[2 : 2 + len(METRIC_KEYS)]
        try:
            metrics[subj] = {k: float(v) for k, v in zip(METRIC_KEYS, vals)}
        except ValueError:
            continue
    return step, metrics


def agg(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), std=float(a.std()), median=float(np.median(a)),
               min=float(a.min()), max=float(a.max()), n=len(a))


def axial_axis(affine):
    """Index of the voxel axis most aligned with superior-inferior (world z)."""
    return int(np.argmax(np.abs(affine[2, :3])))


def mid_slice(vol, affine):
    """Return the middle axial slice, oriented upright for display."""
    ax = axial_axis(affine)
    idx = vol.shape[ax] // 2
    sl = np.take(vol, idx, axis=ax)
    return np.rot90(sl)


def load_vol(path):
    # reorient to canonical RAS so GT/MRI (raw on disk) and predictions
    # (saved in RAS by the trainer) share the same frame before slicing
    img = nib.as_closest_canonical(nib.load(path))
    return np.asanyarray(img.dataobj, dtype=np.float32), img.affine


def write_summary(out_path, step, label_a, label_b, mets_a, mets_b, overlap):
    def block(label, mets, subjects):
        lines = [f"## {label}  (n={len(subjects)} val subjects)"]
        for k in METRIC_KEYS:
            s = agg([mets[s][k] for s in subjects])
            arrow = "(lower better)" if k in LOWER_BETTER else "(higher better)"
            lines.append(f"  {k:<16} mean={s['mean']:.4f}  std={s['std']:.4f}  "
                         f"median={s['median']:.4f}  min={s['min']:.4f}  max={s['max']:.4f}  {arrow}")
        return "\n".join(lines)

    val_a = sorted(mets_a)
    val_b = sorted(mets_b)
    out = [
        "Model type: UNet baseline (brain)",
        f"Snapshot: step {step}",
        f"Model A = {label_a}   Model B = {label_b}",
        "Metrics reused verbatim from each run's val_rankings.txt (training validation loop).",
        "",
        "=" * 70,
        "Aggregate over each model's OWN val set",
        "=" * 70,
        block(f"Model A ({label_a})", mets_a, val_a),
        "",
        block(f"Model B ({label_b})", mets_b, val_b),
        "",
        "=" * 70,
        f"Paired aggregate over the {len(overlap)} SHARED val subjects",
        "=" * 70,
        block(f"Model A ({label_a})", mets_a, overlap),
        "",
        block(f"Model B ({label_b})", mets_b, overlap),
        "",
        "Shared subjects: " + ", ".join(overlap),
    ]
    with open(out_path, "w") as f:
        f.write("\n".join(out) + "\n")


def write_csv(out_path, label_a, label_b, mets_a, mets_b, overlap):
    overlap_set = set(overlap)
    rows = ["subject_id,model,split,in_overlap," + ",".join(METRIC_KEYS)]
    for label, mets in [(label_a, mets_a), (label_b, mets_b)]:
        for subj in sorted(mets):
            vals = ",".join(f"{mets[subj][k]:.6f}" for k in METRIC_KEYS)
            rows.append(f"{subj},{label},{label},{subj in overlap_set},{vals}")
    with open(out_path, "w") as f:
        f.write("\n".join(rows) + "\n")


def plot_error_chart(out_path, label_a, label_b, mets_a, mets_b, step):
    val_a, val_b = sorted(mets_a), sorted(mets_b)
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(4 * len(METRIC_KEYS), 4.5))
    for ax, k in zip(axes, METRIC_KEYS):
        sa = agg([mets_a[s][k] for s in val_a])
        sb = agg([mets_b[s][k] for s in val_b])
        ax.bar([0, 1], [sa["mean"], sb["mean"]], yerr=[sa["std"], sb["std"]],
               capsize=6, color=["#d95f02", "#1b9e77"], width=0.6)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"A\n{label_a}", f"B\n{label_b}"])
        arrow = "↓" if k in LOWER_BETTER else "↑"
        ax.set_title(f"{k} {arrow}")
        for x, s in zip([0, 1], [sa, sb]):
            ax.text(x, s["mean"], f"{s['mean']:.3f}", ha="center", va="bottom", fontsize=9)
    fig.suptitle(f"UNet baseline (brain) — mean ± std over each model's own val set "
                 f"(n=60 each, step {step})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_paired(out_path, label_a, label_b, mets_a, mets_b, overlap, step):
    fig, axes = plt.subplots(1, len(METRIC_KEYS), figsize=(4 * len(METRIC_KEYS), 4.8))
    for ax, k in zip(axes, METRIC_KEYS):
        ya = [mets_a[s][k] for s in overlap]
        yb = [mets_b[s][k] for s in overlap]
        for a, b in zip(ya, yb):
            ax.plot([0, 1], [a, b], color="0.7", lw=0.8, zorder=1)
        ax.scatter([0] * len(ya), ya, color="#d95f02", zorder=2, s=18)
        ax.scatter([1] * len(yb), yb, color="#1b9e77", zorder=2, s=18)
        ax.plot([0, 1], [np.mean(ya), np.mean(yb)], color="k", lw=2.5, marker="o", zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"A\n{label_a}", f"B\n{label_b}"])
        arrow = "↓" if k in LOWER_BETTER else "↑"
        ax.set_title(f"{k} {arrow}")
    fig.suptitle(f"UNet baseline (brain) — paired on {len(overlap)} shared val subjects "
                 f"(step {step}); black = mean", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_overlap_slices(out_path, root, run_a, run_b, snapshot, label_a, label_b,
                        mets_a, mets_b, overlap, hu_window):
    vmin, vmax = hu_window
    n = len(overlap)
    fig, axes = plt.subplots(n, 4, figsize=(13, 3.6 * n))
    if n == 1:
        axes = axes[None, :]
    col_titles = ["Input MRI", "GT CT", f"Model A: {label_a}", f"Model B: {label_b}"]
    for r, subj in enumerate(overlap):
        paths = get_subject_paths(root, subj)
        mri, mri_aff = load_vol(paths["mri"])
        ct, ct_aff = load_vol(paths["ct"])
        pa, pa_aff = load_vol(os.path.join(run_a, "predictions", snapshot, f"pred_{subj}.nii.gz"))
        pb, pb_aff = load_vol(os.path.join(run_b, "predictions", snapshot, f"pred_{subj}.nii.gz"))

        mri_s = mid_slice(mri, mri_aff)
        lo, hi = np.percentile(mri_s, [1, 99])
        panels = [
            (mri_s, lo, hi, "gray"),
            (mid_slice(ct, ct_aff), vmin, vmax, "gray"),
            (mid_slice(pa, pa_aff), vmin, vmax, "gray"),
            (mid_slice(pb, pb_aff), vmin, vmax, "gray"),
        ]
        for c, (img, lo_, hi_, cmap) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=cmap, vmin=lo_, vmax=hi_)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(col_titles[c], fontsize=11)
        ma, mb = mets_a[subj], mets_b[subj]
        axes[r, 0].set_ylabel(subj, fontsize=11, rotation=0, labelpad=30, va="center")

        def metric_str(m):
            return (f"MAE {m['mae_hu']:.1f}  SSIM {m['ssim']:.3f}  PSNR {m['psnr']:.1f}\n"
                    f"Dice-all {m['dice_score_all']:.3f}  Dice-bone {m['dice_score_bone']:.3f}")

        # place metrics as a caption under each prediction panel (own gap, no overlap)
        axes[r, 2].text(0.5, -0.06, metric_str(ma), transform=axes[r, 2].transAxes,
                        ha="center", va="top", fontsize=8)
        axes[r, 3].text(0.5, -0.06, metric_str(mb), transform=axes[r, 3].transAxes,
                        ha="center", va="top", fontsize=8)
    fig.subplots_adjust(left=0.06, right=0.99, top=1 - 0.9 / n, bottom=0.01,
                        wspace=0.05, hspace=0.4)
    fig.suptitle(f"UNet baseline (brain) — shared val subjects, middle axial slice "
                 f"(CT window [{vmin:.0f}, {vmax:.0f}] HU)", fontsize=14, y=1 - 0.25 / n)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_a", default=DEFAULT_RUN_A)
    p.add_argument("--run_b", default=DEFAULT_RUN_B)
    p.add_argument("--label_a", default="center-wise")
    p.add_argument("--label_b", default="random")
    p.add_argument("--split_a", default="splits/brain_center_wise_split.txt")
    p.add_argument("--split_b", default="splits/brain_random_split.txt")
    p.add_argument("--root_dir", default=DEFAULT_ROOT)
    p.add_argument("--snapshot", default="last")
    p.add_argument("--out_dir", default="evaluation_results/brain_unet_centerwise_vs_random")
    p.add_argument("--prefix", default="unet_brain")
    p.add_argument("--hu_window", default="-100,100")  # brain soft-tissue window
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    hu_window = tuple(float(x) for x in args.hu_window.split(","))

    step_a, mets_a = parse_val_rankings(os.path.join(args.run_a, "val_rankings.txt"))
    step_b, mets_b = parse_val_rankings(os.path.join(args.run_b, "val_rankings.txt"))
    if step_a != step_b:
        print(f"[WARN] step mismatch: A={step_a} B={step_b}")
    step = step_a

    val_a = set(get_split_subjects(args.split_a, "val"))
    val_b = set(get_split_subjects(args.split_b, "val"))
    overlap = sorted(val_a & val_b)

    # sanity: parsed subjects should match the split's val set
    for name, mets, vs in [("A", mets_a, val_a), ("B", mets_b, val_b)]:
        missing = vs - set(mets)
        if missing:
            print(f"[WARN] run {name}: {len(missing)} val subjects absent from val_rankings: "
                  f"{sorted(missing)}")
    print(f"Step {step} | A={len(mets_a)} subj, B={len(mets_b)} subj | overlap={len(overlap)}")

    pre = lambda name: os.path.join(args.out_dir, f"{args.prefix}_{name}")
    write_summary(pre("metrics_summary.txt"), step, args.label_a, args.label_b,
                  mets_a, mets_b, overlap)
    write_csv(pre("metrics_summary.csv"), args.label_a, args.label_b, mets_a, mets_b, overlap)
    plot_error_chart(pre("error_chart.png"), args.label_a, args.label_b, mets_a, mets_b, step)
    plot_paired(pre("paired_overlap.png"), args.label_a, args.label_b, mets_a, mets_b, overlap, step)
    plot_overlap_slices(pre("overlap_slices.png"), args.root_dir, args.run_a, args.run_b,
                        args.snapshot, args.label_a, args.label_b, mets_a, mets_b, overlap, hu_window)

    print("Wrote:")
    for name in ["metrics_summary.txt", "metrics_summary.csv", "error_chart.png",
                 "paired_overlap.png", "overlap_slices.png"]:
        print("  " + pre(name))


if __name__ == "__main__":
    main()
