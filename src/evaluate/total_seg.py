import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

# Add project root and src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from common.data import get_region_key

# --- configure inline ---
PRED_DIR = "/home/minsukc/MRI2CT/wandb/runs/_y0c2wryv/predictions/epoch_260"
FAST = False  # True = 3mm fast mode
# ------------------------

DATA_ROOT = "/home/minsukc/MRI2CT/dataset/1.5mm_registered_flat_masked"
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


# ==========================================
# 2. EVALUATION LOGIC
# ==========================================
def compute_dice(seg_a, seg_b, labels):
    """Computes Dice score for specified labels between two segmentation volumes."""
    if seg_a.shape != seg_b.shape:
        print(f"\n   ⚠️ SHAPE MISMATCH: GT {seg_a.shape} vs Pred {seg_b.shape}. Skipping.")
        return None

    dice_scores = {}
    for name, val in labels.items():
        mask_a = np.isin(seg_a, val) if isinstance(val, list) else (seg_a == val)
        mask_b = np.isin(seg_b, val) if isinstance(val, list) else (seg_b == val)

        inter = np.sum(mask_a * mask_b)
        union = np.sum(mask_a) + np.sum(mask_b)

        if union == 0:
            dice_scores[name] = np.nan
        else:
            dice_scores[name] = (2.0 * inter) / (union)

    return dice_scores


STRUCTURES_CSV = os.path.join(os.path.dirname(__file__), "structures.csv")
GROUP_COLORS = {
    "organs": "#4C72B0",
    "vertebrae": "#8B6914",
    "cardiac": "#C44E52",
    "skeleton": "#55A868",
    "muscles": "#DD8452",
    "ribs": "#9B59B6",
    "test": "#888888",
}


def _load_group_map():
    """Returns {structure_name: group} from structures.csv."""
    sdf = pd.read_csv(STRUCTURES_CSV)
    return dict(zip(sdf["structure"], sdf["group"]))


def _organ_bar_chart(ax, means, stds, title, n_subjects, overall_mean, group_map=None):
    x = np.arange(len(means))
    colors = []
    if group_map:
        for organ in means.index:
            grp = group_map.get(organ, "test")
            colors.append(GROUP_COLORS.get(grp, "#888888"))
    else:
        colors = ["steelblue"] * len(means)

    ax.bar(x, means.values, yerr=stds.values, capsize=2, color=colors, alpha=0.85, error_kw={"linewidth": 0.5})
    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=90, fontsize=5)
    ax.set_ylabel("Dice Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"{title} (n={n_subjects})")
    # overall_mean is passed in as true mean over all (subj x organ) pairs
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=0.8, label=f"mean={overall_mean:.3f}")

    if group_map:
        from matplotlib.patches import Patch

        legend_handles = [Patch(color=c, label=g) for g, c in GROUP_COLORS.items() if g != "test"]
        legend_handles.insert(0, plt.Line2D([0], [0], color="red", linestyle="--", label=f"mean={overall_mean:.3f}"))
        ax.legend(handles=legend_handles, fontsize=7, loc="upper right")
    else:
        ax.legend(fontsize=8)


def plot_results(df, pred_dir):
    organ_cols = [c for c in df.columns if c not in ["subj_id", "region", "avg_dice"]]
    organ_df = df[organ_cols].dropna(axis=1, how="all")
    run_tag = f"{pred_dir.split('/')[-3]}/{pred_dir.split('/')[-1]}"
    group_map = _load_group_map()

    def flat_mean(odf):
        vals = odf.values.flatten()
        return float(np.nanmean(vals))

    # 1. Overall organ bar chart (colored by group, sorted by Dice)
    means = organ_df.mean().sort_values(ascending=False)
    stds = organ_df.std().reindex(means.index)
    overall_mean = flat_mean(organ_df)
    fig, ax = plt.subplots(figsize=(max(14, len(means) * 0.18), 6))
    _organ_bar_chart(ax, means, stds, f"Mean Dice per Organ — {run_tag}", len(df), overall_mean, group_map)
    plt.tight_layout()
    out_all = os.path.join(pred_dir, "dice_per_organ.png")
    plt.savefig(out_all, dpi=150)
    plt.close(fig)
    print(f"   Saved: {out_all}")

    # 2. Group-mean summary bar chart (flat mean per group over all subj x organ pairs)
    organ_groups = pd.Series({organ: group_map.get(organ, "test") for organ in organ_df.columns})

    def flat_std(odf):
        vals = odf.values.flatten()
        return float(np.nanstd(vals))

    group_flat_means = {grp: flat_mean(organ_df.loc[:, organ_groups == grp]) for grp in organ_groups.unique()}
    group_flat_stds = {grp: flat_std(organ_df.loc[:, organ_groups == grp]) for grp in organ_groups.unique()}
    group_means = pd.Series(group_flat_means).sort_values(ascending=False)
    group_stds = pd.Series(group_flat_stds).reindex(group_means.index)
    fig, ax = plt.subplots(figsize=(8, 5))
    grp_colors = [GROUP_COLORS.get(g, "#888888") for g in group_means.index]
    ax.bar(np.arange(len(group_means)), group_means.values, yerr=group_stds.values, capsize=4, color=grp_colors, alpha=0.85, error_kw={"linewidth": 1})
    ax.set_xticks(np.arange(len(group_means)))
    ax.set_xticklabels(group_means.index, fontsize=10)
    ax.set_ylabel("Mean Dice Score")
    ax.set_ylim(0, 1)
    ax.set_title(f"Mean Dice by Structural Group — {run_tag} (n={len(df)})")
    ax.axhline(overall_mean, color="red", linestyle="--", linewidth=0.8, label=f"overall mean={overall_mean:.3f}")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_group = os.path.join(pred_dir, "dice_by_group.png")
    plt.savefig(out_group, dpi=150)
    plt.close(fig)
    print(f"   Saved: {out_group}")

    # 3. Per-region organ bar charts (colored by group)
    for region, group in df.groupby("region"):
        region_organ_df = group[organ_cols].dropna(axis=1, how="all")
        r_means = region_organ_df.mean().sort_values(ascending=False)
        r_stds = region_organ_df.std().reindex(r_means.index)
        r_overall_mean = flat_mean(region_organ_df)
        fig, ax = plt.subplots(figsize=(max(14, len(r_means) * 0.18), 6))
        _organ_bar_chart(ax, r_means, r_stds, f"Mean Dice per Organ [{region}] — {run_tag}", len(group), r_overall_mean, group_map)
        plt.tight_layout()
        out_region = os.path.join(pred_dir, f"dice_per_organ_{region}.png")
        plt.savefig(out_region, dpi=150)
        plt.close(fig)
        print(f"   Saved: {out_region}")


def plot_seg_overlay(pred_dir, data_root, n_slices=6):
    """
    For each subject, one figure with:
      Row 0     : plain GT CT (anatomy reference)
      Row 1–N   : one row per structural group, overlay filtered to that group's label IDs
                  GT-only=red, pred-only=cyan, overlap=yellow
    """
    gt_res_dir = "."
    pred_seg_files = sorted(glob.glob(os.path.join(pred_dir, "*_pred_totalseg.nii.gz")))
    if not pred_seg_files:
        print(f"❌ No *_pred_totalseg.nii.gz found in {pred_dir}")
        return

    # Build group → label_ids from structures.csv (preserving display order)
    sdf = pd.read_csv(STRUCTURES_CSV)
    group_order = list(dict.fromkeys(sdf["group"]))  # insertion order, drop "test"
    group_order = [g for g in group_order if g != "test"]
    group_labels = {g: sdf.loc[sdf["group"] == g, "id"].tolist() for g in group_order}

    COLORS = {
        "gt": np.array([1.0, 0.1, 0.1]),
        "pred": np.array([0.0, 0.85, 1.0]),
        "overlap": np.array([1.0, 0.95, 0.0]),
    }
    ALPHA = 0.65
    n_rows = 1 + len(group_order)  # GT row + one per group

    for pred_seg_path in pred_seg_files:
        subj_id = os.path.basename(pred_seg_path).replace("_pred_totalseg.nii.gz", "")
        subj_dir = os.path.join(data_root, gt_res_dir, subj_id)

        gt_ct_path = next((os.path.join(subj_dir, f) for f in ["ct.nii.gz", "ct.nii"] if os.path.exists(os.path.join(subj_dir, f))), None)
        gt_seg_path = os.path.join(subj_dir, "ct_totalseg.nii.gz")
        if gt_ct_path is None or not os.path.exists(gt_seg_path):
            print(f"⚠️ Skipping {subj_id}: missing GT files in {subj_dir}")
            continue

        gt_ct = nib.load(gt_ct_path).get_fdata()
        gt_seg = nib.load(gt_seg_path).get_fdata().astype(int)
        pred_seg = nib.load(pred_seg_path).get_fdata().astype(int)

        pred_ct_path = os.path.join(pred_dir, f"pred_{subj_id}.nii.gz")
        pred_ct = nib.load(pred_ct_path).get_fdata() if os.path.exists(pred_ct_path) else None

        D = gt_ct.shape[2]
        slice_indices = np.linspace(0.1 * D, 0.9 * D, n_slices, dtype=int)

        n_rows_subj = n_rows + (1 if pred_ct is not None else 0)  # +1 for pred CT row
        fig, axes = plt.subplots(n_rows_subj, n_slices, figsize=(3.5 * n_slices, 3.5 * n_rows_subj))
        fig.subplots_adjust(top=0.96, left=0.1, right=0.99, bottom=0.02, hspace=0.05, wspace=0.05)
        fig.suptitle(f"{subj_id}  |  GT-only=red  Pred-only=cyan  Overlap=yellow", fontsize=12, y=0.99)

        for col, z in enumerate(slice_indices):
            ct_slice = gt_ct[:, :, z]
            ct_norm = np.clip(ct_slice, -1024, 1024)
            ct_norm = (ct_norm - ct_norm.min()) / (ct_norm.max() - ct_norm.min() + 1e-8)
            bg = np.stack([ct_norm, ct_norm, ct_norm], axis=-1)

            # Row 0: plain GT CT
            axes[0, col].imshow(bg.transpose(1, 0, 2), origin="lower")
            axes[0, col].set_title(f"z={z}", fontsize=8)
            axes[0, col].axis("off")

            # Row 1: plain Pred CT (if available)
            row_offset = 1
            if pred_ct is not None:
                pc_slice = pred_ct[:, :, z]
                pc_norm = np.clip(pc_slice, -1024, 1024)
                pc_norm = (pc_norm - pc_norm.min()) / (pc_norm.max() - pc_norm.min() + 1e-8)
                axes[1, col].imshow(np.stack([pc_norm, pc_norm, pc_norm], axis=-1).transpose(1, 0, 2), origin="lower")
                axes[1, col].axis("off")
                row_offset = 2

            # Remaining rows: one per group
            for ri, grp in enumerate(group_order):
                ids = group_labels[grp]
                gt_mask = np.isin(gt_seg[:, :, z], ids)
                pred_mask = np.isin(pred_seg[:, :, z], ids)
                gt_only = gt_mask & ~pred_mask
                pred_only = pred_mask & ~gt_mask
                overlap = gt_mask & pred_mask

                ov = bg.copy()
                for mask, color in [(gt_only, COLORS["gt"]), (pred_only, COLORS["pred"]), (overlap, COLORS["overlap"])]:
                    ov[mask] = (1 - ALPHA) * bg[mask] + ALPHA * color

                axes[row_offset + ri, col].imshow(ov.transpose(1, 0, 2), origin="lower")
                axes[row_offset + ri, col].axis("off")

        # Row labels: set AFTER axis("off") so they aren't suppressed
        top_adj, bot_adj = 0.96, 0.02
        row_names = ["GT CT"] + (["Pred CT"] if pred_ct is not None else []) + group_order
        for ri, name in enumerate(row_names):
            y = top_adj - (ri + 0.5) / n_rows_subj * (top_adj - bot_adj)
            fig.text(0.01, y, name, fontsize=11, fontweight="bold", ha="left", va="center")

        out_path = os.path.join(pred_dir, f"seg_overlay_{subj_id}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"   Saved: {out_path}")


def run_functional_eval(pred_dir, data_root, device="gpu", fast=False):
    gt_res_dir = "."

    output_csv = os.path.join(pred_dir, "functional_eval_1.5mm.csv")

    if not os.path.exists(pred_dir):
        print(f"❌ Error: Prediction directory not found: {pred_dir}")
        return

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "pred_*.nii.gz")))
    pred_files = [f for f in pred_files if not f.endswith("_seg.nii.gz")]

    if not pred_files:
        print(f"❌ No predicted volumes (pred_*.nii.gz) found in {pred_dir}")
        return

    print("🚀 Starting Functional Evaluation")
    print(f"   Directory: {pred_dir}")
    print(f"   Target GT Res: {DATA_ROOT}\n")

    all_results = []

    for pf in tqdm(pred_files, desc="Evaluating Subjects"):
        subj_id = os.path.basename(pf).replace("pred_", "").replace(".nii.gz", "")

        # 1. Locate Ground Truth CT
        gt_ct_path = None
        subj_dir = os.path.join(data_root, gt_res_dir, subj_id)
        for ext in ["ct.nii.gz", "ct.nii"]:
            p = os.path.join(subj_dir, ext)
            if os.path.exists(p):
                gt_ct_path = p
                break

        if not gt_ct_path:
            tqdm.write(f"   ⚠️ Skipping {subj_id}: Ground truth CT not found in {gt_res_dir}/{subj_id}")
            continue

        # 2. Run TotalSegmentator on BOTH (Explicit naming)
        gt_seg_path = os.path.join(subj_dir, "ct_totalseg.nii.gz")
        pred_seg_path = os.path.join(pred_dir, f"{subj_id}_pred_totalseg.nii.gz")

        try:
            if not os.path.exists(gt_seg_path):
                tqdm.write(f"   🪄 Segmenting GT: {subj_id}...")
                totalsegmentator(gt_ct_path, gt_seg_path, task="total", ml=True, fast=fast, device=device, quiet=True)
            else:
                tqdm.write(f"   ✓ GT seg found: {subj_id}")

            if not os.path.exists(pred_seg_path):
                tqdm.write(f"   🪄 Segmenting Pred: {subj_id}...")
                totalsegmentator(pf, pred_seg_path, task="total", ml=True, fast=fast, device=device, quiet=True)
        except Exception as e:
            tqdm.write(f"   💥 Failed to segment {subj_id}: {e}")
            continue

        # 3. Load and Compute Dice
        try:
            gt_seg = nib.load(gt_seg_path).get_fdata().astype(int)
            pred_seg = nib.load(pred_seg_path).get_fdata().astype(int)

            # Use TotalSegmentator's internal map for consistent naming and exhaustive coverage
            from totalsegmentator.map_to_binary import class_map

            ts_map = class_map["total"]
            all_ts_organs = {name: lid for lid, name in ts_map.items()}

            scores = compute_dice(gt_seg, pred_seg, all_ts_organs)
            if scores is None:
                continue

            scores["subj_id"] = subj_id
            scores["region"] = get_region_key(subj_id)

            # Average only over labels that existed in GT or were hallucinated in Pred (non-NaN)
            organ_vals = [v for k, v in scores.items() if k not in ["subj_id", "region"]]
            valid_vals = [v for v in organ_vals if not np.isnan(v)]
            scores["avg_dice"] = np.mean(valid_vals) if valid_vals else 0.0

            all_results.append(scores)
            tqdm.write(f"   ✅ {subj_id} | Avg Dice: {scores['avg_dice']:.4f} ({len(valid_vals)} organs)")

        except Exception as e:
            tqdm.write(f"   ⚠️ Error processing {subj_id}: {e}")

    # 4. Save
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ["subj_id", "region", "avg_dice"] + [c for c in df.columns if c not in ["subj_id", "region", "avg_dice"]]
        df[cols].to_csv(output_csv, index=False)
        print(f"\n✨ Done! Results: {output_csv}")
        print(df.groupby("region")["avg_dice"].mean().to_string())
        print("\n📊 Generating visualizations...")
        plot_results(df, pred_dir)
        plot_seg_overlay(pred_dir, data_root)
    else:
        print("❌ No evaluation results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI2CT Functional Evaluation")
    parser.add_argument("path", type=str, nargs="?", default=PRED_DIR, help="Path to predictions (default: PRED_DIR)")
    parser.add_argument("--fast", action="store_true", default=FAST, help="Run TotalSegmentator in fast mode (3mm)")
    parser.add_argument("--overlay-only", action="store_true", help="Skip segmentation/dice; just generate overlay PNGs from existing *_pred_totalseg.nii.gz")

    args = parser.parse_args()
    if args.overlay_only:
        plot_seg_overlay(args.path, DATA_ROOT)
    else:
        run_functional_eval(args.path, DATA_ROOT, device=DEVICE, fast=args.fast)
