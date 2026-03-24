import argparse
import glob
import os
import sys

import nibabel as nib
import numpy as np
import pandas as pd
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

# Add project root and src to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from common.data import get_region_key


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


def run_functional_eval(pred_dir, data_root, device="gpu", fast=False):
    # Only using 1.5mm registered data as per user request
    gt_res_dir = "1.5x1.5x1.5mm_registered"

    output_csv = os.path.join(pred_dir, "functional_eval_1.5mm.csv")

    if not os.path.exists(pred_dir):
        print(f"❌ Error: Prediction directory not found: {pred_dir}")
        return

    pred_files = sorted(glob.glob(os.path.join(pred_dir, "pred_*.nii.gz")))
    pred_files = [f for f in pred_files if not f.endswith("_seg.nii.gz")]

    if not pred_files:
        print(f"❌ No predicted volumes (pred_*.nii.gz) found in {pred_dir}")
        return

    print(f"🚀 Starting Functional Evaluation")
    print(f"   Directory: {pred_dir}")
    print(f"   Target GT Res: 1.5x1.5x1.5mm_registered\n")

    all_results = []

    for pf in tqdm(pred_files, desc="Evaluating Subjects"):
        subj_id = os.path.basename(pf).replace("pred_", "").replace(".nii.gz", "")

        # 1. Locate Ground Truth CT
        gt_ct_path = None
        for split in ["val", "test", "train"]:
            subj_dir = os.path.join(data_root, gt_res_dir, split, subj_id)
            for ext in ["ct.nii.gz", "ct.nii"]:
                p = os.path.join(subj_dir, ext)
                if os.path.exists(p):
                    gt_ct_path = p
                    break
            if gt_ct_path:
                break

        if not gt_ct_path:
            tqdm.write(f"   ⚠️ Skipping {subj_id}: Ground truth CT not found in {gt_res_dir}/[val,test,train]")
            continue

        # 2. Run TotalSegmentator on BOTH (Explicit naming)
        gt_seg_path = os.path.join(pred_dir, f"{subj_id}_gt_totalseg.nii.gz")
        pred_seg_path = os.path.join(pred_dir, f"{subj_id}_pred_totalseg.nii.gz")

        try:
            if not os.path.exists(gt_seg_path):
                tqdm.write(f"   🪄 Segmenting GT: {subj_id}...")
                totalsegmentator(gt_ct_path, gt_seg_path, task="total", ml=True, fast=fast, device=device, quiet=True)

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
    else:
        print("❌ No evaluation results generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI2CT Evaluation")
    parser.add_argument("path", type=str, help="Path to predictions")
    parser.add_argument("--data", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined", help="Dataset root")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--fast", action="store_true", help="Run TotalSegmentator in fast mode (3mm)")

    args = parser.parse_args()
    run_functional_eval(args.path, args.data, device=args.device, fast=args.fast)
