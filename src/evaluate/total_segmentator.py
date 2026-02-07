import os
import glob
import argparse
import pandas as pd
import nibabel as nib
import numpy as np
from tqdm import tqdm
from totalsegmentator.python_api import totalsegmentator

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from src.data import get_region_key, REGION_MAPS

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
            dice_scores[name] = 1.0
        else:
            dice_scores[name] = (2.0 * inter) / (union)
            
    return dice_scores

def run_functional_eval(pred_dir, data_root, gt_res_arg, device="gpu"):
    # Map simplified arg to directory name
    res_map = {"1x1x1": "1.0x1.0x1.0mm", "3x3x3": "3.0x3.0x3.0mm"}
    gt_res_dir = res_map[gt_res_arg]
    
    output_csv = os.path.join(pred_dir, f"functional_eval_{gt_res_arg}.csv")
    
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
    print(f"   Target GT Res: {gt_res_arg} ({gt_res_dir})\n")
    
    all_results = []

    for pf in tqdm(pred_files, desc="Evaluating Subjects"):
        subj_id = os.path.basename(pf).replace("pred_", "").replace(".nii.gz", "")
        
        # 1. Locate Ground Truth
        gt_seg_path = None
        for split in ["val", "test", "train"]:
            p = os.path.join(data_root, gt_res_dir, split, subj_id, "ct_seg.nii.gz")
            if os.path.exists(p):
                gt_seg_path = p
                break
        
        if not gt_seg_path:
            tqdm.write(f"   ⚠️ Skipping {subj_id}: Ground truth not found in {gt_res_dir}/[val,test,train]")
            continue

        # 2. Run TotalSegmentator
        pred_seg_path = pf.replace(".nii.gz", "_seg.nii.gz")
        if not os.path.exists(pred_seg_path):
            try:
                totalsegmentator(pf, pred_seg_path, task="total", ml=True, fast=False, device=device, quiet=True)
            except Exception as e:
                tqdm.write(f"   💥 Failed to segment {subj_id}: {e}")
                continue
        
        # 3. Load and Compute Dice
        try:
            gt_seg = nib.load(gt_seg_path).get_fdata()
            pred_seg = nib.load(pred_seg_path).get_fdata()
            
            region_key = get_region_key(subj_id)
            organs = REGION_MAPS.get(region_key, {})
            
            scores = compute_dice(gt_seg, pred_seg, organs)
            if scores is None: continue

            scores['subj_id'] = subj_id
            scores['region'] = region_key
            
            organ_vals = [v for k, v in scores.items() if k not in ['subj_id', 'region']]
            scores['avg_dice'] = np.mean(organ_vals) if organ_vals else 0.0
            
            all_results.append(scores)
            tqdm.write(f"   ✅ {subj_id} | Avg Dice: {scores['avg_dice']:.4f}")

        except Exception as e:
            tqdm.write(f"   ⚠️ Error processing {subj_id}: {e}")

    # 4. Save
    if all_results:
        df = pd.DataFrame(all_results)
        cols = ['subj_id', 'region', 'avg_dice'] + [c for c in df.columns if c not in ['subj_id', 'region', 'avg_dice']]
        df[cols].to_csv(output_csv, index=False)
        print(f"\n✨ Done! Results: {output_csv}")
        print(df.groupby('region')['avg_dice'].mean().to_string())
    else:
        print("❌ No evaluation results generated.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI2CT Evaluation")
    parser.add_argument("path", type=str, help="Path to predictions")
    parser.add_argument("--data", type=str, default="./dataset", help="Dataset root")
    parser.add_argument("--gt_res", type=str, default="1x1x1", choices=["1x1x1", "3x3x3"], help="GT resolution")
    parser.add_argument("--device", type=str, default="gpu", choices=["gpu", "cpu"])
    
    args = parser.parse_args()
    run_functional_eval(args.path, args.data, args.gt_res, device=args.device)