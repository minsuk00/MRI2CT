import os
import time
import argparse
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm
import math

def discover_subjects(data_root, target_list=None):
    """
    Scans data_root/train, data_root/val, and data_root/test for subjects.
    
    Args:
        data_root: Root directory containing split folders.
        target_list: Optional list of subject IDs to process.
        
    Returns:
        list: A list of dictionaries with subject details.
    """
    valid_subjects = []
    splits = ["train", "val", "test"]

    print(f"Scanning subjects in {data_root}...")

    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path): continue
            
        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        for subj_id in candidates:
            if target_list and subj_id not in target_list: continue

            subj_path = os.path.join(split_path, subj_id)
            mr_path = os.path.join(subj_path, "mr.nii.gz")
            ct_path = os.path.join(subj_path, "ct.nii.gz")

            if os.path.exists(mr_path) and os.path.exists(ct_path):
                valid_subjects.append({
                    "id": subj_id,
                    "path": subj_path,
                    "split": split,
                    "mr": mr_path,
                    "ct": ct_path
                })
    return valid_subjects

def run_sharded_segmentation(data_dir, part_idx=0, total_parts=1, device="gpu", fast_mode=False):
    """
    Run TotalSegmentator on a shard of subjects. Useful for cluster array jobs.
    
    Args:
        data_dir: Path to the directory containing processed resolution data.
        part_idx: Index of this job part (0-based).
        total_parts: Total number of parts to split the subjects into.
        device: Device to use (gpu or cpu).
        fast_mode: If True, uses the fast version of TotalSegmentator.
    """
    # 1. Discover ALL Subjects
    all_subjects = discover_subjects(data_dir, target_list=None)
    
    if not all_subjects:
        print(f"No valid subjects found in {data_dir}.")
        return

    # 2. Shard the data
    total_subjs = len(all_subjects)
    chunk_size = math.ceil(total_subjs / total_parts)
    start_idx = part_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_subjs)
    
    my_subjects = all_subjects[start_idx:end_idx]

    print(f"\nStarting Job Part [{part_idx+1}/{total_parts}]")
    print(f"   Processing indices {start_idx} to {end_idx} ({len(my_subjects)} subjects)")
    print(f"   Device: {device} | Fast Mode: {fast_mode}")

    # 3. Process Only Assigned Subjects
    for subj in tqdm(my_subjects, desc=f"Part {part_idx+1}", unit="subj"):
        ct_out = os.path.join(subj['path'], "ct_seg.nii.gz")
        mr_out = os.path.join(subj['path'], "mr_seg.nii.gz")
        body_out = os.path.join(subj['path'], "body_mask.nii.gz")
        
        # --- Body Mask ---
        if not (os.path.exists(body_out) and not os.path.isdir(body_out)):
            if os.path.isdir(body_out):
                import shutil
                shutil.rmtree(body_out)
                
            try:
                totalsegmentator(
                    input=subj['ct'], output=body_out, 
                    task="body", device=device, quiet=True, ml=True
                )
            except Exception as e:
                tqdm.write(f"Body Mask Failed {subj['id']}: {e}")

        # --- CT Segmentation ---
        if not os.path.exists(ct_out):
            try:
                totalsegmentator(
                    input=subj['ct'], output=ct_out, ml=True, 
                    task="total", fast=fast_mode, device=device, quiet=True
                )
            except Exception as e:
                tqdm.write(f"CT Seg Failed {subj['id']}: {e}")

        # --- MR Segmentation ---
        if not os.path.exists(mr_out):
            try:
                totalsegmentator(
                    input=subj['mr'], output=mr_out, ml=True, 
                    task="total_mr", fast=fast_mode, device=device, quiet=True
                )
            except Exception as e:
                tqdm.write(f"MR Seg Failed {subj['id']}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="Index of this job part (0-based)")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts")
    args = parser.parse_args()
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"
    DATA_DIR = os.path.join(ROOT, "1.0x1.0x1.0mm")
    DEVICE = "gpu" 
    FAST_MODE = False

    run_sharded_segmentation(
        data_dir=DATA_DIR,
        part_idx=args.part,
        total_parts=args.total_parts,
        device=DEVICE,
        fast_mode=FAST_MODE
    )
