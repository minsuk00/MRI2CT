import argparse
import gc
import glob
import math
import os
import subprocess
import sys

import torch
from tqdm import tqdm

# Ensure we can import register_single_subject and anatomix
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../../"))
if script_dir not in sys.path:
    sys.path.append(script_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from register_single_subject import convex_adam

# ==========================================
# 1. CONFIGURATION
# ==========================================
# python src/preprocess/register_batch_sharded.py --part 0 --total_parts 1 --limit 1
# python src/preprocess/register_batch_sharded.py --part 0 --total_parts 1

# Output from mask_native_batch.py
BASE_OUT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/native_masked"
CKPT_PATH = "anatomix/model-weights/best_val_net_G.pth"


# ==========================================
# 2. UTILITIES
# ==========================================
def discover_subjects(data_root):
    if not os.path.exists(data_root):
        print(f"❌ Error: {data_root} does not exist.")
        return []

    subjects = []
    # Subjects are patient ID folders in BASE_OUT
    folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for pid in folders:
        subj_dir = os.path.join(data_root, pid)
        # Check for mr and ct files (mask_native_batch saves as .nii)
        mr_files = glob.glob(os.path.join(subj_dir, "mr.nii*"))
        ct_files = glob.glob(os.path.join(subj_dir, "ct.nii*"))

        if mr_files and ct_files:
            subjects.append({"id": pid, "path": subj_dir, "mr": mr_files[0], "ct": ct_files[0]})
    return subjects


# ==========================================
# 3. WORKER FUNCTION
# ==========================================
def register_subject(subj, fail_log):
    """
    Wraps the registration logic to ensure all tensors and the model
    are local and destroyed after the function returns, reclaiming memory.
    """
    tqdm.write(f"👉 Processing Subject: {subj['id']}")

    try:
        convex_adam(
            ckpt_path=CKPT_PATH,
            expname="synthrad",
            lambda_weight=1.25,
            grid_sp=3,
            disp_hw=4,
            selected_niter=80,
            selected_smooth=0,
            grid_sp_adam=1,
            ic=True,
            result_path=subj["path"],
            fixed_image=subj["ct"],
            moving_image=subj["mr"],
            fixed_minclip=-450,
            fixed_maxclip=450,
            moving_minclip=None,
            moving_maxclip=None,
        )
        # Report VRAM usage
        peak_vram = torch.cuda.max_memory_allocated() / (1024**3)
        tqdm.write(f"📊 Peak VRAM: {peak_vram:.2f} GB")
        return True

    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        tqdm.write(f"💥 Registration Failed for {subj['id']}: {error_msg}")
        with open(fail_log, "a") as f:
            f.write(f"{subj['id']} | {error_msg}\n")
        return False
    finally:
        # Aggressive cleanup after every iteration (success or failure)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        gc.collect()


# ==========================================
# 4. MAIN LOOP
# ==========================================
def run_batch_registration(part_idx=0, total_parts=1, limit=None):
    all_subjects = discover_subjects(BASE_OUT)

    if not all_subjects:
        print(f"❌ No valid subjects found in {BASE_OUT}.")
        return

    # Filter Pending Subjects BEFORE Sharding
    pending_subjects = []
    for subj in all_subjects:
        mr_name = os.path.basename(subj["mr"])
        movsavename = mr_name.replace(".nii.gz", "").replace(".nii", "")
        # Expected output .nii file
        out_name = f"moved_{movsavename}_g3_hw4_l1.25_ga1_icTrue_synthrad.nii"
        out_path = os.path.join(subj["path"], out_name)
        if not os.path.exists(out_path):
            pending_subjects.append(subj)

    if not pending_subjects:
        print("✅ All subjects already processed.")
        return

    if limit:
        pending_subjects = pending_subjects[:limit]

    total_subjs = len(pending_subjects)
    chunk_size = math.ceil(total_subjs / total_parts)
    start_idx = part_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_subjs)

    my_subjects = pending_subjects[start_idx:end_idx]

    # Error tracking file for this specific shard
    fail_log = os.path.join(project_root, f"register_failures_part_{part_idx}.txt")

    print(f"\n🚀 Starting Registration Part [{part_idx + 1}/{total_parts}]")
    print(f"   Processing {len(my_subjects)} subjects (indices {start_idx} to {end_idx})")

    for subj in tqdm(my_subjects, desc=f"Part {part_idx + 1}"):
        register_subject(subj, fail_log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="Index of this job part (0-based)")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts")
    parser.add_argument("--limit", type=int, default=None, help="Limit total subjects (for testing)")
    args = parser.parse_args()

    run_batch_registration(args.part, args.total_parts, args.limit)
