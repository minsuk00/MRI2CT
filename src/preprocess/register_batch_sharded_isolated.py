import argparse
import gc
import glob
import math
import multiprocessing as mp
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

# ==========================================
# 1. CONFIGURATION
# ==========================================
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
    folders = sorted([d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))])

    for pid in folders:
        subj_dir = os.path.join(data_root, pid)
        mr_files = glob.glob(os.path.join(subj_dir, "mr.nii*"))
        ct_files = glob.glob(os.path.join(subj_dir, "ct.nii*"))

        if mr_files and ct_files:
            subjects.append({"id": pid, "path": subj_dir, "mr": mr_files[0], "ct": ct_files[0]})
    return subjects


# ==========================================
# 3. WORKER FUNCTION (Runs in separate process)
# ==========================================
def register_subject_worker(subj, fail_log):
    """
    Performs registration in a subprocess and exits to clear VRAM.
    """
    try:
        # Import inside worker to ensure clean state
        from register_single_subject import convex_adam

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
    except Exception as e:
        error_msg = str(e).replace("\n", " ")
        print(f"💥 Registration Failed for {subj['id']}: {error_msg}")
        with open(fail_log, "a") as f:
            f.write(f"{subj['id']} | {error_msg}\n")


# ==========================================
# 4. MAIN LOOP
# ==========================================
def run_batch_registration(part_idx=0, total_parts=1, limit=None):
    # Ensure spawn method for CUDA safety
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    all_subjects = discover_subjects(BASE_OUT)

    if not all_subjects:
        print(f"❌ No valid subjects found in {BASE_OUT}.")
        return

    # Filter Pending Subjects
    pending_subjects = []
    for subj in all_subjects:
        mr_name = os.path.basename(subj["mr"])
        movsavename = mr_name.replace(".nii.gz", "").replace(".nii", "")
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
    fail_log = os.path.join(project_root, f"register_failures_part_{part_idx}.txt")

    print(f"\n🚀 Starting Isolated Registration Part [{part_idx + 1}/{total_parts}]")
    print(f"   Processing {len(my_subjects)} subjects (Pending total: {total_subjs})")

    for subj in tqdm(my_subjects, desc=f"Part {part_idx + 1}"):
        tqdm.write(f"👉 Processing Subject: {subj['id']}")

        # Launch subprocess
        p = mp.Process(target=register_subject_worker, args=(subj, fail_log))
        p.start()
        p.join()  # Wait for completion

        # Periodic cleanup in main process
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="Index of this job part (0-based)")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts")
    parser.add_argument("--limit", type=int, default=None, help="Limit total subjects (for testing)")
    args = parser.parse_args()

    run_batch_registration(args.part, args.total_parts, args.limit)
