import argparse
import math
import os

from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"
# DATA_DIR = os.path.join(ROOT, "3.0x3.0x3.0mm")
# DATA_DIR = os.path.join(ROOT, "1.0x1.0x1.0mm")
DATA_DIR = os.path.join(ROOT, "1.5x1.5x1.5mm")

# --- PERFORMANCE SETTINGS ---
DEVICE = "gpu"
# FAST_MODE = True: Uses 3mm model (Fastest)
FAST_MODE = False


# ==========================================
# 2. UTILITIES
# ==========================================
def discover_subjects(data_root, target_list=None):
    valid_subjects = []
    splits = ["train", "val", "test"]

    print(f"🔎 Scanning subjects in {data_root}...")

    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue

        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])

        for subj_id in candidates:
            if target_list and subj_id not in target_list:
                continue

            subj_path = os.path.join(split_path, subj_id)
            mr_path = os.path.join(subj_path, "mr.nii.gz")
            ct_path = os.path.join(subj_path, "ct.nii.gz")

            if os.path.exists(mr_path) and os.path.exists(ct_path):
                valid_subjects.append({"id": subj_id, "path": subj_path, "split": split, "mr": mr_path, "ct": ct_path})
    return valid_subjects


# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_batch_segmentation(part_idx=0, total_parts=1, make_plots=False):
    # 1. Discover ALL Subjects first
    all_subjects = discover_subjects(DATA_DIR, target_list=None)

    if not all_subjects:
        print(f"❌ No valid subjects found in {DATA_DIR}.")
        return

    # 2. Shard the data (Split into chunks)
    total_subjs = len(all_subjects)
    chunk_size = math.ceil(total_subjs / total_parts)
    start_idx = part_idx * chunk_size
    end_idx = min(start_idx + chunk_size, total_subjs)

    my_subjects = all_subjects[start_idx:end_idx]

    print(f"\n🚀 Starting Job Part [{part_idx + 1}/{total_parts}]")
    print(f"   Processing indices {start_idx} to {end_idx} ({len(my_subjects)} subjects)")
    print(f"   Device: {DEVICE} | Fast Mode: {FAST_MODE}")

    # 3. Process Only Assigned Subjects
    for subj in tqdm(my_subjects, desc=f"Part {part_idx + 1}", unit="subj"):
        ct_out = os.path.join(subj["path"], "ct_seg.nii.gz")
        mr_out = os.path.join(subj["path"], "mr_seg.nii.gz")
        body_out = os.path.join(subj["path"], "body_mask.nii.gz")

        # --- Body Mask ---
        if os.path.exists(body_out) and not os.path.isdir(body_out):
            pass
        else:
            if os.path.isdir(body_out):
                import shutil

                shutil.rmtree(body_out)

            try:
                totalsegmentator(input=subj["ct"], output=body_out, task="body", fast=FAST_MODE, device=DEVICE, quiet=True, ml=True)
            except Exception as e:
                tqdm.write(f"💥 Body Mask Failed {subj['id']}: {e}")

        # --- CT Segmentation ---
        if not os.path.exists(ct_out):
            try:
                totalsegmentator(input=subj["ct"], output=ct_out, ml=True, task="total", fast=FAST_MODE, device=DEVICE, quiet=True)
            except Exception as e:
                tqdm.write(f"💥 CT Seg Failed {subj['id']}: {e}")

        # --- MR Segmentation ---
        if not os.path.exists(mr_out):
            try:
                totalsegmentator(input=subj["mr"], output=mr_out, ml=True, task="total_mr", fast=FAST_MODE, device=DEVICE, quiet=True)
            except Exception as e:
                tqdm.write(f"💥 MR Seg Failed {subj['id']}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", type=int, default=0, help="Index of this job part (0-based)")
    parser.add_argument("--total_parts", type=int, default=1, help="Total number of parts")
    args = parser.parse_args()

    run_batch_segmentation(args.part, args.total_parts)
