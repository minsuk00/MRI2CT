import glob
import os
import random
from concurrent.futures import ProcessPoolExecutor

import nibabel as nib
import numpy as np
import torchio as tio
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
INPUT_DIR = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/native_registered"
OUTPUT_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm_registered"

# Split Ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Target spacing (mm)
TARGET_SPACING = (1.5, 1.5, 1.5)

# Hardware Constraint
MAX_WORKERS = 12


def get_region_key(subj_id):
    """Determines region key from subject ID. Raises ValueError if unknown."""
    mapping = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if not subj_id or len(subj_id) < 2:
        raise ValueError(f"Subject ID too short: {subj_id}")
    code_2 = subj_id[1:3].upper()
    code_1 = subj_id[1:2].upper()
    if code_2 in mapping:
        return mapping[code_2]
    if code_1 in mapping:
        return mapping[code_1]
    raise ValueError(f"Unknown region for subject: {subj_id}")


def process_single_subject(args):
    """
    Worker function: Resamples to 1.5mm, converts scalars to int16, and saves as .nii
    """
    subj_id, split_name, region_dir = args
    subj_src_path = os.path.join(INPUT_DIR, region_dir, subj_id)
    subj_out_path = os.path.join(OUTPUT_ROOT, split_name, subj_id)

    ct_path = os.path.join(subj_src_path, "ct.nii.gz")
    mask_path = os.path.join(subj_src_path, "mask.nii.gz")
    moved_mr_path = os.path.join(subj_src_path, "mr_moved.nii.gz")

    # Check for existence of all required files
    missing = []
    if not os.path.exists(ct_path):
        missing.append("ct.nii.gz")
    if not os.path.exists(mask_path):
        missing.append("mask.nii.gz")
    if not os.path.exists(moved_mr_path):
        missing.append("mr_moved.nii.gz")

    if missing:
        return subj_id, False, f"Missing files: {', '.join(missing)}"

    try:
        os.makedirs(subj_out_path, exist_ok=True)

        # 1. Load into TorchIO Subject
        subject = tio.Subject(ct=tio.ScalarImage(ct_path), mask=tio.LabelMap(mask_path), moved_mr=tio.ScalarImage(moved_mr_path))

        # 2. Resample (1.5mm isotropic)
        resample = tio.Resample(TARGET_SPACING)
        resampled_subject = resample(subject)

        # 3. Convert and Save as .nii (uncompressed)
        for name, image in resampled_subject.items():
            data = image.data.numpy()[0]  # (1, H, W, D) -> (H, W, D)

            if name in ["ct", "moved_mr"]:
                data = np.round(data).astype(np.int16)
            else:
                data = data.astype(np.uint8)

            new_img = nib.Nifti1Image(data, image.affine)
            out_file = os.path.join(subj_out_path, f"{name}.nii")
            nib.save(new_img, out_file)

        return subj_id, True, "Success"
    except Exception as e:
        return subj_id, False, str(e)


def main():
    print("🚀 Starting Stratified Resampling (1.5mm Isotropic + Int16 + .nii)")

    # --- Step 1: Discover All Subjects across region subfolders ---
    regions = ["AB", "HN", "TH", "brain", "pelvis"]
    all_tasks_data = []  # List of (subj_id, region_dir)

    for r_dir in regions:
        dir_path = os.path.join(INPUT_DIR, r_dir)
        if not os.path.exists(dir_path):
            continue
        subs = [d for d in os.listdir(dir_path) if os.path.isdir(os.path.join(dir_path, d))]
        for s in subs:
            all_tasks_data.append((s, r_dir))

    print(f"✅ Found {len(all_tasks_data)} total subjects in input directories.")

    # --- Step 2: Stratified Split ---
    region_groups = {}
    for sid, r_dir in all_tasks_data:
        r_key = get_region_key(sid)
        if r_key not in region_groups:
            region_groups[r_key] = []
        region_groups[r_key].append((sid, r_dir))

    random.seed(42)

    print("\n📊 Stratified Split Summary:")
    header = f"{'Region':<15} | {'Total':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6}"
    print(header)
    print("-" * len(header))

    processing_tasks = []

    for region, entries in sorted(region_groups.items()):
        random.shuffle(entries)
        n = len(entries)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        r_train = entries[:n_train]
        r_val = entries[n_train : n_train + n_val]
        r_test = entries[n_train + n_val :]

        for sid, r_dir in r_train:
            processing_tasks.append((sid, "train", r_dir))
        for sid, r_dir in r_val:
            processing_tasks.append((sid, "val", r_dir))
        for sid, r_dir in r_test:
            processing_tasks.append((sid, "test", r_dir))

        print(f"{region:<15} | {n:<6} | {len(r_train):<6} | {len(r_val):<6} | {len(r_test):<6}")

    # --- Step 3: Execute Batch Processing ---
    print(f"\n🏃 Processing {len(processing_tasks)} subjects using {MAX_WORKERS} workers...")

    results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(process_single_subject, processing_tasks), total=len(processing_tasks)):
            results.append(result)

    # Final Report
    success_count = sum(1 for _, success, _ in results if success)
    print(f"\n✅ Finished! Successfully processed: {success_count}/{len(processing_tasks)}")

    failures = [(sid, msg) for sid, success, msg in results if not success]
    if failures:
        # Saving log to the same directory as the script
        log_path = os.path.join(os.path.dirname(__file__), "resample_all_failures.txt") if "__file__" in locals() else "resample_all_failures.txt"
        with open(log_path, "w") as f:
            for sid, msg in failures:
                f.write(f"{sid}: {msg}\n")
        print(f"⚠️ {len(failures)} subjects failed (missing registration or error). Details in resample_all_failures.txt")


if __name__ == "__main__":
    main()
