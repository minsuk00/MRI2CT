import os

import torch
from torchio import LabelMap, ScalarImage
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input Datasets
INPUT_DIRS = ["/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2025/Task1", "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2023/Task1"]

# Output Path
OUTPUT_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"
BASE_OUT = os.path.join(OUTPUT_ROOT, "native_masked")

# Set to None to process ALL volumes
NUM_VOLUMES = None
# NUM_VOLUMES = 10


def to_masked_nifti(mr_path: str, ct_path: str, mask_path: str, out_mr: str, out_ct: str, out_mask: str) -> bool:
    """
    Masks MR and CT images using the provided mask and saves them as NIfTI files.
    """
    try:
        # Read the images and mask
        mr_img = ScalarImage(mr_path)
        ct_img = ScalarImage(ct_path)
        mask_img = LabelMap(mask_path)
        mask_data = mask_img.data.bool()

        # Mask the images
        # MRI: use minimum value for background
        mr_masked_tensor = torch.where(mask_data, mr_img.data, mr_img.data.min())
        # CT: use minimum value for background (usually ~ -1024)
        ct_masked_tensor = torch.where(mask_data, ct_img.data, ct_img.data.min())

        mr_masked = ScalarImage(tensor=mr_masked_tensor, affine=mr_img.affine)
        ct_masked = ScalarImage(tensor=ct_masked_tensor, affine=ct_img.affine)

        # Save to nifti
        mr_masked.save(out_mr)
        ct_masked.save(out_ct)
        mask_img.save(out_mask)
        return True
    except Exception as e:
        print(f"❌ Error processing {mr_path}: {e}")
        return False


def main():
    print("🚀 Starting Native Resolution Masking")
    print(f"   Output Directory: {BASE_OUT}")

    # --- Step 1: Discover All Subjects ---
    all_subjects = []

    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            print(f"❌ WARNING: Directory not found: {input_dir}")
            continue

        print(f"🔎 Scanning: {input_dir}...")
        try:
            # Structure: InputDir -> Region -> PatientID
            regions = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        except Exception as e:
            print(f"❌ Error reading {input_dir}: {e}")
            continue

        for region in regions:
            region_path = os.path.join(input_dir, region)
            patients = sorted(os.listdir(region_path))

            for pid in patients:
                p_path = os.path.join(region_path, pid)
                if os.path.isdir(p_path):
                    all_subjects.append({"id": pid, "path": p_path, "region": region, "dataset": os.path.basename(input_dir)})

    if NUM_VOLUMES is not None:
        print(f"⚠️ DEBUG MODE: Limiting to first {NUM_VOLUMES} subjects.")
        all_subjects = all_subjects[:NUM_VOLUMES]

    total_subjs = len(all_subjects)
    print(f"✅ Found {total_subjs} total subjects.")
    if total_subjs == 0:
        return

    # --- Step 2: Process and Save ---
    os.makedirs(BASE_OUT, exist_ok=True)

    success_count = 0
    for subj in tqdm(all_subjects):
        pid = subj["id"]
        src_path = subj["path"]

        # Define output patient directory
        out_patient_dir = os.path.join(BASE_OUT, pid)
        os.makedirs(out_patient_dir, exist_ok=True)

        # Naming convention for output (using .nii for faster loading)
        out_mr = os.path.join(out_patient_dir, "mr.nii")
        out_ct = os.path.join(out_patient_dir, "ct.nii")
        out_mask = os.path.join(out_patient_dir, "mask.nii")

        # Skip if already done
        if os.path.exists(out_mr) and os.path.exists(out_ct) and os.path.exists(out_mask):
            success_count += 1
            continue

        # Find input files (checking multiple extensions)
        def find_file(base, name):
            for ext in [".mha", ".nii", ".nii.gz"]:
                p = os.path.join(base, name + ext)
                if os.path.exists(p):
                    return p
            return None

        mr_path = find_file(src_path, "mr")
        ct_path = find_file(src_path, "ct")
        mask_path = find_file(src_path, "mask")

        if not mr_path or not ct_path or not mask_path:
            print(f"⚠️ Missing files for {pid} in {src_path}")
            continue

        if to_masked_nifti(mr_path, ct_path, mask_path, out_mr, out_ct, out_mask):
            success_count += 1

    print("\n" + "-" * 50)
    print(f"🏁 Processing Complete. Successfully processed {success_count}/{total_subjs} subjects.")
    print(f"📂 Data saved to: {BASE_OUT}")


if __name__ == "__main__":
    main()
