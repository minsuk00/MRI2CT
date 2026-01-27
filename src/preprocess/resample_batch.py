import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import random

# ==========================================
# 1. CONFIGURATION
# ==========================================
# Input Datasets (Add/Remove as needed)
INPUT_DIRS = [
    "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2025/Task1",
    "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2023/Task1" 
]

# Correct Output Path
OUTPUT_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"

# Split Ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Target spacing (mm)
TARGET_SPACING = [3.0, 3.0, 3.0]

# Set to None to process ALL volumes
NUM_VOLUMES = None 
# NUM_VOLUMES = 5  # Debug mode

def main():
    print(f"🚀 Starting Multi-Dataset Resampling & Splitting")
    print(f"   Target Spacing: {TARGET_SPACING}")
    print(f"   Output Root: {OUTPUT_ROOT}")

    # --- Step 1: Discover All Subjects ---
    all_subjects = []
    
    for input_dir in INPUT_DIRS:
        if not os.path.exists(input_dir):
            print(f"❌ WARNING: Directory not found: {input_dir}")
            continue
            
        print(f"🔎 Scanning: {input_dir}...")
        try:
            # Assumes structure: InputDir -> Region -> PatientID
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
                    all_subjects.append({
                        "id": pid,
                        "path": p_path,
                        "region": region,
                        "dataset": os.path.basename(input_dir)
                    })

    # Debug limit
    if NUM_VOLUMES is not None:
        print(f"⚠️ DEBUG MODE: Limiting to first {NUM_VOLUMES} subjects.")
        all_subjects = all_subjects[:NUM_VOLUMES]

    total_subjs = len(all_subjects)
    print(f"✅ Found {total_subjs} total subjects.")
    if total_subjs == 0: return

    # --- Step 2: Shuffle & Assign Splits ---
    # Random seed ensures the same split every time you run this script on the same data
    random.seed(42) 
    random.shuffle(all_subjects)

    n_train = int(total_subjs * SPLIT_RATIOS["train"])
    n_val = int(total_subjs * SPLIT_RATIOS["val"])
    
    splits = {
        "train": all_subjects[:n_train],
        "val": all_subjects[n_train:n_train+n_val],
        "test": all_subjects[n_train+n_val:]
    }

    # --- Step 3: Resample and Save to Split Folders ---
    # Example Output: .../SynthRAD_combined/3.0x3.0x3.0mm/train/12345/
    spacing_str = "x".join([str(s) for s in TARGET_SPACING])
    base_out = os.path.join(OUTPUT_ROOT, f"{spacing_str}mm") 

    for split_name, subjects in splits.items():
        print(f"\n📂 Processing SPLIT: {split_name.upper()} ({len(subjects)} volumes)")
        
        # Define split directory
        split_dir = os.path.join(base_out, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        # Separate viz directory per split
        viz_dir = os.path.join(base_out, "_visualizations", split_name) 

        for idx, subj in enumerate(tqdm(subjects)):
            # Save visualization for the first patient in each split to check quality
            process_subject(subj, split_dir, viz_dir, save_viz=(idx == 0))

    print("\n" + "-" * 50)
    print("🏁 Batch Processing Complete.")

def process_subject(subj, output_dir, viz_dir, save_viz=False):
    """
    Reads MR/CT, resamples them, and saves to the output_dir/PatientID/
    """
    pid = subj['id']
    src_path = subj['path']
    
    mr_path = os.path.join(src_path, "mr.nii.gz")
    ct_path = os.path.join(src_path, "ct.nii.gz")
    mask_path = os.path.join(src_path, "mask.nii.gz")

    # Basic validation
    if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
        return

    # Create patient folder inside the split directory
    out_patient_dir = os.path.join(output_dir, pid)
    os.makedirs(out_patient_dir, exist_ok=True)

    # Naming convention: keeping it simple or appending suffix
    out_mr = os.path.join(out_patient_dir, "mr.nii.gz")
    out_ct = os.path.join(out_patient_dir, "ct.nii.gz")
    out_mask = os.path.join(out_patient_dir, "mask.nii.gz")

    # Skip if done
    if os.path.exists(out_mr) and os.path.exists(out_ct):
        return 

    try:
        # Resample Images
        mr_orig, mr_res = resample_volume(mr_path, out_mr, TARGET_SPACING)
        ct_orig, ct_res = resample_volume(ct_path, out_ct, TARGET_SPACING)
        
        # Resample Mask (Nearest Neighbor) if exists
        mask_orig, mask_res = None, None
        if os.path.exists(mask_path):
            mask_orig, mask_res = resample_volume(mask_path, out_mask, TARGET_SPACING, is_mask=True)

        # QC Plot
        if save_viz:
            orig_dict = {'MRI': minmax(mr_orig), 'CT': minmax(ct_orig, -450, 450), 'Mask': mask_orig}
            res_dict = {'MRI': minmax(mr_res), 'CT': minmax(ct_res, -450, 450), 'Mask': mask_res}
            save_comparison_plot(orig_dict, res_dict, pid, viz_dir)

    except Exception as e:
        print(f"❌ Failed on {pid}: {e}")

# ==========================================
# UTILITIES
# ==========================================
def minmax(arr, minclip=None, maxclip=None):
    if minclip is not None and maxclip is not None:
        arr = np.clip(arr, minclip, maxclip)
    denom = arr.max() - arr.min()
    if denom == 0: return np.zeros_like(arr)
    return (arr - arr.min()) / denom

def get_middle_slice(arr):
    if arr is None: return np.zeros((100,100))
    z_mid = arr.shape[0] // 2
    return arr[z_mid, :, :]

def save_comparison_plot(orig_dict, res_dict, patient_id, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    def plot_ax(ax, data, title):
        ax.imshow(get_middle_slice(data), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plot_ax(axes[0,0], orig_dict['MRI'], "Original MRI")
    plot_ax(axes[0,1], orig_dict['CT'], "Original CT")
    plot_ax(axes[0,2], orig_dict['Mask'], "Original Mask")

    plot_ax(axes[1,0], res_dict['MRI'], "Resampled MRI")
    plot_ax(axes[1,1], res_dict['CT'], "Resampled CT")
    plot_ax(axes[1,2], res_dict['Mask'], "Resampled Mask")

    plt.suptitle(f"Patient: {patient_id}", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"{patient_id}_comparison.png"))
    plt.close()

def resample_volume(in_path, out_path, target_spacing, is_mask=False):
    img = sitk.ReadImage(in_path)
    original_array = sitk.GetArrayFromImage(img) 

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Compute new dimensions
    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputPixelType(img.GetPixelIDValue())

    # Important: Nearest Neighbor for masks to keep integer labels
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    new_img = resample.Execute(img)
    sitk.WriteImage(new_img, out_path)
    
    return original_array, sitk.GetArrayFromImage(new_img)

if __name__ == "__main__":
    main()