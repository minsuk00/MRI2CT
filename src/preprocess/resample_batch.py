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
# Input Datasets
INPUT_DIRS = [
    "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2025/Task1",
    "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2023/Task1" 
]

# Correct Output Path
OUTPUT_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"

# Split Ratios
SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}

# Target spacing (mm)
# TARGET_SPACING = [3.0, 3.0, 3.0]
# TARGET_SPACING = [1.0, 1.0, 1.0]
TARGET_SPACING = [1.5, 1.5, 1.5]

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

    # --- Step 2: Stratified Shuffle & Assign Splits ---
    random.seed(42)

    # Group by region
    region_groups = {}
    for subj in all_subjects:
        r = subj["region"]
        if r not in region_groups:
            region_groups[r] = []
        region_groups[r].append(subj)

    splits = {"train": [], "val": [], "test": []}

    print("\n📊 Stratified Split Summary:")
    header = f"{'Region':<15} | {'Total':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6}"
    print(header)
    print("-" * len(header))

    for region, subjects in sorted(region_groups.items()):
        random.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * SPLIT_RATIOS["train"])
        n_val = int(n * SPLIT_RATIOS["val"])

        r_train = subjects[:n_train]
        r_val = subjects[n_train:n_train + n_val]
        r_test = subjects[n_train + n_val:]

        splits["train"].extend(r_train)
        splits["val"].extend(r_val)
        splits["test"].extend(r_test)

        print(f"{region:<15} | {n:<6} | {len(r_train):<6} | {len(r_val):<6} | {len(r_test):<6}")

    print("-" * len(header))
    print(f"{'TOTAL':<15} | {total_subjs:<6} | {len(splits['train']):<6} | {len(splits['val']):<6} | {len(splits['test']):<6}")

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

        visualized_regions = set()
        for subj in tqdm(subjects):
            region = subj['region']
            # Save visualization for the first successful patient of EACH region in this split
            should_viz = region not in visualized_regions
            success = process_subject(subj, split_dir, viz_dir, save_viz=should_viz)
            
            if success and should_viz:
                visualized_regions.add(region)

    print("\n" + "-" * 50)
    print("🏁 Batch Processing Complete.")

def process_subject(subj, output_dir, viz_dir, save_viz=False):
    """
    Reads MR/CT, resamples them, and saves to the output_dir/PatientID/
    Returns True if resampling was successful (or already done), False otherwise.
    """
    pid = subj['id']
    src_path = subj['path']
    
    mr_path = os.path.join(src_path, "mr.nii.gz")
    ct_path = os.path.join(src_path, "ct.nii.gz")
    mask_path = os.path.join(src_path, "mask.nii.gz")

    # Basic validation
    if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
        return False

    # Create patient folder inside the split directory
    out_patient_dir = os.path.join(output_dir, pid)
    os.makedirs(out_patient_dir, exist_ok=True)

    # Naming convention
    out_mr = os.path.join(out_patient_dir, "mr.nii.gz")
    out_ct = os.path.join(out_patient_dir, "ct.nii.gz")
    out_mask = os.path.join(out_patient_dir, "mask.nii.gz")

    # Skip if done
    if os.path.exists(out_mr) and os.path.exists(out_ct):
        # Even if already done, we check if we need to generate a plot
        if save_viz:
            viz_file = os.path.join(viz_dir, f"{pid}_{subj['region']}_comparison.png")
            if not os.path.exists(viz_file):
                try:
                    # Quick resample for viz only
                    _, mr_res = resample_volume(mr_path, None, TARGET_SPACING)
                    _, ct_res = resample_volume(ct_path, None, TARGET_SPACING)
                    mask_res = None
                    if os.path.exists(mask_path):
                        _, mask_res = resample_volume(mask_path, None, TARGET_SPACING, is_mask=True)
                    
                    orig_dict = {'MRI': minmax(sitk.GetArrayFromImage(sitk.ReadImage(mr_path))), 
                                 'CT': minmax(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), -1024, 1024), 
                                 'Mask': None}
                    res_dict = {'MRI': minmax(mr_res), 'CT': minmax(ct_res, -1024, 1024), 'Mask': mask_res}
                    save_comparison_plot(orig_dict, res_dict, pid, viz_dir, subj['region'])
                except: pass
        return True 

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
            orig_dict = {'MRI': minmax(mr_orig), 'CT': minmax(ct_orig, -1024, 1024), 'Mask': mask_orig}
            res_dict = {'MRI': minmax(mr_res), 'CT': minmax(ct_res, -1024, 1024), 'Mask': mask_res}
            save_comparison_plot(orig_dict, res_dict, pid, viz_dir, subj['region'])
        
        return True

    except Exception as e:
        print(f"❌ Failed on {pid}: {e}")
        return False

# ==========================================
# UTILITIES
# ==========================================
def minmax(arr, minclip=None, maxclip=None):
    # if minclip is not None and maxclip is not None:
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
    denom = arr.max() - arr.min()
    # if denom == 0: return np.zeros_like(arr)
    return (arr - arr.min()) / denom

def get_middle_slice(arr):
    if arr is None: return np.zeros((100,100))
    z_mid = arr.shape[0] // 2
    return arr[z_mid, :, :]

def save_comparison_plot(orig_dict, res_dict, patient_id, save_dir, region):
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

    plt.suptitle(f"Patient: {patient_id} ({region})", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"{patient_id}_{region}_comparison.png"))
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
    if out_path is not None:
        sitk.WriteImage(new_img, out_path)
    
    return original_array, sitk.GetArrayFromImage(new_img)

if __name__ == "__main__":
    main()