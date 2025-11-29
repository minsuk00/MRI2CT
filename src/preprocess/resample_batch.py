import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm

INPUT_DIR = "/scratch/jjparkcv_root/jjparkcv98/minsukc/SynthRAD2025/Task1"
# OUTPUT_DIR = "/home/minsukc/MRI2CT/data"
OUTPUT_DIR = "/scratch/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/data"

# Number of volumes to process PER REGION. Set to None to process all.
# NUM_VOLUMES = None  # e.g., 5 or None
NUM_VOLUMES = 10

# Target spacing in x y z (mm)
TARGET_SPACING = [3.0, 3.0, 3.0]
# TARGET_SPACING = [1.5, 1.5, 1.5]

def main():
    # Setup directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    viz_dir = os.path.join(OUTPUT_DIR, "_visualizations")
    
    # Get regions (e.g., AB, HN, TH)
    try:
        regions = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    except FileNotFoundError:
        print(f"❌ Error: Input directory not found: {INPUT_DIR}")
        return
    
    print(f"🚀 Starting Resampling Job")
    print(f"   Input: {INPUT_DIR}")
    print(f"   Output: {OUTPUT_DIR}")
    print(f"   Target Spacing: {TARGET_SPACING}")
    print(f"   Limit per region: {'ALL' if NUM_VOLUMES is None else NUM_VOLUMES}")
    print("-" * 50)

    total_processed = 0

    for region in regions:
        region_path = os.path.join(INPUT_DIR, region)
        patients = sorted(os.listdir(region_path))
        
        # Filter: Take only first N patients if argument is provided
        if NUM_VOLUMES is not None:
            patients_to_process = patients[:NUM_VOLUMES]
        else:
            patients_to_process = patients

        print(f"📂 Processing Region {region}: {len(patients_to_process)} volumes")

        for idx, patient_id in enumerate(tqdm(patients_to_process)):
            patient_dir = os.path.join(region_path, patient_id)
            
            mr_path = os.path.join(patient_dir, "mr.nii.gz")
            ct_path = os.path.join(patient_dir, "ct.nii.gz")
            mask_path = os.path.join(patient_dir, "mask.nii.gz")

            # Check validity
            if not (os.path.exists(mr_path) and os.path.exists(ct_path)):
                continue

            # Create Output Directory
            spacing_str = "x".join([str(s) for s in TARGET_SPACING])
            out_patient_dir = os.path.join(OUTPUT_DIR, f"{patient_id}_{spacing_str}_resampled")
            os.makedirs(out_patient_dir, exist_ok=True)

            # --- Resample ---
            # Returns (original_array, resampled_array)
            mr_orig, mr_res = resample_volume(mr_path, os.path.join(out_patient_dir, "mr_resampled.nii.gz"), TARGET_SPACING)
            ct_orig, ct_res = resample_volume(ct_path, os.path.join(out_patient_dir, "ct_resampled.nii.gz"), TARGET_SPACING)
            
            mask_orig, mask_res = None, None
            if os.path.exists(mask_path):
                mask_orig, mask_res = resample_volume(mask_path, os.path.join(out_patient_dir, "mask_resampled.nii.gz"), TARGET_SPACING, is_mask=True)

            # --- Visualize (Only first patient of the region) ---
            if idx == 0:
                # Prepare dictionaries for plotting function
                # Normalize for display
                orig_dict = {
                    'MRI': minmax(mr_orig),
                    'CT': minmax(ct_orig, -450, 450),
                    'Mask': mask_orig
                }
                res_dict = {
                    'MRI': minmax(mr_res),
                    'CT': minmax(ct_res, -450, 450),
                    'Mask': mask_res
                }
                save_comparison_plot(orig_dict, res_dict, patient_id, viz_dir)
            
            total_processed += 1

    print("-" * 50)
    print(f"✅ Done! Processed {total_processed} total volumes.")
    print(f"📊 Visualizations saved to: {viz_dir}")

def minmax(arr, minclip=None, maxclip=None):
    """Normalize array to 0-1 range with optional clipping."""
    if minclip is not None and maxclip is not None:
        arr = np.clip(arr, minclip, maxclip)
    
    range_val = arr.max() - arr.min()
    if range_val == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / range_val

def get_middle_slice(arr):
    """Extract the middle axial slice."""
    # SimpleITK arrays are usually (z, y, x)
    z_mid = arr.shape[0] // 2
    return arr[z_mid, :, :]

def save_comparison_plot(orig_dict, res_dict, patient_id, save_dir):
    """
    Saves a comparison plot of middle slices.
    orig_dict/res_dict: {'MRI': arr, 'CT': arr, 'Mask': arr}
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Row 1: Originals
    axes[0, 0].imshow(get_middle_slice(orig_dict['MRI']), cmap='gray')
    axes[0, 0].set_title(f"Original MRI")
    axes[0, 1].imshow(get_middle_slice(orig_dict['CT']), cmap='gray')
    axes[0, 1].set_title(f"Original CT")
    if orig_dict['Mask'] is not None:
        axes[0, 2].imshow(get_middle_slice(orig_dict['Mask']), cmap='gray')
        axes[0, 2].set_title(f"Original Mask")
    else:
        axes[0, 2].axis('off')

    # Row 2: Resampled
    axes[1, 0].imshow(get_middle_slice(res_dict['MRI']), cmap='gray')
    axes[1, 0].set_title(f"Resampled MRI")
    axes[1, 1].imshow(get_middle_slice(res_dict['CT']), cmap='gray')
    axes[1, 1].set_title(f"Resampled CT")
    if res_dict['Mask'] is not None:
        axes[1, 2].imshow(get_middle_slice(res_dict['Mask']), cmap='gray')
        axes[1, 2].set_title(f"Resampled Mask")
    else:
        axes[1, 2].axis('off')

    for ax_row in axes:
        for ax in ax_row:
            ax.axis('off')

    plt.suptitle(f"Patient: {patient_id}", fontsize=16)
    save_path = os.path.join(save_dir, f"{patient_id}_comparison.png")
    plt.savefig(save_path)
    plt.close()
    # print(f"🖼️ Saved visualization to {save_path}")

def resample_volume(in_path, out_path, target_spacing, is_mask=False):
    """
    Resamples a NIfTI file to target spacing.
    Returns: The numpy array of the resampled image.
    """
    img = sitk.ReadImage(in_path)
    
    # If visualization is needed, we need the original array too
    # SimpleITK GetArrayFromImage returns (z, y, x)
    original_array = sitk.GetArrayFromImage(img) 

    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    # Calculate new size
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

    # Use Nearest Neighbor for masks to preserve integer labels, Linear for images
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    new_img = resample.Execute(img)
    sitk.WriteImage(new_img, out_path)
    
    return original_array, sitk.GetArrayFromImage(new_img)

if __name__ == "__main__":
    main()