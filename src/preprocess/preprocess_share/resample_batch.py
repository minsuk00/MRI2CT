import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from tqdm import tqdm
import random

def minmax(arr, minclip=None, maxclip=None):
    """
    Perform min-max normalization on a numpy array, with optional clipping.
    
    Args:
        arr: Input numpy array.
        minclip: Minimum value for clipping.
        maxclip: Maximum value for clipping.
        
    Returns:
        Normalized numpy array.
    """
    if minclip is not None and maxclip is not None:
        arr = np.clip(arr, minclip, maxclip)
    denom = arr.max() - arr.min()
    if denom == 0:
        return np.zeros_like(arr)
    return (arr - arr.min()) / denom

def get_middle_slice(arr):
    """
    Extract the middle axial slice from a 3D volume.
    
    Args:
        arr: 3D numpy array.
        
    Returns:
        2D middle slice.
    """
    if arr is None:
        return np.zeros((100, 100))
    z_mid = arr.shape[0] // 2
    return arr[z_mid, :, :]

def save_comparison_plot(orig_dict, res_dict, patient_id, save_dir, region):
    """
    Save a PNG plot comparing original and resampled volumes.
    
    Args:
        orig_dict: Dictionary containing original MRI, CT, and Mask data.
        res_dict: Dictionary containing resampled MRI, CT, and Mask data.
        patient_id: ID of the patient.
        save_dir: Directory to save the plot.
        region: Anatomical region.
    """
    os.makedirs(save_dir, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    def plot_ax(ax, data, title):
        ax.imshow(get_middle_slice(data), cmap='gray')
        ax.set_title(title)
        ax.axis('off')

    plot_ax(axes[0, 0], orig_dict['MRI'], "Original MRI")
    plot_ax(axes[0, 1], orig_dict['CT'], "Original CT")
    plot_ax(axes[0, 2], orig_dict['Mask'], "Original Mask")

    plot_ax(axes[1, 0], res_dict['MRI'], "Resampled MRI")
    plot_ax(axes[1, 1], res_dict['CT'], "Resampled CT")
    plot_ax(axes[1, 2], res_dict['Mask'], "Resampled Mask")

    plt.suptitle(f"Patient: {patient_id} ({region})", fontsize=16)
    plt.savefig(os.path.join(save_dir, f"{patient_id}_{region}_comparison.png"))
    plt.close()

def resample_volume(in_path, out_path, target_spacing, is_mask=False):
    """
    Resample a NIfTI volume to a target isotropic spacing.
    
    Args:
        in_path: Path to the input NIfTI file.
        out_path: Path to save the resampled NIfTI file.
        target_spacing: List of 3 floats representing the target spacing in mm.
        is_mask: If True, use Nearest Neighbor interpolation; otherwise, use Linear.
        
    Returns:
        tuple: (original_array, resampled_array)
    """
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

def get_image_path(base_path, filename_no_ext):
    """Helper to find file with either .nii.gz or .mha extension"""
    for ext in ['.nii.gz', '.mha']:
        full_path = os.path.join(base_path, f"{filename_no_ext}{ext}")
        if os.path.exists(full_path):
            return full_path
    return None

def process_subject(subj, output_dir, viz_dir, target_spacing, save_viz=False):
    """
    Read MR/CT, resample them, and save to output directory.
    
    Args:
        subj: Dictionary containing subject information.
        output_dir: Root directory for processed outputs.
        viz_dir: Directory for visualization plots.
        target_spacing: Target spacing for resampling.
        save_viz: Whether to save a QC plot.
        
    Returns:
        bool: True if successful, False otherwise.
    """
    pid = subj['id']
    src_path = subj['path']
    
    mr_path = get_image_path(src_path, "mr")
    ct_path = get_image_path(src_path, "ct")
    mask_path = get_image_path(src_path, "mask")

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
        if save_viz:
            viz_file = os.path.join(viz_dir, f"{pid}_{subj['region']}_comparison.png")
            if not os.path.exists(viz_file):
                try:
                    _, mr_res = resample_volume(mr_path, None, target_spacing)
                    _, ct_res = resample_volume(ct_path, None, target_spacing)
                    mask_res = None
                    if os.path.exists(mask_path):
                        _, mask_res = resample_volume(mask_path, None, target_spacing, is_mask=True)
                    
                    orig_dict = {'MRI': minmax(sitk.GetArrayFromImage(sitk.ReadImage(mr_path))), 
                                 'CT': minmax(sitk.GetArrayFromImage(sitk.ReadImage(ct_path)), -450, 450), 
                                 'Mask': None}
                    res_dict = {'MRI': minmax(mr_res), 'CT': minmax(ct_res, -450, 450), 'Mask': mask_res}
                    save_comparison_plot(orig_dict, res_dict, pid, viz_dir, subj['region'])
                except: pass
        return True 

    try:
        # Resample Images
        mr_orig, mr_res = resample_volume(mr_path, out_mr, target_spacing)
        ct_orig, ct_res = resample_volume(ct_path, out_ct, target_spacing)
        
        # Resample Mask (Nearest Neighbor) if exists
        mask_orig, mask_res = None, None
        if os.path.exists(mask_path):
            mask_orig, mask_res = resample_volume(mask_path, out_mask, target_spacing, is_mask=True)

        # QC Plot
        if save_viz:
            orig_dict = {'MRI': minmax(mr_orig), 'CT': minmax(ct_orig, -450, 450), 'Mask': mask_orig}
            res_dict = {'MRI': minmax(mr_res), 'CT': minmax(ct_res, -450, 450), 'Mask': mask_res}
            save_comparison_plot(orig_dict, res_dict, pid, viz_dir, subj['region'])
        
        return True

    except Exception as e:
        print(f"FAILED on {pid}: {e}")
        return False

def run_resampling_pipeline(input_dirs, output_root, split_ratios, target_spacing, num_volumes=None):
    """
    Main pipeline to discover, split, and resample subjects.
    
    Args:
        input_dirs: List of directories to scan for subjects.
        output_root: Directory to save the combined results.
        split_ratios: Dictionary with train/val/test split proportions.
        target_spacing: List of 3 floats representing target resolution.
        num_volumes: Optional limit for the number of subjects to process.
    """
    print(f"Starting Multi-Dataset Resampling & Splitting")
    print(f"   Target Spacing: {target_spacing}")
    print(f"   Output Root: {output_root}")

    # --- Step 1: Discover All Subjects ---
    all_subjects = []
    for input_dir in input_dirs:
        if not os.path.exists(input_dir):
            print(f"WARNING: Directory not found: {input_dir}")
            continue
            
        print(f"Scanning: {input_dir}...")
        try:
            regions = sorted([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])
        except Exception as e:
            print(f"Error reading {input_dir}: {e}")
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

    if num_volumes is not None:
        print(f"DEBUG MODE: Limiting to first {num_volumes} subjects.")
        all_subjects = all_subjects[:num_volumes]

    total_subjs = len(all_subjects)
    print(f"Found {total_subjs} total subjects.")
    if total_subjs == 0:
        return

    # --- Step 2: Stratified Shuffle & Assign Splits ---
    random.seed(42)
    region_groups = {}
    for subj in all_subjects:
        r = subj["region"]
        if r not in region_groups:
            region_groups[r] = []
        region_groups[r].append(subj)

    splits = {"train": [], "val": [], "test": []}
    print("\nStratified Split Summary:")
    header = f"{'Region':<15} | {'Total':<6} | {'Train':<6} | {'Val':<6} | {'Test':<6}"
    print(header)
    print("-" * len(header))

    for region, subjects in sorted(region_groups.items()):
        random.shuffle(subjects)
        n = len(subjects)
        n_train = int(n * split_ratios["train"])
        n_val = int(n * split_ratios["val"])

        r_train = subjects[:n_train]
        r_val = subjects[n_train:n_train + n_val]
        r_test = subjects[n_train + n_val:]

        splits["train"].extend(r_train)
        splits["val"].extend(r_val)
        splits["test"].extend(r_test)
        print(f"{region:<15} | {n:<6} | {len(r_train):<6} | {len(r_val):<6} | {len(r_test):<6}")

    print("-" * len(header))
    print(f"{'TOTAL':<15} | {total_subjs:<6} | {len(splits['train']):<6} | {len(splits['val']):<6} | {len(splits['test']):<6}")

    # --- Step 3: Resample and Save ---
    spacing_str = "x".join([str(s) for s in target_spacing])
    base_out = os.path.join(output_root, f"{spacing_str}mm") 

    for split_name, subjects in splits.items():
        print(f"\nProcessing SPLIT: {split_name.upper()} ({len(subjects)} volumes)")
        split_dir = os.path.join(base_out, split_name)
        os.makedirs(split_dir, exist_ok=True)
        viz_dir = os.path.join(base_out, "_visualizations", split_name) 

        visualized_regions = set()
        for subj in tqdm(subjects):
            region = subj['region']
            should_viz = region not in visualized_regions
            success = process_subject(subj, split_dir, viz_dir, target_spacing, save_viz=should_viz)
            if success and should_viz:
                visualized_regions.add(region)

    print("\n" + "-" * 50)
    print("Batch Processing Complete.")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    INPUT_DIRS = [
        "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/temp/synthRAD2023",
        "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/temp/synthRAD2025/Task1"
    ]
    OUTPUT_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/temp/synthRAD_combined"
    SPLIT_RATIOS = {"train": 0.7, "val": 0.1, "test": 0.2}
    TARGET_SPACING = [3.0, 3.0, 3.0]
    NUM_VOLUMES = None # Set to an integer for testing. None to process all

    run_resampling_pipeline(
        input_dirs=INPUT_DIRS,
        output_root=OUTPUT_ROOT,
        split_ratios=SPLIT_RATIOS,
        target_spacing=TARGET_SPACING,
        num_volumes=NUM_VOLUMES
    )
