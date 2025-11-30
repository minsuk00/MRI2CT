import os
import time
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT = "/home/minsukc/MRI2CT"
DATA_DIR = os.path.join(ROOT, "data")

# OPTION A: Define specific folders to process (leave None to process ALL)
TARGET_LIST = None 
# TARGET_LIST = ["1ABA005_3.0x3.0x3.0_resampled"] 
# TARGET_LIST = ["1ABA005_3.0x3.0x3.0_resampled","1ABA009_3.0x3.0x3.0_resampled","1ABA011_3.0x3.0x3.0_resampled","1ABA012_3.0x3.0x3.0_resampled","1ABA014_3.0x3.0x3.0_resampled","1ABA018_3.0x3.0x3.0_resampled","1ABA019_3.0x3.0x3.0_resampled","1ABA025_3.0x3.0x3.0_resampled","1ABA029_3.0x3.0x3.0_resampled","1ABA030_3.0x3.0x3.0_resampled","1HNA001_3.0x3.0x3.0_resampled","1HNA004_3.0x3.0x3.0_resampled","1HNA006_3.0x3.0x3.0_resampled","1HNA008_3.0x3.0x3.0_resampled","1HNA010_3.0x3.0x3.0_resampled","1HNA012_3.0x3.0x3.0_resampled","1HNA013_3.0x3.0x3.0_resampled","1HNA014_3.0x3.0x3.0_resampled","1HNA015_3.0x3.0x3.0_resampled","1HNA018_3.0x3.0x3.0_resampled","1THA001_3.0x3.0x3.0_resampled","1THA002_3.0x3.0x3.0_resampled","1THA003_3.0x3.0x3.0_resampled","1THA004_3.0x3.0x3.0_resampled","1THA005_3.0x3.0x3.0_resampled","1THA010_3.0x3.0x3.0_resampled","1THA011_3.0x3.0x3.0_resampled","1THA013_3.0x3.0x3.0_resampled","1THA015_3.0x3.0x3.0_resampled","1THA016_3.0x3.0x3.0_resampled"]


# --- PERFORMANCE SETTINGS ---
DEVICE = "gpu" 
# False = High Res (Slow, ~5 mins/scan). True = Low Res (Fast, ~30s/scan)
FAST_MODE = False 
# FAST_MODE = True

# ==========================================
# 2. UTILITIES
# ==========================================
def get_region_from_id(subject_id):
    """
    Parses the subject ID to determine the anatomy region.
    Logic: Looks at the 2 characters following the leading '1'.
    e.g., 1ABA... -> AB -> abdomen
    """
    # 1. Basic Format Check
    if len(subject_id) < 2 or not subject_id.startswith("1"):
        raise ValueError(
            f"⚠️ Invalid subject ID '{subject_id}'. Expected format '1XX...'"
        )

    # 2. Extract Code
    region_code = subject_id[1:3].upper()

    # 3. Map to regions
    mapping = {
        # SynthRAD 2025 (2-char codes)
        "AB": "abdomen",
        "TH": "thorax",
        "HN": "head_neck",
        # SynthRAD 2023 (1-char codes)
        "B": "brain",
        "P": "pelvis" 
    }

    # Extract candidates
    code_2 = subject_id[1:3].upper()
    code_1 = subject_id[1:2].upper()
    
    # Match
    if code_2 in mapping:
        return mapping[code_2]
    elif code_1 in mapping:
        return mapping[code_1]

    raise ValueError(
        f"⚠️ Region code '{region_code}' in '{subject_id}' is not recognized..."
    )

def discover_subjects(data_dir, target_list=None):
    """
    Scans the data directory for valid subjects containing necessary NIfTI files.
    """
    valid_subjects = []
    
    # Determine candidates
    if target_list:
        candidates = target_list
    else:
        # Get all subdirectories
        candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"🔎 Scanning {len(candidates)} candidates in {data_dir}...")

    for subj_id in candidates:
        subj_path = os.path.join(data_dir, subj_id)
        
        # Define required files
        mr_path = os.path.join(subj_path, "mr_resampled.nii.gz")
        ct_path = os.path.join(subj_path, "ct_resampled.nii.gz")

        # Check existence
        if os.path.exists(mr_path) and os.path.exists(ct_path):
            region = get_region_from_id(subj_id)
            valid_subjects.append({
                "id": subj_id,
                "path": subj_path,
                "region": region,
                "mr": mr_path,
                "ct": ct_path
            })
        else:
            if target_list: # Only warn if user specifically asked for this one
                print(f"   ❌ Missing files for {subj_id}. Skipping.")

    return valid_subjects

def save_qa_plot(ct_path, ct_seg_path, mr_path, mr_seg_path, save_path):
    """Generates a slice view of CT and MRI with segmentation overlays."""
    try:
        ct_dat = nib.load(ct_path).get_fdata()
        ct_seg = nib.load(ct_seg_path).get_fdata()
        mr_dat = nib.load(mr_path).get_fdata()
        mr_seg = nib.load(mr_seg_path).get_fdata()

        # Normalize for display
        ct_disp = np.clip(ct_dat, -450, 450)
        ct_disp = (ct_disp - ct_disp.min()) / (ct_disp.max() - ct_disp.min() + 1e-8)
        mr_disp = (mr_dat - mr_dat.min()) / (mr_dat.max() - mr_dat.min() + 1e-8)

        # Pick middle slice
        sliceidx = ct_dat.shape[2] // 2

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # CT Row
        axs[0,0].imshow(np.rot90(ct_disp[:, :, sliceidx]), cmap="gray")
        axs[0,0].set_title("CT Image")
        axs[0,1].imshow(np.rot90(ct_disp[:, :, sliceidx]), cmap="gray")
        axs[0,1].imshow(np.rot90(ct_seg[:, :, sliceidx]), cmap="tab20", alpha=0.3)
        axs[0,1].set_title("CT + Seg Overlay")
        axs[0,2].imshow(np.rot90(ct_seg[:, :, sliceidx]), cmap="tab20")
        axs[0,2].set_title("CT Seg Mask")

        # MR Row
        axs[1,0].imshow(np.rot90(mr_disp[:, :, sliceidx]), cmap="gray")
        axs[1,0].set_title("MR Image")
        axs[1,1].imshow(np.rot90(mr_disp[:, :, sliceidx]), cmap="gray")
        axs[1,1].imshow(np.rot90(mr_seg[:, :, sliceidx]), cmap="tab20", alpha=0.3)
        axs[1,1].set_title("MR + Seg Overlay")
        axs[1,2].imshow(np.rot90(mr_seg[:, :, sliceidx]), cmap="tab20")
        axs[1,2].set_title("MR Seg Mask")

        for ax in axs.flatten(): ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        # print(f"   📸 QA Plot saved")
        
    except Exception as e:
        print(f"   ⚠️ QA Plot Error: {e}")

# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_batch_segmentation():
    # 1. Discover Subjects
    subjects = discover_subjects(DATA_DIR, target_list=TARGET_LIST)
    
    if not subjects:
        print("❌ No valid subjects found. Check your paths.")
        return

    print(f"\n🚀 Starting TotalSegmentator Batch for {len(subjects)} subjects")
    print(f"   Device: {DEVICE}")
    print(f"   Fast Mode: {FAST_MODE}\n")
    
    if not torch.cuda.is_available() and DEVICE == "gpu":
        print("   ⚠️ WARNING: No GPU detected. Processing will be slow.")

    # 2. Process
    for subj in tqdm(subjects, desc="Processing Subjects", unit="subj"):
        start_time = time.time()
        
        ct_out = os.path.join(subj['path'], "ct_seg.nii.gz")
        mr_out = os.path.join(subj['path'], "mr_seg.nii.gz")
        qa_path = os.path.join(subj['path'], "segmentation_qa.png")

        # --- CT Segmentation ---
        if os.path.exists(ct_out):
            tqdm.write("✅ CT Segmentation exists.")
        else:
            tqdm.write("🧠 Running CT Seg (Task: total)...")
            try:
                totalsegmentator(
                    input=subj['ct'], output=ct_out, ml=True, 
                    task="total", fastest=FAST_MODE, device=DEVICE, quiet = True
                )
                # quiet = True, preview = True, skip_saving = True
            except Exception as e:
                tqdm.write(f"💥 CT Seg Failed: {e}")

        # --- MR Segmentation ---
        if os.path.exists(mr_out):
            tqdm.write("✅ MR Segmentation exists.")
        else:
            tqdm.write("🧠 Running MR Seg (Task: total_mr)...")
            try:
                totalsegmentator(
                    input=subj['mr'], output=mr_out, ml=True, 
                    task="total_mr", fastest=FAST_MODE, device=DEVICE, quiet = True
                )
            except Exception as e:
                tqdm.write(f"💥 MR Seg Failed: {e}")

        # --- QA Visualization ---
        if os.path.exists(ct_out) and os.path.exists(mr_out):
            if not os.path.exists(qa_path):
                save_qa_plot(subj['ct'], ct_out, subj['mr'], mr_out, qa_path)
        
        tqdm.write(f"✨ Subject done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_batch_segmentation()