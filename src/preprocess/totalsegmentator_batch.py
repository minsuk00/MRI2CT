import os
import time
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE: Point to the output of the previous step
ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"
# DATA_DIR = os.path.join(ROOT, "3.0x3.0x3.0mm") # The specific resolution folder
DATA_DIR = os.path.join(ROOT, "1.0x1.0x1.0mm")

# Target specific subjects or None for ALL
TARGET_LIST = None 

# --- PERFORMANCE SETTINGS ---
DEVICE = "gpu" 
# False = High Res (Slow, ~5 mins/scan). True = Low Res (Fast, ~30s/scan)
FAST_MODE = False 
# FAST_MODE = True

# ==========================================
# 2. UTILITIES
# ==========================================
def discover_subjects(data_root, target_list=None):
    """
    Scans data_root/train, data_root/val, and data_root/test for subjects.
    """
    valid_subjects = []
    splits = ["train", "val", "test"]

    print(f"🔎 Scanning subjects in {data_root}...")

    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue
            
        # Get patient IDs in this split
        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        for subj_id in candidates:
            # Filter if TARGET_LIST is set
            if target_list and subj_id not in target_list:
                continue

            subj_path = os.path.join(split_path, subj_id)
            
            # UPDATE: Filenames match the output of resample_and_split.py
            mr_path = os.path.join(subj_path, "mr.nii.gz")
            ct_path = os.path.join(subj_path, "ct.nii.gz")

            if os.path.exists(mr_path) and os.path.exists(ct_path):
                valid_subjects.append({
                    "id": subj_id,
                    "path": subj_path,
                    "split": split,
                    "mr": mr_path,
                    "ct": ct_path
                })
            else:
                if target_list: 
                    print(f"   ❌ Missing files for {subj_id} in {split}. Skipping.")

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
        
    except Exception as e:
        print(f"   ⚠️ QA Plot Error: {e}")

# ==========================================
# 3. MAIN LOOP
# ==========================================
def run_batch_segmentation():
    # 1. Discover Subjects
    subjects = discover_subjects(DATA_DIR, target_list=TARGET_LIST)
    
    if not subjects:
        print(f"❌ No valid subjects found in {DATA_DIR}. Check your paths.")
        return

    print(f"\n🚀 Starting TotalSegmentator Batch for {len(subjects)} subjects")
    print(f"   Device: {DEVICE}")
    print(f"   Fast Mode: {FAST_MODE}\n")
    
    if not torch.cuda.is_available() and DEVICE == "gpu":
        print("   ⚠️ WARNING: No GPU detected. Processing will be slow.")

    # 2. Process
    for subj in tqdm(subjects, desc="Processing Subjects", unit="subj"):
        start_time = time.time()
        
        # Outputs saved in the patient's folder
        ct_out = os.path.join(subj['path'], "ct_seg.nii.gz")
        mr_out = os.path.join(subj['path'], "mr_seg.nii.gz")
        body_out = os.path.join(subj['path'], "body_mask.nii.gz")
        qa_path = os.path.join(subj['path'], "segmentation_qa.png")

        # --- Body Mask (Binary) ---
        if os.path.exists(body_out) and not os.path.isdir(body_out):
            pass
        else:
            if os.path.isdir(body_out):
                import shutil
                shutil.rmtree(body_out)

            tqdm.write(f"[{subj['id']}] Running Body Mask...")
            try:
                totalsegmentator(
                    input=subj['ct'], output=body_out, 
                    task="body", device=DEVICE, ml=True
                )
            except Exception as e:
                tqdm.write(f"💥 Body Mask Failed for {subj['id']}: {e}")

        # --- CT Segmentation ---
        if os.path.exists(ct_out):
            # tqdm.write(f"[{subj['id']}] CT Seg exists.") 
            pass
        else:
            tqdm.write(f"[{subj['id']}] Running CT Seg...")
            try:
                totalsegmentator(
                    input=subj['ct'], output=ct_out, ml=True, 
                    # task="total", fast=FAST_MODE, device=DEVICE, quiet=True
                    task="total", fast=FAST_MODE, device=DEVICE
                )
            except Exception as e:
                tqdm.write(f"💥 CT Seg Failed for {subj['id']}: {e}")

        # --- MR Segmentation ---
        if os.path.exists(mr_out):
            pass
        else:
            tqdm.write(f"[{subj['id']}] Running MR Seg...")
            try:
                totalsegmentator(
                    input=subj['mr'], output=mr_out, ml=True, 
                    # task="total_mr", fast=FAST_MODE, device=DEVICE, quiet=True
                    task="total_mr", fast=FAST_MODE, device=DEVICE
                )
            except Exception as e:
                tqdm.write(f"💥 MR Seg Failed for {subj['id']}: {e}")

        # --- QA Visualization ---
        # Only creating QA plot if both segments exist and plot doesn't exist
        if os.path.exists(ct_out) and os.path.exists(mr_out):
            save_qa_plot(subj['ct'], ct_out, subj['mr'], mr_out, qa_path)
            # if not os.path.exists(qa_path):
            #     save_qa_plot(subj['ct'], ct_out, subj['mr'], mr_out, qa_path)
        
        # Optional: Print time per subject
        tqdm.write(f"✨ {subj['id']} done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    run_batch_segmentation()