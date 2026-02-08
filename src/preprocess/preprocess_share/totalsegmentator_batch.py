import os
import time
import torch
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
from tqdm import tqdm

def discover_subjects(data_root, target_list=None):
    """
    Scans data_root/train, data_root/val, and data_root/test for subjects.
    
    Args:
        data_root: Root directory containing split folders.
        target_list: Optional list of subject IDs to process.
        
    Returns:
        list: A list of dictionaries with subject details.
    """
    valid_subjects = []
    splits = ["train", "val", "test"]

    print(f"Scanning subjects in {data_root}...")

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
                valid_subjects.append({
                    "id": subj_id,
                    "path": subj_path,
                    "split": split,
                    "mr": mr_path,
                    "ct": ct_path
                })
            else:
                if target_list: 
                    print(f"   Missing files for {subj_id} in {split}. Skipping.")

    return valid_subjects

def save_qa_plot(ct_path, ct_seg_path, mr_path, mr_seg_path, save_path):
    """
    Generates a slice view of CT and MRI with segmentation overlays for quality assurance.
    
    Args:
        ct_path: Path to CT image.
        ct_seg_path: Path to CT segmentation.
        mr_path: Path to MRI image.
        mr_seg_path: Path to MRI segmentation.
        save_path: Path to save the QA plot.
    """
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
        print(f"   WARNING: QA Plot Error: {e}")

def run_batch_segmentation(data_dir, target_list=None, device="gpu", fast_mode=False):
    """
    Run TotalSegmentator on a batch of subjects.
    
    Args:
        data_dir: Path to the directory containing processed resolution data.
        target_list: Optional list of subject IDs to process.
        device: Device to use (gpu or cpu).
        fast_mode: If True, uses the fast (low-res) version of TotalSegmentator.
    """
    subjects = discover_subjects(data_dir, target_list=target_list)
    
    if not subjects:
        print(f"No valid subjects found in {data_dir}. Check your paths.")
        return

    print(f"\nStarting TotalSegmentator Batch for {len(subjects)} subjects")
    print(f"   Device: {device}")
    print(f"   Fast Mode: {fast_mode}\n")
    
    if not torch.cuda.is_available() and device == "gpu":
        print("   WARNING: No GPU detected. Processing will be slow.")

    for subj in tqdm(subjects, desc="Processing Subjects", unit="subj"):
        start_time = time.time()
        
        ct_out = os.path.join(subj['path'], "ct_seg.nii.gz")
        mr_out = os.path.join(subj['path'], "mr_seg.nii.gz")
        body_out = os.path.join(subj['path'], "body_mask.nii.gz")
        qa_path = os.path.join(subj['path'], "segmentation_qa.png")

        # --- Body Mask (Binary) ---
        if not (os.path.exists(body_out) and not os.path.isdir(body_out)):
            if os.path.isdir(body_out):
                import shutil
                shutil.rmtree(body_out)

            tqdm.write(f"[{subj['id']}] Running Body Mask...")
            try:
                totalsegmentator(
                    input=subj['ct'], output=body_out, 
                    task="body", device=device, ml=True
                )
            except Exception as e:
                tqdm.write(f"Body Mask Failed for {subj['id']}: {e}")

        # --- CT Segmentation ---
        if not os.path.exists(ct_out):
            tqdm.write(f"[{subj['id']}] Running CT Seg...")
            try:
                totalsegmentator(
                    input=subj['ct'], output=ct_out, ml=True, 
                    task="total", fast=fast_mode, device=device
                )
            except Exception as e:
                tqdm.write(f"CT Seg Failed for {subj['id']}: {e}")

        # --- MR Segmentation ---
        if not os.path.exists(mr_out):
            tqdm.write(f"[{subj['id']}] Running MR Seg...")
            try:
                totalsegmentator(
                    input=subj['mr'], output=mr_out, ml=True, 
                    task="total_mr", fast=fast_mode, device=device
                )
            except Exception as e:
                tqdm.write(f"MR Seg Failed for {subj['id']}: {e}")

        # --- QA Visualization ---
        if os.path.exists(ct_out) and os.path.exists(mr_out):
            save_qa_plot(subj['ct'], ct_out, subj['mr'], mr_out, qa_path)
        
        tqdm.write(f"{subj['id']} done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    # ==========================================
    # CONFIGURATION
    # ==========================================
    ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/temp/synthRAD_combined"
    DATA_DIR = os.path.join(ROOT, "1.0x1.0x1.0mm")
    TARGET_LIST = None 
    DEVICE = "gpu" 
    FAST_MODE = False # False = High Res, True = Low Res

    run_batch_segmentation(
        data_dir=DATA_DIR,
        target_list=TARGET_LIST,
        device=DEVICE,
        fast_mode=FAST_MODE
    )
