import os
import glob
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

# --- EXTERNAL DEPENDENCIES ---
try:
    from anatomix.registration import convex_adam
except ImportError:
    print("⚠️ WARNING: anatomix library not found. Dummy mode enabled.")
    def convex_adam(**kwargs): pass

# ==========================================
# 1. CONFIGURATION & BEST PARAMETERS
# ==========================================
ROOT = "/home/minsukc/MRI2CT"
DATA_DIR = os.path.join(ROOT, "data")
CKPT_PATH = os.path.join(ROOT, "anatomix/model-weights/anatomix.pth")
OUTPUT_ROOT = os.path.join(ROOT, "final_results_batch")

# *** ENTER YOUR TUNED PARAMETERS HERE ***
# NOTE: All parameters defining size (grid_sp) or counts (niter, smooth) must be INTEGERS.
BEST_PARAMS = {
    'lambda_weight': 0.75,  
    'grid_sp': 2,           # FIXED: Changed from 2.0 to 2
    'selected_smooth': 0,   # FIXED: Changed from float to int
    'selected_niter': 80,   # Using 80 as a safe default if not provided, ensure it's int
    'disp_hw': 1            # Using 1 as a safe default
}

# Define your subjects
SUBJECTS_CONFIG = [
    { "id": "1ABA103_3x3x3_resampled", "region": "abdomen" },
    { "id": "1ABB116_3x3x3_resampled", "region": "abdomen" },
    { "id": "1ABB164_3x3x3_resampled", "region": "abdomen" },
    { "id": "1THA267_3x3x3_resampled", "region": "thorax" },
    { "id": "1THB050_3x3x3_resampled", "region": "thorax" },
    { "id": "1THB211_3x3x3_resampled", "region": "thorax" },
    { "id": "1HNA038_3x3x3_resampled", "region": "head_neck" },
    { "id": "1HNA119_3x3x3_resampled", "region": "head_neck" },
    { "id": "1HNC073_3x3x3_resampled", "region": "head_neck" },
]

# ==========================================
# 2. REGION MAPS (For Accurate Dice)
# ==========================================
REGION_MAPS = {
    "abdomen": {
        "Spleen": (1, 1), "Kidney_R": (2, 2), "Kidney_L": (3, 3), 
        "Liver": (5, 5), "Stomach": (6, 6), "Pancreas": (7, 7)
    },
    "thorax": {
        "Heart": (51, 22), "Aorta": (52, 23), "Esophagus": (15, 12)
    },
    "head_neck": {
        "Brain": (90, 50)
    }
}

# ==========================================
# 3. HELPER FUNCTIONS (Preprocessing)
# ==========================================
def minmax(arr, minclip=None, maxclip=None):
    if minclip is not None and maxclip is not None:
        arr = np.clip(arr, minclip, maxclip)
    denom = arr.max() - arr.min()
    if denom == 0: return np.zeros_like(arr)
    return (arr - arr.min()) / denom

def pad_to_multiple_np(arr, multiple=16):
    D, H, W = arr.shape
    pad_D = (multiple - D % multiple) % multiple
    pad_H = (multiple - H % multiple) % multiple
    pad_W = (multiple - W % multiple) % multiple
    if pad_D == 0 and pad_H == 0 and pad_W == 0:
        return arr, (0,0,0)
    return np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant'), (pad_D, pad_H, pad_W)

def unpad_np(arr, pad_vals):
    pad_D, pad_H, pad_W = pad_vals
    D_end = None if pad_D == 0 else -pad_D
    H_end = None if pad_H == 0 else -pad_H
    W_end = None if pad_W == 0 else -pad_W
    return arr[:D_end, :H_end, :W_end]

def save_nifti(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), path)

def make_rgb_overlay(ct_slice, mr_slice):
    """Creates a Red-Green overlay (Red=CT, Green=MRI)."""
    ct_norm = (ct_slice - ct_slice.min()) / (ct_slice.max() - ct_slice.min() + 1e-8)
    mr_norm = (mr_slice - mr_slice.min()) / (mr_slice.max() - mr_slice.min() + 1e-8)
    rgb = np.zeros((*ct_norm.shape, 3), dtype=np.float32)
    rgb[..., 0] = ct_norm # Red
    rgb[..., 1] = mr_norm # Green
    return rgb

# ==========================================
# 4. EVALUATION FUNCTION
# ==========================================
def compute_dice_region(gt, pred, region_name):
    target_map = REGION_MAPS.get(region_name)
    if not target_map: return {}, 0.0

    gt_present = np.unique(gt)
    pred_present = np.unique(pred)
    
    organ_scores = {}
    scores_list = []

    for organ, (ct_id, mr_id) in target_map.items():
        if ct_id not in gt_present or mr_id not in pred_present:
            continue

        y_true = (gt == ct_id)
        y_pred = (pred == mr_id)
        inter = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)
        dice = (2.0 * inter) / (total + 1e-6)
        
        organ_scores[organ] = dice
        scores_list.append(dice)

    avg = np.mean(scores_list) if scores_list else 0.0
    return organ_scores, avg

# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def run_batch_pipeline():
    print(f"🚀 Starting Batch Registration with Best Params: {BEST_PARAMS}")
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    summary_results = []
    
    # --- Ensure required parameters are integers ---
    params = BEST_PARAMS.copy()
    params['grid_sp'] = int(params.get('grid_sp', 2))
    params['selected_smooth'] = int(params.get('selected_smooth', 0))
    params['selected_niter'] = int(params.get('selected_niter', 80))
    params['disp_hw'] = int(params.get('disp_hw', 1))

    for s_idx, subj in enumerate(SUBJECTS_CONFIG):
        subj_id = subj['id']
        region = subj['region']
        
        print(f"\n[{s_idx+1}/{len(SUBJECTS_CONFIG)}] Processing: {subj_id} ({region})")
        subj_dir = os.path.join(DATA_DIR, subj_id)
        result_dir = os.path.join(OUTPUT_ROOT, subj_id)
        os.makedirs(result_dir, exist_ok=True)

        # --- A. Load & Preprocess RAW Data ---
        raw_files = {
            'fixed': os.path.join(subj_dir, "ct_resampled.nii.gz"),
            'moving': os.path.join(subj_dir, "mr_resampled.nii.gz"),
            'mask': os.path.join(subj_dir, "mask_resampled.nii.gz"),
            'fixed_seg': os.path.join(subj_dir, "ct_seg.nii.gz"),
            'moving_seg': os.path.join(subj_dir, "mr_seg.nii.gz"),
        }
        
        try:
            # 1. Load Raw Data
            nii_fixed = nib.load(raw_files['fixed'])
            dat_fixed = nii_fixed.get_fdata()
            dat_moving = nib.load(raw_files['moving']).get_fdata()
            affine = nii_fixed.affine
            
            # 2. Normalize & Get Segments
            dat_fixed_norm = minmax(dat_fixed, -450, 450)
            dat_moving_norm = minmax(dat_moving)
            dat_mask = nib.load(raw_files['mask']).get_fdata()
            dat_fseg = nib.load(raw_files['fixed_seg']).get_fdata()
            dat_mseg = nib.load(raw_files['moving_seg']).get_fdata()

            # 3. Pad ALL Volumes
            pad_fixed, pad_vals = pad_to_multiple_np(dat_fixed_norm, 16)
            pad_moving, _ = pad_to_multiple_np(dat_moving_norm, 16)
            pad_mask, _ = pad_to_multiple_np(dat_mask, 16)
            pad_fseg, _ = pad_to_multiple_np(dat_fseg, 16)
            pad_mseg, _ = pad_to_multiple_np(dat_mseg, 16)

            # 4. Define & Save Temp Padded Files
            temp_paths = {k: os.path.join(result_dir, f"temp_padded_{k}.nii.gz") for k in ['fixed', 'moving', 'mask', 'fixed_seg', 'moving_seg']}
            save_nifti(pad_fixed, affine, temp_paths['fixed'])
            save_nifti(pad_moving, affine, temp_paths['moving'])
            save_nifti(pad_mask, affine, temp_paths['mask'])
            save_nifti(pad_fseg, affine, temp_paths['fixed_seg'])
            save_nifti(pad_mseg, affine, temp_paths['moving_seg'])

        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            continue

        # --- B. Run Registration ---
        try:
            convex_adam(
                ckpt_path=CKPT_PATH,
                expname="final",
                result_path=result_dir,
                lambda_weight=params['lambda_weight'],
                grid_sp=params['grid_sp'],
                selected_smooth=params['selected_smooth'],
                disp_hw=params['disp_hw'],
                selected_niter=params['selected_niter'], 
                grid_sp_adam=params['grid_sp'], # Typically grid_sp_adam = grid_sp
                ic=True, use_mask=True, warp_seg=True,
                fixed_image=temp_paths['fixed'], fixed_mask=temp_paths['mask'], fixed_seg=temp_paths['fixed_seg'],
                fixed_minclip=-450, fixed_maxclip=450,
                moving_image=temp_paths['moving'], moving_mask=temp_paths['mask'], moving_seg=temp_paths['moving_seg']
            )
        except Exception as e:
            print(f"   💥 Registration failed: {e}")
            
            # --- Cleanup Temp ---
            for p in temp_paths.values(): os.remove(p)
            continue


        # --- C. Unpad & Save Final Results ---
        try:
            moved_img_path = glob.glob(os.path.join(result_dir, "moved_temp_padded_moving*.nii.gz"))[0]
            moved_seg_path = glob.glob(os.path.join(result_dir, "labels_moved_temp_padded_moving*.nii.gz"))[0]
            
            # Unpad back to original shape
            moved_img_unpad = unpad_np(nib.load(moved_img_path).get_fdata(), pad_vals)
            moved_seg_unpad = unpad_np(nib.load(moved_seg_path).get_fdata(), pad_vals)
            
            # Save Final (using original affine)
            final_img_path = os.path.join(result_dir, "final_moved_mr.nii.gz")
            final_seg_path = os.path.join(result_dir, "final_moved_seg.nii.gz")
            save_nifti(moved_img_unpad, affine, final_img_path)
            save_nifti(moved_seg_unpad, affine, final_seg_path)

            # --- D. Evaluate ---
            organ_scores, avg_dice = compute_dice_region(dat_fseg, moved_seg_unpad, region)
            print(f"   ✅ Registration Dice ({region}): {avg_dice:.4f}")
            
            summary_results.append({
                'subject': subj_id, 'region': region, 'avg_dice': avg_dice, **organ_scores
            })

            # --- E. Cleanup Temp ---
            for p in temp_paths.values(): os.remove(p)
            for p in glob.glob(os.path.join(result_dir, "moved_temp_padded_moving*.nii.gz")): os.remove(p)
            for p in glob.glob(os.path.join(result_dir, "labels_moved_temp_padded_moving*.nii.gz")): os.remove(p)


        except Exception as e:
            print(f"   ⚠️ Post-processing/Evaluation Error: {e}")
            # Ensure temp files are cleaned even on failure
            for p in temp_paths.values(): os.remove(p)
            continue


    # Save Summary CSV
    if summary_results:
        pd.DataFrame(summary_results).to_csv(os.path.join(OUTPUT_ROOT, "batch_summary.csv"), index=False)
        print("\n🏁 Batch Processing Complete. Results saved to batch_summary.csv")

if __name__ == "__main__":
    run_batch_pipeline()