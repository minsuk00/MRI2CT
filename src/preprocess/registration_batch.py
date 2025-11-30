import os
import glob
import time
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.metrics import f1_score
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="monai")

from anatomix.registration import convex_adam

# ==========================================
# 1. CONFIGURATION
# ==========================================
ROOT = "/home/minsukc/MRI2CT"
DATA_DIR = os.path.join(ROOT, "data")
CKPT_PATH = os.path.join(ROOT, "anatomix/model-weights/anatomix.pth")

# Target specific subjects or set to None for ALL
TARGET_LIST = None 
# TARGET_LIST = ["1ABA005_3.0x3.0x3.0_resampled", "1HNA001_3.0x3.0x3.0_resampled", "1THA001_3.0x3.0x3.0_resampled"]
# TARGET_LIST = ["1ABA005_3.0x3.0x3.0_resampled"]

# *** BEST PARAMETERS FROM TUNING ***
BEST_PARAMS = {
    'lambda_weight': 0.5,  
    'grid_sp': 3,           
    'selected_smooth': 1,   
    'selected_niter': 80,   
    'disp_hw': 1            
}

# ==========================================
# 2. REGION MAPS
# ==========================================
REGION_MAPS = {
    "abdomen": {
        "Spleen": (1, 1), "Kidney_R": (2, 2), "Kidney_L": (3, 3), "Liver": (5, 5), "Stomach": (6, 6)
    },
    "thorax": {
        "Heart": (51, 22), "Lung_L": ([10, 11], 10), "Lung_R": ([12, 13, 14], 11)
    },
    "head_neck": {
        "Brain": (90, 50)
    },
    "brain": {},
    "pelvis": {}
}


# ==========================================
# 3. UTILITIES
# ==========================================
def minmax(arr, minclip=None, maxclip=None):
    if not (minclip is None) & (maxclip is None):
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
    padded = np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant')
    return padded, (pad_D, pad_H, pad_W)

def unpad_np(arr, pad_vals):
    pad_D, pad_H, pad_W = pad_vals
    s_d = slice(None, -pad_D) if pad_D > 0 else slice(None)
    s_h = slice(None, -pad_H) if pad_H > 0 else slice(None)
    s_w = slice(None, -pad_W) if pad_W > 0 else slice(None)
    return arr[s_d, s_h, s_w]

def save_nifti(arr, affine, path):
    nib.save(nib.Nifti1Image(arr, affine), path)

def get_region_from_id(subject_id):
    if len(subject_id) < 2 or not subject_id.startswith("1"):
        raise ValueError(
            f"⚠️ Invalid subject ID '{subject_id}'. Expected format '1XX...'"
        )
        
    mapping = { "AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis" }
    code_2 = subject_id[1:3].upper()
    code_1 = subject_id[1:2].upper()
    
    if code_2 in mapping: 
        return mapping[code_2]
    if code_1 in mapping: 
        return mapping[code_1]
    
    raise ValueError(
        f"⚠️ Region code '{region_code}' in '{subject_id}' is not recognized..."
    )

def discover_subjects(data_dir, target_list=None):
    if target_list:
        candidates = target_list
    else:
        candidates = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])

    print(f"🔎 Scanning {len(candidates)} candidates in {data_dir}...")
    required_files = ["mr_resampled.nii.gz", "ct_resampled.nii.gz", "mask_resampled.nii.gz", "ct_seg.nii.gz", "mr_seg.nii.gz"]
    valid = []

    for subj_id in candidates:
        subj_path = os.path.join(data_dir, subj_id)
        missing = [f for f in required_files if not os.path.exists(os.path.join(subj_path, f))]
        if missing:
            if target_list: print(f"   ❌ Skipping {subj_id}: Missing {missing}")
            continue
        region = get_region_from_id(subj_id)
        valid.append({ "id": subj_id, "path": subj_path, "region": region })

    print(f"✅ Found {len(valid)} valid subjects.")
    return valid

# ==========================================
# 4. VISUALIZATION & EVALUATION
# ==========================================
def create_random_colormap(num_classes=120, seed=42):
    num_classes = int(num_classes)
    np.random.seed(seed)
    colors = np.random.rand(num_classes + 1, 4) 
    colors[:, 3] = 1.0 
    colors[0, :] = [0, 0, 0, 0] 
    return ListedColormap(colors)

def create_binary_colormap(color_name):
    if color_name == 'red': c = [1, 0, 0, 0.6] 
    elif color_name == 'green': c = [0, 1, 0, 0.6] 
    else: c = [1, 1, 1, 1]
        
    cmap = np.zeros((2, 4))
    cmap[0, :] = [0, 0, 0, 0]
    cmap[1, :] = c 
    return ListedColormap(cmap)

def get_target_label_mask(seg_volume, region_name):
    if region_name not in REGION_MAPS: 
        return np.zeros_like(seg_volume)
    target_ids = []
    
    for _, (ct_ids, mr_ids) in REGION_MAPS[region_name].items():
        if isinstance(ct_ids, list): target_ids.extend(ct_ids)
        else: target_ids.append(ct_ids)
            
        if isinstance(mr_ids, list): target_ids.extend(mr_ids)
        else: target_ids.append(mr_ids)
    mask = np.isin(seg_volume, target_ids)
    return np.where(mask, seg_volume, 0) 

def save_registration_vis(fixed_path, moving_path, warped_path, fseg_path, mseg_path, wseg_path, disp_path, region_name, save_path):
    try:
        # Load Data
        fix = nib.load(fixed_path).get_fdata()
        mov = nib.load(moving_path).get_fdata()
        wrp = nib.load(warped_path).get_fdata()
        
        fix_seg = nib.load(fseg_path).get_fdata().astype(np.int32)
        mov_seg = nib.load(mseg_path).get_fdata().astype(np.int32)
        wrp_seg = nib.load(wseg_path).get_fdata().astype(np.int32)
        
        disp = nib.load(disp_path).get_fdata()

        # Select Middle Slice
        z = int(fix.shape[2] // 2)

        # Helper: Normalize and rotate
        def get_sl(vol, normalize=False):
            s = np.rot90(vol[..., z])
            if normalize:
                denom = s.max() - s.min()
                if denom > 0: s = (s - s.min()) / denom
            return s

        # Prepare Slices (Full Volume)
        f_sl = get_sl(fix, True)
        m_sl = get_sl(mov, True)
        w_sl = get_sl(wrp, True)

        fs_sl = get_sl(fix_seg)
        ms_sl = get_sl(mov_seg)
        ws_sl = get_sl(wrp_seg)

        # Prepare Slices (Target Regions Only)
        # Note: Relies on get_target_label_mask being available in global scope
        fix_filt = get_target_label_mask(fix_seg, region_name)
        mov_filt = get_target_label_mask(mov_seg, region_name)
        wrp_filt = get_target_label_mask(wrp_seg, region_name)

        fs_filt_sl = get_sl(fix_filt)
        ms_filt_sl = get_sl(mov_filt)
        ws_filt_sl = get_sl(wrp_filt)

        # Calculate Displacement Magnitude
        # Check dims: if 5D (x,y,z,1,3) squeeze, if 4D (x,y,z,3) use as is
        if disp.ndim == 5: disp = disp.squeeze() 
        # Assuming channels are last (x, y, z, 3). If channels are first, use axis=0
        axis_ch = -1 if disp.shape[-1] == 3 else 0
        disp_mag_vol = np.linalg.norm(disp, axis=axis_ch)
        d_sl = get_sl(disp_mag_vol)

        # Colormaps
        max_label = max(fix_seg.max(), mov_seg.max(), wrp_seg.max())
        seg_cmap = create_random_colormap(int(max_label))
        red_cmap = create_binary_colormap('red')
        green_cmap = create_binary_colormap('green')
        mag_cmap = 'jet'

        # Plotting Setup (4 Rows x 4 Cols)
        fig, axes = plt.subplots(4, 4, figsize=(20, 20))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)

        # --- ROW 1: Intensity Inputs & Result ---
        axes[0,0].imshow(f_sl, cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title("Fixed (CT)")
        
        axes[0,1].imshow(m_sl, cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title("Moving (MRI)")
        
        axes[0,2].imshow(w_sl, cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title("Warped (Reg. MRI)")

        axes[0,3].axis('off') # EMPTY

        # --- ROW 2: Overlays & Displacement ---
        # Pre-Registration Overlay
        rgb_pre = np.zeros((*f_sl.shape, 3))
        rgb_pre[..., 0] = f_sl # R = CT
        rgb_pre[..., 1] = m_sl # G = Moving
        axes[1,0].imshow(rgb_pre)
        axes[1,0].set_title("Pre-Reg (R=CT, G=MR)")

        # Post-Registration Overlay
        rgb_post = np.zeros((*f_sl.shape, 3))
        rgb_post[..., 0] = f_sl # R = CT
        rgb_post[..., 1] = w_sl # G = Warped
        axes[1,1].imshow(rgb_post)
        axes[1,1].set_title("Post-Reg (R=CT, G=Warped MR)")

        # Displacement with Colorbar
        im_d = axes[1,2].imshow(d_sl, cmap=mag_cmap)
        axes[1,2].set_title("Disp. Magnitude")
        cbar = plt.colorbar(im_d, ax=axes[1,2], fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=8)

        axes[1,3].axis('off') # EMPTY

        # --- ROW 3: Segmentation (ALL) ---
        axes[2,0].imshow(fs_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,0].set_title("Fixed Seg (All)")

        axes[2,1].imshow(ms_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,1].set_title("Moving Seg (All)")

        axes[2,2].imshow(ws_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,2].set_title("Warped Seg (All)")

        # Mask Overlay
        f_bin = (fs_sl > 0).astype(int)
        w_bin = (ws_sl > 0).astype(int)
        
        axes[2,3].imshow(f_sl, cmap='gray', alpha=0.5) 
        axes[2,3].imshow(f_bin, cmap=red_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[2,3].imshow(w_bin, cmap=green_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[2,3].set_title("All Seg Overlap (R=Fix, G=Warped MR)")

        # --- ROW 4: Target Regions (FILTERED) ---
        axes[3,0].imshow(fs_filt_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[3,0].set_title("Fixed Seg (Targets)")

        axes[3,1].imshow(ms_filt_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[3,1].set_title("Moving Seg (Targets)")

        axes[3,2].imshow(ws_filt_sl, cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[3,2].set_title("Warped Seg (Targets)")

        # Target Mask Overlay
        f_filt_bin = (fs_filt_sl > 0).astype(int)
        w_filt_bin = (ws_filt_sl > 0).astype(int)

        axes[3,3].imshow(f_sl, cmap='gray', alpha=0.5)
        axes[3,3].imshow(f_filt_bin, cmap=red_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[3,3].imshow(w_filt_bin, cmap=green_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[3,3].set_title("Target Overlap (R=Fix, G=Warped MR)")

        # Cleanup axes
        for r in range(4):
            for c in range(4):
                if not (r==1 and c==2): # Don't turn off axis for Disp (colorbar needs it sometimes)
                   axes[r,c].axis('off')
                else:
                   axes[r,c].set_xticks([])
                   axes[r,c].set_yticks([])

        plt.savefig(save_path)
        plt.close(fig)
    except Exception as e:
        print(f"⚠️ Visualization failed: {e}")
        
def compute_dice_region(gt, pred, region_name):
    target_map = REGION_MAPS.get(region_name)
    if not target_map:
        return {}, 0.0

    gt_present = np.unique(gt)
    pred_present = np.unique(pred)
    organ_scores = {}
    scores_list = []

    def is_present(ids, present_list):
        if isinstance(ids, list): return any(i in present_list for i in ids)
        return ids in present_list

    def make_mask(volume, ids):
        if isinstance(ids, list): return np.isin(volume, ids)
        return volume == ids

    for organ, (ct_ids, mr_ids) in target_map.items():
        if not is_present(ct_ids, gt_present) or not is_present(mr_ids, pred_present):
            continue

        y_true = make_mask(gt, ct_ids)
        y_pred = make_mask(pred, mr_ids)
        inter = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)
        dice = (2.0 * inter) / (total + 1e-6)
        
        organ_scores[organ] = dice
        scores_list.append(dice)

    avg = np.mean(scores_list) if scores_list else 0.0
    print("Avg Dice:", avg)
    return organ_scores, avg
    
# ==========================================
# 5. MAIN PIPELINE
# ==========================================
def run_batch_pipeline():
    subjects = discover_subjects(DATA_DIR, target_list=TARGET_LIST)
    if not subjects:
        return

    print(f"\n🚀 Starting Batch Registration with Best Params: {BEST_PARAMS}")
    summary_results = []
    
    params = BEST_PARAMS.copy()
    params['grid_sp'] = int(params.get('grid_sp', 2))
    params['selected_smooth'] = int(params.get('selected_smooth', 0))
    params['selected_niter'] = int(params.get('selected_niter', 80))
    params['disp_hw'] = int(params.get('disp_hw', 1))

    for subj in tqdm(subjects, desc="Batch Progress", unit="subj"):
        subj_id = subj['id']
        region = subj['region']
        subj_dir = subj['path']
        
        # Local output directory per subject
        result_dir = os.path.join(subj_dir, "registration_output")
        os.makedirs(result_dir, exist_ok=True)

        # # If both the moved image and segmentation exist, skip this subject
        # final_img_path = os.path.join(result_dir, "moved_mr.nii.gz")
        # final_seg_path = os.path.join(result_dir, "labels_moved.nii.gz")
        # if os.path.exists(final_img_path) and os.path.exists(final_seg_path):
        #     Optional: tqmd.write(f"Skipping {subj_id} (Already done)")
        #     continue

        # --- A. Load & Preprocess ---
        raw_files = {
            'fixed': os.path.join(subj_dir, "ct_resampled.nii.gz"),
            'moving': os.path.join(subj_dir, "mr_resampled.nii.gz"),
            'mask': os.path.join(subj_dir, "mask_resampled.nii.gz"),
            'fixed_seg': os.path.join(subj_dir, "ct_seg.nii.gz"),
            'moving_seg': os.path.join(subj_dir, "mr_seg.nii.gz"),
        }
        
        try:
            nii_fixed = nib.load(raw_files['fixed'])
            dat_fixed = nii_fixed.get_fdata()
            dat_moving = nib.load(raw_files['moving']).get_fdata()
            affine = nii_fixed.affine
            
            dat_fixed_norm = minmax(dat_fixed, -450, 450)
            dat_moving_norm = minmax(dat_moving)
            dat_mask = nib.load(raw_files['mask']).get_fdata()
            dat_fseg = nib.load(raw_files['fixed_seg']).get_fdata()
            dat_mseg = nib.load(raw_files['moving_seg']).get_fdata()

            # Pad
            pad_fixed, pad_vals = pad_to_multiple_np(dat_fixed_norm, 16)
            pad_moving, _ = pad_to_multiple_np(dat_moving_norm, 16)
            pad_mask, _ = pad_to_multiple_np(dat_mask, 16)
            pad_fseg, _ = pad_to_multiple_np(dat_fseg, 16)
            pad_mseg, _ = pad_to_multiple_np(dat_mseg, 16)

            # Temp Files
            temp_paths = {k: os.path.join(result_dir, f"temp_padded_{k}.nii.gz") for k in ['fixed', 'moving', 'mask', 'fixed_seg', 'moving_seg']}
            save_nifti(pad_fixed, affine, temp_paths['fixed'])
            save_nifti(pad_moving, affine, temp_paths['moving'])
            save_nifti(pad_mask, affine, temp_paths['mask'])
            save_nifti(pad_fseg, affine, temp_paths['fixed_seg'])
            save_nifti(pad_mseg, affine, temp_paths['moving_seg'])

        except Exception as e:
            tqdm.write(f"   ❌ Preprocessing failed for {subj_id}: {e}")
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
                grid_sp_adam=params['grid_sp'],
                ic=True, use_mask=True, warp_seg=True,
                fixed_image=temp_paths['fixed'], fixed_mask=temp_paths['mask'], fixed_seg=temp_paths['fixed_seg'],
                fixed_minclip=-450, fixed_maxclip=450,
                moving_image=temp_paths['moving'], moving_mask=temp_paths['mask'], moving_seg=temp_paths['moving_seg']
            )
        except Exception as e:
            tqdm.write(f"   💥 Registration failed: {e}")
            for p in temp_paths.values(): os.remove(p)
            continue

        # --- C. Unpad & Save Final Results ---
        try:
            # Glob search for output
            warped_pattern = os.path.join(result_dir, "moved*.nii.gz")
            label_pattern = os.path.join(result_dir, "labels_moved_temp_padded*.nii.gz")
            disp_pattern = os.path.join(result_dir, "disp_temp_padded_moving*.nii.gz")

            moved_img_path = glob.glob(warped_pattern)[0]
            moved_seg_path = glob.glob(label_pattern)[0]
            disp_path = glob.glob(disp_pattern)[0]

            # Load and Unpad
            moved_img_unpad = unpad_np(nib.load(moved_img_path).get_fdata(), pad_vals)
            moved_seg_unpad = unpad_np(nib.load(moved_seg_path).get_fdata(), pad_vals)
            disp_unpad = unpad_np(nib.load(disp_path).get_fdata(), pad_vals)
            
            # Save Final (using original affine)
            final_img_path = os.path.join(result_dir, "moved_mr.nii.gz")
            final_seg_path = os.path.join(result_dir, "labels_moved.nii.gz")
            final_disp_path = os.path.join(result_dir, "disp.nii.gz")
            save_nifti(moved_img_unpad, affine, final_img_path)
            save_nifti(moved_seg_unpad, affine, final_seg_path)
            save_nifti(disp_unpad, affine, final_disp_path)

            # Evaluate
            organ_scores, avg_dice = compute_dice_region(dat_fseg, moved_seg_unpad, region)
            
            # Vis
            vis_path = os.path.join(result_dir, "registration_qc.png")
            save_registration_vis(
                raw_files['fixed'], raw_files['moving'], final_img_path,
                raw_files['fixed_seg'], raw_files['moving_seg'], final_seg_path, final_disp_path,
                region, vis_path
            )

            # Store result
            summary_results.append({
                'subject': subj_id, 'region': region, 'avg_dice': avg_dice, **organ_scores
            })

            # Cleanup
            for p in temp_paths.values():
                os.remove(p)
            os.remove(moved_img_path)
            os.remove(moved_seg_path)
            os.remove(disp_path)

        except Exception as e:
            tqdm.write(f"   ⚠️ Post-processing failed: {e}")
            for p in temp_paths.values(): os.remove(p)
            continue

    # Save Summary CSV
    if summary_results:
        # Save to root data dir or project root
        summary_path = os.path.join(DATA_DIR, "_registration_summary.csv")
        pd.DataFrame(summary_results).to_csv(summary_path, index=False)
        print(f"\n🏁 Batch Processing Complete. Results saved to {summary_path}")

if __name__ == "__main__":
    run_batch_pipeline()