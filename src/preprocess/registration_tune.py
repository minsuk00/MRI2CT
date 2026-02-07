import os
import glob
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from sklearn.metrics import f1_score
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import warnings
import datetime
import contextlib
import io
import gc
warnings.filterwarnings("ignore", category=UserWarning, module='monai.utils.module')
from anatomix.registration import convex_adam, load_model
import torch

# 100% Safe Speed Optimization (No effect on precision)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

ROOT = "/home/minsukc/MRI2CT"
DATA_DIR = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.0x1.0x1.0mm"
CKPT = os.path.join(ROOT, "anatomix/model-weights/best_val_net_G.pth")
RES_MULT = 32
# CKPT = os.path.join(ROOT, "anatomix/model-weights/anatomix.pth")
SCRATCH_ROOT =  "/scratch/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/"
OUTPUT_DIR = os.path.join(DATA_DIR, "temp_tuning_results")


# --- 1. Define Subjects ---
# TARGET_SUBJECTS = None 
TARGET_SUBJECTS = ["1ABA005", "1ABA033", "1BA014", "1BA085", "1HNA013", "1HNA001", "1PA021", "1PA145", "1THA010", "1THA022"]  

# export PYTORCH_ALLOC_CONF=expandable_segments:True
# --- 2. Define Parameter Search Grid ---
grid = {
    'lambda_weight': [0.5, 0.75, 1.5], 
    'disp_hw': [1, 2, 4], 
    'grid_sp': [2, 4, 6],       
    'grid_sp_adam': [2],     
    'selected_niter': [50],  # Will Use 80 for actual run
    'smooth': [0, 1],
}

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
    "brain": {"Brain": (90, 50)},
    "pelvis": {"Bladder": (21, 16), "Sacrum": (25, 18), "Femur_L": (75, 36), "Femur_R": (76, 37)}
}

# ==========================================
# UTILITIES & CONFIG
# ==========================================
def minmax(arr, minclip=None, maxclip=None):
    if not (minclip is None) & (maxclip is None):
        arr = np.clip(arr, minclip, maxclip)
    # Avoid division by zero
    denom = arr.max() - arr.min()
    if denom == 0: 
        return np.zeros_like(arr)
    arr = (arr - arr.min()) / denom
    return arr

def pad_to_multiple_np(arr, multiple=16):
    D, H, W = arr.shape
    pad_D = (multiple - D % multiple) % multiple
    pad_H = (multiple - H % multiple) % multiple
    pad_W = (multiple - W % multiple) % multiple
    # Only pad if necessary
    if pad_D == 0 and pad_H == 0 and pad_W == 0:
        return arr, (0,0,0)
    
    padded = np.pad(arr, ((0, pad_D), (0, pad_H), (0, pad_W)), mode='constant')
    return padded, (pad_D, pad_H, pad_W)

def get_region_from_id(subject_id):
    """
    Parses subject ID to determine anatomy region.
    Supports SynthRAD2025 (AB, TH, HN) and SynthRAD2023 (B, P).
    """
    if len(subject_id) < 2 or not subject_id.startswith("1"):
        raise ValueError(
            f"⚠️ Invalid subject ID '{subject_id}'. Expected format '1XX...'"
        )
        
    mapping = {
        "AB": "abdomen",
        "TH": "thorax",
        "HN": "head_neck",
        "B": "brain",
        "P": "pelvis" 
    }

    code_2 = subject_id[1:3].upper()
    code_1 = subject_id[1:2].upper()

    if code_2 in mapping: 
        return mapping[code_2]
    if code_1 in mapping: 
        return mapping[code_1]
    
    raise ValueError(
        f"⚠️ Region code '{region_code}' in '{subject_id}' is not recognized..."
    )

def discover_tuning_subjects(data_dir, target_list=None):
    valid_configs = []
    splits = ["train", "val", "test"]
    
    required_files = ["mr.nii.gz", "ct.nii.gz", "ct_seg.nii.gz", "mr_seg.nii.gz"]

    print(f"🔎 Scanning subjects in {data_dir}...")

    for split in splits:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path): continue
        
        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        
        for subj_id in candidates:
            # Filter targets
            if target_list and subj_id not in target_list: continue
            
            subj_full_path = os.path.join(split_path, subj_id)
            missing = [f for f in required_files if not os.path.exists(os.path.join(subj_full_path, f))]
            
            if missing:
                if target_list: print(f"   ❌ Skipping {subj_id}: Missing {missing}")
                continue

            valid_configs.append({
                "id": subj_id,
                "path": subj_full_path, # Keep full path
                "region": get_region_from_id(subj_id)
            })

    print(f"✅ Found {len(valid_configs)} valid subjects for tuning.")
    return valid_configs

def prepare_subject_data(subj_conf):
    target_dir = subj_conf['path']
    subj_id = subj_conf['id']
    
    # Try body_mask.nii.gz first, fallback to mask.nii.gz
    body_mask_path = os.path.join(target_dir, "body_mask.nii.gz")
    fallback_mask_path = os.path.join(target_dir, "mask.nii.gz")
    selected_mask_path = body_mask_path if os.path.isfile(body_mask_path) else fallback_mask_path

    raw_paths = {
        'mr': os.path.join(target_dir, "mr.nii.gz"),
        'ct': os.path.join(target_dir, "ct.nii.gz"),
        'ct_seg': os.path.join(target_dir, "ct_seg.nii.gz"),
        'mr_seg': os.path.join(target_dir, "mr_seg.nii.gz"),
        'mask': selected_mask_path, 
    }
    
    print(f"   Processing {subj_id}: Normalizing & Padding...")

    # 2. Load Data
    mr_nii = nib.load(raw_paths['mr'])
    ct_nii = nib.load(raw_paths['ct'])
    mask_nii = nib.load(raw_paths['mask'])
    ct_seg_nii = nib.load(raw_paths['ct_seg'])
    mr_seg_nii = nib.load(raw_paths['mr_seg'])

    mr_data = mr_nii.get_fdata()
    ct_data = ct_nii.get_fdata()
    mask_data = mask_nii.get_fdata()
    ct_seg_data = ct_seg_nii.get_fdata()
    mr_seg_data = mr_seg_nii.get_fdata()

    # 3. MinMax Normalization
    mri_norm = minmax(mr_data) 
    ct_norm = minmax(ct_data, minclip=-450, maxclip=450)
    
    # 4. Pad to multiple of 16
    mr_pad, _      = pad_to_multiple_np(mri_norm, RES_MULT)
    ct_pad, _      = pad_to_multiple_np(ct_norm, RES_MULT)
    mask_pad, _    = pad_to_multiple_np(mask_data, RES_MULT)
    ct_seg_pad, _ = pad_to_multiple_np(ct_seg_data, RES_MULT)
    mr_seg_pad, _ = pad_to_multiple_np(mr_seg_data, RES_MULT)

    # 5. Define Output Paths (Temporary)
    temp_dir = os.path.join(target_dir, "temp_tuning")
    os.makedirs(temp_dir, exist_ok=True)
    
    temp_paths = {
        'fixed':       os.path.join(temp_dir, "ct_padded_temp.nii.gz"),
        'fixed_mask':  os.path.join(temp_dir, "mask_padded_temp.nii.gz"), 
        'fixed_seg':   os.path.join(temp_dir, "ct_seg_padded_temp.nii.gz"),
        'moving':      os.path.join(temp_dir, "mr_padded_temp.nii.gz"),
        'moving_mask': os.path.join(temp_dir, "mask_padded_temp.nii.gz"),
        'moving_seg':  os.path.join(temp_dir, "mr_seg_padded_temp.nii.gz"),
    }

    # 6. Save Temporary Files
    affine = np.eye(4) 
    nib.save(nib.Nifti1Image(mr_pad, affine), temp_paths['moving'])
    nib.save(nib.Nifti1Image(ct_pad, affine), temp_paths['fixed'])
    nib.save(nib.Nifti1Image(mask_pad, affine), temp_paths['fixed_mask']) 
    nib.save(nib.Nifti1Image(ct_seg_pad, affine), temp_paths['fixed_seg'])
    nib.save(nib.Nifti1Image(mr_seg_pad, affine), temp_paths['moving_seg'])

    return temp_paths

def cleanup_subject_data(temp_paths):
    """ Deletes the temporary padded files. """
    for p in temp_paths.values():
        if os.path.exists(p):
            os.remove(p)
    print("   🧹 Temporary files cleaned.")


def create_random_colormap(num_classes=120, seed=42):
    """
    Creates a discrete colormap where:
    - 0 is TRANSPARENT
    - 1..N are Random High-Contrast Colors
    This fixes the issue of small labels disappearing into background bins.
    """
    np.random.seed(seed)
    # Generate random RGB colors
    colors = np.random.rand(num_classes + 1, 4) 
    # Set Alpha = 1.0 (Opaque) for labels > 0
    colors[:, 3] = 1.0
    # Set Index 0 (Background) to Transparent
    colors[0, :] = [0, 0, 0, 0] 
    
    return ListedColormap(colors)
    
def get_target_label_mask(seg_volume, region_name):
    """
    Creates a mask containing ONLY the labels defined in REGION_MAPS.
    """
    if region_name not in REGION_MAPS:
        return np.zeros_like(seg_volume)
    
    target_ids = []
    # Collect all CT and MR target IDs
    for _, (ct_ids, mr_ids) in REGION_MAPS[region_name].items():
        if isinstance(ct_ids, list): target_ids.extend(ct_ids)
        else: target_ids.append(ct_ids)
        
        if isinstance(mr_ids, list): target_ids.extend(mr_ids)
        else: target_ids.append(mr_ids)
    
    # Create mask where voxel value is in target_ids
    mask = np.isin(seg_volume, target_ids)
    return np.where(mask, seg_volume, 0) # Keep original label values, zero out others

def create_binary_colormap(color_name):
    """Creates a colormap: 0=Transparent, 1=Specific Color."""
    # Matplotlib colors are RGBA 0-1
    if color_name == 'red': c = [1, 0, 0, 0.6] # Red with alpha
    elif color_name == 'green': c = [0, 1, 0, 0.6] # Green with alpha
    else: c = [1, 1, 1, 1]
    
    cmap = np.zeros((2, 4))
    cmap[0, :] = [0, 0, 0, 0] # Background transparent
    cmap[1, :] = c # Foreground color
    return ListedColormap(cmap)
    
def save_registration_vis(paths, region_name, save_path):
    """
    Creates a 3-row visual comparison: 
    Row 1: Intensity (CT, MRI, Warped, Overlay)
    Row 2: All Labels (CT Seg, MRI Seg, Warped Seg, Overlay)
    Row 3: Target Labels (Filtered by Region Map)
    """
    try:
        # 1. Load Data
        fix = nib.load(paths['fixed']).get_fdata()
        mov = nib.load(paths['moving']).get_fdata()
        wrp = nib.load(paths['warped']).get_fdata()
        
        fix_seg = nib.load(paths['fixed_seg']).get_fdata()
        mov_seg = nib.load(paths['moving_seg']).get_fdata()
        wrp_seg = nib.load(paths['warped_seg']).get_fdata()
        
        # 2. Select Middle Slice
        z = fix.shape[2] // 2
        
        # 3. Create Filtered Masks (Target Labels Only)
        fix_seg_filt = get_target_label_mask(fix_seg, region_name)
        mov_seg_filt = get_target_label_mask(mov_seg, region_name)
        wrp_seg_filt = get_target_label_mask(wrp_seg, region_name)

        # 4. Helper for slice extraction & Norm
        def sl(vol, normalize=False):
            s = np.rot90(vol[..., z])
            if normalize:
                denom = s.max() - s.min()
                if denom > 0: s = (s - s.min()) / denom
            return s

        # Determine max label value for consistent color mapping
        max_label = max(fix_seg.max(), mov_seg.max(), wrp_seg.max())
        seg_cmap = create_random_colormap(int(max_label))
        red_cmap = create_binary_colormap('red')
        green_cmap = create_binary_colormap('green')

        # 5. Plotting
        fig, axes = plt.subplots(3, 4, figsize=(20, 15))
        plt.subplots_adjust(hspace=0.3, wspace=0.1)
        
        # --- Row 1: Intensity Images ---
        axes[0,0].imshow(sl(fix, True), cmap='gray', vmin=0, vmax=1)
        axes[0,0].set_title("Fixed (CT)")
        
        axes[0,1].imshow(sl(mov, True), cmap='gray', vmin=0, vmax=1)
        axes[0,1].set_title("Moving (MRI)")
        
        axes[0,2].imshow(sl(wrp, True), cmap='gray', vmin=0, vmax=1)
        axes[0,2].set_title("Warped MRI")
        
        # Overlay (Red=CT, Green=Warped)
        rgb = np.zeros((*sl(fix).shape, 3))
        rgb[..., 0] = sl(fix, True)
        rgb[..., 1] = sl(wrp, True)
        axes[0,3].imshow(rgb)
        axes[0,3].set_title("Overlay (R=CT, G=Warp)")

        # --- Row 2: All Labels ---
        # Use vmin=0 and vmax=max_label for consistent colors
        axes[1,0].imshow(sl(fix_seg), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[1,0].set_title("Fixed Seg (All)")
        
        axes[1,1].imshow(sl(mov_seg), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[1,1].set_title("Moving Seg (All)")
        
        axes[1,2].imshow(sl(wrp_seg), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[1,2].set_title("Warped Seg (All)")
        
        # Overlay Segs (Edges or transparency)
        fix_bin = (sl(fix_seg) > 0).astype(int)
        wrp_bin = (sl(wrp_seg) > 0).astype(int)
        axes[1,3].imshow(fix_bin, cmap=red_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[1,3].imshow(wrp_bin, cmap=green_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[1,3].set_title("Fixed CT vs Warped MR (All)")

        # --- Row 3: Target Labels Only ---
        # Use vmin=0 and vmax=max_label for consistent colors
        axes[2,0].imshow(sl(fix_seg_filt), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,0].set_title("Fixed Seg (Targets)")
        
        axes[2,1].imshow(sl(mov_seg_filt), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,1].set_title("Moving Seg (Targets)")
        
        axes[2,2].imshow(sl(wrp_seg_filt), cmap=seg_cmap, interpolation='nearest', vmin=0, vmax=max_label)
        axes[2,2].set_title("Warped Seg (Targets)")
        
        # Overlay Targets
        fix_filt_bin = (sl(fix_seg_filt) > 0).astype(int)
        wrp_filt_bin = (sl(wrp_seg_filt) > 0).astype(int)
        axes[2,3].imshow(fix_filt_bin, cmap=red_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[2,3].imshow(wrp_filt_bin, cmap=green_cmap, interpolation='nearest', vmin=0, vmax=1)
        axes[2,3].set_title("Fixed CT vs Warped MR (Targets)")

        for ax in axes.flatten():
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    except Exception as e:
        tqdm.write(f"      ⚠️ Visualization failed: {e}")
        
# ==========================================
# DYNAMIC DICE FUNCTION
# ==========================================
def compute_dice_for_region(gt_path, pred_path, region_name):
    try:
        gt = nib.load(gt_path).get_fdata().astype(np.uint16)
        pred = nib.load(pred_path).get_fdata().astype(np.uint16)
    except Exception as e:
        tqdm.write(f"   ⚠️ Error loading NIfTIs: {e}")
        return 0.0

    target_map = REGION_MAPS.get(region_name)
    if not target_map: 
        tqdm.write(f"   ⚠️ No organ map defined for region '{region_name}'. Skipping Dice.")
        return 0.0

    gt_present = np.unique(gt)
    pred_present = np.unique(pred)
    scores = []
    
    # Helper to check if any ID in a list (or single int) is present in the volume
    def is_present(ids, present_list):
        if isinstance(ids, list):
            return any(i in present_list for i in ids)
        return ids in present_list

    # Helper to create boolean mask from int or list of ints
    def make_mask(volume, ids):
        if isinstance(ids, list):
            return np.isin(volume, ids)
        return volume == ids

    for organ, (ct_ids, mr_ids) in target_map.items():
        if not is_present(ct_ids, gt_present) or not is_present(mr_ids, pred_present):
            continue
            
        y_true = make_mask(gt, ct_ids)
        y_pred = make_mask(pred, mr_ids)
        
        inter = np.sum(y_true * y_pred)
        total = np.sum(y_true) + np.sum(y_pred)
        
        dice = (2.0 * inter) / (total + 1e-6)
        scores.append(dice)
        tqdm.write(f"   {organ:<15} | Dice: {dice:.4f}")

    if not scores:
        return 0.0
    avg = np.mean(scores)
    tqdm.write(f"   ✅ Subject Avg: {avg:.4f}")
    return avg

# ==========================================
# TUNING LOGIC
# ==========================================
def save_and_print_results(results, output_root):
    if not results:
        print("No results to aggregate.")
        return

    df = pd.DataFrame(results)
    valid_df = df[df['dice'] > 1e-6]
    
    # Group by Configuration Parameters and calculate Mean Dice across all subjects
    param_cols = [c for c in df.columns if c not in ['subject', 'region', 'dice', 'config_id']]
    
    # Group by config_id
    summary = valid_df.groupby('config_id').agg({
        'dice': 'mean',
        **{c: 'first' for c in param_cols} # Keep param values
    }).sort_values(by='dice', ascending=False)

    # Save Raw with timestamp
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(os.path.join(output_root, f"raw_results_{ts}.csv"), index=False)
    summary.to_csv(os.path.join(output_root, f"summary_results_{ts}.csv"), index=False)

    print("\n" + "="*60)
    print(f"🏆 FINAL TUNING RESULTS (Ranked by Avg Dice) | {ts}")
    print("="*60)
    print(summary.head(10).to_string(columns=['dice'] + param_cols))
    
    if not summary.empty:
        best = summary.iloc[0]
        print("\n⭐ Best Configuration:")
        for k in param_cols:
            print(f"   {k}: {best[k]}")
        print(f"   Avg Dice: {best['dice']:.4f}")

def run_tuning_session(data_dir, output_root, ckpt_path, subjects_config, param_grid):
    os.makedirs(output_root, exist_ok=True)
    results = []

    # Preload Model
    print(f"📦 Preloading model from {ckpt_path}...")
    model = load_model(ckpt_path)

    # 1. Expand Parameter Grid
    keys, values = zip(*param_grid.items())
    experiments = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"🔎 Starting Tuning Session")
    print(f"   Subjects: {len(subjects_config)}")
    print(f"   Configs:  {len(experiments)}")
    print(f"   Total Runs: {len(subjects_config) * len(experiments)}")

    # 2. Iterate SUBJECTS first (Efficient disk usage)
    subj_pbar = tqdm(subjects_config, desc="Subjects", unit="subj")
    for s_idx, subj_conf in enumerate(subj_pbar):
        subj_id = subj_conf['id']
        region = subj_conf['region']

        subj_pbar.set_description(f"Subjects ({subj_id})")

        # --- A. Preprocess (Create Temp Files) ---
        try:
            temp_paths = prepare_subject_data(subj_conf)
        except Exception as e:
            print(f"   ❌ Preprocessing failed: {e}")
            continue

        # --- B. Iterate Parameters on this subject ---
        exp_pbar = tqdm(enumerate(experiments), total=len(experiments), desc="Configs", leave=False)
        for i, params in exp_pbar:
            # Create a unique string for this config
            param_str = "_".join([f"{k}{v}" for k,v in params.items()])
            
            # Store outputs in a subfolder per subject/config
            result_dir = os.path.join(output_root, subj_id, f"conf_{i:02d}_{param_str}")
            os.makedirs(result_dir, exist_ok=True)
            
            tqdm.write(f"   🔹 Config {i}: {params}")

            lam = params.get('lambda_weight', 0.75)
            grid_sp = params.get('grid_sp', 4)
            grid_sp_adam = params.get('grid_sp_adam', 2)
            smooth = params.get('smooth', 0)       
            disp_hw = params.get('disp_hw', 1)     
            niter = params.get('selected_niter', 80) 

            try:
                # Run Registration on TEMP PADDED files
                # Suppress "Using pre-loaded model" etc by capturing stdout
                with contextlib.redirect_stdout(io.StringIO()):
                    convex_adam(
                        ckpt_path=ckpt_path,
                        expname="tune",
                        result_path=result_dir,
                        lambda_weight=lam,
                        grid_sp=grid_sp,
                        selected_smooth=smooth,
                        disp_hw=disp_hw, selected_niter=niter, grid_sp_adam=grid_sp_adam,
                        ic=False, use_mask=True, warp_seg=True,
                        fixed_image=temp_paths['fixed'], 
                        fixed_mask=temp_paths['fixed_mask'], 
                        fixed_seg=temp_paths['fixed_seg'],
                        fixed_minclip=-450, fixed_maxclip=450,
                        moving_image=temp_paths['moving'], 
                        moving_mask=temp_paths['moving_mask'], 
                        moving_seg=temp_paths['moving_seg'],
                        model=model
                    )

                # Locate Output Files, Compute Dice Score, Visualize output
                warped_img_candidates = glob.glob(os.path.join(result_dir, "moved_mr_padded_temp*.nii.gz"))
                warped_seg_candidates = glob.glob(os.path.join(result_dir, "labels_moved_*.nii.gz"))
                dice = 0.0
                if warped_seg_candidates:
                    warped_seg_path = warped_seg_candidates[0]
                    # Calc Dice
                    dice = compute_dice_for_region(temp_paths['fixed_seg'], warped_seg_path, region)
                    
                    # Generate Visualization
                    if warped_img_candidates:
                        warped_img_path = warped_img_candidates[0]
                        vis_path = os.path.join(result_dir, "vis_registration.png")
                        
                        vis_paths = {
                            'fixed': temp_paths['fixed'],
                            'moving': temp_paths['moving'],
                            'warped': warped_img_path,
                            'fixed_seg': temp_paths['fixed_seg'],
                            'moving_seg': temp_paths['moving_seg'],
                            'warped_seg': warped_seg_path
                        }
                        
                        save_registration_vis(vis_paths, region, vis_path)
                else:
                    tqdm.write("      ⚠️ Output missing.")

                # Store Result
                rec = params.copy()
                rec['subject'] = subj_id
                rec['region'] = region
                rec['dice'] = dice
                rec['config_id'] = i # Helpful for grouping later
                results.append(rec)

                exp_pbar.set_postfix({"Dice": f"{dice:.4f}"})


            except Exception as e:
                error_msg = str(e)
                tqdm.write(f"      💥 Registration Crash: {error_msg}")
                
                # Record the failure in the CSV
                rec = params.copy()
                rec['subject'] = subj_id
                rec['region'] = region
                rec['config_id'] = i
                
                if "out of memory" in error_msg.lower():
                    rec['dice'] = -1.0 # OOM indicator
                else:
                    rec['dice'] = -2.0 # Other error indicator
                results.append(rec)
                
            finally:
                gc.collect()
                torch.cuda.empty_cache()

        # --- C. Cleanup (Delete Temp Files) ---
        cleanup_subject_data(temp_paths)

    # 3. Final Aggregation
    save_and_print_results(results, output_root)

if __name__ == "__main__":
    subjects_config = discover_tuning_subjects(DATA_DIR, target_list=TARGET_SUBJECTS)
    if not subjects_config:
        print("❌ No subjects found. Exiting.")
        exit()
        
    run_tuning_session(DATA_DIR, OUTPUT_DIR, CKPT, subjects_config, grid)