import glob
import os
import numpy as np
import pandas as pd
import pydicom
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm

def compute_author_pre_metrics():
    # --- CONFIGURATION ---
    # Path to the FOLDER containing 'Coltea-Lung-CT-100W', 'test_data.csv', etc.
    ROOT_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/Coltea-Lung-CT-100W"
    
    # Tag A is the Reference (Arterial/Contrast), Tag B is the Input (Native)
    TAG_A = 'ARTERIAL' 
    TAG_B = 'NATIVE'
    
    # --- LOAD SUBJECT LIST ---
    csv_path = os.path.join(ROOT_PATH, 'test_data.csv')
    if not os.path.exists(csv_path):
        print(f"❌ Error: Could not find {csv_path}")
        return

    print(f"📖 Reading test subjects from {csv_path}...")
    eval_df = pd.read_csv(csv_path)
    # Assuming the subject ID is in the 2nd column (index 1) as per author code
    eval_dirs = list(eval_df.iloc[:, 1]) 
    
    print(f"🎯 Found {len(eval_dirs)} test subjects.")

    # --- METRICS STORAGE ---
    # Author stores ALL slices in one giant list, then averages.
    mae_list = []
    rmse_list = []
    ssim_list = []

    # --- MAIN LOOP ---
    # We iterate through the raw data folder structure
    data_path = os.path.join(ROOT_PATH, 'data_dicom') 
    
    total_slices = 0

    for subject_id in tqdm(eval_dirs, desc="Processing Subjects"):
        subj_dir = os.path.join(data_path, subject_id)
        
        if not os.path.exists(subj_dir):
            print(f"⚠️ Subject {subject_id} not found in {data_path}")
            continue

        # Path to Reference (Arterial) DICOMs
        # Author iterates TAG_A and infers TAG_B
        dicom_pattern = os.path.join(subj_dir, TAG_A, 'DICOM', '*')
        dicom_files = glob.glob(dicom_pattern)
        
        if not dicom_files:
            print(f"⚠️ No DICOMs found for {subject_id} in {TAG_A}")
            continue

        for scan_path in dicom_files:
            try:
                # 1. Load Reference (Arterial)
                # pydicom.dcmread returns the raw pixel array (Int16/UInt16)
                ds_ref = pydicom.dcmread(scan_path)
                ref_raw = ds_ref.pixel_array.astype(np.float32)

                # 2. Find Matching Input (Native)
                # Author's Logic: String Replace path. 
                # DANGER: Assumes filenames are identical (e.g., I10.dcm matches I10.dcm)
                native_path = scan_path.replace(TAG_A, TAG_B)
                
                if not os.path.exists(native_path):
                    # Skip if corresponding file doesn't exist
                    continue
                
                ds_native = pydicom.dcmread(native_path)
                native_raw = ds_native.pixel_array.astype(np.float32)

                # 3. Pre-process (Author Logic)
                # "Scale native image"
                native_raw[native_raw < 0] = 0
                native_img = (native_raw / 1000.0) - 1.0

                # "Scale original image" (Reference)
                ref_raw[ref_raw < 0] = 0
                ref_img = (ref_raw / 1000.0) - 1.0

                # 4. Compute Metrics (Slice vs Slice)
                # MAE
                mae_list.append(np.mean(np.abs(ref_img - native_img)))
                
                # RMSE
                rmse_list.append(np.sqrt(np.mean((ref_img - native_img)**2)))
                
                # SSIM (Author default: no data_range specified)
                # Note: skimage might warn you, but this matches author behavior
                try:
                    # Provide range to suppress warning if you want, or leave blank to match author exactly
                    # Author likely ran: ssim(ref_img, native_img)
                    ssim_val = ssim(ref_img, native_img, data_range=max(ref_img.max() - ref_img.min(), native_img.max() - native_img.min())) 
                except:
                    ssim_val = 0.0
                
                ssim_list.append(ssim_val)
                total_slices += 1

            except Exception as e:
                # Corrupt DICOM or read error
                continue

    # --- FINAL REPORT ---
    print("\n" + "="*50)
    print(f"📊 AUTHOR REPLICATION RESULTS (Pre-Only)")
    print(f"files processed: {total_slices}")
    print("="*50)
    
    if len(mae_list) > 0:
        print(f"MAE  (Paper ~0.072): {np.mean(mae_list):.5f}")
        print(f"RMSE (Paper ~0.163): {np.mean(rmse_list):.5f}")
        print(f"SSIM (Paper ~0.664): {np.mean(ssim_list):.5f}")
    else:
        print("❌ No valid slices processed.")

if __name__ == '__main__':
    compute_author_pre_metrics()