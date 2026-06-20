import os
import multiprocessing as mp
import nibabel as nib
import numpy as np
import pandas as pd
from tqdm import tqdm

PROJ = "/home/minsukc/MRI2CT"
DATA = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
SEG = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/cads/seg"
MERGED_DIR = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/cads/merged"
CSV = os.path.join(PROJ, "src", "preprocess", "cads_labelmap.csv")
TASKS = [551, 552, 553, 554, 555, 556, 557, 558, 559]
FILE_NAME = "cads_grouped_35_labels_seg.nii.gz"

def load_map():
    m = pd.read_csv(CSV)
    m = m[m.src_index != 0].copy() # drop Background no-ops
    m["paint_id"] = m.final_id.astype(int)
    m["paint_label"] = m.final_label
    m = m.sort_values("priority", kind="stable") # paint low->high; MANDATORY
    return m

def process_subject(args):
    subj, m = args
    try:
        seg_dir = os.path.join(SEG, subj)
        parts, ref = {}, None
        for t in TASKS:
            f = os.path.join(seg_dir, f"{subj}_part_{t}.nii.gz")
            if not os.path.exists(f):
                return subj, False, f"MISSING part {t}"
            img = nib.load(f)
            if ref is None:
                ref = img
            else:
                if img.shape != ref.shape:
                    return subj, False, f"shape mismatch part {t}"
            parts[t] = np.asarray(img.dataobj).astype(np.int16)

        # Create uint8 array (since max label is 35)
        out = np.zeros(ref.shape, dtype=np.uint8)
        
        # Priority painting logic EXACTLY as in merge_demo.py
        for _, r in m.iterrows():
            part = parts[int(r.src_model)]
            out[part == int(r.src_index)] = int(r.paint_id)

        # Save to MERGED_DIR
        out_nii = nib.Nifti1Image(out, ref.affine, ref.header)
        out_nii.header.set_data_dtype(np.uint8)
        
        target_path = os.path.join(MERGED_DIR, f"{subj}_{FILE_NAME}")
        nib.save(out_nii, target_path)

        # Create symlink in the main dataset folder
        subj_data_dir = os.path.join(DATA, subj)
        if os.path.exists(subj_data_dir):
            symlink_path = os.path.join(subj_data_dir, FILE_NAME)
            # Safe relative target computation
            rel_target = os.path.relpath(target_path, subj_data_dir)
            
            if os.path.islink(symlink_path) or os.path.exists(symlink_path):
                os.remove(symlink_path)
            
            os.symlink(rel_target, symlink_path)
            
        return subj, True, None
    except Exception as e:
        return subj, False, str(e)

def main():
    os.makedirs(MERGED_DIR, exist_ok=True)
    m = load_map()
    
    subjects = sorted([d for d in os.listdir(SEG) if os.path.isdir(os.path.join(SEG, d))])
    print(f"Found {len(subjects)} subjects in {SEG}")
    
    args = [(subj, m) for subj in subjects]
    
    success = 0
    failed = []
    
    print(f"Starting multiprocessing pool with 16 workers...")
    with mp.Pool(16) as pool:
        results = list(tqdm(pool.imap_unordered(process_subject, args), total=len(subjects)))
        
    for subj, status, err in results:
        if status:
            success += 1
        else:
            failed.append((subj, err))
            
    print(f"\nFinished. Successfully merged: {success}/{len(subjects)}")
    if failed:
        print(f"Failed ({len(failed)}):")
        for subj, err in failed[:10]:
            print(f"  {subj}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more.")

if __name__ == "__main__":
    main()
