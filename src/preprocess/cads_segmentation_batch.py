import os
import torch
import shutil
import argparse
import sys
from tqdm import tqdm

# Add project root and CADS to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CADS"))

try:
    from cads.utils.libs import setup_nnunet_env, get_model_weights_dir, check_or_download_model_weights
    from cads.utils.inference import predict
except ImportError:
    print("Error: CADS package not found. Please ensure CADS is in the project root.")
    sys.exit(1)

# --- Global Config ---
DEFAULT_GPFS_WEIGHTS_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/cads_weights"
TASK_ID = 559

def discover_subjects(data_root):
    valid_subjects = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path): continue
        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        for subj_id in candidates:
            subj_path = os.path.join(split_path, subj_id)
            ct_path = os.path.join(subj_path, "ct.nii.gz")
            if os.path.exists(ct_path):
                valid_subjects.append({"id": subj_id, "path": subj_path, "ct": ct_path})
    return valid_subjects

def main():
    parser = argparse.ArgumentParser(description="Batch CADS Segmentation for MRI2CT")
    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk_size", type=int, default=50)
    args = parser.parse_args()

    os.environ["CADS_WEIGHTS_PATH"] = DEFAULT_GPFS_WEIGHTS_PATH
    setup_nnunet_env()
    model_folder = get_model_weights_dir()
    check_or_download_model_weights(TASK_ID)

    subjects = discover_subjects(args.data_dir)
    to_process = []
    for s in subjects:
        p_final = os.path.join(s['path'], "cads_ct_seg.nii.gz")
        if not os.path.exists(p_final) or args.force:
            to_process.append(s)

    if not to_process:
        print("All subjects already processed.")
        return

    print(f"Remaining: {len(to_process)} subjects.")
    use_cpu = not torch.cuda.is_available()

    pbar = tqdm(total=len(to_process), desc="CADS Segmentation")
    for i in range(0, len(to_process), args.chunk_size):
        chunk = to_process[i:i + args.chunk_size]
        
        batch_temp_dir = os.path.join(args.data_dir, f"cads_batch_tmp_{i}")
        os.makedirs(batch_temp_dir, exist_ok=True)
        tmp_in = os.path.join(batch_temp_dir, "input")
        tmp_out = os.path.join(batch_temp_dir, "output")
        os.makedirs(tmp_in); os.makedirs(tmp_out)

        # 1. Link existing CT (renaming to Subject ID for CADS)
        chunk_ct_paths = []
        for subj in chunk:
            ct_link = os.path.join(tmp_in, f"{subj['id']}.nii.gz")
            if os.path.exists(ct_link): os.remove(ct_link)
            os.symlink(subj['ct'], ct_link)
            chunk_ct_paths.append(ct_link)

        # 2. Run CADS Prediction
        try:
            predict(
                files_in=chunk_ct_paths,
                folder_out=tmp_out,
                model_folder=model_folder,
                task_ids=[TASK_ID],
                folds='all',
                use_cpu=use_cpu,
                preprocess_cads=True,
                postprocess_cads=True,
                save_all_combined_seg=False,
                verbose=False
            )

            # 3. Move results to final destination
            for subj in chunk:
                pid = subj['id']
                raw_seg = os.path.join(tmp_out, pid, f"{pid}_part_{TASK_ID}.nii.gz")
                final_seg = os.path.join(subj['path'], "cads_ct_seg.nii.gz")
                
                if os.path.exists(raw_seg):
                    shutil.move(raw_seg, final_seg)
                else:
                    print(f"\nWarning: Segmentation failed for subject {pid}")

        except Exception as e:
            print(f"\nError during chunk processing: {e}")
        finally:
            if os.path.exists(batch_temp_dir):
                shutil.rmtree(batch_temp_dir)
            if not use_cpu:
                torch.cuda.empty_cache()
            pbar.update(len(chunk))

    pbar.close()
    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
