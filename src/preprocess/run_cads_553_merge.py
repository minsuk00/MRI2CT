import argparse
import glob
import os
import shutil
import sys

import nibabel as nib
import numpy as np
import torch
from tqdm import tqdm

# Add project root and CADS to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(PROJECT_ROOT)
sys.path.append(os.path.join(PROJECT_ROOT, "CADS"))

try:
    from cads.utils.inference import predict
    from cads.utils.libs import check_or_download_model_weights, get_model_weights_dir, setup_nnunet_env
except ImportError:
    print("Error: CADS package not found. Please ensure CADS is in the project root.")
    sys.exit(1)

# --- Global Config ---
DEFAULT_GPFS_WEIGHTS_PATH = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/cads_weights"
TASK_ID = 553
BRAIN_LABEL_553 = 9
NEW_BRAIN_LABEL = 11


def get_region_key(subj_id):
    """Determines region key from subject ID (e.g., 1ABA005 -> abdomen)."""
    mapping = {"AB": "abdomen", "TH": "thorax", "HN": "head_neck", "B": "brain", "P": "pelvis"}
    if not subj_id or len(subj_id) < 2:
        return "abdomen"
    code_2 = subj_id[1:3].upper()
    code_1 = subj_id[1:2].upper()
    if code_2 in mapping:
        return mapping[code_2]
    if code_1 in mapping:
        return mapping[code_1]
    return "abdomen"


def discover_subjects(data_root):
    valid_subjects = []
    splits = ["train", "val", "test"]
    for split in splits:
        split_path = os.path.join(data_root, split)
        if not os.path.exists(split_path):
            continue
        candidates = sorted([d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))])
        for subj_id in candidates:
            subj_path = os.path.join(split_path, subj_id)
            ct_path = os.path.join(subj_path, "ct.nii.gz")
            seg_559_path = os.path.join(subj_path, "cads_ct_seg_559.nii.gz")
            if os.path.exists(ct_path) and os.path.exists(seg_559_path):
                valid_subjects.append({"id": subj_id, "path": subj_path, "ct": ct_path, "seg_559": seg_559_path, "region": get_region_key(subj_id)})
    return valid_subjects


def merge_brain_label(subj_path, seg_559_path, seg_553_path):
    """
    Load 559, load 553, extract label 9 from 553, and set it to 11 in 559.
    Save as cads_ct_seg.nii.gz
    """
    final_path = os.path.join(subj_path, "cads_ct_seg.nii.gz")

    img_559 = nib.load(seg_559_path)
    data_559 = img_559.get_fdata().astype(np.uint8)

    img_553 = nib.load(seg_553_path)
    data_553 = img_553.get_fdata().astype(np.uint8)

    brain_mask = data_553 == BRAIN_LABEL_553
    count = np.sum(brain_mask)
    if count > 0:
        data_559[brain_mask] = NEW_BRAIN_LABEL
        print(f"  [Merge] Added {count} brain voxels for {os.path.basename(subj_path)}")
        new_img = nib.Nifti1Image(data_559, img_559.affine, img_559.header)
        nib.save(new_img, final_path)
    else:
        print(f"  [Merge] No brain voxels found for {os.path.basename(subj_path)}. Copying original 559.")
        shutil.copy(seg_559_path, final_path)


def main():
    parser = argparse.ArgumentParser(description="Batch CADS 553 Brain Extraction & Merge for MRI2CT")
    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--chunk_idx", type=int, default=0, help="Chunk index for parallel processing")
    parser.add_argument("--num_chunks", type=int, default=1, help="Total number of chunks")
    parser.add_argument("--batch_size", type=int, default=5, help="Number of subjects per prediction batch")
    args = parser.parse_args()

    os.environ["CADS_WEIGHTS_PATH"] = DEFAULT_GPFS_WEIGHTS_PATH
    setup_nnunet_env()
    model_folder = get_model_weights_dir()
    check_or_download_model_weights(TASK_ID)

    subjects = discover_subjects(args.data_dir)

    # Chunking
    chunk_size = (len(subjects) + args.num_chunks - 1) // args.num_chunks
    start_idx = args.chunk_idx * chunk_size
    end_idx = min(start_idx + chunk_size, len(subjects))
    chunk_subjects = subjects[start_idx:end_idx]

    print(f"Chunk {args.chunk_idx}/{args.num_chunks}: Processing {len(chunk_subjects)} subjects.")

    to_process = []
    for s in chunk_subjects:
        final_seg = os.path.join(s["path"], "cads_ct_seg.nii.gz")

        # Optimization: Only run 553 on brain/head_neck
        if s["region"] not in ["brain", "head_neck"]:
            if not os.path.exists(final_seg) or args.force:
                print(f"  [Skip] {s['id']} is {s['region']}. Copying 559 directly.")
                shutil.copy(s["seg_559"], final_seg)
            continue

        # If it IS brain/head_neck, check if processed
        if not os.path.exists(final_seg) or args.force:
            to_process.append(s)

    if not to_process:
        print("All target subjects in this chunk already processed.")
        return

    print(f"Remaining target subjects to run CADS 553: {len(to_process)}.")
    use_cpu = not torch.cuda.is_available()

    pbar = tqdm(total=len(to_process), desc=f"Chunk {args.chunk_idx} - CADS 553")

    for i in range(0, len(to_process), args.batch_size):
        batch = to_process[i : i + args.batch_size]

        # Temp dir for nnunet prediction
        batch_temp_dir = os.path.join(args.data_dir, f"cads_553_tmp_c{args.chunk_idx}_{i}")
        os.makedirs(batch_temp_dir, exist_ok=True)
        tmp_in = os.path.join(batch_temp_dir, "input")
        tmp_out = os.path.join(batch_temp_dir, "output")
        os.makedirs(tmp_in)
        os.makedirs(tmp_out)

        # 1. Link existing CT (renaming to Subject ID for CADS)
        batch_ct_paths = []
        for subj in batch:
            ct_link = os.path.join(tmp_in, f"{subj['id']}.nii.gz")
            if os.path.exists(ct_link):
                os.remove(ct_link)
            os.symlink(subj["ct"], ct_link)
            batch_ct_paths.append(ct_link)

        # 2. Run CADS Prediction for Task 553
        try:
            predict(
                files_in=batch_ct_paths,
                folder_out=tmp_out,
                model_folder=model_folder,
                task_ids=[TASK_ID],
                folds="all",
                use_cpu=use_cpu,
                preprocess_cads=True,
                postprocess_cads=True,
                save_all_combined_seg=False,
                verbose=False,
            )

            # 3. Merge results
            for subj in batch:
                pid = subj["id"]
                raw_seg_553 = os.path.join(tmp_out, pid, f"{pid}_part_{TASK_ID}.nii.gz")

                if os.path.exists(raw_seg_553):
                    # Save the 553 result for reference
                    perm_seg_553 = os.path.join(subj["path"], "cads_ct_seg_553.nii.gz")
                    shutil.copy(raw_seg_553, perm_seg_553)

                    merge_brain_label(subj["path"], subj["seg_559"], raw_seg_553)
                else:
                    print(f"\nWarning: Segmentation 553 failed for subject {pid}")

        except Exception as e:
            print(f"\nError during batch processing: {e}")
        finally:
            if os.path.exists(batch_temp_dir):
                shutil.rmtree(batch_temp_dir)
            if not use_cpu:
                torch.cuda.empty_cache()
            pbar.update(len(batch))

    pbar.close()
    print(f"\nChunk {args.chunk_idx} complete.")


if __name__ == "__main__":
    main()
