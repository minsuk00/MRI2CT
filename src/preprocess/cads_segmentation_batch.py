import os
import torch
import shutil
import argparse
from tqdm import tqdm
import sys
import tempfile
import SimpleITK as sitk

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

def resample_sitk(image_path, target_spacing=(1.5, 1.5, 1.5), is_mask=False, out_path=None):
    """Resamples a NIfTI image to a target isotropic spacing."""
    img = sitk.ReadImage(image_path)
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()

    new_size = [
        int(round(osz * ospc / tspc))
        for osz, ospc, tspc in zip(original_size, original_spacing, target_spacing)
    ]

    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(target_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(img.GetDirection())
    resample.SetOutputOrigin(img.GetOrigin())
    resample.SetOutputPixelType(img.GetPixelIDValue())
    
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)

    new_img = resample.Execute(img)
    if out_path:
        sitk.WriteImage(new_img, out_path)
        return out_path
    return new_img

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
    parser.add_argument("--data_dir", type=str, default="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/3.0x3.0x3.0mm")
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
        p3 = os.path.join(s['path'], "cads_ct_seg.nii.gz")
        p15 = os.path.join(s['path'], "cads_ct_seg_1.5mm.nii.gz")
        if not (os.path.exists(p3) and os.path.exists(p15)) or args.force:
            to_process.append(s)

    if not to_process:
        print("All subjects already processed.")
        return

    print(f"Remaining: {len(to_process)} subjects.")
    use_cpu = not torch.cuda.is_available()

    for i in range(0, len(to_process), args.chunk_size):
        chunk = to_process[i:i + args.chunk_size]
        print(f"\n--- Processing Chunk {i//args.chunk_size + 1} ---")

        batch_temp_dir = os.path.join(args.data_dir, f"cads_batch_tmp_{i}")
        os.makedirs(batch_temp_dir, exist_ok=True)
        tmp_in = os.path.join(batch_temp_dir, "input")
        tmp_out = os.path.join(batch_temp_dir, "output")
        os.makedirs(tmp_in); os.makedirs(tmp_out)

        # 1. Manually resample input CT to 1.5mm
        chunk_ct_15_paths = []
        for subj in chunk:
            ct_15_path = os.path.join(tmp_in, f"{subj['id']}.nii.gz")
            resample_sitk(subj['ct'], (1.5, 1.5, 1.5), is_mask=False, out_path=ct_15_path)
            chunk_ct_15_paths.append(ct_15_path)

        # 2. Run CADS Prediction
        try:
            predict(
                files_in=chunk_ct_15_paths,
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

            # 3. Handle Outputs
            for subj in chunk:
                pid = subj['id']
                raw_seg_15 = os.path.join(tmp_out, pid, f"{pid}_part_{TASK_ID}.nii.gz")
                final_seg_3mm = os.path.join(subj['path'], "cads_ct_seg.nii.gz")
                final_seg_15mm = os.path.join(subj['path'], "cads_ct_seg_1.5mm.nii.gz")
                
                if os.path.exists(raw_seg_15):
                    # Save the high-res 1.5mm version
                    shutil.copy(raw_seg_15, final_seg_15mm)
                    # Downsample to 3.0mm for current training
                    resample_sitk(raw_seg_15, (3.0, 3.0, 3.0), is_mask=True, out_path=final_seg_3mm)
                else:
                    print(f"Warning: Failed for {pid}")

        except Exception as e:
            print(f"Error: {e}")
        finally:
            shutil.rmtree(batch_temp_dir)
            torch.cuda.empty_cache()

    print("\nDone.")

if __name__ == "__main__":
    main()
