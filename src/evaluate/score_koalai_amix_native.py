"""Score koalAI fold-0 predictions using amix-native metrics.

Reads validation_revert_norm/<subj_id>.mha per region (HU predictions),
resamples to the same grid as the validators' NIfTI outputs,
then computes MAE_HU/PSNR/SSIM (full-vol + body-masked) using the same
compute_metrics / compute_metrics_body functions as the other validators.

Output: evaluation_results/baselines_latest_20260529/koalai/validate_metrics.txt

Usage:
    python src/evaluate/score_koalai_amix_native.py
    python src/evaluate/score_koalai_amix_native.py --split_file splits/center_wise_split.txt
"""

import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch

_REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "src"))
from src.common.data import get_region_key, get_split_subjects  # noqa: E402
from common.eval_utils import write_metrics_txt  # noqa: E402
from src.common.utils import compute_metrics, compute_metrics_body  # noqa: E402

SYNTHRAD_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked"
NNSYN_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace/results"

KOALAI_REGION_DS = {
    "thorax": "Dataset962_synthrad2025_task1_mri2ct_thorax",
    "abdomen": "Dataset960_synthrad2025_task1_mri2ct_abdomen",
    "head_neck": "Dataset964_synthrad2025_task1_mri2ct_head_neck",
    "brain": "Dataset966_synthrad2025_task1_mri2ct_brain",
    "pelvis": "Dataset968_synthrad2025_task1_mri2ct_pelvis",
}
KOALAI_TRAINER = "nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres"

CT_HU_MIN, CT_HU_MAX = -1024.0, 1024.0
HU_RANGE = CT_HU_MAX - CT_HU_MIN  # 2048


def find_gt_paths(subj_id):
    root = os.path.join(SYNTHRAD_ROOT, subj_id)
    for fn in ("ct.nii.gz", "ct.nii"):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            ct_path = p
            break
    else:
        raise FileNotFoundError(f"No ct.nii[.gz] for {subj_id}")
    for fn in ("mask.nii.gz", "mask.nii"):
        p = os.path.join(root, fn)
        if os.path.exists(p):
            mask_path = p
            break
    else:
        raise FileNotFoundError(f"No mask.nii[.gz] for {subj_id}")
    return ct_path, mask_path


def load_koalai_pred(subj_id):
    region = get_region_key(subj_id)
    ds = KOALAI_REGION_DS[region]
    mha_path = os.path.join(NNSYN_ROOT, ds, KOALAI_TRAINER, "fold_0", "validation_revert_norm", f"{subj_id}.mha")
    img = sitk.ReadImage(mha_path)
    # sitk: [z,y,x] in LPS → transpose to [x,y,z] in LPS, then flip x+y to get RAS
    arr = sitk.GetArrayFromImage(img).transpose(2, 1, 0).astype(np.float32)
    arr = np.flip(arr, axis=(0, 1)).copy()
    return arr


def load_gt_and_mask(subj_id):
    ct_path, mask_path = find_gt_paths(subj_id)
    ct = nib.as_closest_canonical(nib.load(ct_path)).get_fdata().astype(np.float32)
    mask = nib.as_closest_canonical(nib.load(mask_path)).get_fdata().astype(np.float32)
    mask = (mask > 0.5).astype(np.float32)
    return ct, mask


def clip_to_01(arr, hu_min=CT_HU_MIN, hu_max=CT_HU_MAX):
    return np.clip((arr - hu_min) / (hu_max - hu_min), 0.0, 1.0)


def align_shapes(pred, gt):
    """Pad or crop pred to match gt shape (same grid, just handle edge padding diffs)."""
    if pred.shape == gt.shape:
        return pred
    out = np.full(gt.shape, 0.0, dtype=np.float32)
    s = tuple(slice(0, min(a, b)) for a, b in zip(pred.shape, gt.shape))
    out[s] = pred[s]
    return out


def to_tensor(arr, device):
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)


def score_subject(subj_id, device):
    t0 = time.time()

    pred_arr = load_koalai_pred(subj_id)   # HU, (x,y,z)
    gt_arr, mask_arr = load_gt_and_mask(subj_id)  # (x,y,z) in RAS

    pred_arr = align_shapes(pred_arr, gt_arr)
    mask_arr = align_shapes(mask_arr, gt_arr)

    pred_01 = clip_to_01(pred_arr)
    gt_01 = clip_to_01(gt_arr)

    pred_t = to_tensor(pred_01, device)
    gt_t = to_tensor(gt_01, device)
    mask_t = to_tensor(mask_arr, device)

    full = compute_metrics(pred_t, gt_t, data_range=1.0, hu_range=HU_RANGE)
    body = compute_metrics_body(pred_t, gt_t, mask_t, hu_range=HU_RANGE)

    metrics = {
        "mae_hu": full["mae_hu"],
        "psnr": full["psnr"],
        "ssim": full["ssim"],
        "body_mae_hu": body["mae_hu"],
        "body_psnr": body["psnr"],
        "body_ssim": body["ssim"],
        "time_sec": time.time() - t0,
    }
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", default="splits/center_wise_split.txt")
    parser.add_argument("--split_name", default="val")
    parser.add_argument("--out_dir", default="evaluation_results/baselines_latest_20260529/koalai")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    subjects = get_split_subjects(args.split_file, args.split_name)
    print(f"[score-koalai] {len(subjects)} subjects, device={device}")

    records = []
    for i, subj in enumerate(subjects):
        try:
            metrics = score_subject(subj, device)
            records.append({"subj_id": subj, "metrics": metrics})
            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{len(subjects)}] {subj}  MAE={metrics['mae_hu']:.1f}HU  PSNR={metrics['psnr']:.2f}  SSIM={metrics['ssim']:.3f}")
        except Exception as e:
            print(f"  [SKIP] {subj}: {e}")

    out_path = os.path.join(args.out_dir, "validate_metrics.txt")
    write_metrics_txt(
        out_path,
        header_lines=[
            "Validation report — KoalAI (fold-0 OOD val, amix-native metrics)",
            "checkpoint: nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres/fold_0/checkpoint_final.pth",
            f"split_file: {args.split_file}   split_name: {args.split_name}",
            f"subjects: {len(records)}",
            "clip: [-1024, 1024] HU -> [0, 1];  hu_range=2048 for MAE_HU",
        ],
        per_subject=records,
        metric_keys=["mae_hu", "psnr", "ssim", "body_mae_hu", "body_psnr", "body_ssim", "time_sec"],
    )
    print(f"[score-koalai] wrote {out_path}")


if __name__ == "__main__":
    main()
