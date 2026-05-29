"""SynthRAD-native image metrics (MAE + PSNR) for unet, amix_v1_4 predictions.

Metric spec (from koalAI ImageMetrics):
  - clip to [-1024, 3000] HU (PSNR only; MAE uses raw values but masked)
  - body-masked: divides by mask.sum() (not total voxels)
  - PSNR data_range = 4024

KoalAI results are already in evaluation_results/koalai_native/fold0/ — read from there.

Usage:
    python src/evaluate/score_synthrad_metric.py
"""

import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import numpy as np
import nibabel as nib
import SimpleITK as sitk
import pandas as pd
from skimage.metrics import peak_signal_noise_ratio
from glob import glob

from src.common.data import get_region_key, get_split_subjects  # noqa: E402

NNSYN_ORIGIN = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace/origin"
EVAL_ROOT = "evaluation_results/baselines_latest_20260529"
DR = [-1024., 3000.]
DR_RANGE = DR[1] - DR[0]  # 4024

REGION_DS = {
    "abdomen":  "synthrad2025_task1_mri2ct_abdomen",
    "thorax":   "synthrad2025_task1_mri2ct_thorax",
    "head_neck":"synthrad2025_task1_mri2ct_head_neck",
    "brain":    "synthrad2025_task1_mri2ct_brain",
    "pelvis":   "synthrad2025_task1_mri2ct_pelvis",
}


def load_origin_gt(subj_id):
    region = get_region_key(subj_id)
    ds = REGION_DS[region]
    gt_path = os.path.join(NNSYN_ORIGIN, ds, "TARGET_IMAGES", f"{subj_id}_0000.mha")
    mask_path = os.path.join(NNSYN_ORIGIN, ds, "MASKS", f"{subj_id}.mha")
    # sitk [z,y,x] LPS → [x,y,z] → flip x+y → RAS
    def read_mha(p):
        arr = sitk.GetArrayFromImage(sitk.ReadImage(p)).transpose(2, 1, 0).astype(np.float32)
        return np.flip(arr, axis=(0, 1)).copy()
    return read_mha(gt_path), read_mha(mask_path)


def load_pred_nifti(path):
    return nib.load(path).get_fdata().astype(np.float32)


def find_pred(model_root, subj_id):
    flat = os.path.join(model_root, subj_id, "sample.nii.gz")
    if os.path.exists(flat):
        return flat
    sharded = glob(os.path.join(model_root, "shard_*_of_*", subj_id, "sample.nii.gz"))
    if sharded:
        return sharded[0]
    raise FileNotFoundError(f"No sample.nii.gz for {subj_id} under {model_root}")


def synthrad_metrics(pred_arr, gt_arr, mask_arr):
    mask = (mask_arr > 0.5).astype(np.float32)
    # MAE: body-masked, divide by body count
    mae = float(np.sum(np.abs(gt_arr * mask - pred_arr * mask)) / mask.sum())
    # PSNR: clip to [-1024,3000], body voxels only, data_range=4024
    gt_c  = np.clip(gt_arr,   DR[0], DR[1])
    pred_c = np.clip(pred_arr, DR[0], DR[1])
    gt_body   = gt_c[mask == 1]
    pred_body = pred_c[mask == 1]
    psnr = float(peak_signal_noise_ratio(gt_body, pred_body, data_range=DR_RANGE))
    return {"synthrad_mae": mae, "synthrad_psnr": psnr}


def score_model(model_key, model_root, subjects):
    records = []
    n_ok = 0
    for subj in subjects:
        try:
            pred_arr = load_pred_nifti(find_pred(model_root, subj))
            gt_arr, mask_arr = load_origin_gt(subj)
            if pred_arr.shape != gt_arr.shape:
                # align shapes (pad/crop)
                out = np.full(gt_arr.shape, DR[0], dtype=np.float32)
                s = tuple(slice(0, min(a, b)) for a, b in zip(pred_arr.shape, gt_arr.shape))
                out[s] = pred_arr[s]
                pred_arr = out
            m = synthrad_metrics(pred_arr, gt_arr, mask_arr)
            records.append({"subj_id": subj, **m})
            n_ok += 1
        except Exception as e:
            print(f"  [SKIP] {subj}: {e}")
    print(f"  {model_key}: {n_ok}/{len(subjects)} scored")
    return records


def build_table(all_records, subjects):
    from src.common.data import get_region_key
    REGIONS = ["abdomen", "brain", "head_neck", "pelvis", "thorax"]
    lines = []
    lines.append("=" * 80)
    lines.append("SynthRAD metric: clip[-1024,3000], body-masked MAE, PSNR data_range=4024")
    lines.append("NOTE: unet/amix preds capped at ~1024 HU (training clip) → penalized on bone>1024")
    lines.append("=" * 80)
    hdr = f"{'region':<12} {'model':<16} {'n':>4}  {'MAE_HU':>8}  {'PSNR':>7}"
    lines.append(hdr)
    lines.append("-" * 55)

    models = list(all_records.keys())
    total = {m: [] for m in models}
    for region in REGIONS:
        subjs_r = [s for s in subjects if get_region_key(s) == region]
        for m in models:
            recs = [r for r in all_records[m] if r["subj_id"] in subjs_r]
            if not recs:
                continue
            mae  = np.mean([r["synthrad_mae"]  for r in recs])
            psnr = np.mean([r["synthrad_psnr"] for r in recs])
            lines.append(f"{region:<12} {m:<16} {len(recs):>4}  {mae:>8.2f}  {psnr:>7.2f}")
            total[m].extend(recs)
        lines.append("")

    lines.append("-" * 55)
    for m in models:
        recs = total[m]
        if not recs:
            continue
        mae  = np.mean([r["synthrad_mae"]  for r in recs])
        psnr = np.mean([r["synthrad_psnr"] for r in recs])
        lines.append(f"{'MACRO avg':<12} {m:<16} {len(recs):>4}  {mae:>8.2f}  {psnr:>7.2f}")

    return "\n".join(lines)


def main():
    subjects = get_split_subjects("splits/center_wise_split.txt", "val")

    models = {
        "UNet ep799":      os.path.join(EVAL_ROOT, "unet"),
        "Amix v1.4 ep799": os.path.join(EVAL_ROOT, "amix_v1_4"),
    }

    all_records = {}
    for name, root in models.items():
        print(f"Scoring {name}...")
        all_records[name] = score_model(name, root, subjects)

    # Pull koalAI from pre-existing native results
    print("Loading koalAI native results...")
    koalai_recs = []
    REGION_IDS = {"abdomen": 960, "brain": 966, "head_neck": 964, "pelvis": 968, "thorax": 962}
    KOALAI_TRAINER = "nnUNetTrainer_nnsyn_loss_map__nnUNetResEncUNetLPlans__3d_fullres"
    NNSYN_RESULTS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace/results"
    for region, ds_id in REGION_IDS.items():
        ds = f"Dataset{ds_id}_synthrad2025_task1_mri2ct_{region}"
        csv_path = os.path.join(NNSYN_RESULTS, ds, KOALAI_TRAINER, "fold_0",
                                "validation_revert_norm_results", "results_individual.csv")
        # fallback: check eval_results dir
        if not os.path.exists(csv_path):
            csv_path = os.path.join("evaluation_results/koalai_native/fold0", region, "results_individual.csv")
        if not os.path.exists(csv_path):
            print(f"  [SKIP] koalAI {region}: no results_individual.csv")
            continue
        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            koalai_recs.append({
                "subj_id": row.get("patient_id", row.get("id", "")),
                "synthrad_mae":  float(row["mae"]),
                "synthrad_psnr": float(row["psnr"]),
            })
    all_records["KoalAI ep999"] = koalai_recs
    print(f"  KoalAI: {len(koalai_recs)} subjects loaded")

    table = build_table(all_records, subjects)
    print("\n" + table)

    out_path = os.path.join(EVAL_ROOT, "comparison_synthrad_metric.txt")
    with open(out_path, "w") as f:
        f.write(table + "\n")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
