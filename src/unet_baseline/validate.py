"""Standalone post-training evaluator for the plain U-Net MR→CT baseline.

Mirrors `UnetTrainer.validate()`: SW inference with the U-Net directly on MRI,
teacher Dice, NIfTI + TXT output. Architecture flags are read from the
checkpoint's saved config.
"""
import argparse
import gc
import os
import sys
import time

import nibabel as nib
import torch
from anatomix.model.network import Unet
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_split_subjects,
)
from common.eval_utils import (
    default_teacher_specs,
    default_validate_dir,
    dual_teacher_dice,
    extract_checkpoint_info,
    format_checkpoint_info,
    load_subject_segs,
    load_teachers,
    write_metrics_txt,
)
from common.utils import (
    clean_state_dict,
    compute_metrics,
    compute_metrics_body,
    unpad,
)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--split_name", default="val")
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--max_subjects", type=int, default=None)
    parser.add_argument("--shard_idx", type=int, default=0)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--teacher_weights_path", default="auto",
                        help="'none' to disable Dice; any other value runs the canonical dual teachers.")
    # batch=1: sliding-window results are batch-invariant, and the wider CADS-35
    # (v2) teacher overflows 32-bit conv indexing at a 256^3 window with batch>1.
    parser.add_argument("--teacher_sw_batch_size", type=int, default=1)
    parser.add_argument("--teacher_sw_overlap", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_on = bool(args.teacher_weights_path and args.teacher_weights_path.lower() != "none")

    print(f"[VAL-UNET] 📥 Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {}) or {}
    ckpt_info = extract_checkpoint_info(args.checkpoint, ckpt_dict=ckpt)
    ckpt_info_str = format_checkpoint_info(ckpt_info)
    print(f"[VAL-UNET] checkpoint: {ckpt_info_str}")

    if args.split_file is None:
        args.split_file = cfg.get("split_file", "splits/center_wise_split.txt")
    if args.root_dir is None:
        from common.config import DEFAULT_CONFIG
        args.root_dir = cfg.get("root_dir", DEFAULT_CONFIG["root_dir"])

    if args.out_dir is None:
        args.out_dir = default_validate_dir(args.checkpoint, prefix="validate")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[VAL-UNET] output dir: {args.out_dir}")

    # Build U-Net from saved config
    model = Unet(
        dimension=3,
        input_nc=cfg.get("input_nc", 1),
        output_nc=cfg.get("output_nc", 1),
        num_downs=cfg.get("num_downs", 4),
        ngf=cfg.get("ngf", 16),
        norm=cfg.get("norm", "batch"),
        final_act="sigmoid",
    ).to(device)
    state = clean_state_dict(ckpt["model_state_dict"])
    model.load_state_dict(state, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    teachers = load_teachers(default_teacher_specs(), device) if dice_on else []

    val_subj = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects is not None:
        val_subj = val_subj[: args.max_subjects]
    if args.num_shards and args.num_shards > 1:
        val_subj = val_subj[args.shard_idx :: args.num_shards]
        print(f"[VAL-UNET] shard {args.shard_idx}/{args.num_shards} → {len(val_subj)} subjects")
    # Segs are loaded per-teacher (dual label spaces), not through the cached pipeline.
    val_dicts = build_data_dicts(args.root_dir, val_subj, load_seg=False, load_body_mask=True)
    print(f"[VAL-UNET] 📂 {len(val_dicts)} subjects (split={args.split_name})")

    val_xform = get_cached_transforms(
        patch_size=cfg.get("patch_size", 128),
        res_mult=cfg.get("res_mult", 32),
        enforce_ras=True,
        mri_norm=cfg.get("mri_norm", "minmax"),
        ct_range=tuple(cfg.get("ct_range", (-1024, 1024))),
        load_seg=False,
        load_body_mask=True,
    )
    cache_dir = default_monai_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    ds = PersistentDataset(data=val_dicts, transform=val_xform, cache_dir=cache_dir, hash_transform=pickle_hashing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    val_ps = cfg.get("val_patch_size", 256)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 1)

    per_subject = []
    for batch in tqdm(loader, desc="validate", dynamic_ncols=True):
        subj_id = batch["subj_id"][0]
        mri = batch["mri"].to(device).float()
        ct  = batch["ct"].to(device).float()
        body_mask = batch["body_mask"].to(device).float() if "body_mask" in batch else None
        orig_shape = batch["original_shape"][0].tolist()
        affine = batch["ct_affine"][0].cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        # fp32 inference (no bf16 autocast): bf16's 7-bit mantissa quantizes a
        # single-pass regressor's output to ~4-8 HU steps, which posterizes soft
        # tissue when viewed in a narrow window (e.g. brain [-100,100] HU).
        pred = sliding_window_inference(
            inputs=mri, roi_size=(val_ps,) * 3,
            sw_batch_size=val_sw_batch_size,
            predictor=model, overlap=val_sw_overlap, device=device,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()  # CUDA is async; sync so time_sec reflects real GPU compute
        elapsed = time.time() - t0

        pred_unpad = unpad(pred.float(), orig_shape)
        ct_unpad   = unpad(ct, orig_shape)
        mask_unpad = unpad(body_mask, orig_shape) if body_mask is not None else None

        met = compute_metrics(pred_unpad, ct_unpad, hu_range=2048)
        record = {"mae_hu": met["mae_hu"], "psnr": met["psnr"],
                  "ssim": met["ssim"], "grad_diff": met["grad_diff"]}
        if mask_unpad is not None:
            bm = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad, hu_range=2048)
            record["body_mae_hu"] = bm["mae_hu"]
            record["body_psnr"]   = bm["psnr"]
            record["body_ssim"]   = bm["ssim"]

        if teachers:
            seg_by_file = load_subject_segs(args.root_dir, subj_id, teachers, device)
            sw = dict(val_patch_size=val_ps, sw_batch_size=args.teacher_sw_batch_size,
                      overlap=args.teacher_sw_overlap)
            full, bod = dual_teacher_dice(teachers, pred_unpad, seg_by_file, device,
                                          body_mask=mask_unpad, sw_kwargs=sw)
            record.update(full)
            record.update(bod)

        record["time_sec"] = elapsed
        per_subject.append({"subj_id": subj_id, "metrics": record})

        subj_dir = os.path.join(args.out_dir, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        pred_hu = (pred_unpad * 2048.0 - 1024.0).float().cpu().numpy().squeeze()
        ct_hu   = (ct_unpad   * 2048.0 - 1024.0).float().cpu().numpy().squeeze()
        nib.save(nib.Nifti1Image(pred_hu, affine), os.path.join(subj_dir, "sample.nii.gz"))
        nib.save(nib.Nifti1Image(ct_hu,   affine), os.path.join(subj_dir, "target.nii.gz"))

        tqdm.write(
            f"  {subj_id} | {elapsed:6.1f}s | MAE={record['mae_hu']:6.1f}HU "
            f"PSNR={record['psnr']:5.2f} SSIM={record['ssim']:.3f}"
            + (f" | dice12={record.get('dice_score_all', float('nan')):.3f}"
               f" dice35={record.get('dice_score_all_cads35', float('nan')):.3f}"
               if teachers else "")
        )
        del pred, pred_unpad, ct_unpad, mri, ct
        gc.collect()
        torch.cuda.empty_cache()

    metric_keys = []
    for r in per_subject:
        for k in r["metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)
    header = [
        "Validation report — UNet baseline",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint info: {ckpt_info_str}",
        f"split_file: {args.split_file}   split_name: {args.split_name}",
        f"input_nc: {cfg.get('input_nc', 1)}   num_downs: {cfg.get('num_downs', 4)}   ngf: {cfg.get('ngf', 16)}   norm: {cfg.get('norm', 'batch')}",
        f"val_patch_size: {val_ps}   val_sw_overlap: {val_sw_overlap}   val_sw_batch_size: {val_sw_batch_size}",
        f"teachers: {', '.join(t['tag'] for t in teachers) if teachers else 'disabled'}",
        f"subjects: {len(per_subject)}",
    ]
    if args.num_shards and args.num_shards > 1:
        header.insert(1, f"shard: {args.shard_idx}/{args.num_shards}")
    write_metrics_txt(
        os.path.join(args.out_dir, "validate_metrics.txt"),
        header_lines=header, per_subject=per_subject, metric_keys=metric_keys,
    )


if __name__ == "__main__":
    main()
