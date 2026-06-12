"""Standalone post-training evaluator for MAISI ControlNet on SynthRAD.

Loads a saved ControlNet checkpoint together with the frozen VAE + denoising
UNet (NV-Generate-CT), iterates the val split, samples a synthetic CT per
subject, decodes to image space, computes MAE_HU / PSNR / SSIM (+ body) and
optional Dice / Bone Dice via the Baby U-Net teacher, then saves per-subject
NIfTI predictions plus a TXT report next to the checkpoint.

This mirrors `MAISITrainer.validate()` exactly — same `_sample` /
`_decode` plumbing — but without instantiating the full trainer (no WandB,
no optimizer, no training data loader).

Teacher note: the teacher was trained on amix preproc (ct_range=[-1024, 1024]),
while MAISI predictions are in [-1000, 1000] HU. We pass the prediction in
MAISI's [0, 1] mapping directly to the teacher — matching exactly what
`trainer.validate()` does today, so the standalone numbers compare 1:1 with
the inline ones.
"""
import argparse
import gc
import json
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.bundle import ConfigParser
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import SlidingWindowInferer
from monai.networks.utils import copy_model_state
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_split_subjects,
)
from common.eval_utils import (
    compute_dice,
    default_validate_dir,
    extract_checkpoint_info,
    format_checkpoint_info,
    load_teacher_model,
    run_teacher_sw,
    write_metrics_txt,
)
from common.utils import clean_state_dict, compute_metrics, compute_metrics_body, unpad


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "autoencoder_v1.pt")
DIFFUSION_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "diff_unet_3d_rflow-ct.pt")
NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "NV-Generate-CTMR", "configs", "config_network_rflow.json")
DEFAULT_TEACHER = "/home/minsukc/MRI2CT/ckpt/seg_baby_unet/seg_baby_unet_epoch_749.pth"


# ---------------------------------------------------------------------------
# Model setup (mirror of MAISITrainer._setup_models, no Trainer instance)
# ---------------------------------------------------------------------------
def build_maisi_models(args, device):
    with open(args.network_config_path, "r") as f:
        model_def = json.load(f)
    model_def["controlnet_def"]["conditioning_embedding_in_channels"] = 1
    model_def["autoencoder_def"]["num_splits"] = 8

    parser = ConfigParser()
    parser.update(model_def)

    # VAE (frozen)
    autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True).to(device)
    ae_ckpt = torch.load(args.autoencoder_path, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["unet_state_dict"] if "unet_state_dict" in ae_ckpt else ae_ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    # Denoising UNet (frozen)
    unet = parser.get_parsed_content("diffusion_unet_def", instantiate=True).to(device)
    unet_ckpt = torch.load(args.diffusion_path, map_location=device, weights_only=False)
    unet.load_state_dict(unet_ckpt["unet_state_dict"], strict=True)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    scale_factor = unet_ckpt.get("scale_factor", 1.0)
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.to(device)
    else:
        scale_factor = torch.tensor(float(scale_factor), device=device)
    print(f"[VAL-MAISI] 📈 Scale Factor: {scale_factor.item():.6f}")

    # ControlNet (trainable in training; loaded from checkpoint here)
    controlnet = parser.get_parsed_content("controlnet_def", instantiate=True).to(device)
    copy_model_state(controlnet, unet.state_dict())  # match trainer's init pattern

    print(f"[VAL-MAISI] 📥 Loading ControlNet from {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cn_state = ckpt.get("model_state_dict", ckpt)
    controlnet.load_state_dict(clean_state_dict(cn_state), strict=True)
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad = False

    noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)

    return autoencoder, unet, controlnet, noise_scheduler, scale_factor, ckpt


# ---------------------------------------------------------------------------
# Sampling + decoding (mirrors trainer._sample / _decode / _decode_sliding_window)
# ---------------------------------------------------------------------------
@torch.no_grad()
def sample_latent(controlnet, unet, noise_scheduler, mr, spacing, num_steps, device):
    latent_shape = (1, 4, mr.shape[2] // 4, mr.shape[3] // 4, mr.shape[4] // 4)
    latents = torch.randn(latent_shape, device=device)

    try:
        num_voxels = int(torch.prod(torch.tensor(latent_shape[2:])).item())
        noise_scheduler.set_timesteps(num_inference_steps=num_steps, input_img_size_numel=num_voxels)
    except (TypeError, AttributeError):
        noise_scheduler.set_timesteps(num_inference_steps=num_steps)

    all_timesteps = noise_scheduler.timesteps.to(device)
    all_next_timesteps = torch.cat(
        (all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=device))
    )

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for t, next_t in zip(all_timesteps, all_next_timesteps):
            t_tensor = torch.tensor([t], device=device).float()
            class_labels = torch.ones(latents.shape[0], dtype=torch.long, device=device)
            down_res, mid_res = controlnet(
                x=latents, timesteps=t_tensor, controlnet_cond=mr, class_labels=class_labels,
            )
            model_output = unet(
                x=latents, timesteps=t_tensor, spacing_tensor=spacing,
                class_labels=class_labels,
                down_block_additional_residuals=down_res,
                mid_block_additional_residual=mid_res,
            )
            latents, _ = noise_scheduler.step(model_output, t, latents, next_t)

    return latents.float()


@torch.no_grad()
def decode_latent(autoencoder, latent, scale_factor, val_sw_overlap, val_sw_batch_size, device):
    z = latent / scale_factor
    roi_size = [96, 88, 64]
    needs_sw = any(s > limit for s, limit in zip(z.shape[2:], roi_size))
    if needs_sw:
        inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=1, overlap=val_sw_overlap,
            mode="gaussian", sw_device=device, device=torch.device("cpu"),
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon = inferer(z, autoencoder.decode_stage_2_outputs)
        recon = recon.to(device)
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon = autoencoder.decode_stage_2_outputs(z)
    return torch.clamp(recon, 0.0, 1.0).float()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to MAISI ControlNet checkpoint (.pt)")
    parser.add_argument("--split_file", default="splits/center_wise_split.txt")
    parser.add_argument("--split_name", default="val")
    parser.add_argument("--root_dir", default=None, help="Override data root (defaults to DEFAULT_CONFIG)")
    parser.add_argument("--num_inference_steps", type=int, default=10)
    parser.add_argument("--val_sw_batch_size", type=int, default=1)
    parser.add_argument("--val_sw_overlap", type=float, default=0.4)
    parser.add_argument("--patch_size", type=int, default=128, help="Cached pipeline pad-target (matches training)")
    parser.add_argument("--val_patch_size", type=int, default=256, help="Teacher sliding-window patch size")
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (defaults to <ckpt_dir>/validate_<ts>/)")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Limit to first N subjects (smoke testing).")
    # Teacher / Dice
    parser.add_argument("--teacher_weights_path", default=DEFAULT_TEACHER,
                        help="Baby U-Net teacher weights. Pass 'none' to skip Dice.")
    parser.add_argument("--n_classes", type=int, default=12)
    parser.add_argument("--dice_bone_idx", type=int, default=5)
    parser.add_argument("--teacher_sw_batch_size", type=int, default=2)
    parser.add_argument("--teacher_sw_overlap", type=float, default=0.25)
    # MAISI model paths (override if non-default)
    parser.add_argument("--autoencoder_path", default=AUTOENCODER_PATH)
    parser.add_argument("--diffusion_path", default=DIFFUSION_PATH)
    parser.add_argument("--network_config_path", default=NETWORK_CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_on = bool(args.teacher_weights_path and args.teacher_weights_path.lower() != "none")

    if args.root_dir is None:
        from common.config import DEFAULT_CONFIG
        args.root_dir = DEFAULT_CONFIG["root_dir"]

    if args.out_dir is None:
        args.out_dir = default_validate_dir(args.checkpoint, prefix="validate")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[VAL-MAISI] output dir: {args.out_dir}")

    # --- Models ---
    autoencoder, unet, controlnet, noise_scheduler, scale_factor, ckpt = build_maisi_models(args, device)
    ckpt_info = extract_checkpoint_info(args.checkpoint, ckpt_dict=ckpt)
    ckpt_info_str = format_checkpoint_info(ckpt_info)
    print(f"[VAL-MAISI] checkpoint: {ckpt_info_str}")
    # ckpt is now metadata-only (no need to hold the model state dict in memory)
    del ckpt

    # --- Teacher ---
    teacher = None
    if dice_on:
        teacher = load_teacher_model(
            args.teacher_weights_path, device=device, n_classes_minus_bg=args.n_classes - 1
        )

    # --- Val data ---
    val_subj = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects is not None:
        val_subj = val_subj[: args.max_subjects]
    val_dicts = build_data_dicts(args.root_dir, val_subj, load_seg=dice_on, load_body_mask=True)
    print(f"[VAL-MAISI] 📂 {len(val_dicts)} subjects (split={args.split_name})")

    # MAISI preproc: ct_range=(-1000, 1000), mri_norm="percentile", res_mult=32.
    val_xform = get_cached_transforms(
        patch_size=args.patch_size,
        res_mult=32,
        enforce_ras=True,
        mri_norm="percentile",
        ct_range=(-1000, 1000),
        load_seg=dice_on,
        load_ct_image=True,
        load_ct_latent_from=None,
    )
    cache_dir = default_monai_cache_dir()
    os.makedirs(cache_dir, exist_ok=True)
    ds = PersistentDataset(data=val_dicts, transform=val_xform, cache_dir=cache_dir, hash_transform=pickle_hashing)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Inference loop ---
    per_subject = []

    for batch in tqdm(loader, desc="validate", dynamic_ncols=True):
        subj_id = batch["subj_id"][0]
        mr = batch["mri"].to(device)
        ct = batch["ct"].to(device)
        spacing = batch["ct_spacing"].float().to(device) * 100.0
        orig_shape = batch["original_shape"][0].tolist()
        body_mask = batch["body_mask"].to(device).float() if "body_mask" in batch else None
        seg = batch["seg"].to(device).long() if (dice_on and "seg" in batch) else None
        affine = batch["ct_affine"][0].cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        pred_latent = sample_latent(controlnet, unet, noise_scheduler, mr, spacing,
                                    args.num_inference_steps, device)
        pred_ct_norm = decode_latent(autoencoder, pred_latent, scale_factor,
                                     args.val_sw_overlap, args.val_sw_batch_size, device)
        if device.type == "cuda":
            torch.cuda.synchronize()  # CUDA is async; sync so time_sec reflects real GPU compute
        elapsed = time.time() - t0

        # HU + matched [0,1]
        gt_hu_raw = (ct * 2000.0) - 1000.0
        gt_hu = torch.clamp(gt_hu_raw, -1000.0, 1000.0)
        pred_hu = (pred_ct_norm * 2000.0) - 1000.0
        gt_matched = (gt_hu + 1000.0) / 2000.0

        # Unpad
        pred_unpad = unpad(pred_ct_norm.float(), orig_shape)
        gt_unpad = unpad(gt_matched.float(), orig_shape)
        pred_hu_unpad = unpad(pred_hu.float(), orig_shape)
        gt_hu_unpad = unpad(gt_hu.float(), orig_shape)
        mask_unpad = unpad(body_mask, orig_shape) if body_mask is not None else None
        seg_unpad = unpad(seg, orig_shape) if seg is not None else None

        # Image metrics (hu_range=2000 because MAISI clips to [-1000, 1000])
        met = compute_metrics(pred_unpad, gt_unpad, hu_range=2000)
        record = {
            "mae_hu": met["mae_hu"], "psnr": met["psnr"],
            "ssim": met["ssim"], "grad_diff": met["grad_diff"],
        }

        # Air-excluded MAE (NVIDIA MAISI convention)
        hu_mask = gt_hu_unpad > -900
        if hu_mask.any():
            record["mae_hu_air_excluded"] = torch.mean(
                torch.abs(pred_hu_unpad[hu_mask] - gt_hu_unpad[hu_mask])
            ).item()

        if mask_unpad is not None:
            bm = compute_metrics_body(pred_unpad, gt_unpad, mask_unpad.float(), hu_range=2000)
            record["body_mae_hu"] = bm["mae_hu"]
            record["body_psnr"]   = bm["psnr"]
            record["body_ssim"]   = bm["ssim"]

        # Dice (teacher fed MAISI [0,1] directly — matches trainer.validate())
        if teacher is not None and seg_unpad is not None:
            pred_logits = run_teacher_sw(
                teacher, pred_unpad.float(), device=device,
                val_patch_size=args.val_patch_size,
                sw_batch_size=args.teacher_sw_batch_size,
                overlap=args.teacher_sw_overlap,
            )
            dice = compute_dice(pred_logits, seg_unpad, bone_idx=args.dice_bone_idx)
            record.update(dice)
            if mask_unpad is not None:
                dice_b = compute_dice(pred_logits, seg_unpad, mask=mask_unpad, bone_idx=args.dice_bone_idx)
                record["body_dice_score_all"] = dice_b["dice_score_all"]
                if "dice_score_bone" in dice_b:
                    record["body_dice_score_bone"] = dice_b["dice_score_bone"]
            del pred_logits

        record["time_sec"] = elapsed
        per_subject.append({"subj_id": subj_id, "metrics": record})

        # NIfTI (HU)
        subj_dir = os.path.join(args.out_dir, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        nib.save(nib.Nifti1Image(pred_hu_unpad.float().cpu().numpy().squeeze(), affine),
                 os.path.join(subj_dir, "sample.nii.gz"))
        nib.save(nib.Nifti1Image(gt_hu_unpad.float().cpu().numpy().squeeze(), affine),
                 os.path.join(subj_dir, "target.nii.gz"))

        tqdm.write(
            f"  {subj_id} | {elapsed:6.1f}s | MAE={record['mae_hu']:6.1f}HU "
            f"PSNR={record['psnr']:5.2f} SSIM={record['ssim']:.3f}"
            + (f" | dice_all={record.get('dice_score_all', float('nan')):.3f}"
               f" bone={record.get('dice_score_bone', float('nan')):.3f}"
               if teacher else "")
        )

        # Memory cleanup per subject (large volumes)
        del pred_latent, pred_ct_norm, gt_hu_raw, gt_hu, pred_hu, gt_matched
        del pred_unpad, gt_unpad, pred_hu_unpad, gt_hu_unpad, mr, ct
        gc.collect()
        torch.cuda.empty_cache()

    # --- TXT report ---
    metric_keys = []
    for r in per_subject:
        for k in r["metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)
    header = [
        "Validation report — MAISI ControlNet",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint info: {ckpt_info_str}",
        f"split_file: {args.split_file}   split_name: {args.split_name}",
        f"num_inference_steps: {args.num_inference_steps}   val_sw_overlap: {args.val_sw_overlap}",
        f"teacher: {args.teacher_weights_path if teacher is not None else 'disabled'}",
        f"subjects: {len(per_subject)}",
    ]
    write_metrics_txt(
        os.path.join(args.out_dir, "validate_metrics.txt"),
        header_lines=header, per_subject=per_subject, metric_keys=metric_keys,
    )


if __name__ == "__main__":
    main()
