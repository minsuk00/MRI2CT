"""Post-training evaluation for MC-IDDPM on SynthRAD.

Runs the published 50-step respaced sampler over each val subject via MONAI's
SlidingWindowInferer with (128, 128, 4) patches and 50% overlap. Optionally
Monte-Carlo averages across N runs (paper: N=5). Writes per-subject NIfTI
predictions and a TXT report with per-subject + aggregate metrics.

Metrics use the amix preproc ([-1024, 1024] HU, span 2048), matching what the
UNet/Amix/MAISI/cWDM baselines report — column names are plain `psnr`, `ssim`,
`mae_hu`, etc. so the file slots directly into cross-model comparison.

Dice / Bone Dice (optional) use the Baby U-Net teacher, which was trained on
the same amix preproc, so it consumes `pred_amix01`.
"""
import argparse
import os
import sys
import time

import nibabel as nib
import numpy as np
import torch
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import sliding_window_inference
from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    LoadImaged,
    Orientationd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
)
from tqdm import tqdm

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

# Cloned MC-IDDPM modules (imported via baselines.mc_ddpm.__init__ sys.path shim).
from baselines.mc_ddpm.diffusion.Create_diffusion import create_gaussian_diffusion
from baselines.mc_ddpm.network.Diffusion_model_transformer import SwinVITModel

from baselines.mc_ddpm.data import CT_CLIP, PATCH

from common.data import (
    RecordAffineD,
    RecordOriginalShapeD,
    StripMetaD,
    build_data_dicts,
    default_monai_cache_dir,
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
from common.utils import clean_state_dict, compute_metrics, compute_metrics_body, unpad

# Paper preproc span — used only to invert the model output back to HU.
CT_SPAN_PAPER = CT_CLIP[1] - CT_CLIP[0]  # 1650 - (-1024) = 2674
CT_SPAN_AMIX = 2048  # [-1024, 1024]; the yardstick all metrics report on.



def build_cached_xform_with_seg(load_body_mask=True, load_seg=False):
    """Local copy of `mc_ddpm.data.build_cached_xform`, extended with seg loading.

    Lives here (rather than `data.py`) so the validate script can fetch seg
    without altering training-side data plumbing. Seg is loaded as a uint8
    label map at the same padded resolution as ct/mri/body_mask; no intensity
    scaling. PersistentDataset hashes transform spec, so this gets its own
    cache entry separate from the training cache.
    """
    img_keys = ["mri", "ct"]
    mask_keys = ["body_mask"] if load_body_mask else []
    seg_keys = ["seg"] if load_seg else []
    spatial_keys = img_keys + mask_keys + seg_keys

    xforms = [
        LoadImaged(keys=spatial_keys, image_only=True),
        EnsureChannelFirstd(keys=spatial_keys),
        Orientationd(keys=spatial_keys, axcodes="RAS"),
        ScaleIntensityRanged(keys=["ct"], a_min=CT_CLIP[0], a_max=CT_CLIP[1],
                             b_min=-1.0, b_max=1.0, clip=True),
        ScaleIntensityd(keys=["mri"], minv=-1.0, maxv=1.0),
        RecordOriginalShapeD(ref_key="ct"),
        RecordAffineD(ref_key="ct", key_prefix="ct"),
        SpatialPadd(keys=img_keys, spatial_size=PATCH, method="end",
                    mode="constant", constant_values=-1.0),
    ]
    if mask_keys:
        xforms.append(
            SpatialPadd(keys=mask_keys, spatial_size=PATCH, method="end",
                        mode="constant", constant_values=0)
        )
        xforms.append(CastToTyped(keys=mask_keys, dtype=torch.uint8))
    if seg_keys:
        xforms.append(
            SpatialPadd(keys=seg_keys, spatial_size=PATCH, method="end",
                        mode="constant", constant_values=0)
        )
        xforms.append(CastToTyped(keys=seg_keys, dtype=torch.uint8))
    xforms.append(StripMetaD(keys=spatial_keys))
    return Compose(xforms)


def build_model(device, use_checkpoint=False):
    """Construct SwinVITModel exactly as the trainer does."""
    return SwinVITModel(
        image_size=PATCH,
        in_channels=2,
        model_channels=64,
        out_channels=2,
        dims=3,
        sample_kernel=(([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),),
        num_res_blocks=[2, 2, 2, 2],
        attention_resolutions=(32, 16, 8),
        dropout=0.0,
        channel_mult=(1, 2, 3, 4),
        num_classes=None,
        use_checkpoint=use_checkpoint,
        use_fp16=False,
        num_heads=[4, 4, 8, 16],
        window_size=[[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]],
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=False,
    ).to(device)


def paper_to_hu(x_minus1_plus1):
    """[-1, 1] under paper clip -> HU. Inverse of `ScaleIntensityRanged(a,b → [-1,1])`."""
    return (x_minus1_plus1 + 1.0) * 0.5 * CT_SPAN_PAPER + CT_CLIP[0]


def hu_to_amix01(x_hu):
    """HU -> [0, 1] under amix preproc (clip to [-1024, 1024], scale)."""
    x = torch.clamp(x_hu, -1024.0, 1024.0)
    return (x - (-1024.0)) / CT_SPAN_AMIX


def diffusion_sample(diffusion, model, condition):
    # AMP autocast matches the upstream notebook (`with torch.cuda.amp.autocast()`
    # around inferer(...)). bfloat16 (not fp16) — A40 is Ampere with native bf16
    # tensor cores, and bf16's wider exponent range avoids the overflow/NaN
    # risks fp16 hits in diffusion sampling. Matches the trainer's choice.
    with torch.autocast(device_type="cuda" if condition.device.type == "cuda" else "cpu",
                        dtype=torch.bfloat16):
        return diffusion.p_sample_loop(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
            condition=condition,
            clip_denoised=True,
            progress=False,
            device=condition.device,
        )


def run_inference(model, diffusion, mri, original_shape, device, sw_batch_size, overlap, mc_runs):
    """Returns pred at original_shape, in HU. `mc_runs` averages that many sample loops."""
    accum = None
    for _ in range(mc_runs):
        pred = sliding_window_inference(
            inputs=mri,
            roi_size=PATCH,
            sw_batch_size=sw_batch_size,
            predictor=lambda c: diffusion_sample(diffusion, model, c),
            overlap=overlap,
            mode="constant",
            device=device,
        )
        accum = pred if accum is None else (accum + pred)
    pred_avg = accum / float(mc_runs)
    pred_hu = paper_to_hu(pred_avg)
    return unpad(pred_hu, original_shape)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pt)")
    parser.add_argument("--split_file", default="splits/center_wise_split.txt")
    parser.add_argument("--split_name", default="val")
    parser.add_argument("--root_dir", default=None, help="Override data root (defaults to DEFAULT_CONFIG)")
    parser.add_argument("--ddim_steps", type=int, default=50, help="Sampling step count (paper: 50)")
    parser.add_argument("--mc_runs", type=int, default=1, help="Monte-Carlo averaging runs (paper: 5)")
    parser.add_argument("--sw_batch_size", type=int, default=4)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--out_dir", default=None,
                        help="Output dir (defaults to <ckpt_dir>/validate_<ts>/)")
    parser.add_argument("--max_subjects", type=int, default=None,
                        help="Limit to first N subjects (smoke testing).")
    parser.add_argument("--shard_idx", type=int, default=0,
                        help="SLURM-array shard index (0-based). Use with --num_shards>1.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help=">1 enables round-robin sharding: each shard gets subjects[shard_idx::num_shards].")
    parser.add_argument("--use_checkpoint", action="store_true",
                        help="Use gradient-checkpointed SwinVITModel (saves VRAM during inference)")
    # Teacher / Dice ----------------------------------------------------------
    parser.add_argument("--teacher_weights_path", default="auto",
                        help="'none' to disable Dice; any other value runs the canonical dual teachers.")
    parser.add_argument("--teacher_val_patch_size", type=int, default=128)
    parser.add_argument("--teacher_sw_batch_size", type=int, default=2)
    parser.add_argument("--teacher_sw_overlap", type=float, default=0.25)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dice_on = bool(args.teacher_weights_path and args.teacher_weights_path.lower() != "none")

    # --- Resolve paths ---
    if args.root_dir is None:
        from common.config import DEFAULT_CONFIG
        args.root_dir = DEFAULT_CONFIG["root_dir"]

    if args.out_dir is None:
        args.out_dir = default_validate_dir(args.checkpoint, prefix="validate")
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[VAL] output dir: {args.out_dir}")

    # --- Build model + diffusion ---
    print(f"[VAL] 🏗️ Building model + diffusion (steps=1000, respaced=[{args.ddim_steps}])")
    model = build_model(device, use_checkpoint=args.use_checkpoint)
    diffusion = create_gaussian_diffusion(
        steps=1000, learn_sigma=True, sigma_small=False, noise_schedule="linear",
        use_kl=False, predict_xstart=False, rescale_timesteps=True,
        rescale_learned_sigmas=True, timestep_respacing=[args.ddim_steps],
    )

    # --- Load checkpoint ---
    print(f"[VAL] 📥 Loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state = clean_state_dict(ckpt.get("model_state_dict", ckpt))
    model.load_state_dict(state, strict=True)
    model.eval()

    ckpt_info = extract_checkpoint_info(args.checkpoint, ckpt_dict=ckpt)
    ckpt_info_str = format_checkpoint_info(ckpt_info)
    print(f"[VAL] checkpoint: {ckpt_info_str}")

    # --- Teachers (legacy 12-class + CADS 35-class) ---
    teachers = load_teachers(default_teacher_specs(), device) if dice_on else []

    # --- Build val dataset ---
    subjects = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects is not None:
        subjects = subjects[: args.max_subjects]
    if args.num_shards and args.num_shards > 1:
        subjects = subjects[args.shard_idx :: args.num_shards]
        print(f"[VAL] shard {args.shard_idx}/{args.num_shards} → {len(subjects)} subjects")
    # Segs are loaded per-teacher (dual label spaces), not through the cached pipeline.
    dicts = build_data_dicts(args.root_dir, subjects, load_seg=False, load_body_mask=True)
    print(f"[VAL] 📂 {len(dicts)} subjects (split={args.split_name}, max_subjects={args.max_subjects})")

    xform = build_cached_xform_with_seg(load_body_mask=True, load_seg=False)
    # hash_transform=pickle_hashing → key by transform spec, not just data path,
    # so this validate cache doesn't collide with the training cache (different
    # CT clip range / scale / load_seg).
    ds = PersistentDataset(
        data=dicts, transform=xform,
        cache_dir=default_monai_cache_dir(), hash_transform=pickle_hashing,
    )
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    # --- Inference loop ---
    per_subject = []

    for batch in tqdm(loader, desc="validate", dynamic_ncols=True):
        subj_id = batch["subj_id"][0]
        mri = batch["mri"].to(device).float()        # (1, 1, D, H, W) in [-1, 1]
        ct  = batch["ct"].to(device).float()         # (1, 1, D, H, W) in [-1, 1]
        body_mask = batch["body_mask"].to(device).float() if "body_mask" in batch else None
        original_shape = batch["original_shape"][0]
        ct_affine = batch["ct_affine"][0].cpu().numpy()

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        pred_hu_unpad = run_inference(
            model, diffusion, mri, original_shape, device, args.sw_batch_size,
            args.overlap, args.mc_runs,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()  # CUDA is async; sync so time_sec reflects real GPU compute
        elapsed = time.time() - t0

        # GT in HU
        ct_hu_unpad = unpad(paper_to_hu(ct), original_shape)
        mask_unpad = unpad(body_mask, original_shape) if body_mask is not None else None

        # Metrics on the amix [-1024, 1024] yardstick (span 2048).
        pred_amix01 = hu_to_amix01(pred_hu_unpad)
        ct_amix01   = hu_to_amix01(ct_hu_unpad)
        met = compute_metrics(pred_amix01, ct_amix01, hu_range=CT_SPAN_AMIX)

        record = {
            "mae_hu":    met["mae_hu"],
            "psnr":      met["psnr"],
            "ssim":      met["ssim"],
            "grad_diff": met["grad_diff"],
        }

        # Body-masked
        if mask_unpad is not None:
            mb = compute_metrics_body(pred_amix01, ct_amix01, mask_unpad, hu_range=CT_SPAN_AMIX)
            record["body_mae_hu"] = mb["mae_hu"]
            record["body_psnr"]   = mb["psnr"]
            record["body_ssim"]   = mb["ssim"]

        # Dice — teacher consumes amix-clipped [0,1] (same preproc it was trained on).
        # Both teachers: 12-class unsuffixed, 35-class `_cads35`.
        if teachers:
            seg_by_file = load_subject_segs(args.root_dir, subj_id, teachers, device)
            sw = dict(val_patch_size=args.teacher_val_patch_size, sw_batch_size=1,
                      overlap=args.teacher_sw_overlap)
            full, bod = dual_teacher_dice(teachers, pred_amix01.float(), seg_by_file, device,
                                          body_mask=mask_unpad, sw_kwargs=sw)
            record.update(full)
            record.update(bod)

        record["time_sec"] = elapsed
        per_subject.append({"subj_id": subj_id, "metrics": record})

        # Save NIfTI (HU)
        subj_dir = os.path.join(args.out_dir, subj_id)
        os.makedirs(subj_dir, exist_ok=True)
        pred_np = pred_hu_unpad.float().cpu().numpy().squeeze()
        ct_np   = ct_hu_unpad.float().cpu().numpy().squeeze()
        nib.save(nib.Nifti1Image(pred_np, ct_affine), os.path.join(subj_dir, "sample.nii.gz"))
        nib.save(nib.Nifti1Image(ct_np,   ct_affine), os.path.join(subj_dir, "target.nii.gz"))

        tqdm.write(
            f"  {subj_id} | {elapsed:6.1f}s | amix MAE={met['mae_hu']:6.1f}HU "
            f"PSNR={met['psnr']:5.2f} SSIM={met['ssim']:.3f}"
            + (f" | dice12={record.get('dice_score_all', float('nan')):.3f} dice35={record.get('dice_score_all_cads35', float('nan')):.3f}" if teachers else "")
        )

    # --- TXT report ---
    metric_keys = []
    for r in per_subject:
        for k in r["metrics"]:
            if k not in metric_keys:
                metric_keys.append(k)
    header = [
        "Validation report — MC-IDDPM",
        f"checkpoint: {args.checkpoint}",
        f"checkpoint info: {ckpt_info_str}",
        f"split_file: {args.split_file}   split_name: {args.split_name}",
        f"ddim_steps: {args.ddim_steps}   mc_runs: {args.mc_runs}   overlap: {args.overlap}",
        f"ct_span_paper: {CT_SPAN_PAPER}   ct_span_amix: {CT_SPAN_AMIX}",
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
