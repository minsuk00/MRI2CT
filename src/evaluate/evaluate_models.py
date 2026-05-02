"""
Standalone evaluation script for MRI2CT models.

Usage:
    python src/evaluate/evaluate_models.py \
        --split_file splits/thorax_center_wise_split.txt \
        --checkpoints amix:/path/to/amix.pt unet:/path/to/unet.pt \
                      mcddpm:/path/to/mcddpm.pt maisi:/path/to/maisi.pt \
        [--output results.json]

Model type can be omitted for auto-detection (e.g. just /path/to/ckpt.pt).
"""

import argparse
import gc
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
import torchio as tio
from tqdm import tqdm

# Path setup
_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_REPO_DIR, "MC-DDPM"))

from common.config import DEFAULT_CONFIG
from common.data import DataPreprocessing, build_tio_subjects, get_split_subjects
from common.loss import get_class_dice
from common.utils import clean_state_dict, compute_metrics, compute_metrics_body, unpad

_GPFS_ROOT = DEFAULT_CONFIG["root_dir"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_pad_offset(batch):
    po = batch.get("pad_offset", 0)
    if torch.is_tensor(po):
        return int(po[0])
    if isinstance(po, (list, tuple)):
        return int(po[0])
    return int(po)


def _resolve_root_dir(cfg_root: str, override: str = None) -> str:
    """Return the best root_dir: override > config path (if accessible) > GPFS default."""
    if override:
        return override
    if cfg_root and os.path.isdir(cfg_root):
        return cfg_root
    print(f"  [INFO] root_dir '{cfg_root}' not accessible, falling back to GPFS default.")
    return _GPFS_ROOT


def _check_masks(root_dir, val_subjects):
    """Returns set of subject IDs that have a real mask.nii.gz file. Warns about missing ones."""
    has_mask = set()
    for sid in val_subjects:
        subj_dir = os.path.join(root_dir, sid)
        if os.path.exists(os.path.join(subj_dir, "mask.nii.gz")) or os.path.exists(os.path.join(subj_dir, "mask.nii")):
            has_mask.add(sid)
        else:
            print(f"  [WARNING] mask.nii.gz not found for {sid} — will use full-volume metrics for this subject.")
    return has_mask


def _build_val_loader(root_dir, val_subjects, preprocess, load_mask=False, load_seg=False):
    val_objs = build_tio_subjects(root_dir, val_subjects, load_seg=load_seg, use_weighted_sampler=load_mask)
    val_ds = tio.SubjectsDataset(val_objs, transform=preprocess)
    return tio.SubjectsLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)


def _compute_teacher_dice(pred_unpad, seg_unpad, teacher_model, patch_size, val_sw_batch_size, val_sw_overlap, device, cfg, body_mask_tensor=None):
    """Run teacher BabyUNet on pred CT and compute dice_score_all and dice_score_bone."""
    from monai.inferers import sliding_window_inference

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred_probs = sliding_window_inference(
            inputs=pred_unpad.float(),
            roi_size=(patch_size, patch_size, patch_size),
            sw_batch_size=val_sw_batch_size,
            predictor=teacher_model,
            overlap=val_sw_overlap,
            device=device,
        )

    bone_idx = cfg.get("dice_bone_idx", 5)
    class_dices, bone_dice = get_class_dice(pred_probs, seg_unpad, mask=body_mask_tensor, bone_idx=bone_idx)

    result = {}
    excl_bg = cfg.get("dice_exclude_background", True)
    result["dice_score_all"] = (class_dices[1:].mean() if excl_bg else class_dices.mean()).item()
    if bone_dice is not None:
        result["dice_score_bone"] = bone_dice.item()

    return result


def detect_type(ckpt: dict) -> str:
    if "controlnet_state_dict" in ckpt:
        return "maisi"
    cfg = ckpt.get("config", {})
    # MAISI: has num_inference_steps or autoencoder_path
    if "num_inference_steps" in cfg or "autoencoder_path" in cfg:
        return "maisi"
    if "diffusion_steps" in cfg:
        return "mcddpm"
    if "ngf" in cfg or cfg.get("model_type") == "unet_baseline":
        return "unet"
    if "anatomix_weights" in cfg:
        return "amix"
    raise ValueError("Cannot auto-detect model type from checkpoint keys. Use 'type:path' format.")


# ---------------------------------------------------------------------------
# AMIX Validator
# Matches src/amix/trainer.py validate() lines 462-587
# ---------------------------------------------------------------------------


@torch.inference_mode()
def validate_amix(ckpt_path: str, val_subjects: list, device: torch.device, root_dir_override: str = None, body_mask: bool = False) -> dict:
    from anatomix.model.network import Unet
    from monai.inferers import sliding_window_inference

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    root_dir = _resolve_root_dir(cfg.get("root_dir"), root_dir_override)
    patch_size = cfg.get("patch_size", 128)
    res_mult = cfg.get("res_mult", 32)
    anatomix_weights = cfg.get("anatomix_weights", "v1_3")
    val_sw_batch_size = cfg.get("val_sw_batch_size", 4)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)
    feat_instance_norm = cfg.get("feat_instance_norm", False)
    pass_mri = cfg.get("pass_mri_to_translator", False)

    # Build Anatomix feature extractor
    if anatomix_weights == "v1":
        feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
        fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/anatomix.pth")
    else:  # v1_2, v1_3
        feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
        if anatomix_weights == "v1_2":
            fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/best_val_net_G_v1_2.pth")
        else:
            fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/best_val_net_G_real_v1_3.pth")

    # Load feat extractor weights: prefer checkpoint, fall back to file
    if "feat_extractor_state_dict" in ckpt:
        feat_extractor.load_state_dict(clean_state_dict(ckpt["feat_extractor_state_dict"]))
        print(f"  [AMIX] Loaded feat extractor from checkpoint.")
    elif os.path.exists(fe_ckpt_path):
        raw = torch.load(fe_ckpt_path, map_location=device, weights_only=False)
        feat_extractor.load_state_dict(clean_state_dict(raw))
        print(f"  [AMIX] Loaded feat extractor from {fe_ckpt_path}")
    else:
        print(f"  [WARNING] Anatomix weights not found at {fe_ckpt_path}")

    feat_extractor.eval()
    for p in feat_extractor.parameters():
        p.requires_grad = False

    # Teacher model for dice scores (matches training validate_dice=True)
    teacher_model = None
    validate_dice = cfg.get("validate_dice", False)
    if validate_dice:
        teacher_weights = cfg.get("teacher_weights_path")
        if teacher_weights and os.path.exists(teacher_weights):
            from anatomix.segmentation.segmentation_utils import load_model_v1_2

            n_classes = cfg.get("n_classes", 12)
            teacher_model = load_model_v1_2(pretrained_ckpt=teacher_weights, n_classes=n_classes - 1, device=device, compile_model=False)
            teacher_model.to(device=device, dtype=torch.bfloat16).eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print(f"  [AMIX] Loaded teacher model for dice computation.")
        else:
            print(f"  [AMIX] Warning: validate_dice=True but teacher_weights_path not found ({teacher_weights}). Skipping dice.")

    # Build translator
    translator_input_nc = 17 if pass_mri else 16
    translator = Unet(
        dimension=3,
        input_nc=translator_input_nc,
        output_nc=1,
        num_downs=4,
        ngf=16,
        final_act="sigmoid",
    ).to(device)
    translator.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    translator.eval()

    subjects_with_mask = _check_masks(root_dir, val_subjects) if body_mask else set()
    preprocess = DataPreprocessing(patch_size=patch_size, res_mult=res_mult, use_weighted_sampler=body_mask)
    val_loader = _build_val_loader(root_dir, val_subjects, preprocess, load_mask=body_mask, load_seg=teacher_model is not None)

    val_metrics = defaultdict(list)
    for batch in tqdm(val_loader, desc="[AMIX] Validating", leave=False):
        mri = batch["mri"][tio.DATA].to(device)
        ct = batch["ct"][tio.DATA].to(device)
        orig_shape = batch["original_shape"][0].tolist()
        pad_offset = _get_pad_offset(batch)
        subj_id = batch["subj_id"][0] if isinstance(batch["subj_id"], (list, tuple)) else batch["subj_id"]

        def combined_forward(x):
            f = feat_extractor(x)
            if feat_instance_norm:
                f = torch.nn.functional.instance_norm(f)
            if pass_mri:
                f = torch.cat([f, x], dim=1)
            return translator(f)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = sliding_window_inference(
                inputs=mri,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=val_sw_batch_size,
                predictor=combined_forward,
                overlap=val_sw_overlap,
                device=device,
            )

        pred_unpad = unpad(pred, orig_shape, pad_offset)
        ct_unpad = unpad(ct, orig_shape, pad_offset)
        mask_unpad = None
        if body_mask and subj_id in subjects_with_mask and "prob_map" in batch:
            mask_unpad = unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset)
            met = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad)
        else:
            met = compute_metrics(pred_unpad, ct_unpad)

        if teacher_model is not None and "seg" in batch:
            seg_unpad = unpad(batch["seg"][tio.DATA].to(device), orig_shape, pad_offset)
            met.update(_compute_teacher_dice(pred_unpad, seg_unpad, teacher_model, patch_size, val_sw_batch_size, val_sw_overlap, device, cfg, body_mask_tensor=mask_unpad))

        for k, v in met.items():
            val_metrics[k].append(v)
        del mri, ct, pred, pred_unpad, ct_unpad
        gc.collect()

    return {k: float(np.mean(v)) for k, v in val_metrics.items()}


# ---------------------------------------------------------------------------
# UNet Baseline Validator
# Matches src/unet_baseline/train.py validate() lines 405-507
# ---------------------------------------------------------------------------


@torch.inference_mode()
def validate_unet(ckpt_path: str, val_subjects: list, device: torch.device, root_dir_override: str = None, body_mask: bool = False) -> dict:
    from anatomix.model.network import Unet
    from monai.inferers import sliding_window_inference

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    root_dir = _resolve_root_dir(cfg.get("root_dir"), root_dir_override)
    patch_size = cfg.get("patch_size", 128)
    res_mult = cfg.get("res_mult", 16)
    input_nc = cfg.get("input_nc", 1)
    output_nc = cfg.get("output_nc", 1)
    num_downs = cfg.get("num_downs", 4)
    ngf = cfg.get("ngf", 16)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 8)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)

    model = Unet(
        dimension=3,
        input_nc=input_nc,
        output_nc=output_nc,
        num_downs=num_downs,
        ngf=ngf,
        final_act="sigmoid",
    ).to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.eval()

    # Teacher model for dice scores
    teacher_model = None
    if cfg.get("validate_dice", False):
        teacher_weights = cfg.get("teacher_weights_path")
        if teacher_weights and os.path.exists(teacher_weights):
            from anatomix.segmentation.segmentation_utils import load_model_v1_2

            n_classes = cfg.get("n_classes", 12)
            teacher_model = load_model_v1_2(pretrained_ckpt=teacher_weights, n_classes=n_classes - 1, device=device, compile_model=False)
            teacher_model.to(device=device, dtype=torch.bfloat16).eval()
            for p in teacher_model.parameters():
                p.requires_grad = False
            print(f"  [UNet] Loaded teacher model for dice computation.")
        else:
            print(f"  [UNet] Warning: validate_dice=True but teacher_weights_path not found ({teacher_weights}). Skipping dice.")

    subjects_with_mask = _check_masks(root_dir, val_subjects) if body_mask else set()
    preprocess = DataPreprocessing(patch_size=patch_size, res_mult=res_mult, use_weighted_sampler=body_mask)
    val_loader = _build_val_loader(root_dir, val_subjects, preprocess, load_mask=body_mask, load_seg=teacher_model is not None)

    val_metrics = defaultdict(list)
    for batch in tqdm(val_loader, desc="[UNet] Validating", leave=False):
        mri = batch["mri"][tio.DATA].to(device)
        ct = batch["ct"][tio.DATA].to(device)
        orig_shape = batch["original_shape"][0].tolist()
        pad_offset = _get_pad_offset(batch)
        subj_id = batch["subj_id"][0] if isinstance(batch["subj_id"], (list, tuple)) else batch["subj_id"]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = sliding_window_inference(
                inputs=mri,
                roi_size=(patch_size, patch_size, patch_size),
                sw_batch_size=val_sw_batch_size,
                predictor=model,
                overlap=val_sw_overlap,
                device=device,
            )

        pred_unpad = unpad(pred, orig_shape, pad_offset)
        ct_unpad = unpad(ct, orig_shape, pad_offset)
        mask_unpad = None
        if body_mask and subj_id in subjects_with_mask and "prob_map" in batch:
            mask_unpad = unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset)
            met = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad)
        else:
            met = compute_metrics(pred_unpad, ct_unpad)

        if teacher_model is not None and "seg" in batch:
            seg_unpad = unpad(batch["seg"][tio.DATA].to(device), orig_shape, pad_offset)
            met.update(_compute_teacher_dice(pred_unpad, seg_unpad, teacher_model, patch_size, val_sw_batch_size, val_sw_overlap, device, cfg, body_mask_tensor=mask_unpad))

        for k, v in met.items():
            val_metrics[k].append(v)
        del mri, ct, pred, pred_unpad, ct_unpad
        gc.collect()

    return {k: float(np.mean(v)) for k, v in val_metrics.items()}


# ---------------------------------------------------------------------------
# MC-DDPM Validator
# Matches src/mc_ddpm_baseline/trainer.py validate() lines 210-255
# ---------------------------------------------------------------------------


@torch.inference_mode()
def validate_mcddpm(ckpt_path: str, val_subjects: list, device: torch.device, root_dir_override: str = None, body_mask: bool = False) -> dict:
    from diffusion.Create_diffusion import create_gaussian_diffusion
    from monai.inferers import SlidingWindowInferer
    from network.Diffusion_model_transformer import SwinVITModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    root_dir = _resolve_root_dir(cfg.get("root_dir"), root_dir_override)
    patch_size = cfg.get("patch_size", (64, 64, 4))
    # patch_size may be stored as list when loaded from JSON
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    learn_sigma = cfg.get("learn_sigma", True)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 4)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.5)

    # Reconstruct diffusion schedule from saved config
    timestep_respacing = cfg.get("timestep_respacing", [50])
    if isinstance(timestep_respacing, list):
        timestep_respacing = timestep_respacing  # keep as list (expected by create_gaussian_diffusion)

    diffusion = create_gaussian_diffusion(
        steps=cfg.get("diffusion_steps", 1000),
        learn_sigma=learn_sigma,
        sigma_small=cfg.get("sigma_small", False),
        noise_schedule=cfg.get("noise_schedule", "linear"),
        use_kl=cfg.get("use_kl", False),
        predict_xstart=cfg.get("predict_xstart", True),
        rescale_timesteps=cfg.get("rescale_timesteps", True),
        rescale_learned_sigmas=cfg.get("rescale_learned_sigmas", True),
        timestep_respacing=timestep_respacing,
    )

    out_channels = 2 if learn_sigma else 1
    # attention_resolutions and channel_mult may be stored as lists
    attention_resolutions = cfg.get("attention_resolutions", (32, 16, 8))
    channel_mult = cfg.get("channel_mult", (1, 2, 3, 4))
    sample_kernel = cfg.get("sample_kernel", (([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),))

    model = SwinVITModel(
        image_size=patch_size,
        in_channels=2,
        model_channels=cfg.get("num_channels", 64),
        out_channels=out_channels,
        dims=3,
        sample_kernel=sample_kernel,
        num_res_blocks=cfg.get("num_res_blocks", [2, 2, 2, 2]),
        attention_resolutions=attention_resolutions,
        dropout=cfg.get("dropout", 0.0),
        channel_mult=channel_mult,
        num_classes=None,
        use_checkpoint=cfg.get("use_checkpoint", False),
        use_fp16=False,
        num_heads=cfg.get("num_heads", [4, 4, 8, 16]),
        window_size=cfg.get("window_size", [[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]]),
        num_head_channels=64,
        num_heads_upsample=-1,
        use_scale_shift_norm=cfg.get("use_scale_shift_norm", True),
        resblock_updown=cfg.get("resblock_updown", False),
        use_new_attention_order=False,
    ).to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.eval()

    patch_size_scalar = max(patch_size) if isinstance(patch_size, (list, tuple)) else patch_size
    preprocess = DataPreprocessing(
        patch_size=patch_size_scalar,
        res_mult=1,
        enforce_ras=False,
        use_weighted_sampler=body_mask,
    )
    val_loader = _build_val_loader(root_dir, val_subjects, preprocess, load_mask=body_mask)

    inferer = SlidingWindowInferer(
        roi_size=patch_size,
        sw_batch_size=val_sw_batch_size,
        overlap=val_sw_overlap,
        mode="constant",
    )

    def diffusion_sampling(condition):
        shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4])
        return diffusion.p_sample_loop(model, shape, condition=condition, clip_denoised=True)

    val_metrics = defaultdict(list)
    for batch in tqdm(val_loader, desc="[MCDDPM] Validating", leave=False):
        mri = batch["mri"][tio.DATA].to(device)
        ct = batch["ct"][tio.DATA].to(device)
        orig_shape = batch["original_shape"][0].tolist()
        pad_offset = _get_pad_offset(batch)

        mri_scaled = mri * 2.0 - 1.0
        with torch.amp.autocast("cuda"):
            pred = inferer(mri_scaled, diffusion_sampling)
        pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

        pred_unpad = unpad(pred, orig_shape, pad_offset)
        ct_unpad = unpad(ct, orig_shape, pad_offset)
        if body_mask and "prob_map" in batch:
            mask_unpad = unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset)
            met = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad)
        else:
            met = compute_metrics(pred_unpad, ct_unpad)
        for k, v in met.items():
            val_metrics[k].append(v)
        del mri, ct, pred, pred_unpad, ct_unpad
        gc.collect()

    return {k: float(np.mean(v)) for k, v in val_metrics.items()}


# ---------------------------------------------------------------------------
# MAISI Validator
# Matches src/maisi_baseline/trainer.py validate()
# Instantiates MAISITrainer directly to reuse complex _sample/_decode logic.
# ---------------------------------------------------------------------------


def validate_maisi(ckpt_path: str, val_subjects: list, split_file: str, device: torch.device, root_dir_override: str = None, body_mask: bool = False) -> dict:
    from common.utils import cleanup_gpu
    from maisi_baseline.trainer import MAISITrainer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))

    # Override for eval-only mode — no wandb, no staging, no sanity check, no resume
    config["wandb"] = False
    config["stage_data"] = False
    config["root_dir"] = _resolve_root_dir(config.get("root_dir"), root_dir_override)
    config["use_weighted_sampler"] = body_mask  # load body mask for masked metrics
    config["sanity_check"] = False
    config["resume_wandb_id"] = None
    config["augment"] = False
    config["full_val"] = True
    config["split_file"] = os.path.abspath(split_file)

    trainer = MAISITrainer(config)

    # Manually load trained controlnet weights (bypasses wandb-based _load_resume)
    # Support both checkpoint formats: model_state_dict (new) and controlnet_state_dict (old)
    controlnet_state = ckpt.get("model_state_dict") or ckpt.get("controlnet_state_dict")
    trainer.controlnet.load_state_dict(clean_state_dict(controlnet_state))
    trainer.controlnet.eval()

    if "scale_factor" in ckpt:
        sf = ckpt["scale_factor"]
        if isinstance(sf, torch.Tensor):
            trainer.scale_factor = sf.to(device)
        else:
            trainer.scale_factor = torch.tensor(float(sf), device=device)
        print(f"  [MAISI] Restored scale_factor = {trainer.scale_factor.item():.6f}")

    if body_mask:
        print("  [MAISI] Warning: body_mask not supported for MAISI (uses internal val loop). Computing full-volume metrics.")
    avg_met = trainer.validate(0)

    del trainer
    cleanup_gpu()

    return {k: float(v) for k, v in avg_met.items()}


# ---------------------------------------------------------------------------
# Table printer
# ---------------------------------------------------------------------------


def print_table(results: list):
    cols = ["mae_hu", "ssim", "psnr", "dice_score_all", "dice_score_bone"]
    col_names = ["MAE HU", "SSIM", "PSNR", "Dice (All)", "Dice (Bone)"]

    try:
        from tabulate import tabulate

        rows = []
        for r in results:
            m = r["metrics"]
            rows.append([r["label"], r["ckpt_name"]] + [m.get(c, float("nan")) for c in cols])
        headers = ["Model", "Checkpoint"] + col_names
        print(tabulate(rows, headers=headers, floatfmt=".4f", tablefmt="simple"))
    except ImportError:
        # Manual fixed-width fallback
        w_model, w_ckpt, w_val = 8, 30, 11
        header = f"{'Model':<{w_model}}  {'Checkpoint':<{w_ckpt}}" + "".join(f"  {n:>{w_val}}" for n in col_names)
        sep = "-" * len(header)
        print(sep)
        print(header)
        print(sep)
        for r in results:
            m = r["metrics"]
            vals = [m.get(c, float("nan")) for c in cols]
            line = f"{r['label']:<{w_model}}  {r['ckpt_name']:<{w_ckpt}}" + "".join(f"  {v:>{w_val}.4f}" for v in vals)
            print(line)
        print(sep)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import warnings

    import matplotlib

    matplotlib.use("Agg")
    warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
    warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")

    parser = argparse.ArgumentParser(description="Evaluate MRI2CT models on a validation split")
    parser.add_argument("--split_file", required=True, help="Path to split file")
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        metavar="TYPE:PATH",
        help="Checkpoint(s) as 'type:path' where type ∈ {amix, unet, mcddpm, maisi}. Type may be omitted for auto-detection.",
    )
    parser.add_argument("--output", type=str, default=None, help="Optional path to save results as JSON")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root_dir", type=str, default=None, help="Override data root_dir from checkpoint config (useful when checkpoint has a stale /tmp path)")
    parser.add_argument("--body_mask", action="store_true", help="Compute metrics on body region only (requires mask.nii.gz in each subject dir)")
    args = parser.parse_args()

    device = torch.device(args.device)
    val_subjects = get_split_subjects(args.split_file, "val")
    print(f"[INFO] Val subjects ({len(val_subjects)}): {val_subjects[:5]}{'...' if len(val_subjects) > 5 else ''}")

    results = []

    for ckpt_spec in args.checkpoints:
        if ":" in ckpt_spec and not os.path.exists(ckpt_spec):
            # Split only if the raw string isn't itself an existing path
            model_type, ckpt_path = ckpt_spec.split(":", 1)
        else:
            ckpt_path = ckpt_spec
            ckpt_tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_type = detect_type(ckpt_tmp)
            del ckpt_tmp
            print(f"[INFO] Auto-detected model type: {model_type}")

        # Include parent dir name to distinguish same-named checkpoints from different runs
        parent = os.path.basename(os.path.dirname(ckpt_path))
        ckpt_name = f"{parent}/{os.path.basename(ckpt_path)}"
        print(f"\n{'=' * 60}")
        print(f"Evaluating [{model_type.upper()}]  {ckpt_name}")
        print(f"{'=' * 60}")

        try:
            if model_type == "amix":
                metrics = validate_amix(ckpt_path, val_subjects, device, root_dir_override=args.root_dir, body_mask=args.body_mask)
            elif model_type == "unet":
                metrics = validate_unet(ckpt_path, val_subjects, device, root_dir_override=args.root_dir, body_mask=args.body_mask)
            elif model_type == "mcddpm":
                metrics = validate_mcddpm(ckpt_path, val_subjects, device, root_dir_override=args.root_dir, body_mask=args.body_mask)
            elif model_type == "maisi":
                metrics = validate_maisi(ckpt_path, val_subjects, args.split_file, device, root_dir_override=args.root_dir, body_mask=args.body_mask)
            else:
                print(f"[ERROR] Unknown model type '{model_type}'. Choose from: amix, unet, mcddpm, maisi")
                continue

            results.append({"label": model_type, "ckpt_name": ckpt_name, "ckpt_path": ckpt_path, "metrics": metrics})

            # Print per-model summary
            key_metrics = {k: metrics.get(k) for k in ["mae_hu", "ssim", "psnr", "dice_score_bone_threshold"] if k in metrics}
            print(f"  {key_metrics}")

        except Exception as e:
            import traceback

            print(f"[ERROR] Failed to evaluate {ckpt_name}: {e}")
            traceback.print_exc()

        gc.collect()
        torch.cuda.empty_cache()

    if results:
        print(f"\n{'=' * 60}")
        print("RESULTS TABLE")
        print(f"{'=' * 60}")
        print_table(results)

        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
            print(f"\n[INFO] Results saved to {args.output}")
    else:
        print("[WARNING] No results to display.")
