"""Shared per-family dispatch for the 5 trained models (UNet, Amix, MAISI, cWDM, MC-IDDPM).

Used by `OOD_test.py` (CHAOS abdomen) and `evaluate_mrnet_knee.py` (MRNet knee).
Both drivers do the same thing — load each model, run inference on each MR volume —
they just disagree on how to discover volumes and how to lay out the output figures.

Public API:
    MODELS                          : dict of name → {family, ckpt_path}
    preprocess_for_family(family, vol_path, cfg) → (tensor, orig_shape, affine, voxel_sizes)
    load_for_family(family, ckpt_path, device)   → (family_tag, bundle, cfg, epoch)
    infer_for_family(family, bundle, cfg, mri_tensor, voxel_sizes, device) → pred
    to_hu(family, pred)                          → tensor in HU
"""

import json
import os
import sys

import numpy as np
import torch
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.transforms import (
    Compose,
    DivisiblePad,
    EnsureChannelFirst,
    LoadImage,
    Orientation,
    ScaleIntensity,
    ScaleIntensityRangePercentiles,
    SpatialPad,
)

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)
_SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
if _SRC_ROOT not in sys.path:
    sys.path.append(_SRC_ROOT)

from anatomix.model.network import Unet  # noqa: E402

from common.utils import clean_state_dict, unpad  # noqa: E402,F401 (unpad re-exported for drivers)


# ─── configuration ───────────────────────────────────────────────────────────
GPFS = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT"
RUNS = os.path.join(GPFS, "wandb_logs/runs")

MODELS = {
    "unet_300k": {
        "family": "unet",
        "ckpt_path": os.path.join(RUNS, "20260507_0952_9xmodnhn", "unet_baseline_epoch00600.pt"),
    },
    "amix_v1_4_300k": {
        "family": "amix",
        "ckpt_path": os.path.join(RUNS, "20260509_1413_6hjye9gp", "anatomix_translator_epoch00600.pt"),
    },
    "maisi": {
        "family": "maisi",
        "ckpt_path": os.path.join(RUNS, "20260511_0244_5hprtpwl", "checkpoint_last.pt"),
    },
    "cwdm": {
        "family": "cwdm",
        "ckpt_path": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/wandb/run-20260518_175723-smg8thkr/files/checkpoints/synthrad_last.pt",
    },
    "mcddpm": {
        "family": "mcddpm",
        "ckpt_path": os.path.join(RUNS, "20260515_0836_a3g28rez", "checkpoint_last.pt"),
    },
}

# Diffusion sampling step counts.
NUM_STEPS_MAISI = 30
DDIM_STEPS_CWDM = 100
DDIM_STEPS_MCDDPM = 50
MCDDPM_MC_RUNS = 1

# MC-IDDPM constants (mirrored from baselines/mc_ddpm/data.py).
MCDDPM_PATCH = (128, 128, 4)
MCDDPM_CT_CLIP = (-1024, 1650)
MCDDPM_CT_SPAN_PAPER = MCDDPM_CT_CLIP[1] - MCDDPM_CT_CLIP[0]  # 2674

# MAISI assets.
MAISI_AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "autoencoder_v1.pt")
MAISI_DIFFUSION_PATH = os.path.join(PROJECT_ROOT, "ckpt", "nv-generate-ct", "models", "diff_unet_3d_rflow-ct.pt")
MAISI_NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "baselines", "NV-Generate-CTMR", "configs", "config_network_rflow.json")

# Anatomix feature-extractor weights for the amix family.
AMIX_CKPTS = {
    "v1":   "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth",
    "v1_2": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth",
    "v1_3": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_real_v1_3.pth",
    "v1_4": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth",
}


# ─── preprocessing ────────────────────────────────────────────────────────────
def preprocess_volume(vol_path, mri_norm_mode, min_size, res_mult, pad_value):
    """Load → channel-first → RAS → normalize → end-pad to ≥min_size and multiple of res_mult.

    mri_norm_mode: 'minmax_01' | 'percentile_01' | 'minmax_-1_1'
    min_size: int (the smaller of patch_size and res_mult is fine) — used as
        SpatialPad floor.
    res_mult: int — DivisiblePad divisor. If None, only SpatialPad is applied.
    pad_value: float constant fill for both pads.

    Returns (tensor (1, D, H, W), orig_shape, affine (4,4), voxel_sizes (3,)).
    """
    if mri_norm_mode == "minmax_01":
        norm = ScaleIntensity(minv=0.0, maxv=1.0)
    elif mri_norm_mode == "minmax_-1_1":
        norm = ScaleIntensity(minv=-1.0, maxv=1.0)
    elif mri_norm_mode == "percentile_01":
        norm = ScaleIntensityRangePercentiles(lower=0.0, upper=99.5, b_min=0.0, b_max=1.0, clip=True)
    else:
        raise ValueError(f"Unknown mri_norm_mode: {mri_norm_mode}")

    pre_pad = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        norm,
    ])
    img = pre_pad(vol_path)
    orig_shape = list(img.shape[1:])
    affine = np.asarray(img.affine.cpu()) if hasattr(img.affine, "cpu") else np.asarray(img.affine)
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)  # mm

    img = SpatialPad(spatial_size=(min_size,) * 3, method="end",
                     mode="constant", constant_values=pad_value)(img)
    if res_mult is not None:
        img = DivisiblePad(k=res_mult, method="end",
                           mode="constant", constant_values=pad_value)(img)
    return img.float(), orig_shape, affine, voxel_sizes


def preprocess_to_patch(vol_path, mri_norm_mode, patch, pad_value):
    """Variant for MC-IDDPM: pad to exact PATCH (D, H, W) tuple (end-pad)."""
    if mri_norm_mode == "minmax_-1_1":
        norm = ScaleIntensity(minv=-1.0, maxv=1.0)
    else:
        raise ValueError(mri_norm_mode)
    pre_pad = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        Orientation(axcodes="RAS"),
        norm,
    ])
    img = pre_pad(vol_path)
    orig_shape = list(img.shape[1:])
    affine = np.asarray(img.affine.cpu()) if hasattr(img.affine, "cpu") else np.asarray(img.affine)
    voxel_sizes = np.linalg.norm(affine[:3, :3], axis=0)
    img = SpatialPad(spatial_size=patch, method="end",
                     mode="constant", constant_values=pad_value)(img)
    return img.float(), orig_shape, affine, voxel_sizes


# ─── UNet baseline ────────────────────────────────────────────────────────────
def load_unet(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    epoch = state.get("epoch", "?")
    sd = clean_state_dict(state["model_state_dict"])
    model = Unet(
        dimension=3,
        input_nc=cfg.get("input_nc", 1),
        output_nc=cfg.get("output_nc", 1),
        num_downs=cfg.get("num_downs", 4),
        ngf=cfg.get("ngf", 16),
        norm=cfg.get("norm", "batch"),
        final_act="sigmoid",
    ).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model, cfg, epoch


@torch.inference_mode()
def infer_unet(model, mri_tensor, cfg, device):
    val_ps = cfg.get("val_patch_size", cfg.get("patch_size", 128))
    sw_batch = cfg.get("val_sw_batch_size", 1)
    sw_overlap = cfg.get("val_sw_overlap", 0.25)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        return sliding_window_inference(
            inputs=mri_tensor, roi_size=(val_ps,) * 3,
            sw_batch_size=sw_batch, predictor=model,
            overlap=sw_overlap, device=device,
        )


# ─── Amix translator ──────────────────────────────────────────────────────────
def build_feat_extractor(anat_weights, feat_norm, device):
    if anat_weights == "v1":
        model = Unet(3, 1, 16, 4, 16).to(device); k = "v1"
    elif anat_weights in ("v2", "v1_2"):
        model = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear", pooling="Avg", use_bias=True).to(device); k = "v1_2"
    elif anat_weights == "v1_3":
        model = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear", pooling="Avg", use_bias=True).to(device); k = "v1_3"
    elif anat_weights == "v1_4":
        model = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(device); k = "v1_4"
    else:
        raise ValueError(f"Unknown anatomix_weights: {anat_weights}")
    sd = clean_state_dict(torch.load(AMIX_CKPTS[k], map_location=device))
    model.load_state_dict(sd, strict=True)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return model


def load_amix(ckpt_path, device):
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = state.get("config", {})
    epoch = state.get("epoch", "?")
    sd = clean_state_dict(state["model_state_dict"])

    pass_mri = cfg.get("pass_mri_to_translator", False)
    input_nc = 17 if pass_mri else 16
    translator = Unet(
        dimension=3, input_nc=input_nc, output_nc=1,
        num_downs=4, ngf=16, final_act="sigmoid",
    ).to(device)
    translator.load_state_dict(sd, strict=True)
    translator.eval()

    feat_extractor = build_feat_extractor(
        cfg.get("anatomix_weights", "v1_3"),
        cfg.get("feat_norm", "instance"),
        device,
    )
    return translator, feat_extractor, cfg, epoch


def make_amix_forward(feat_extractor, translator, cfg):
    feat_in = cfg.get("feat_instance_norm", False)
    feat_scale = cfg.get("feat_scale_down", 1)
    pass_mri = cfg.get("pass_mri_to_translator", False)

    def forward(x):
        with torch.no_grad():
            f = feat_extractor(x)
        if feat_in:
            f = torch.nn.functional.instance_norm(f)
        if feat_scale != 1:
            f = f / feat_scale
        if pass_mri:
            f = torch.cat([f, x], dim=1)
        return translator(f)

    return forward


@torch.inference_mode()
def infer_amix(translator, feat_extractor, cfg, mri_tensor, device):
    val_ps = cfg.get("val_patch_size", cfg.get("patch_size", 128))
    sw_batch = cfg.get("val_sw_batch_size", 1)
    sw_overlap = cfg.get("val_sw_overlap", 0.25)
    predictor = make_amix_forward(feat_extractor, translator, cfg)
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        return sliding_window_inference(
            inputs=mri_tensor, roi_size=(val_ps,) * 3,
            sw_batch_size=sw_batch, predictor=predictor,
            overlap=sw_overlap, device=device,
        )


# ─── MAISI ────────────────────────────────────────────────────────────────────
def load_maisi(ckpt_path, device):
    from monai.bundle import ConfigParser
    from monai.networks.utils import copy_model_state

    with open(MAISI_NETWORK_CONFIG_PATH, "r") as f:
        model_def = json.load(f)
    model_def["controlnet_def"]["conditioning_embedding_in_channels"] = 1
    model_def["autoencoder_def"]["num_splits"] = 8
    parser = ConfigParser()
    parser.update(model_def)

    autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True).to(device)
    ae_ckpt = torch.load(MAISI_AUTOENCODER_PATH, map_location=device, weights_only=False)
    autoencoder.load_state_dict(ae_ckpt["unet_state_dict"] if "unet_state_dict" in ae_ckpt else ae_ckpt)
    autoencoder.eval()
    for p in autoencoder.parameters():
        p.requires_grad = False

    unet = parser.get_parsed_content("diffusion_unet_def", instantiate=True).to(device)
    unet_ckpt = torch.load(MAISI_DIFFUSION_PATH, map_location=device, weights_only=False)
    unet.load_state_dict(unet_ckpt["unet_state_dict"], strict=True)
    unet.eval()
    for p in unet.parameters():
        p.requires_grad = False

    scale_factor = unet_ckpt.get("scale_factor", 1.0)
    if isinstance(scale_factor, torch.Tensor):
        scale_factor = scale_factor.to(device)
    else:
        scale_factor = torch.tensor(float(scale_factor), device=device)

    controlnet = parser.get_parsed_content("controlnet_def", instantiate=True).to(device)
    copy_model_state(controlnet, unet.state_dict())
    cn_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cn_state = cn_ckpt.get("model_state_dict", cn_ckpt)
    controlnet.load_state_dict(clean_state_dict(cn_state), strict=True)
    controlnet.eval()
    for p in controlnet.parameters():
        p.requires_grad = False

    noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)
    epoch = cn_ckpt.get("epoch", "?") if isinstance(cn_ckpt, dict) else "?"
    return (autoencoder, unet, controlnet, noise_scheduler, scale_factor), epoch


@torch.no_grad()
def _maisi_sample_latent(controlnet, unet, scheduler, mr, spacing, num_steps, device):
    latent_shape = (1, 4, mr.shape[2] // 4, mr.shape[3] // 4, mr.shape[4] // 4)
    latents = torch.randn(latent_shape, device=device)
    try:
        num_voxels = int(torch.prod(torch.tensor(latent_shape[2:])).item())
        scheduler.set_timesteps(num_inference_steps=num_steps, input_img_size_numel=num_voxels)
    except (TypeError, AttributeError):
        scheduler.set_timesteps(num_inference_steps=num_steps)
    all_t = scheduler.timesteps.to(device)
    all_next = torch.cat((all_t[1:], torch.tensor([0], dtype=all_t.dtype, device=device)))
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        for t, nt in zip(all_t, all_next):
            t_tensor = torch.tensor([t], device=device).float()
            class_labels = torch.ones(latents.shape[0], dtype=torch.long, device=device)
            d, m = controlnet(x=latents, timesteps=t_tensor, controlnet_cond=mr, class_labels=class_labels)
            out = unet(
                x=latents, timesteps=t_tensor, spacing_tensor=spacing,
                class_labels=class_labels,
                down_block_additional_residuals=d,
                mid_block_additional_residual=m,
            )
            latents, _ = scheduler.step(out, t, latents, nt)
    return latents.float()


@torch.no_grad()
def _maisi_decode(autoencoder, latent, scale_factor, device):
    z = latent / scale_factor
    roi_size = [96, 88, 64]
    needs_sw = any(s > limit for s, limit in zip(z.shape[2:], roi_size))
    if needs_sw:
        inferer = SlidingWindowInferer(
            roi_size=roi_size, sw_batch_size=1, overlap=0.4,
            mode="gaussian", sw_device=device, device=torch.device("cpu"),
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon = inferer(z, autoencoder.decode_stage_2_outputs)
        recon = recon.to(device)
    else:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            recon = autoencoder.decode_stage_2_outputs(z)
    return torch.clamp(recon, 0.0, 1.0).float()


@torch.inference_mode()
def infer_maisi(bundle, mri_tensor, voxel_sizes, device):
    autoencoder, unet, controlnet, scheduler, scale_factor = bundle
    spacing = torch.tensor(voxel_sizes, device=device).float().unsqueeze(0) * 100.0  # (1, 3)
    latent = _maisi_sample_latent(controlnet, unet, scheduler, mri_tensor, spacing, NUM_STEPS_MAISI, device)
    return _maisi_decode(autoencoder, latent, scale_factor, device)


# ─── cWDM ─────────────────────────────────────────────────────────────────────
_CWDM_PATH_DONE = False


def _setup_cwdm_path():
    global _CWDM_PATH_DONE
    if _CWDM_PATH_DONE:
        return
    cwdm_root = os.path.join(PROJECT_ROOT, "baselines", "cwdm")
    if cwdm_root not in sys.path:
        sys.path.insert(0, cwdm_root)
    _CWDM_PATH_DONE = True


# Args mirror sbatch/validate_cwdm.sh:70-83 (the model+diffusion config the
# checkpoint was trained with).
CWDM_ARGS = dict(
    image_size=224,
    num_channels=64, num_res_blocks=2, num_heads=1,
    num_heads_upsample=-1, num_head_channels=-1,
    attention_resolutions="",
    channel_mult="1,2,2,4,4",
    dropout=0.0, class_cond=False, use_checkpoint=False,
    use_scale_shift_norm=False, resblock_updown=True, use_fp16=False,
    use_new_attention_order=False, dims=3, num_groups=32,
    in_channels=16, out_channels=8,
    bottleneck_attention=False, resample_2d=False, additive_skips=False,
    mode="i2i", use_freq=False, predict_xstart=True,
    learn_sigma=False, diffusion_steps=1000, noise_schedule="linear",
    timestep_respacing="",  # train-time diffusion has no respacing
    use_kl=False, rescale_timesteps=False, rescale_learned_sigmas=False,
    dataset="synthrad",
)


def load_cwdm(ckpt_path, device):
    _setup_cwdm_path()
    from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D
    from guided_diffusion.script_util import (
        create_gaussian_diffusion,
        create_model_and_diffusion,
        model_and_diffusion_defaults,
    )

    keys = model_and_diffusion_defaults().keys()
    arguments = {k: CWDM_ARGS[k] for k in keys}
    model, _train_diff = create_model_and_diffusion(**arguments)

    # Respaced diffusion for DDIM sampling.
    sample_diff = create_gaussian_diffusion(
        steps=arguments["diffusion_steps"],
        learn_sigma=arguments["learn_sigma"],
        noise_schedule=arguments["noise_schedule"],
        use_kl=arguments["use_kl"],
        predict_xstart=arguments["predict_xstart"],
        rescale_timesteps=arguments["rescale_timesteps"],
        rescale_learned_sigmas=arguments["rescale_learned_sigmas"],
        timestep_respacing=f"ddim{DDIM_STEPS_CWDM}",
        mode="i2i",
    )

    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(sd)
    # cWDM UNetModel overrides .to() and returns None (multi-device shim); call separately.
    model.to(device)
    model.eval()

    dwt = DWT_3D("haar")
    idwt = IDWT_3D("haar")
    return (model, sample_diff, dwt, idwt), "?"


@torch.inference_mode()
def infer_cwdm(bundle, mri_tensor, device):
    model, sample_diff, dwt, idwt = bundle
    LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(mri_tensor)
    cond_dwt = torch.cat([LLL / 3.0, LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
    B, _, D, H, W = mri_tensor.shape
    noise = torch.randn(B, 8, D // 2, H // 2, W // 2, device=device)
    x0_wav = sample_diff.p_sample_loop(
        model=model, shape=noise.shape, noise=noise, cond=cond_dwt,
        clip_denoised=True, model_kwargs={}, progress=False,
    )
    return idwt(
        x0_wav[:, 0:1] * 3.0,
        x0_wav[:, 1:2], x0_wav[:, 2:3], x0_wav[:, 3:4],
        x0_wav[:, 4:5], x0_wav[:, 5:6], x0_wav[:, 6:7], x0_wav[:, 7:8],
    ).clamp(0.0, 1.0)


# ─── MC-IDDPM ─────────────────────────────────────────────────────────────────
def load_mcddpm(ckpt_path, device):
    from baselines.mc_ddpm.diffusion.Create_diffusion import create_gaussian_diffusion as mc_create_diffusion
    from baselines.mc_ddpm.network.Diffusion_model_transformer import SwinVITModel

    model = SwinVITModel(
        image_size=MCDDPM_PATCH, in_channels=2, model_channels=64, out_channels=2,
        dims=3, sample_kernel=(([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),),
        num_res_blocks=[2, 2, 2, 2], attention_resolutions=(32, 16, 8), dropout=0.0,
        channel_mult=(1, 2, 3, 4), num_classes=None, use_checkpoint=False, use_fp16=False,
        num_heads=[4, 4, 8, 16], window_size=[[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]],
        num_head_channels=64, num_heads_upsample=-1, use_scale_shift_norm=True,
        resblock_updown=False, use_new_attention_order=False,
    ).to(device)
    diffusion = mc_create_diffusion(
        steps=1000, learn_sigma=True, sigma_small=False, noise_schedule="linear",
        use_kl=False, predict_xstart=False, rescale_timesteps=True,
        rescale_learned_sigmas=True, timestep_respacing=[DDIM_STEPS_MCDDPM],
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = clean_state_dict(ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt)
    model.load_state_dict(state, strict=True)
    model.eval()
    epoch = ckpt.get("epoch", "?") if isinstance(ckpt, dict) else "?"
    return (model, diffusion), epoch


def _mcddpm_predictor(diffusion, model, condition):
    with torch.autocast(device_type=("cuda" if condition.device.type == "cuda" else "cpu"),
                        dtype=torch.bfloat16):
        return diffusion.p_sample_loop(
            model,
            (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4]),
            condition=condition, clip_denoised=True, progress=False, device=condition.device,
        )


@torch.inference_mode()
def infer_mcddpm(bundle, mri_tensor, device):
    model, diffusion = bundle
    accum = None
    for _ in range(MCDDPM_MC_RUNS):
        pred = sliding_window_inference(
            inputs=mri_tensor, roi_size=MCDDPM_PATCH,
            sw_batch_size=4, predictor=lambda c: _mcddpm_predictor(diffusion, model, c),
            overlap=0.5, mode="constant", device=device,
        )
        accum = pred if accum is None else accum + pred
    pred_avg = accum / float(MCDDPM_MC_RUNS)
    return (pred_avg + 1.0) * 0.5 * MCDDPM_CT_SPAN_PAPER + MCDDPM_CT_CLIP[0]  # HU


# ─── HU conversion ────────────────────────────────────────────────────────────
def to_hu(family, pred):
    if family in ("unet", "amix", "cwdm"):
        return pred * 2048.0 - 1024.0
    if family == "maisi":
        return pred * 2000.0 - 1000.0
    if family == "mcddpm":
        return pred  # already HU
    raise ValueError(family)


# ─── dispatch helpers ─────────────────────────────────────────────────────────
def preprocess_for_family(family, vol_path, cfg):
    if family == "unet" or family == "amix":
        mri_norm = cfg.get("mri_norm", "minmax")
        norm_mode = "percentile_01" if mri_norm == "percentile" else "minmax_01"
        return preprocess_volume(
            vol_path,
            mri_norm_mode=norm_mode,
            min_size=cfg.get("patch_size", 128),
            res_mult=cfg.get("res_mult", 32),
            pad_value=0.0,
        )
    if family == "maisi":
        return preprocess_volume(
            vol_path,
            mri_norm_mode="percentile_01",
            min_size=128, res_mult=32, pad_value=0.0,
        )
    if family == "cwdm":
        return preprocess_volume(
            vol_path,
            mri_norm_mode="minmax_-1_1",
            min_size=128, res_mult=32, pad_value=-1.0,
        )
    if family == "mcddpm":
        return preprocess_to_patch(
            vol_path,
            mri_norm_mode="minmax_-1_1",
            patch=MCDDPM_PATCH, pad_value=-1.0,
        )
    raise ValueError(family)


def load_for_family(family, ckpt_path, device):
    if family == "unet":
        model, cfg, epoch = load_unet(ckpt_path, device)
        return ("unet", (model,), cfg, epoch)
    if family == "amix":
        translator, feat_extractor, cfg, epoch = load_amix(ckpt_path, device)
        return ("amix", (translator, feat_extractor), cfg, epoch)
    if family == "maisi":
        bundle, epoch = load_maisi(ckpt_path, device)
        return ("maisi", bundle, {}, epoch)
    if family == "cwdm":
        bundle, epoch = load_cwdm(ckpt_path, device)
        return ("cwdm", bundle, {}, epoch)
    if family == "mcddpm":
        bundle, epoch = load_mcddpm(ckpt_path, device)
        return ("mcddpm", bundle, {}, epoch)
    raise ValueError(family)


def infer_for_family(family, bundle, cfg, mri_tensor, voxel_sizes, device):
    if family == "unet":
        return infer_unet(bundle[0], mri_tensor, cfg, device)
    if family == "amix":
        return infer_amix(bundle[0], bundle[1], cfg, mri_tensor, device)
    if family == "maisi":
        return infer_maisi(bundle, mri_tensor, voxel_sizes, device)
    if family == "cwdm":
        return infer_cwdm(bundle, mri_tensor, device)
    if family == "mcddpm":
        return infer_mcddpm(bundle, mri_tensor, device)
    raise ValueError(family)
