"""
Visualize MRI2CT model predictions: 1 subject per body region.

One figure per region with columns: MRI | GT CT | [Model Pred | Residual] × N_models
Also saves predicted CT and GT CT as NIfTI (.nii.gz) for each subject.

Usage:
    python src/evaluate/visualize_predictions.py \
        --checkpoints amix:/path/to/amix.pt unet:/path/to/unet.pt \
        [--split_file splits/center_wise_split.txt] \
        [--root_dir /path/to/data] \
        [--output_dir viz_results/] \
        [--body_mask]

Subject selection: scans root_dir for all subject folders and picks 1 per region.
If --split_file is given, restricts candidates to val subjects in that split.
"""

import argparse
import gc
import os
import sys
import tempfile
import warnings

import matplotlib
import numpy as np
import torch
import torchio as tio

matplotlib.use("Agg")
warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_DIR = os.path.abspath(os.path.join(_SRC_DIR, ".."))
sys.path.insert(0, _SRC_DIR)
sys.path.insert(0, os.path.join(_REPO_DIR, "MC-DDPM"))

from common.config import DEFAULT_CONFIG
from common.data import DataPreprocessing, build_tio_subjects, get_region_key, get_split_subjects
from common.utils import clean_state_dict, unpad

_GPFS_ROOT = DEFAULT_CONFIG["root_dir"]

# ---------------------------------------------------------------------------
# Default subjects to visualize — edit this list directly.
# 1 per non-thorax region + 3 thorax center A + 3 thorax center B.
# Override at runtime with --subjects.
# ---------------------------------------------------------------------------

DEFAULT_SUBJECTS = [
    # abdomen
    "1ABA005",
    # brain
    "1BA001",
    # head & neck
    "1HNA001",
    # pelvis
    "1PA001",
    # thorax — 3 center A (1THA), 3 center B (1THB)
    "1THA001",
    "1THA002",
    "1THA003",
    "1THB006",
    "1THB011",
    "1THB016",
]

# ---------------------------------------------------------------------------
# Helpers shared with evaluate_models.py
# ---------------------------------------------------------------------------


def _get_pad_offset(batch):
    po = batch.get("pad_offset", 0)
    if torch.is_tensor(po):
        return int(po[0])
    if isinstance(po, (list, tuple)):
        return int(po[0])
    return int(po)


def _resolve_root_dir(cfg_root, override=None):
    if override:
        return override
    if cfg_root and os.path.isdir(cfg_root):
        return cfg_root
    return _GPFS_ROOT


def detect_type(ckpt):
    if "controlnet_state_dict" in ckpt:
        return "maisi"
    cfg = ckpt.get("config", {})
    if "num_inference_steps" in cfg or "autoencoder_path" in cfg:
        return "maisi"
    if "diffusion_steps" in cfg:
        return "mcddpm"
    if "ngf" in cfg or cfg.get("model_type") == "unet_baseline":
        return "unet"
    if "anatomix_weights" in cfg:
        return "amix"
    raise ValueError("Cannot auto-detect model type. Use 'type:path' format.")


# ---------------------------------------------------------------------------
# Subject selection
# ---------------------------------------------------------------------------


def _select_subjects(root_dir, split_file=None):
    """Return [(region, subj_id), ...], 1 per region. Scans root_dir unless split_file given."""
    if split_file is not None:
        candidates = get_split_subjects(split_file, "val")
        print(f"[INFO] Using {len(candidates)} val subjects from split file.")
    else:
        try:
            all_entries = sorted(os.listdir(root_dir))
            candidates = [e for e in all_entries if os.path.isdir(os.path.join(root_dir, e))]
            print(f"[INFO] Scanning root_dir: found {len(candidates)} subject directories.")
        except Exception as e:
            raise RuntimeError(f"Cannot list root_dir '{root_dir}': {e}")

    seen = {}
    for sid in sorted(candidates):
        r = get_region_key(sid)
        if r not in seen:
            seen[r] = sid

    result = sorted(seen.items())
    print(f"[INFO] Selected 1 subject per region: { {r: s for r, s in result} }")
    return result


# ---------------------------------------------------------------------------
# Single-subject data loading
# ---------------------------------------------------------------------------


def _load_subject_batch(root_dir, subj_id, patch_size, res_mult, body_mask):
    """Load a single subject through DataPreprocessing. Returns (batch_dict, ct_affine)."""
    import nibabel as nib

    subj_objs = build_tio_subjects(root_dir, [subj_id], use_weighted_sampler=body_mask)
    if not subj_objs:
        raise RuntimeError(f"Could not build subject {subj_id} from {root_dir}")

    ct_path = os.path.join(root_dir, subj_id, "ct.nii.gz")
    if not os.path.exists(ct_path):
        ct_path = os.path.join(root_dir, subj_id, "ct.nii")
    affine = nib.load(ct_path).affine

    preprocess = DataPreprocessing(patch_size=patch_size, res_mult=res_mult, use_weighted_sampler=body_mask)
    ds = tio.SubjectsDataset(subj_objs, transform=preprocess)
    loader = tio.SubjectsLoader(ds, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    return batch, affine


# ---------------------------------------------------------------------------
# Per-model inference — returns {"pred", "ct", "mri", "mask", "affine"}
# All tensors: (1, 1, H, W, D) float32 CPU, unpadded to original spatial shape
# CT/pred in [0, 1] normalized space (0→-1024 HU, 1→+1024 HU)
# ---------------------------------------------------------------------------


@torch.inference_mode()
def _infer_amix(ckpt_path, subj_id, root_dir, device, body_mask):
    from anatomix.model.network import Unet
    from monai.inferers import sliding_window_inference

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    patch_size = cfg.get("patch_size", 128)
    res_mult = cfg.get("res_mult", 32)
    anatomix_weights = cfg.get("anatomix_weights", "v1_3")
    val_sw_batch_size = cfg.get("val_sw_batch_size", 4)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)
    feat_instance_norm = cfg.get("feat_instance_norm", False)
    pass_mri = cfg.get("pass_mri_to_translator", False)

    if anatomix_weights == "v1":
        feat_extractor = Unet(3, 1, 16, 4, 16).to(device)
        fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/anatomix.pth")
    else:
        feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(device)
        if anatomix_weights == "v1_2":
            fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/best_val_net_G_v1_2.pth")
        else:
            fe_ckpt_path = os.path.join(_REPO_DIR, "anatomix/model-weights/best_val_net_G_real_v1_3.pth")

    if "feat_extractor_state_dict" in ckpt:
        feat_extractor.load_state_dict(clean_state_dict(ckpt["feat_extractor_state_dict"]))
    elif os.path.exists(fe_ckpt_path):
        raw = torch.load(fe_ckpt_path, map_location=device, weights_only=False)
        feat_extractor.load_state_dict(clean_state_dict(raw))
    feat_extractor.eval()
    for p in feat_extractor.parameters():
        p.requires_grad = False

    translator_input_nc = 17 if pass_mri else 16
    translator = Unet(dimension=3, input_nc=translator_input_nc, output_nc=1, num_downs=4, ngf=16, final_act="sigmoid").to(device)
    translator.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    translator.eval()

    batch, affine = _load_subject_batch(root_dir, subj_id, patch_size, res_mult, body_mask)
    mri = batch["mri"][tio.DATA].to(device)
    ct = batch["ct"][tio.DATA].to(device)
    orig_shape = batch["original_shape"][0].tolist()
    pad_offset = _get_pad_offset(batch)

    def combined_forward(x, _fe=feat_extractor, _tr=translator):
        f = _fe(x)
        if feat_instance_norm:
            f = torch.nn.functional.instance_norm(f)
        if pass_mri:
            f = torch.cat([f, x], dim=1)
        return _tr(f)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        pred = sliding_window_inference(
            inputs=mri,
            roi_size=(patch_size, patch_size, patch_size),
            sw_batch_size=val_sw_batch_size,
            predictor=combined_forward,
            overlap=val_sw_overlap,
            device=device,
        )

    result = {
        "pred": unpad(pred, orig_shape, pad_offset).cpu().float(),
        "ct": unpad(ct, orig_shape, pad_offset).cpu().float(),
        "mri": unpad(mri, orig_shape, pad_offset).cpu().float(),
        "mask": unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset).cpu().float() if body_mask and "prob_map" in batch else None,
        "affine": affine,
    }
    del feat_extractor, translator, mri, ct, pred
    gc.collect()
    return result


@torch.inference_mode()
def _infer_unet(ckpt_path, subj_id, root_dir, device, body_mask):
    from anatomix.model.network import Unet
    from monai.inferers import sliding_window_inference

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    patch_size = cfg.get("patch_size", 128)
    res_mult = cfg.get("res_mult", 16)
    input_nc = cfg.get("input_nc", 1)
    output_nc = cfg.get("output_nc", 1)
    num_downs = cfg.get("num_downs", 4)
    ngf = cfg.get("ngf", 16)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 8)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.25)

    model = Unet(dimension=3, input_nc=input_nc, output_nc=output_nc, num_downs=num_downs, ngf=ngf, final_act="sigmoid").to(device)
    model.load_state_dict(clean_state_dict(ckpt["model_state_dict"]))
    model.eval()

    batch, affine = _load_subject_batch(root_dir, subj_id, patch_size, res_mult, body_mask)
    mri = batch["mri"][tio.DATA].to(device)
    ct = batch["ct"][tio.DATA].to(device)
    orig_shape = batch["original_shape"][0].tolist()
    pad_offset = _get_pad_offset(batch)

    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        pred = sliding_window_inference(
            inputs=mri,
            roi_size=(patch_size, patch_size, patch_size),
            sw_batch_size=val_sw_batch_size,
            predictor=model,
            overlap=val_sw_overlap,
            device=device,
        )

    result = {
        "pred": unpad(pred, orig_shape, pad_offset).cpu().float(),
        "ct": unpad(ct, orig_shape, pad_offset).cpu().float(),
        "mri": unpad(mri, orig_shape, pad_offset).cpu().float(),
        "mask": unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset).cpu().float() if body_mask and "prob_map" in batch else None,
        "affine": affine,
    }
    del model, mri, ct, pred
    gc.collect()
    return result


@torch.inference_mode()
def _infer_mcddpm(ckpt_path, subj_id, root_dir, device, body_mask):
    from diffusion.Create_diffusion import create_gaussian_diffusion
    from monai.inferers import SlidingWindowInferer
    from network.Diffusion_model_transformer import SwinVITModel

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get("config", {})

    patch_size = cfg.get("patch_size", (64, 64, 4))
    if isinstance(patch_size, list):
        patch_size = tuple(patch_size)
    learn_sigma = cfg.get("learn_sigma", True)
    val_sw_batch_size = cfg.get("val_sw_batch_size", 4)
    val_sw_overlap = cfg.get("val_sw_overlap", 0.5)
    timestep_respacing = cfg.get("timestep_respacing", [50])

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
    batch, affine = _load_subject_batch(root_dir, subj_id, patch_size_scalar, 1, body_mask)
    mri = batch["mri"][tio.DATA].to(device)
    ct = batch["ct"][tio.DATA].to(device)
    orig_shape = batch["original_shape"][0].tolist()
    pad_offset = _get_pad_offset(batch)

    inferer = SlidingWindowInferer(roi_size=patch_size, sw_batch_size=val_sw_batch_size, overlap=val_sw_overlap, mode="constant")

    def diffusion_sampling(condition, _model=model, _diff=diffusion):
        shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4])
        return _diff.p_sample_loop(_model, shape, condition=condition, clip_denoised=True)

    mri_scaled = mri * 2.0 - 1.0
    with torch.amp.autocast("cuda"):
        pred = inferer(mri_scaled, diffusion_sampling)
    pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

    result = {
        "pred": unpad(pred, orig_shape, pad_offset).cpu().float(),
        "ct": unpad(ct, orig_shape, pad_offset).cpu().float(),
        "mri": unpad(mri, orig_shape, pad_offset).cpu().float(),
        "mask": unpad(batch["prob_map"][tio.DATA].to(device), orig_shape, pad_offset).cpu().float() if body_mask and "prob_map" in batch else None,
        "affine": affine,
    }
    del model, mri, ct, pred
    gc.collect()
    return result


def _infer_maisi(ckpt_path, subj_id, root_dir, device):
    """Inference for MAISI. Uses a temp split file with a single val subject."""
    import torch.nn.functional as F

    from common.utils import cleanup_gpu
    from maisi_baseline.trainer import MAISITrainer

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(f"val {subj_id}\n")
        tmp_split = f.name

    try:
        config["wandb"] = False
        config["stage_data"] = False
        config["root_dir"] = root_dir
        config["sanity_check"] = False
        config["resume_wandb_id"] = None
        config["augment"] = False
        config["full_val"] = True
        config["split_file"] = os.path.abspath(tmp_split)

        class _CapturingTrainer(MAISITrainer):
            def validate(self_t, epoch):
                self_t._viz_result = None
                self_t.controlnet.eval()
                for i, batch_list in enumerate(self_t.val_loader):
                    if i not in self_t.val_indices_to_run:
                        continue
                    subj_data = batch_list[0]
                    mr = subj_data["mri"]["data"].unsqueeze(0).to(self_t.device)
                    ct = subj_data["ct"]["data"].unsqueeze(0).to(self_t.device)
                    spacing = subj_data["original_spacing"].unsqueeze(0).to(self_t.device) * 100.0
                    orig_shape = subj_data["original_spatial_shape"].tolist()

                    pred_latent = self_t._sample(mr, spacing, num_steps=self_t.cfg.num_inference_steps)
                    pred_ct_norm = self_t._decode(pred_latent)

                    pred_resized = F.interpolate(pred_ct_norm.float(), size=orig_shape, mode="trilinear", align_corners=False)
                    gt_matched = (torch.clamp((ct * 2048.0) - 1024.0, -1000.0, 1000.0) + 1000.0) / 2000.0
                    gt_resized = F.interpolate(gt_matched.float(), size=orig_shape, mode="trilinear", align_corners=False)
                    mr_resized = F.interpolate(mr.float(), size=orig_shape, mode="trilinear", align_corners=False)

                    self_t._viz_result = {
                        "pred": pred_resized.cpu(),
                        "ct": gt_resized.cpu(),
                        "mri": mr_resized.cpu(),
                        "mask": None,
                    }
                    break  # Only 1 subject needed
                return {}

        trainer = _CapturingTrainer(config)
        controlnet_state = ckpt.get("model_state_dict") or ckpt.get("controlnet_state_dict")
        trainer.controlnet.load_state_dict(clean_state_dict(controlnet_state))
        trainer.controlnet.eval()

        if "scale_factor" in ckpt:
            sf = ckpt["scale_factor"]
            trainer.scale_factor = sf.to(device) if isinstance(sf, torch.Tensor) else torch.tensor(float(sf), device=device)
            print(f"  [MAISI] Restored scale_factor = {trainer.scale_factor.item():.6f}")

        trainer.validate(0)
        result = trainer._viz_result
        del trainer
        cleanup_gpu()
    finally:
        os.unlink(tmp_split)

    if result is None:
        raise RuntimeError(f"MAISI inference returned no result for {subj_id}")

    import nibabel as nib

    ct_path = os.path.join(root_dir, subj_id, "ct.nii.gz")
    if not os.path.exists(ct_path):
        ct_path = os.path.join(root_dir, subj_id, "ct.nii")
    result["affine"] = nib.load(ct_path).affine
    return result


# ---------------------------------------------------------------------------
# NIfTI output
# ---------------------------------------------------------------------------


def _save_nifti_ct(tensor, affine, path):
    """Save a CT tensor (normalized [0,1]) as NIfTI in HU space: 0→-1024, 1→+1024."""
    import nibabel as nib

    data_hu = tensor.squeeze().cpu().float().numpy() * 2048.0 - 1024.0
    nib.save(nib.Nifti1Image(data_hu, affine), path)


def _save_nifti_mri(tensor, affine, path):
    """Save an MRI tensor (normalized [0,1]) as NIfTI float."""
    import nibabel as nib

    data = tensor.squeeze().cpu().float().numpy()
    nib.save(nib.Nifti1Image(data, affine), path)


# ---------------------------------------------------------------------------
# Figure generation
# ---------------------------------------------------------------------------


def _make_figure(region, subj_id, mri_np, ct_np, predictions, mask_np, output_dir, body_mask, label_map=None):
    """
    Build and save a comparison figure.
    predictions: list of (label, pred_np) where pred_np is (H, W, D) float32 [0,1]
    mri_np, ct_np: (H, W, D) float32 [0,1]
    mask_np: (H, W, D) float32 or None
    label_map: dict mapping run_id -> nickname, e.g. {"ct5bgykw": "high bone dice"}
    """
    import matplotlib.pyplot as plt

    n_models = len(predictions)
    ncols = 2 + 2 * n_models  # MRI, GT CT, (pred, residual) × n_models
    n_slices = 5

    D = mri_np.shape[-1]
    slice_indices = np.linspace(int(0.1 * D), int(0.9 * D), n_slices, dtype=int)

    # Build column titles with nickname if available
    def _col_title(label):
        run_id = label.split("/")[-1]
        nick = (label_map or {}).get(run_id)
        if nick:
            return f"{nick}\n({run_id})"
        return label.replace("/", "\n")

    col_titles = ["MRI", "GT CT"]
    for label, _ in predictions:
        col_titles += [f"{_col_title(label)}\nPred", f"{_col_title(label)}\nResidual"]

    fig, axes = plt.subplots(n_slices, ncols, figsize=(3 * ncols, 3.5 * n_slices))
    plt.subplots_adjust(wspace=0.03, hspace=0.12)

    if n_slices == 1:
        axes = axes.reshape(1, -1)

    residual_im = None

    for row, z in enumerate(slice_indices):
        col = 0

        # MRI
        ax = axes[row, col]
        ax.imshow(mri_np[:, :, z], cmap="gray", vmin=0, vmax=1)
        if body_mask and mask_np is not None:
            ax.contour(mask_np[:, :, z], levels=[0.5], colors="cyan", linewidths=0.5)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8)
        ax.axis("off")
        col += 1

        # GT CT
        ax = axes[row, col]
        ax.imshow(ct_np[:, :, z], cmap="gray", vmin=0, vmax=1)
        if body_mask and mask_np is not None:
            ax.contour(mask_np[:, :, z], levels=[0.5], colors="cyan", linewidths=0.5)
        if row == 0:
            ax.set_title(col_titles[col], fontsize=8)
        ax.axis("off")
        col += 1

        # Per-model: Pred + Residual
        for _, (label, pred_np) in enumerate(predictions):
            # Pred
            ax = axes[row, col]
            ax.imshow(pred_np[:, :, z], cmap="gray", vmin=0, vmax=1)
            if body_mask and mask_np is not None:
                ax.contour(mask_np[:, :, z], levels=[0.5], colors="cyan", linewidths=0.5)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
            ax.axis("off")
            col += 1

            # Residual
            ax = axes[row, col]
            residual_slice = pred_np[:, :, z] - ct_np[:, :, z]
            im = ax.imshow(residual_slice, cmap="seismic", vmin=-0.5, vmax=0.5)
            if body_mask and mask_np is not None:
                ax.contour(mask_np[:, :, z], levels=[0.5], colors="lime", linewidths=0.5)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=8)
            ax.axis("off")
            col += 1

            if residual_im is None:
                residual_im = im

    if residual_im is not None:
        residual_axes = [axes[:, 3 + 2 * m] for m in range(n_models)]
        flat_res_axes = [ax for col_axes in residual_axes for ax in col_axes]
        cbar = fig.colorbar(residual_im, ax=flat_res_axes, fraction=0.02, pad=0.02, shrink=0.8)
        cbar.set_label("Pred − GT", fontsize=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(f"{region} — {subj_id}", fontsize=13, y=1.001)

    out_path = os.path.join(output_dir, f"{region}_{subj_id}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [FIG] Saved {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Visualize MRI2CT predictions per body region")
    parser.add_argument("--checkpoints", nargs="+", required=True, metavar="TYPE:PATH", help="Checkpoints as 'type:path' (type ∈ {amix,unet,mcddpm,maisi}) or auto-detect")
    parser.add_argument("--split_file", type=str, default=None, help="Optional split file; if given, restricts subjects to val set")
    parser.add_argument("--output_dir", type=str, default="evaluation_results/viz")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--root_dir", type=str, default=None, help="Override data root_dir from checkpoint config")
    parser.add_argument("--subjects", nargs="+", default=None, metavar="SUBJ_ID", help="Explicit subject IDs to visualize (bypasses auto region-selection)")
    parser.add_argument("--label_map", nargs="+", default=None, metavar="ID:NICKNAME", help="Map run IDs to nicknames, e.g. ct5bgykw:'high bone dice'")
    parser.add_argument("--body_mask", action="store_true", help="Overlay body mask contour on all panels")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    # Parse checkpoints
    import re

    checkpoints = []
    for spec in args.checkpoints:
        if ":" in spec and not os.path.exists(spec):
            model_type, ckpt_path = spec.split(":", 1)
        else:
            ckpt_path = spec
            tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            model_type = detect_type(tmp)
            del tmp
            print(f"[INFO] Auto-detected type: {model_type} for {os.path.basename(ckpt_path)}")

        # Build a readable label: use wandb run ID if path follows run-DATE-RUNID/files/ structure
        parent = os.path.basename(os.path.dirname(ckpt_path))
        if parent == "files":
            grandparent = os.path.basename(os.path.dirname(os.path.dirname(ckpt_path)))
            m = re.search(r"-([a-z0-9]+)$", grandparent)
            parent = m.group(1) if m else grandparent
        label = f"{model_type}/{parent}"
        checkpoints.append({"type": model_type, "path": ckpt_path, "label": label})
        print(f"[INFO] Checkpoint: [{model_type}] {ckpt_path}")

    # Determine root_dir from first checkpoint config (if not overridden)
    first_ckpt = torch.load(checkpoints[0]["path"], map_location="cpu", weights_only=False)
    cfg0 = first_ckpt.get("config", {})
    root_dir = _resolve_root_dir(cfg0.get("root_dir"), args.root_dir)
    del first_ckpt
    print(f"[INFO] Data root: {root_dir}")

    # Build [(region, subj_id), ...] to iterate
    if args.subjects:
        subjects_list = [(get_region_key(s), s) for s in args.subjects]
        print(f"[INFO] Using {len(args.subjects)} explicit subject(s): {args.subjects}")
    elif args.split_file:
        subjects_list = _select_subjects(root_dir, args.split_file)
    else:
        subjects_list = [(get_region_key(s), s) for s in DEFAULT_SUBJECTS]
        print(f"[INFO] Using DEFAULT_SUBJECTS ({len(DEFAULT_SUBJECTS)} subjects)")

    if not subjects_list:
        print("[ERROR] No subjects found. Check root_dir or split_file.")
        return

    # Parse label_map: "ct5bgykw:high bone dice" -> {"ct5bgykw": "high bone dice"}
    label_map = {}
    for entry in args.label_map or []:
        k, _, v = entry.partition(":")
        if k and v:
            label_map[k.strip()] = v.strip()

    saved_files = []

    for region, subj_id in subjects_list:
        print(f"\n{'=' * 60}")
        print(f"Region: {region}  Subject: {subj_id}")
        print(f"{'=' * 60}")

        predictions = []  # list of (label, pred_np)
        gt_ct_np = None
        mri_np = None
        mask_np = None
        affine = None

        for ckpt_info in checkpoints:
            mtype = ckpt_info["type"]
            ckpt_path = ckpt_info["path"]
            label = ckpt_info["label"]
            print(f"  Running [{mtype}] {label} on {subj_id}...")

            try:
                # Resolve root_dir per-checkpoint (some may have different stale configs)
                ckpt_tmp = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_root = _resolve_root_dir(ckpt_tmp.get("config", {}).get("root_dir"), args.root_dir)
                del ckpt_tmp

                if mtype == "amix":
                    res = _infer_amix(ckpt_path, subj_id, ckpt_root, device, args.body_mask)
                elif mtype == "unet":
                    res = _infer_unet(ckpt_path, subj_id, ckpt_root, device, args.body_mask)
                elif mtype == "mcddpm":
                    res = _infer_mcddpm(ckpt_path, subj_id, ckpt_root, device, args.body_mask)
                elif mtype == "maisi":
                    res = _infer_maisi(ckpt_path, subj_id, ckpt_root, device)
                else:
                    print(f"  [SKIP] Unknown type '{mtype}'")
                    continue

                pred_np = res["pred"].squeeze().numpy()
                predictions.append((label, pred_np))

                # Use first model's GT/MRI/mask (all models share same subject data)
                if gt_ct_np is None:
                    gt_ct_np = res["ct"].squeeze().numpy()
                    mri_np = res["mri"].squeeze().numpy()
                    affine = res["affine"]
                    if res["mask"] is not None:
                        mask_np = res["mask"].squeeze().numpy()

                # Save prediction NIfTI
                safe_label = label.replace("/", "_")
                pred_nii_path = os.path.join(args.output_dir, f"{region}_{subj_id}_{safe_label}_pred.nii.gz")
                _save_nifti_ct(res["pred"], affine, pred_nii_path)
                print(f"  [NII] Saved {pred_nii_path}")

            except Exception as e:
                import traceback

                print(f"  [ERROR] {label} on {subj_id}: {e}")
                traceback.print_exc()

            gc.collect()
            torch.cuda.empty_cache()

        if gt_ct_np is None:
            print(f"  [SKIP] No successful inference for {subj_id}, skipping figure.")
            continue

        # Save GT CT and MRI NIfTIs (once per subject)
        gt_nii_path = os.path.join(args.output_dir, f"{region}_{subj_id}_gt_ct.nii.gz")
        mri_nii_path = os.path.join(args.output_dir, f"{region}_{subj_id}_mri.nii.gz")
        _save_nifti_ct(torch.tensor(gt_ct_np).unsqueeze(0).unsqueeze(0), affine, gt_nii_path)
        _save_nifti_mri(torch.tensor(mri_np).unsqueeze(0).unsqueeze(0), affine, mri_nii_path)
        print(f"  [NII] Saved {gt_nii_path}")
        print(f"  [NII] Saved {mri_nii_path}")

        # Generate figure
        if predictions:
            fig_path = _make_figure(region, subj_id, mri_np, gt_ct_np, predictions, mask_np, args.output_dir, args.body_mask, label_map=label_map)
            saved_files.append(fig_path)

    print(f"\n{'=' * 60}")
    print(f"Done. {len(saved_files)} figure(s) saved to {args.output_dir}")
    for p in saved_files:
        print(f"  {p}")


if __name__ == "__main__":
    main()
