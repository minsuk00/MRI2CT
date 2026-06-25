"""Full-validation-set evaluator for cWDM on SynthRAD MR->CT.

Runs reduced-step DDPM sampling (via SpacedDiffusion's `timestep_respacing=ddimN`)
on every subject in a chosen split. Computes MAE_HU / PSNR / SSIM (and
body-masked variants) plus Dice / Bone Dice via the Baby U-Net teacher, then
writes a TXT report with per-subject and aggregate metrics and saves NIfTI
predictions next to the checkpoint.

Padding correctness: every volume is padded by `get_cached_transforms` to
multiple-of-res_mult; `original_shape` is recorded *before* padding, so we
slice the prediction back to that shape before computing metrics or saving.

Body masking matches the post-sample step in sample.py: predictions are
multiplied by the body mask (analog of cWDM's `sample[cond_1==0] = 0` background
cleanup for skull-stripped BraTS).

CT range: cWDM uses [-1024, 1024] HU → [0, 1], identical to amix/unet, so the
teacher (trained on amix preproc) can be applied to the prediction directly.
"""
import argparse
import os
import pathlib
import random
import sys
import time

import nibabel as nib
import numpy as np
import torch as th
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing

sys.path.append(".")

from guided_diffusion import dist_util, logger  # noqa: E402
from guided_diffusion.script_util import (  # noqa: E402
    model_and_diffusion_defaults, create_model_and_diffusion, create_gaussian_diffusion,
    add_dict_to_argparser, args_to_dict,
)
from DWT_IDWT.DWT_IDWT_layer import DWT_3D, IDWT_3D  # noqa: E402

# Reuse the project's metric definitions so the table is directly comparable to amix runs.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from common.data import (  # noqa: E402
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_split_subjects,
)
from common.eval_utils import (  # noqa: E402
    default_teacher_specs,
    default_validate_dir,
    dual_teacher_dice,
    extract_checkpoint_info,
    format_checkpoint_info,
    load_subject_segs,
    load_teachers,
    write_metrics_txt,
)
from common.utils import compute_metrics, compute_metrics_body, unpad  # noqa: E402


def _build_val_loader(args, load_seg):
    """Build PersistentDataset → DataLoader for full padded volumes (bs=1)."""
    subjects = get_split_subjects(args.split_file, args.split_name)
    if args.max_subjects and args.max_subjects > 0:
        subjects = subjects[: args.max_subjects]
    # SLURM-array sharding: split subjects across `num_shards` round-robin so
    # each shard gets a mix of large + small volumes (more balanced wall time
    # than contiguous slicing).
    if args.num_shards and args.num_shards > 1:
        subjects = subjects[args.shard_idx :: args.num_shards]
        print(f"[VAL] shard {args.shard_idx}/{args.num_shards} → {len(subjects)} subjects")
    dicts = build_data_dicts(
        args.data_dir, subjects, load_seg=load_seg, load_body_mask=True
    )
    if not dicts:
        raise RuntimeError(f"No usable subjects for split '{args.split_name}' under {args.data_dir}")
    xform = get_cached_transforms(
        patch_size=args.patch_size,
        res_mult=args.res_mult,
        enforce_ras=True,
        mri_norm=args.mri_norm,
        ct_range=(args.ct_range_lo, args.ct_range_hi),
        load_seg=load_seg,
        load_body_mask=True,
    )
    # hash_transform=pickle_hashing → cache key includes the transform spec.
    # Without it, PersistentDataset hashes only the data dict, and a stale
    # entry made by a different transform (e.g. an earlier run without proper
    # padding) silently shadows ours.
    ds = PersistentDataset(
        data=dicts, transform=xform,
        cache_dir=default_monai_cache_dir(), hash_transform=pickle_hashing,
    )
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=0), dicts


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    if not args.output_dir:
        args.output_dir = default_validate_dir(args.model_path, prefix="validate")
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    logger.log(f"[VAL] output dir: {args.output_dir}")

    logger.log("Creating model...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, _train_diffusion = create_model_and_diffusion(**arguments)
    logger.log(f"Loading model from: {args.model_path}")
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    model.eval()

    # Checkpoint metadata for logging — cWDM saves a bare state_dict, so this
    # falls back to filename / sibling-file inspection.
    ckpt_info = extract_checkpoint_info(args.model_path, ckpt_dict=None)
    ckpt_info_str = format_checkpoint_info(ckpt_info)
    logger.log(f"[VAL] checkpoint: {ckpt_info_str}")

    # Respaced diffusion for ddimN-step sampling on the same weights.
    sample_diffusion = create_gaussian_diffusion(
        steps=arguments['diffusion_steps'],
        learn_sigma=arguments['learn_sigma'],
        noise_schedule=arguments['noise_schedule'],
        use_kl=arguments['use_kl'],
        predict_xstart=arguments['predict_xstart'],
        rescale_timesteps=arguments['rescale_timesteps'],
        rescale_learned_sigmas=arguments['rescale_learned_sigmas'],
        timestep_respacing=f"ddim{args.ddim_steps}",
        mode='i2i',
    )

    dwt = DWT_3D("haar")
    idwt = IDWT_3D("haar")

    device = dist_util.dev()

    # Teachers (legacy 12-class + CADS 35-class). Set --teacher_weights_path=none to disable Dice.
    dice_on = bool(args.teacher_weights_path and args.teacher_weights_path.lower() != "none")
    teachers = load_teachers(default_teacher_specs(), device) if dice_on else []

    # Segs are loaded per-teacher (dual label spaces), not through the cached pipeline.
    loader, _dicts = _build_val_loader(args, load_seg=False)

    per_subject = []
    hu_range = args.ct_range_hi - args.ct_range_lo

    for idx, batch in enumerate(loader):
        subj_id = batch['subj_id'][0] if isinstance(batch['subj_id'], (list, tuple)) else batch['subj_id']
        if th.cuda.is_available():
            th.cuda.synchronize()
        t0 = time.time()

        mri = batch['mri'].to(device).float()
        ct_gt = batch['ct'].to(device).float()
        mask = batch['body_mask'].to(device).float() if 'body_mask' in batch else None

        # 8-ch conditioning DWT.
        LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(mri)
        cond_dwt = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        B, _, D_dim, H_dim, W_dim = mri.shape
        noise = th.randn(B, 8, D_dim // 2, H_dim // 2, W_dim // 2, device=device)

        with th.no_grad():
            x0_wav = sample_diffusion.p_sample_loop(
                model=model,
                shape=noise.shape,
                noise=noise,
                cond=cond_dwt,
                clip_denoised=args.clip_denoised,
                model_kwargs={},
                progress=args.progress,
            )

            # IDWT back to image space; LLL band rescaled by *3 (mirror of training).
            pred = idwt(
                x0_wav[:, 0:1] * 3.,
                x0_wav[:, 1:2], x0_wav[:, 2:3], x0_wav[:, 3:4],
                x0_wav[:, 4:5], x0_wav[:, 5:6], x0_wav[:, 6:7], x0_wav[:, 7:8],
            ).clamp(0., 1.)

        # Unpad back to pre-pad shape (RecordOriginalShapeD).
        orig_shape = batch['original_shape']
        if th.is_tensor(orig_shape):
            orig_shape = orig_shape[0].tolist() if orig_shape.ndim > 1 else orig_shape.tolist()
        pred = unpad(pred, orig_shape)
        ct_gt = unpad(ct_gt, orig_shape)
        if mask is not None:
            mask = unpad(mask, orig_shape)
            pred = pred * mask  # background cleanup (sample.py mirror)

        # Image metrics
        met = compute_metrics(pred.float(), ct_gt.float(), hu_range=hu_range)
        record = {
            'mae_hu': met['mae_hu'], 'psnr': met['psnr'],
            'ssim': met['ssim'], 'grad_diff': met['grad_diff'],
        }
        if mask is not None:
            bm = compute_metrics_body(pred.float(), ct_gt.float(), mask.float(), hu_range=hu_range)
            record['body_mae_hu'] = bm['mae_hu']
            record['body_psnr']   = bm['psnr']
            record['body_ssim']   = bm['ssim']

        # Dice — both teachers (12-class unsuffixed, 35-class `_cads35`)
        if teachers:
            seg_by_file = load_subject_segs(args.data_dir, subj_id, teachers, device)
            sw = dict(val_patch_size=args.val_patch_size, sw_batch_size=1, overlap=args.val_sw_overlap)
            full, bod = dual_teacher_dice(teachers, pred.float(), seg_by_file, device,
                                          body_mask=mask, sw_kwargs=sw)
            record.update(full)
            record.update(bod)

        if th.cuda.is_available():
            th.cuda.synchronize()  # CUDA is async; sync so time_sec reflects real GPU compute
        record['time_sec'] = time.time() - t0
        per_subject.append({'subj_id': subj_id, 'metrics': record})

        logger.log(
            f"[{idx+1:3d}/{len(loader)}] {subj_id} "
            + " ".join(f"{k}={v:.4f}" for k, v in record.items() if isinstance(v, float))
        )

        if args.save_nifti:
            aff_t = batch.get('ct_affine', None)
            if aff_t is not None and th.is_tensor(aff_t):
                affine = (aff_t[0] if aff_t.ndim == 3 else aff_t).detach().cpu().numpy()
            else:
                affine = np.eye(4)
            subj_dir = os.path.join(args.output_dir, subj_id)
            pathlib.Path(subj_dir).mkdir(parents=True, exist_ok=True)
            # Save in HU to match the cross-model contract (amix/unet/maisi/mcddpm all
            # save HU). cWDM works in [0,1] over [-1024,1024] (span 2048); convert back.
            # Masked-out background (pred*mask -> 0) maps to -1024 HU (air), like the others.
            span = args.ct_range_hi - args.ct_range_lo
            pred_hu = pred[0, 0].detach().cpu().numpy() * span + args.ct_range_lo
            ct_hu = ct_gt[0, 0].detach().cpu().numpy() * span + args.ct_range_lo
            nib.save(nib.Nifti1Image(pred_hu, affine),
                     os.path.join(subj_dir, 'sample.nii.gz'))
            nib.save(nib.Nifti1Image(ct_hu, affine),
                     os.path.join(subj_dir, 'target.nii.gz'))

    # Write TXT report
    metric_keys = []
    for r in per_subject:
        for k in r['metrics']:
            if k not in metric_keys:
                metric_keys.append(k)
    header = [
        f"Validation report — cWDM",
        f"checkpoint: {args.model_path}",
        f"checkpoint info: {ckpt_info_str}",
        f"split_file: {args.split_file}   split_name: {args.split_name}",
        f"ddim_steps: {args.ddim_steps}   ct_range: [{args.ct_range_lo}, {args.ct_range_hi}]   mri_norm: {args.mri_norm}",
        f"teachers: {', '.join(t['tag'] for t in teachers) if teachers else 'disabled'}",
        f"subjects: {len(per_subject)}",
    ]
    if args.num_shards and args.num_shards > 1:
        header.insert(1, f"shard: {args.shard_idx}/{args.num_shards}")
    write_metrics_txt(
        os.path.join(args.output_dir, 'validate_metrics.txt'),
        header_lines=header, per_subject=per_subject, metric_keys=metric_keys,
    )


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        clip_denoised=True,
        model_path="",
        devices=[0],
        output_dir='',
        mode='i2i',
        renormalize=True,
        image_size=256,
        contr='ct',
        # SynthRAD ------------------------------------------------------------
        dataset='synthrad',
        split_file='splits/center_wise_split.txt',
        split_name='val',
        patch_size=128,
        res_mult=32,
        ct_range_lo=-1024,
        ct_range_hi=1024,
        mri_norm='minmax',
        # Sampling ------------------------------------------------------------
        # Paper-faithful evaluation uses the full 1000-step DDPM (--ddim_steps=1000).
        # Drop to a smaller value (e.g. 50) for fast ablation runs, but final reported
        # metrics for the paper baseline should use 1000.
        ddim_steps=1000,
        progress=False,
        save_nifti=True,
        max_subjects=0,  # 0 = full split; >0 limits to first N (smoke testing)
        shard_idx=0,
        num_shards=1,    # >1 enables SLURM-array round-robin sharding
        # Teacher / Dice ------------------------------------------------------
        teacher_weights_path="auto",  # 'none' disables Dice; else runs the canonical dual teachers
        val_patch_size=128,
        val_sw_batch_size=2,
        val_sw_overlap=0.25,
    )
    defaults.update({k: v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
