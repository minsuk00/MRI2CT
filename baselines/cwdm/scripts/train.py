"""
A script for training a diffusion model for paired image-to-image translation.
"""

import argparse
import os
import numpy as np
import random
import sys
import torch as th

sys.path.append(".")
sys.path.append("..")

from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          create_gaussian_diffusion,
                                          args_to_dict, add_dict_to_argparser)
from guided_diffusion.train_util import TrainLoop
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.synthradloader import SynthRADVolumes
from torch.utils.tensorboard import SummaryWriter


def main():
    args = create_argparser().parse_args()
    seed = args.seed
    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # Set up logging in a single pass so we never write to ./results.
    # Priority: tensorboard (if requested) → wandb run dir → $WANDB_DIR fallback → /tmp fallback.
    summary_writer = None
    wandb_run = None

    if args.use_tensorboard:
        logdir = args.tensorboard_path if args.tensorboard_path else None
        summary_writer = SummaryWriter(log_dir=logdir)
        summary_writer.add_text(
            'config',
            '\n'.join([f'--{k}={repr(v)} <br/>' for k, v in vars(args).items()])
        )

    if args.use_wandb:
        import wandb, datetime as _dt
        extra_tags = [t.strip() for t in (args.wandb_extra_tags or '').split(',') if t.strip()]
        # amix-style auto-name: YYYYMMDD_HHMM_cwdm. Skip when resuming (wandb attaches by id).
        if not args.wandb_run_name and not args.wandb_resume_id:
            args.wandb_run_name = f"{_dt.datetime.now().strftime('%Y%m%d_%H%M')}_cwdm"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or None,
            tags=['cwdm'] + extra_tags,
            config=vars(args),
            id=args.wandb_resume_id or None,
            resume='allow' if args.wandb_resume_id else None,
            dir=(summary_writer.get_logdir() if summary_writer is not None else None),
        )

    # Pick the canonical log/checkpoint dir, then call logger.configure exactly once.
    # cWDM's `get_blob_logdir()` returns `logger.get_dir()`, so checkpoints land here.
    log_dir = None
    if summary_writer is not None:
        log_dir = summary_writer.get_logdir()
    elif wandb_run is not None and wandb_run.dir:
        log_dir = wandb_run.dir
    elif os.environ.get('WANDB_DIR'):
        log_dir = os.path.join(os.environ['WANDB_DIR'], 'cwdm_logs', f'run-{int(__import__("time").time())}')
    else:
        log_dir = os.path.join('/tmp', f'cwdm_logs_{os.environ.get("USER", "user")}', f'run-{int(__import__("time").time())}')
    os.makedirs(log_dir, exist_ok=True)
    logger.configure(dir=log_dir)

    # Auto-discover the latest checkpoint when resuming via wandb id (amix-style ergonomics):
    # if --wandb_resume_id is set but --resume_checkpoint is empty:
    #   1. Prefer the rolling `<dataset>_last.pt` if it exists (newest, written every save_last_interval).
    #   2. Else fall back to the highest-numbered milestone `<dataset>_NNNNNN.pt`.
    if args.use_wandb and args.wandb_resume_id and not args.resume_checkpoint:
        import glob
        wb_dir = os.environ.get('WANDB_DIR', '.')
        run_dir_globs = [f'run-*-{args.wandb_resume_id}', f'offline-run-*-{args.wandb_resume_id}']

        # 1. Look for rolling last.pt first.
        last_pt = []
        for g in run_dir_globs:
            last_pt.extend(glob.glob(os.path.join(wb_dir, 'wandb', g, 'files', 'checkpoints', f'{args.dataset}_last.pt')))
        if last_pt:
            args.resume_checkpoint = last_pt[0]
            logger.log(f"[resume] auto-discovered rolling last checkpoint: {args.resume_checkpoint}")
        else:
            # 2. Fall back to milestones.
            patterns = [
                os.path.join(wb_dir, 'wandb', g, 'files', 'checkpoints', f'{args.dataset}_[0-9]*.pt')
                for g in run_dir_globs
            ]
            cand = []
            for p in patterns:
                cand.extend(glob.glob(p))
            cand = sorted(cand, key=lambda p: os.path.basename(p))
            if cand:
                args.resume_checkpoint = cand[-1]
                logger.log(f"[resume] no last.pt; using highest milestone: {args.resume_checkpoint}")
            else:
                logger.log(f"[resume] wandb_resume_id={args.wandb_resume_id} set but no checkpoint found under {wb_dir}/wandb/*-{args.wandb_resume_id}/...")

    dist_util.setup_dist(devices=args.devices)

    logger.log("Creating model and diffusion...")
    arguments = args_to_dict(args, model_and_diffusion_defaults().keys())
    model, diffusion = create_model_and_diffusion(**arguments)

    # Count + log model size (param count, fp32 footprint).
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    state_size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    logger.log(f"Model: {total_params:,} params ({trainable_params:,} trainable), state_dict ~{state_size_mb:.1f} MB fp32")
    if wandb_run is not None:
        wandb_run.summary['model/total_params'] = total_params
        wandb_run.summary['model/trainable_params'] = trainable_params
        wandb_run.summary['model/state_dict_mb'] = state_size_mb

    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=1000)

    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='train')
    elif args.dataset == 'synthrad':
        ds = SynthRADVolumes(
            root_dir=args.data_dir,
            split_file=args.split_file,
            split_name='train',
            patch_size=args.patch_size,
            res_mult=args.res_mult,
            ct_range=(args.ct_range_lo, args.ct_range_hi),
            mri_norm=args.mri_norm,
        )
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=args.num_workers,
                                     shuffle=True,)

    # Optional in-training val: 1 fixed subject sampled every val_interval steps with reduced-step DDPM.
    val_loader = None
    val_diffusion = None
    if args.dataset == 'synthrad' and args.val_subj_id and args.val_interval > 0:
        try:
            val_ds = SynthRADVolumes(
                root_dir=args.data_dir,
                split_file=args.split_file,
                split_name='val',
                patch_size=args.patch_size,
                res_mult=args.res_mult,
                ct_range=(args.ct_range_lo, args.ct_range_hi),
                mri_norm=args.mri_norm,
                subjects_filter=[args.val_subj_id],
            )
            val_loader = th.utils.data.DataLoader(val_ds, batch_size=1, num_workers=0, shuffle=False)

            # SpacedDiffusion that re-uses the same trained weights but runs val_ddim_steps timesteps.
            val_arg_dict = dict(arguments)  # shallow copy
            val_arg_dict['timestep_respacing'] = f"ddim{args.val_ddim_steps}"
            val_diffusion = create_gaussian_diffusion(
                steps=val_arg_dict['diffusion_steps'],
                learn_sigma=val_arg_dict['learn_sigma'],
                noise_schedule=val_arg_dict['noise_schedule'],
                use_kl=val_arg_dict['use_kl'],
                predict_xstart=val_arg_dict['predict_xstart'],
                rescale_timesteps=val_arg_dict['rescale_timesteps'],
                rescale_learned_sigmas=val_arg_dict['rescale_learned_sigmas'],
                timestep_respacing=val_arg_dict['timestep_respacing'],
                mode='i2i',
            )
            logger.log(
                f"In-training val: subj={args.val_subj_id} every {args.val_interval} steps "
                f"with {args.val_ddim_steps} sampling steps."
            )
        except Exception as e:
            logger.log(f"[val setup] disabled: {e}")
            val_loader = None
            val_diffusion = None

    logger.log("Start training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=datal,
        batch_size=args.batch_size,
        in_channels=args.in_channels,
        image_size=args.image_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        resume_step=args.resume_step,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        dataset=args.dataset,
        summary_writer=summary_writer,
        mode='i2i',
        contr=args.contr,
        use_wandb=(wandb_run is not None),
        wandb_run=wandb_run,
        val_loader=val_loader,
        val_diffusion=val_diffusion,
        val_interval=args.val_interval,
        val_subj_id=args.val_subj_id,
        val_ddim_steps=args.val_ddim_steps,
        save_last_interval=args.save_last_interval,
    ).run_loop()


def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,
        ema_rate="0.9999",
        log_interval=100,
        save_interval=5000,
        resume_checkpoint='',
        resume_step=0,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        dataset='synthrad',
        use_tensorboard=False,
        tensorboard_path='',  # set path to existing logdir for resuming
        devices=[0],
        dims=3,
        learn_sigma=False,
        num_groups=32,
        channel_mult="1,2,2,4,4",
        in_channels=8,
        out_channels=8,
        bottleneck_attention=False,
        num_workers=0,
        mode='default',
        renormalize=True,
        additive_skips=False,
        use_freq=False,
        contr='ct',
        # SynthRAD MR->CT integration ------------------------------------------
        split_file='splits/center_wise_split.txt',
        patch_size=128,
        res_mult=32,
        ct_range_lo=-1024,
        ct_range_hi=1024,
        mri_norm='minmax',
        # WandB ----------------------------------------------------------------
        use_wandb=True,
        wandb_project='mri2ct',
        wandb_run_name='',
        wandb_extra_tags='',
        wandb_resume_id='',
        # In-training validation hook -----------------------------------------
        val_subj_id='1THB011',
        val_interval=20000,
        val_ddim_steps=50,
        # Rolling-last checkpoint cadence: writes <dataset>_last.pt + optlast.pt + last_step.txt,
        # overwritten each fire. Set 0 to disable.
        save_last_interval=1000,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
