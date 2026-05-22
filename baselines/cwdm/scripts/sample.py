"""
A script for sampling from a diffusion model for paired image-to-image translation.
"""

import argparse
import nibabel as nib
import numpy as np
import os
import pathlib
import random
import sys
import torch as th
import torch.nn.functional as F

sys.path.append(".")

from guided_diffusion import (dist_util,
                              logger)
from guided_diffusion.bratsloader import BRATSVolumes
from guided_diffusion.synthradloader import SynthRADVolumes
from guided_diffusion.script_util import (model_and_diffusion_defaults, create_model_and_diffusion,
                                          add_dict_to_argparser, args_to_dict)
from DWT_IDWT.DWT_IDWT_layer import IDWT_3D, DWT_3D

def main():
    args = create_argparser().parse_args()
    seed = args.seed
    dist_util.setup_dist(devices=args.devices)
    logger.configure()

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    diffusion.mode = 'i2i'
    logger.log("Load model from: {}".format(args.model_path))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev([0, 1]) if len(args.devices) > 1 else dist_util.dev())  # allow for 2 devices

    if args.dataset == 'brats':
        ds = BRATSVolumes(args.data_dir, mode='eval')
    elif args.dataset == 'synthrad':
        ds = SynthRADVolumes(
            root_dir=args.data_dir,
            split_file=args.split_file,
            split_name=args.split_name,
            patch_size=args.patch_size,
            res_mult=args.res_mult,
            ct_range=(args.ct_range_lo, args.ct_range_hi),
            mri_norm=args.mri_norm,
        )
    else:
        raise ValueError(f"Unknown dataset '{args.dataset}'")

    datal = th.utils.data.DataLoader(ds,
                                     batch_size=args.batch_size,
                                     num_workers=12 if args.dataset == 'brats' else 0,
                                     shuffle=False,)

    model.eval()
    idwt = IDWT_3D("haar")
    dwt = DWT_3D("haar")

    th.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    for batch in iter(datal):
        if args.dataset == 'synthrad':
            batch['ct']  = batch['ct'].to(dist_util.dev())
            batch['mri'] = batch['mri'].to(dist_util.dev())
            subj = batch['subj_id'][0] if isinstance(batch['subj_id'], (list, tuple)) else batch['subj_id']
            print(subj)

            target = batch['ct']
            cond_1 = batch['mri']
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
            cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        else:
            batch['t1n'] = batch['t1n'].to(dist_util.dev())
            batch['t1c'] = batch['t1c'].to(dist_util.dev())
            batch['t2w'] = batch['t2w'].to(dist_util.dev())
            batch['t2f'] = batch['t2f'].to(dist_util.dev())

            subj = batch['subj'][0].split('validation/')[1][:19]
            print(subj)

            if args.contr == 't1n':
                target = batch['t1n']  # target
                cond_1 = batch['t1c']  # condition
                cond_2 = batch['t2w']  # condition
                cond_3 = batch['t2f']  # condition

            elif args.contr == 't1c':
                target = batch['t1c']
                cond_1 = batch['t1n']
                cond_2 = batch['t2w']
                cond_3 = batch['t2f']

            elif args.contr == 't2w':
                target = batch['t2w']
                cond_1 = batch['t1n']
                cond_2 = batch['t1c']
                cond_3 = batch['t2f']

            elif args.contr == 't2f':
                target = batch['t2f']
                cond_1 = batch['t1n']
                cond_2 = batch['t1c']
                cond_3 = batch['t2w']

            else:
                print("This contrast can't be synthesized.")

            # Conditioning vector (3 MRI modalities for BraTS)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_1)
            cond = th.cat([LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_2)
            cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)
            LLL, LLH, LHL, LHH, HLL, HLH, HHL, HHH = dwt(cond_3)
            cond = th.cat([cond, LLL / 3., LLH, LHL, LHH, HLL, HLH, HHL, HHH], dim=1)

        # Noise — derive shape from cond_1 so any input volume works.
        B, _, D_dim, H_dim, W_dim = cond_1.shape
        noise = th.randn(B, 8, D_dim // 2, H_dim // 2, W_dim // 2).to(dist_util.dev())

        model_kwargs = {}

        sample_fn = diffusion.p_sample_loop

        sample = sample_fn(model=model,
                           shape=noise.shape,
                           noise=noise,
                           cond=cond,
                           clip_denoised=args.clip_denoised,
                           model_kwargs=model_kwargs)

        B, _, D, H, W = sample.size()
        sample = idwt(sample[:, 0, :, :, :].view(B, 1, D, H, W) * 3.,
                      sample[:, 1, :, :, :].view(B, 1, D, H, W),
                      sample[:, 2, :, :, :].view(B, 1, D, H, W),
                      sample[:, 3, :, :, :].view(B, 1, D, H, W),
                      sample[:, 4, :, :, :].view(B, 1, D, H, W),
                      sample[:, 5, :, :, :].view(B, 1, D, H, W),
                      sample[:, 6, :, :, :].view(B, 1, D, H, W),
                      sample[:, 7, :, :, :].view(B, 1, D, H, W))

        sample[sample <= 0] = 0
        sample[sample >= 1] = 1

        if args.dataset == 'synthrad':
            # Unpad to the pre-pad spatial shape recorded by RecordOriginalShapeD,
            # then zero out non-body voxels with the body mask (analog of cWDM's
            # `sample[cond_1 == 0] = 0` background cleanup for skull-stripped BraTS).
            if 'original_shape' in batch:
                orig_shape = batch['original_shape']
                if th.is_tensor(orig_shape):
                    orig_shape = orig_shape[0].tolist() if orig_shape.ndim > 1 else orig_shape.tolist()
                w_o, h_o, d_o = int(orig_shape[0]), int(orig_shape[1]), int(orig_shape[2])
                sample = sample[..., :w_o, :h_o, :d_o]
                target = target[..., :w_o, :h_o, :d_o]
                if 'body_mask' in batch:
                    body_mask = batch['body_mask'].to(dist_util.dev()).float()[..., :w_o, :h_o, :d_o]
                    sample = sample * body_mask
            elif 'body_mask' in batch:
                body_mask = batch['body_mask'].to(dist_util.dev()).float()
                sample = sample * body_mask
        else:
            # BraTS: zero out non-brain via cond_1==0 (skull-stripped); crop depth to 155.
            sample[cond_1 == 0] = 0
            if len(sample.shape) == 5:
                sample = sample.squeeze(dim=1)
            sample = sample[:, :, :, :155]
            if len(target.shape) == 5:
                target = target.squeeze(dim=1)
            target = target[:, :, :, :155]

        # Affine: SynthRAD provides ct_affine; BraTS uses identity.
        affine = np.eye(4)
        if args.dataset == 'synthrad' and 'ct_affine' in batch:
            aff_t = batch['ct_affine']
            affine = (aff_t[0] if aff_t.ndim == 3 else aff_t).detach().cpu().numpy()

        pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(os.path.join(args.output_dir, subj)).mkdir(parents=True, exist_ok=True)

        sample_np = sample.detach().cpu().numpy()
        target_np = target.detach().cpu().numpy()

        # Squeeze channel dim if still present (synthrad keeps (B, 1, D, H, W); brats squeezed earlier).
        if sample_np.ndim == 5:
            sample_np = sample_np[:, 0]
        if target_np.ndim == 5:
            target_np = target_np[:, 0]

        for i in range(sample_np.shape[0]):
            output_name = os.path.join(args.output_dir, subj, 'sample.nii.gz')
            img = nib.Nifti1Image(sample_np[i], affine)
            nib.save(img=img, filename=output_name)
            print(f'Saved to {output_name}')

            output_name = os.path.join(args.output_dir, subj, 'target.nii.gz')
            img = nib.Nifti1Image(target_np[i], affine)
            nib.save(img=img, filename=output_name)

def create_argparser():
    defaults = dict(
        seed=0,
        data_dir="",
        data_mode='validation',
        clip_denoised=True,
        num_samples=1,
        batch_size=1,
        use_ddim=False,
        class_cond=False,
        sampling_steps=0,
        model_path="",
        devices=[0],
        output_dir='./results',
        mode='default',
        renormalize=False,
        image_size=256,
        half_res_crop=False,
        concat_coords=False,  # if true, add 3 (for 3d) or 2 (for 2d) to in_channels
        contr="ct",
        # Default to SynthRAD (this fork's primary use). Pass --dataset=brats to use the original loader.
        dataset='synthrad',
        split_file='splits/center_wise_split.txt',
        split_name='val',
        patch_size=128,
        res_mult=32,
        ct_range_lo=-1024,
        ct_range_hi=1024,
        mri_norm='minmax',
    )
    defaults.update({k:v for k, v in model_and_diffusion_defaults().items() if k not in defaults})
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()

















