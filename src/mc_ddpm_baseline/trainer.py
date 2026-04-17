import copy
import gc
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

import wandb

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Add MC-DDPM root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "MC-DDPM")))

import torchio as tio
from diffusion.Create_diffusion import create_gaussian_diffusion
from diffusion.resampler import UniformSampler
from monai.inferers import SlidingWindowInferer
from network.Diffusion_model_transformer import SwinVITModel

from common.config import Config
from common.data import DataPreprocessing, build_tio_subjects, get_augmentations
from common.trainer_base import BaseTrainer
from common.utils import count_parameters, unpad, visualize_lite


class MCDDPMTrainer(BaseTrainer):
    def __init__(self, config_dict):
        from common.config import DEFAULT_CONFIG

        full_conf = copy.deepcopy(DEFAULT_CONFIG)
        full_conf.update(config_dict)
        cfg = Config(full_conf)
        super().__init__(cfg, prefix="MCDDPM")

        self._find_subjects()
        self._stage_data_local()
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 7. Model Summary Logging
        self._log_model_summary({"MC-DDPM": self.model})

        self._load_resume(self.model, self.optimizer, self.scheduler, self.scaler)

    def _setup_models(self):
        print(f"[{self.prefix}] 🏗️ Building MC-DDPM Models")

        # Diffusion setup
        self.diffusion = create_gaussian_diffusion(
            steps=self.cfg.diffusion_steps,
            learn_sigma=self.cfg.learn_sigma,
            sigma_small=self.cfg.sigma_small,
            noise_schedule=self.cfg.noise_schedule,
            use_kl=self.cfg.use_kl,
            predict_xstart=self.cfg.predict_xstart,
            rescale_timesteps=self.cfg.rescale_timesteps,
            rescale_learned_sigmas=self.cfg.rescale_learned_sigmas,
            timestep_respacing=self.cfg.timestep_respacing,
        )
        self.schedule_sampler = UniformSampler(self.diffusion)

        # Swin-VIT Network
        # The paper concat MR and CT for the input, resulting in in_channels=2
        # Output channels = 2 if learn_sigma=True, else 1
        out_channels = 2 if self.cfg.learn_sigma else 1

        self.model = SwinVITModel(
            image_size=self.cfg.patch_size,
            in_channels=2,
            model_channels=self.cfg.num_channels,
            out_channels=out_channels,
            dims=3,
            sample_kernel=self.cfg.sample_kernel,
            num_res_blocks=self.cfg.num_res_blocks,
            attention_resolutions=self.cfg.attention_resolutions,
            dropout=self.cfg.dropout,
            channel_mult=self.cfg.channel_mult,
            num_classes=None,
            use_checkpoint=self.cfg.use_checkpoint,
            use_fp16=False,
            num_heads=self.cfg.num_heads,
            window_size=self.cfg.window_size,
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=self.cfg.use_scale_shift_norm,
            resblock_updown=self.cfg.resblock_updown,
            use_new_attention_order=False,
        ).to(self.device)

        tot, train = count_parameters(self.model)
        print(f"[{self.prefix}] Model Params: Total={tot:,} | Trainable={train:,}")

    def _setup_data(self, seed=None):
        if seed is not None:
            import random

            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Train Loader
        train_objs = build_tio_subjects(self.cfg.root_dir, self.train_subjects, use_weighted_sampler=self.cfg.use_weighted_sampler, load_seg=False)
        preprocess = DataPreprocessing(patch_size=max(self.cfg.patch_size), enable_safety_padding=False, res_mult=1, enforce_ras=False, use_weighted_sampler=self.cfg.use_weighted_sampler)

        transform_list = [preprocess]
        if self.cfg.augment:
            transform_list.append(get_augmentations())

        transforms = tio.Compose(transform_list)
        train_ds = tio.SubjectsDataset(train_objs, transform=transforms)

        if self.cfg.use_weighted_sampler:
            sampler = tio.WeightedSampler(patch_size=self.cfg.patch_size, probability_map="prob_map")
        else:
            sampler = tio.UniformSampler(patch_size=self.cfg.patch_size)

        queue = tio.Queue(
            subjects_dataset=train_ds,
            samples_per_volume=self.cfg.patches_per_volume,
            max_length=max(self.cfg.patches_per_volume, self.cfg.data_queue_max_length),
            sampler=sampler,
            num_workers=self.cfg.data_queue_num_workers,
            shuffle_patches=True,
            shuffle_subjects=True,
        )
        self.train_loader = tio.SubjectsLoader(queue, batch_size=self.cfg.batch_size, num_workers=0)
        self.train_iter = self._inf_gen(self.train_loader)

        # Val Loader
        # NOTE: use_weighted_sampler here only loads mask.nii.gz as prob_map in the batch —
        # it does NOT do weighted patch sampling (that's training-only via Queue).
        val_objs = build_tio_subjects(self.cfg.root_dir, self.val_subjects, load_seg=False,
                                      use_weighted_sampler=getattr(self.cfg, "val_body_mask", False))
        val_ds = tio.SubjectsDataset(val_objs, transform=preprocess)
        self.val_loader = tio.SubjectsLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Stratified Validation Sampling
        rng = np.random.RandomState(self.cfg.seed)
        from collections import defaultdict
        from common.data import get_region_key

        region_to_indices = defaultdict(list)
        for idx, subj_id in enumerate(self.val_subjects):
            region = get_region_key(subj_id)
            region_to_indices[region].append(idx)

        if self.cfg.full_val:
            self.val_indices_to_run = set(range(len(self.val_subjects)))
            viz_indices = []
            for region, indices in region_to_indices.items():
                viz_indices.extend(rng.choice(indices, min(len(indices), self.cfg.viz_limit), replace=False))
            self.val_viz_indices = set(viz_indices)
        else:
            reduced_indices = [rng.choice(indices) for indices in region_to_indices.values()]
            self.val_indices_to_run = set(reduced_indices)
            self.val_viz_indices = set(reduced_indices)

        print(f"[{self.prefix}] 📊 Validation strategy: full_val={self.cfg.full_val}, running on {len(self.val_indices_to_run)}/{len(self.val_subjects)} volumes.")

    def _setup_opt(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        self.scaler = torch.amp.GradScaler("cuda")
        self.scheduler = None

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0

        steps = self.cfg.steps_per_epoch
        pbar = tqdm(range(steps), desc=f"Ep {epoch}", leave=False, dynamic_ncols=True)

        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            for _ in range(self.cfg.accum_steps):
                batch = next(self.train_iter)
                # the diffusion model expects channel dimension 1, which tio provides [B, 1, X, Y, Z]
                mri = batch["mri"][tio.DATA].to(self.device, non_blocking=True)
                ct = batch["ct"][tio.DATA].to(self.device, non_blocking=True)

                # Modular Synchronized CutOut (via BaseTrainer)
                # Apply on [0, 1] range before scaling to [-1, 1]
                mri, ct, _ = self.apply_cutout(mri, ct)

                # Scale from [0, 1] (from DataPreprocessing) to [-1, 1] (for MC-DDPM)
                mri = mri * 2.0 - 1.0
                ct = ct * 2.0 - 1.0

                t, weights = self.schedule_sampler.sample(mri.shape[0], self.device)

                with torch.amp.autocast("cuda"):
                    # diffusion.training_losses takes (model, target, condition, t)
                    all_loss = self.diffusion.training_losses(self.model, ct, mri, t)
                    loss = (all_loss["loss"] * weights).mean()
                    loss = loss / self.cfg.accum_steps

                self.scaler.scale(loss).backward()
                step_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += step_loss
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            pbar.set_postfix({"loss": f"{step_loss:.4f}"})

            if self.cfg.wandb and step_idx % 50 == 0:
                wandb.log({"train/loss": step_loss, "info/lr": self.optimizer.param_groups[0]["lr"]}, step=self.global_step)

        return total_loss / steps, {}, 0.0

    @torch.inference_mode()
    def validate(self, epoch):
        self.model.eval()
        val_metrics = defaultdict(list)

        inferer = SlidingWindowInferer(roi_size=self.cfg.patch_size, sw_batch_size=self.cfg.val_sw_batch_size, overlap=self.cfg.val_sw_overlap, mode="constant")

        def diffusion_sampling(condition):
            shape = (condition.shape[0], 1, condition.shape[2], condition.shape[3], condition.shape[4])
            sampled_images = self.diffusion.p_sample_loop(self.model, shape, condition=condition, clip_denoised=True)
            return sampled_images

        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            if i not in self.val_indices_to_run:
                continue
            mri = batch["mri"][tio.DATA].to(self.device)
            ct = batch["ct"][tio.DATA].to(self.device)
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]
            pad_offset = int(batch["pad_offset"][0]) if "pad_offset" in batch else 0

            # Scale MRI to [-1, 1] for MC-DDPM inference
            mri_scaled = mri * 2.0 - 1.0

            with torch.amp.autocast("cuda"):
                pred = inferer(mri_scaled, diffusion_sampling)

            # Scale prediction back to [0, 1]
            pred = torch.clamp((pred + 1.0) / 2.0, 0.0, 1.0)

            pred_unpad = unpad(pred, orig_shape, pad_offset)
            ct_unpad = unpad(ct, orig_shape, pad_offset)
            mri_unpad = unpad(mri, orig_shape, pad_offset)

            mask_unpad = None
            if getattr(self.cfg, "val_body_mask", False) and "prob_map" in batch:
                mask_unpad = unpad(batch["prob_map"][tio.DATA].to(self.device), orig_shape, pad_offset)

            met, body_met = self._compute_val_metrics(pred_unpad, ct_unpad, mask_unpad)
            for k, v in met.items():
                val_metrics[k].append(v)
            if body_met is not None:
                for k, v in body_met.items():
                    val_metrics[f"body_{k}"].append(v)

            if i in self.val_viz_indices and self.cfg.wandb:
                from common.utils import visualize_lite
                viz_metrics = {k: met[k] for k in ("ssim", "psnr", "mae_hu") if k in met}
                viz_body = {k: body_met[k] for k in ("ssim", "psnr", "mae_hu") if body_met and k in body_met} or None
                visualize_lite(pred_unpad, ct_unpad, mri_unpad, subj_id, orig_shape, self.global_step, epoch, log_name=f"viz/val_{i}", metrics=viz_metrics, body_metrics=viz_body)

        avg_met = {k: np.mean(v) for k, v in val_metrics.items() if not k.startswith("body_")}
        avg_body = {k[5:]: np.mean(v) for k, v in val_metrics.items() if k.startswith("body_")}
        if self.cfg.wandb:
            wandb.log({f"val/{k}": v for k, v in avg_met.items()}, step=self.global_step)
            if avg_body:
                wandb.log({f"val_body/{k}": v for k, v in avg_body.items()}, step=self.global_step)

        return avg_met

    def _save_checkpoint_wandb(self, epoch, is_last=False):
        if self.cfg.wandb and wandb.run and wandb.run.dir:
            save_dir = wandb.run.dir
        else:
            save_dir = os.path.join(self.gpfs_root, "results", "models", "mcddpm")
        os.makedirs(save_dir, exist_ok=True)
        filename = "checkpoint_last.pt" if is_last else f"{self.run_name}_epoch{epoch:05d}.pt"
        path = os.path.join(save_dir, filename)
        self.save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler, epoch, path)

    def train(self):
        print(f"[{self.prefix}] 🏁 Starting Loop")
        self.global_start_time = time.time()

        if self.cfg.sanity_check and not self.cfg.resume_wandb_id:
            print(f"[{self.prefix}] Running sanity check...")
            avg_met = self.validate(0)
            tqdm.write(f"Ep -1 | Val MAE HU: {avg_met.get('mae_hu', 0):.1f} | SSIM: {avg_met.get('ssim', 0):.4f}")

        for epoch in range(self.start_epoch, self.cfg.total_epochs):
            ep_start = time.time()
            loss, _, gn = self.train_epoch(epoch)

            if epoch % self.cfg.val_interval == 0 or (epoch + 1) == self.cfg.total_epochs:
                avg_met = self.validate(epoch)
                tqdm.write(f"Ep {epoch} | Train: {loss:.4f} | Val MAE HU: {avg_met.get('mae_hu', 0):.1f} | SSIM: {avg_met.get('ssim', 0):.4f}")

            self._save_checkpoint_wandb(epoch, is_last=True)
            if epoch % self.cfg.model_save_interval == 0:
                self._save_checkpoint_wandb(epoch)

        if self.cfg.wandb:
            wandb.finish()
