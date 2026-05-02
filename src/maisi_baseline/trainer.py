import copy
import gc
import json
import os
import random

# Add project root to path
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
import torchio as tio
from monai.bundle import ConfigParser
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.utils import copy_model_state
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import Config
from common.data import DataPreprocessing, build_tio_subjects, get_augmentations, get_region_key
from common.trainer_base import BaseTrainer
from common.utils import anatomix_normalize, count_parameters, unpad, visualize_lite


class MAISITrainer(BaseTrainer):
    def __init__(self, config_dict):
        # 1. Config Setup
        from common.config import DEFAULT_CONFIG

        full_conf = copy.deepcopy(DEFAULT_CONFIG)
        full_conf.update(config_dict)
        cfg = Config(full_conf)
        super().__init__(cfg, prefix="MAISI")

        # 2. Setup Components
        self._find_subjects()
        self._stage_data_local()
        self._setup_models()
        if getattr(self.cfg, "preencoded_latents_dir", None):
            self.cfg.augment = False
            print(f"[{self.prefix}] 💾 Pre-encoded latents enabled. Augmentation disabled.")
            self._pre_encode_all()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 7. Model Summary Logging
        models_to_log = {"ControlNet (Trainable)": self.controlnet, "UNet (Frozen)": self.unet, "Autoencoder (Frozen)": self.autoencoder}
        self._log_model_summary(models_to_log)

        extra_modules = {"unet": self.unet, "autoencoder": self.autoencoder}
        self._load_resume(self.controlnet, self.optimizer, self.scheduler, self.scaler, extra_modules=extra_modules)

    def _setup_models(self):
        print(f"[{self.prefix}] 🏗️ Building MAISI Models")
        with open(self.cfg.network_config_path, "r") as f:
            model_def = json.load(f)

        # Adapt ControlNet for 1-channel MRI conditioning (original trained on 8-ch segmentation)
        model_def["controlnet_def"]["conditioning_embedding_in_channels"] = 1
        # Use moderate splits; sliding window will handle overall volume size
        model_def["autoencoder_def"]["num_splits"] = 8

        parser = ConfigParser()
        parser.update(model_def)

        # 1. VAE (Frozen)
        self.autoencoder = parser.get_parsed_content("autoencoder_def", instantiate=True).to(self.device)
        ae_ckpt = torch.load(self.cfg.autoencoder_path, map_location=self.device, weights_only=False)
        self.autoencoder.load_state_dict(ae_ckpt["unet_state_dict"] if "unet_state_dict" in ae_ckpt else ae_ckpt)
        self.autoencoder.eval()
        for p in self.autoencoder.parameters():
            p.requires_grad = False

        if getattr(self.cfg, "vae_compile", False):
            self.autoencoder.encode_stage_2_inputs = torch.compile(self.autoencoder.encode_stage_2_inputs, mode="default")
            print(f"[{self.prefix}] ⚡ Compiled VAE encoder (fixed 320x256x192 patches)")

        # 2. Denoising UNet (Frozen)
        self.unet = parser.get_parsed_content("diffusion_unet_def", instantiate=True).to(self.device)
        unet_ckpt = torch.load(self.cfg.diffusion_path, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(unet_ckpt["unet_state_dict"], strict=False)

        # Automatic Scale Factor
        if "scale_factor" in unet_ckpt:
            self.scale_factor = unet_ckpt["scale_factor"]
            if isinstance(self.scale_factor, torch.Tensor):
                self.scale_factor = self.scale_factor.to(self.device)
            else:
                self.scale_factor = torch.tensor(self.scale_factor, device=self.device)
        else:
            self.scale_factor = torch.tensor(1.0, device=self.device)

        print(f"[{self.prefix}] 📈 Scale Factor: {self.scale_factor.item():.6f}")
        self.unet.eval()
        for p in self.unet.parameters():
            p.requires_grad = False

        # 3. ControlNet (Trainable)
        self.controlnet = parser.get_parsed_content("controlnet_def", instantiate=True).to(self.device)
        copy_model_state(self.controlnet, self.unet.state_dict())

        self.noise_scheduler = parser.get_parsed_content("noise_scheduler", instantiate=True)

        # 4. Compile Mode (ControlNet, UNet)
        if getattr(self.cfg, "compile_mode", None):
            print(f"[{self.prefix}] ⚡ Compiling models (ControlNet, UNet)...")
            self.controlnet = torch.compile(self.controlnet)
            self.unet = torch.compile(self.unet)

        tot, train = count_parameters(self.controlnet)
        print(f"[{self.prefix}] ControlNet Params: Total={tot:,} | Trainable={train:,}")

        # 5. Teacher Model for Dice Validation
        self.teacher_model = None
        if getattr(self.cfg, "validate_dice", False):
            print(f"[{self.prefix}] 👨‍🏫 Initializing Teacher for Dice Validation...")
            try:
                from anatomix.segmentation.segmentation_utils import load_model_v1_2
                self.teacher_model = load_model_v1_2(pretrained_ckpt=self.cfg.teacher_weights_path, n_classes=self.cfg.n_classes - 1, device=self.device, compile_model=False)
                self.teacher_model.to(device=self.device, dtype=torch.bfloat16)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False
                print(f"[{self.prefix}] ✅ Teacher initialized.")
            except Exception as e:
                print(f"[{self.prefix}] ❌ Failed to init Teacher: {e}")

    @staticmethod
    def _resample_subject(subject):
        """Normalize and pad subject to nearest multiple of 32 for VAE compatibility."""
        subject = tio.ToCanonical()(subject)
        subject["original_spacing"] = torch.from_numpy(np.array(subject["ct"].spacing)).float()
        subject["original_spatial_shape"] = torch.tensor(subject["ct"].spatial_shape)

        # STRICT MAISI NORMALIZATION
        subject["ct"].set_data(anatomix_normalize(subject["ct"].data, clip_range=(-1000, 1000)))
        subject["mri"].set_data(anatomix_normalize(subject["mri"].data, percentile_range=(0.0, 99.5)))

        spatial_keys = ["ct", "mri"]
        if "prob_map" in subject:
            spatial_keys.append("prob_map")
        if "seg" in subject:
            spatial_keys.append("seg")

        current_shape = subject["ct"].spatial_shape
        padding_params = []
        for dim in current_shape:
            target = max(32, (int(dim) + 31) // 32 * 32)
            padding_params.extend([0, int(target - dim)])

        if any(p > 0 for p in padding_params):
            subject = tio.Pad(padding_params, padding_mode=0, include=spatial_keys)(subject)

        return subject

    def _pre_encode_all(self):
        """Pre-encode all train+val CT volumes to VAE latent space and cache to disk."""
        cache_dir = self.cfg.preencoded_latents_dir
        os.makedirs(cache_dir, exist_ok=True)

        all_subjects = list(dict.fromkeys(self.train_subjects + self.val_subjects))
        to_encode = [s for s in all_subjects if not os.path.exists(os.path.join(cache_dir, f"{s}_ct_latent.pt"))]
        print(f"[{self.prefix}] Pre-encoding: {len(to_encode)} to encode, {len(all_subjects) - len(to_encode)} already cached.")

        if not to_encode:
            return

        subj_objs = build_tio_subjects(self.cfg.root_dir, to_encode, use_weighted_sampler=False, load_seg=False)
        for subj_obj in tqdm(subj_objs, desc="Pre-encoding CT latents"):
            sid = subj_obj["subj_id"]
            subj_obj = MAISITrainer._resample_subject(subj_obj)
            ct = subj_obj["ct"].data.unsqueeze(0).to(self.device)
            ct_emb = self._encode_sliding_window(ct) * self.scale_factor
            torch.save(ct_emb.cpu(), os.path.join(cache_dir, f"{sid}_ct_latent.pt"))
            del ct, ct_emb
            gc.collect()
            torch.cuda.empty_cache()

        print(f"[{self.prefix}] Pre-encoding complete.")

    def _setup_data(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # Safeguard: Disable augmentation and enforce 1 step/epoch if running on a single subject (SSO)
        if len(self.train_subjects) == 1:
            if self.cfg.augment:
                print(f"[{self.prefix}] ℹ️ Only 1 subject found. Forcefully disabling augmentations for SSO.")
                self.cfg.augment = False
            if self.cfg.steps_per_epoch > 1:
                print(f"[{self.prefix}] ℹ️ SSO Mode detected: Setting steps_per_epoch=1.")
                self.cfg.steps_per_epoch = 1

        # Full Volume Training Data Pipeline
        train_objs = build_tio_subjects(self.cfg.root_dir, self.train_subjects, use_weighted_sampler=False, load_seg=False)

        transforms = tio.Compose([MAISITrainer._resample_subject, get_augmentations()]) if self.cfg.augment else tio.Compose([MAISITrainer._resample_subject])
        train_ds = tio.SubjectsDataset(train_objs, transform=transforms)

        self.train_loader = DataLoader(train_ds, batch_size=self.cfg.batch_size, shuffle=True, num_workers=self.cfg.dataloader_num_workers, collate_fn=lambda x: x)
        self.train_iter = self._inf_gen(self.train_loader)

        # Val Loader
        # NOTE: use_weighted_sampler here only loads mask.nii.gz as prob_map in the subject —
        # it does NOT do weighted patch sampling (that's training-only via Queue).
        load_seg = getattr(self.cfg, "validate_dice", False)
        val_objs = build_tio_subjects(self.cfg.root_dir, self.val_subjects, load_seg=load_seg, use_weighted_sampler=getattr(self.cfg, "val_body_mask", False))
        val_ds = tio.SubjectsDataset(val_objs, transform=MAISITrainer._resample_subject)
        self.val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: x)

        # Stratified Validation Sampling
        rng = random.Random(self.cfg.seed)
        region_to_indices = defaultdict(list)
        for idx, subj_id in enumerate(self.val_subjects):
            region = get_region_key(subj_id)
            region_to_indices[region].append(idx)

        # Determine indices to run metrics on
        if self.cfg.full_val:
            self.val_indices_to_run = set(range(len(self.val_subjects)))
            # For visualization (pick 2 per region)
            viz_indices = []
            for region, indices in region_to_indices.items():
                sampled = rng.sample(indices, min(self.cfg.viz_limit, len(indices)))
                viz_indices.extend(sampled)
            self.val_viz_indices = set(viz_indices)
        else:
            # Reduced validation: only 1 per region
            reduced_indices = []
            for region, indices in region_to_indices.items():
                sampled = rng.sample(indices, 1)
                reduced_indices.extend(sampled)
            self.val_indices_to_run = set(reduced_indices)
            self.val_viz_indices = set(reduced_indices)  # Visualize all run volumes if reduced

        print(f"[{self.prefix}] 📊 Validation strategy: full_val={self.cfg.full_val}, running on {len(self.val_indices_to_run)}/{len(self.val_subjects)} volumes.")

    def _setup_opt(self):
        self.optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=self.cfg.lr)

        # Standardize: Use fixed steps_per_epoch for comparability with other models
        total_steps = self.cfg.total_epochs * self.cfg.steps_per_epoch
        print(f"[{self.prefix}] 📉 Learning rate scheduler: total_steps={total_steps} (Epochs={self.cfg.total_epochs}, Steps/Epoch={self.cfg.steps_per_epoch})")

        # Original uses PolynomialLR with power 2.0
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_steps, power=2.0)
        self.scaler = torch.amp.GradScaler("cuda")

    def _get_diffusion_target(self, ct_emb, noise, timesteps):
        """Compute the diffusion training target based on the noise scheduler type."""
        if isinstance(self.noise_scheduler, RFlowScheduler):
            return ct_emb - noise
        elif self.noise_scheduler.prediction_type == DDPMPredictionType.EPSILON:
            return noise
        elif self.noise_scheduler.prediction_type == DDPMPredictionType.SAMPLE:
            return ct_emb
        elif self.noise_scheduler.prediction_type == DDPMPredictionType.V_PREDICTION:
            return self.noise_scheduler.get_velocity(ct_emb, noise, timesteps)
        else:
            raise ValueError(f"Unsupported noise scheduler: {type(self.noise_scheduler)}")

    @torch.no_grad()
    def _encode_sliding_window(self, ct_tensor):
        """Encodes full volume CT [-1000, 1000] HU into VAE Latent using sliding window."""
        ct_hu = (ct_tensor * 2000.0) - 1000.0
        ct_norm = torch.clamp((ct_hu + 1000.0) / 2000.0, 0.0, 1.0)

        # ROI [384, 352, 256]
        # Final optimized ROI for A40: Covers ~85% of subjects in 1 patch.
        roi_size = [384, 352, 256]
        overlap = self.cfg.val_sw_overlap

        inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=self.cfg.val_sw_batch_size,
            overlap=overlap,
            mode="gaussian",
            sw_device=self.device,
            device=torch.device("cpu"),
        )

        vol_shape = ct_norm.shape[2:]
        n_patches = 1
        for v, r in zip(vol_shape, roi_size):
            n_patches *= int(np.ceil((v - r) / (r * (1 - overlap))) + 1) if v > r else 1

        if n_patches > 1:
            print(f"[{self.prefix}] 🪟 Encoding Sliding Window: {vol_shape} -> {n_patches} patches (Overlap={overlap})")

        t_sw_start = time.time()
        with torch.amp.autocast("cuda"):
            latent = inferer(ct_norm.to(self.device), self.autoencoder.encode_stage_2_inputs)
        t_sw_dur = time.time() - t_sw_start

        if n_patches > 1:
            print(f"[{self.prefix}] ✅ Encoding SW Finished in {t_sw_dur:.2f}s")

        if self.cfg.wandb:
            wandb.log({"info/sw_encoding_patches": n_patches, "profiling/sw_encoding_ms": t_sw_dur * 1000}, step=self.global_step)

        return latent.to(self.device)

    @torch.no_grad()
    def _decode_sliding_window(self, z):
        """Decode latent back to image space using sliding window to prevent OOM."""
        # VAE latent ROI [96, 88, 64] -> Image space [384, 352, 256]
        roi_size = [96, 88, 64]
        overlap = self.cfg.val_sw_overlap

        inferer = SlidingWindowInferer(
            roi_size=roi_size,
            sw_batch_size=1,
            overlap=overlap,
            mode="gaussian",
            sw_device=self.device,
            device=torch.device("cpu"),
        )

        # Log Patch Count
        vol_shape = z.shape[2:]
        n_patches = 1
        for v, r in zip(vol_shape, roi_size):
            n_patches *= int(np.ceil((v - r) / (r * (1 - overlap))) + 1) if v > r else 1

        if n_patches > 1:
            print(f"[{self.prefix}] 🪟 Decoding Sliding Window: {vol_shape} -> {n_patches} patches (Overlap={overlap})")

        t_sw_start = time.time()
        with torch.amp.autocast("cuda"):
            recon = inferer(z, self.autoencoder.decode_stage_2_outputs)
        t_sw_dur = time.time() - t_sw_start

        if n_patches > 1:
            print(f"[{self.prefix}] ✅ Decoding SW Finished in {t_sw_dur:.2f}s")

        if self.cfg.wandb:
            wandb.log({"info/sw_decoding_patches": n_patches, "profiling/sw_decoding_ms": t_sw_dur * 1000}, step=self.global_step)

        return recon.to(self.device)

    @torch.no_grad()
    def _decode(self, z):
        """Decode latent back to image space. Automatically uses sliding window if latent is large."""
        latent = z / self.scale_factor

        # Latent shape threshold [96, 88, 64]
        if any(s > limit for s, limit in zip(latent.shape[2:], [96, 88, 64])):
            recon = self._decode_sliding_window(latent)
        else:
            with torch.amp.autocast("cuda"):
                recon = self.autoencoder.decode_stage_2_outputs(latent)

        recon = torch.clamp(recon, 0.0, 1.0)
        return recon.to(self.device)

    @torch.no_grad()
    def _sample(self, mr, spacing, num_steps=10):
        """Iterative denoising to sample synthetic CT latent."""
        self.controlnet.eval()
        latent_shape = (1, 4, mr.shape[2] // 4, mr.shape[3] // 4, mr.shape[4] // 4)
        latents = torch.randn(latent_shape, device=self.device).half()
        mr_h = mr.half()
        sp_h = spacing.half()

        try:
            num_voxels = int(torch.prod(torch.tensor(latent_shape[2:])).item())
            self.noise_scheduler.set_timesteps(num_inference_steps=num_steps, input_img_size_numel=num_voxels)
        except (TypeError, AttributeError):
            self.noise_scheduler.set_timesteps(num_inference_steps=num_steps)

        all_timesteps = self.noise_scheduler.timesteps.to(self.device)
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=self.device)))

        with torch.amp.autocast("cuda"):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                t_tensor = torch.tensor([t], device=self.device).float()
                # Modality 1 corresponds to CT (refer to modality_mapping.json)
                class_labels = torch.ones(latents.shape[0], dtype=torch.long, device=self.device)
                down_res, mid_res = self.controlnet(x=latents, timesteps=t_tensor, controlnet_cond=mr_h, class_labels=class_labels)
                model_output = self.unet(x=latents, timesteps=t_tensor, spacing_tensor=sp_h, class_labels=class_labels, down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res)
                latents, _ = self.noise_scheduler.step(model_output, t, latents, next_t)

        return latents.float()

    def train_epoch(self, epoch):
        self.controlnet.train()
        self.unet.eval()  # STRICT PARITY

        total_loss = 0
        total_grad = 0

        # Use fixed steps per epoch for comparability with amix/unet
        steps = self.cfg.steps_per_epoch
        pbar = tqdm(range(steps), desc=f"Ep {epoch}", leave=False, dynamic_ncols=True)

        prof = self.cfg.enable_profiling

        def _t():
            if prof:
                torch.cuda.synchronize()
            return time.time()

        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0
            step_t_load = 0.0
            step_t_encode = 0.0
            step_t_noise = 0.0
            step_t_predict = 0.0
            step_t_backward = 0.0

            for _ in range(self.cfg.accum_steps):
                t0 = _t()
                batch_list = next(self.train_iter)
                step_t_load += _t() - t0

                # Manually batch the list of torchio subjects
                mri_list = [subj["mri"]["data"] for subj in batch_list]
                ct_list = [subj["ct"]["data"] for subj in batch_list]

                # Ensure dimensions match before stacking
                mr = torch.stack(mri_list).to(self.device)
                ct = torch.stack(ct_list).to(self.device)

                # Modular Synchronized CutOut (via BaseTrainer)
                # Note: Maisi uses stack of single-channel tensors: (B, 1, D, H, W)
                mr, ct, _ = self.apply_cutout(mr, ct)

                # STRICT PARITY: The original author does NOT adjust the spacing tensor after resizing the image
                spacing = torch.stack([subj["original_spacing"] for subj in batch_list]).to(self.device) * 100.0

                # 1. Encode CT to latent (load from cache or encode on-the-fly)
                t0 = _t()
                if getattr(self.cfg, "preencoded_latents_dir", None):
                    ct_emb = torch.cat(
                        [torch.load(os.path.join(self.cfg.preencoded_latents_dir, f"{subj['subj_id']}_ct_latent.pt"), map_location=self.device, weights_only=False) for subj in batch_list], dim=0
                    )
                else:
                    ct_emb = self._encode_sliding_window(ct) * self.scale_factor
                step_t_encode += _t() - t0

                with torch.amp.autocast("cuda"):
                    # 2. Diffusion forward: sample timestep and corrupt latent
                    t0 = _t()
                    noise = torch.randn_like(ct_emb).to(self.device)
                    if hasattr(self.noise_scheduler, "sample_timesteps"):
                        timesteps = self.noise_scheduler.sample_timesteps(ct_emb)
                    else:
                        timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (ct_emb.shape[0],), device=ct_emb.device).long()
                    noisy_latent = self.noise_scheduler.add_noise(original_samples=ct_emb, noise=noise, timesteps=timesteps)
                    step_t_noise += _t() - t0

                    # 3. Predict velocity (ControlNet conditions the frozen UNet)
                    t0 = _t()
                    class_labels = torch.ones(noisy_latent.shape[0], dtype=torch.long, device=self.device)
                    down_res, mid_res = self.controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=mr, class_labels=class_labels)
                    model_output = self.unet(
                        x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing, class_labels=class_labels, down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res
                    )
                    step_t_predict += _t() - t0

                    target = self._get_diffusion_target(ct_emb, noise, timesteps)
                    loss = F.l1_loss(model_output.float(), target.float())
                    loss = loss / self.cfg.accum_steps

                t0 = _t()
                self.scaler.scale(loss).backward()
                step_t_backward += _t() - t0
                step_loss += loss.item()

                if step_idx == 0 and self.cfg.wandb:
                    subj_id = batch_list[0]["subj_id"] if hasattr(batch_list[0], "subj_id") else getattr(batch_list[0], "name", None)

                    # FREE MEMORY for visualization decode
                    # model_output, down_res, mid_res are the largest consumers
                    del model_output, down_res, mid_res, noisy_latent, target, noise
                    torch.cuda.empty_cache()

                    decoded_ct_norm = self._decode(ct_emb[0:1])
                    decoded_ct = (decoded_ct_norm * 2000.0 - 1000.0 + 1024.0) / 2048.0
                    self._log_training_patch(mr, ct, decoded_ct, self.global_step, step_idx, subj_id=subj_id)

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            if self.cfg.wandb and self.global_step % 5 == 0:
                log_dict = {
                    "train/loss": step_loss,
                    "train/grad_norm": grad_norm.item(),
                    "info/lr": self.optimizer.param_groups[0]["lr"],
                }
                if prof:
                    s = self.cfg.accum_steps
                    log_dict.update(
                        {
                            "profiling/load_ms": step_t_load / s * 1000,
                            "profiling/encode_ms": step_t_encode / s * 1000,
                            "profiling/noise_ms": step_t_noise / s * 1000,
                            "profiling/predict_ms": step_t_predict / s * 1000,
                            "profiling/backward_ms": step_t_backward / s * 1000,
                        }
                    )
                wandb.log(log_dict, step=self.global_step)

            pbar.set_postfix({"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"})

        return total_loss / steps, total_grad / steps

    @torch.no_grad()
    def validate(self, epoch):
        self.controlnet.eval()
        val_metrics = defaultdict(list)
        val_subject_ids = []
        t_inf_start = time.time()
        prof = self.cfg.enable_profiling

        # Clear training memory before validation
        gc.collect()
        torch.cuda.empty_cache()

        val_ps = getattr(self.cfg, "val_patch_size", self.cfg.patch_size)

        for i, batch_list in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            if i not in self.val_indices_to_run:
                continue

            # Since batch_size=1, batch_list has 1 element
            subj = batch_list[0]
            mr = subj["mri"]["data"].unsqueeze(0).to(self.device)
            ct = subj["ct"]["data"].unsqueeze(0).to(self.device)
            # STRICT PARITY: The original author does NOT adjust the spacing tensor after resizing the image
            spacing = subj["original_spacing"].unsqueeze(0).to(self.device) * 100.0
            orig_shape = subj["original_spatial_shape"].tolist()
            subj_id = subj["subj_id"] if hasattr(subj, "subj_id") else getattr(subj, "name", "unknown")

            # Generate synthetic CT latent
            if prof:
                torch.cuda.synchronize()
                t_sample = time.time()
            pred_latent = self._sample(mr, spacing, num_steps=self.cfg.num_inference_steps)
            if prof:
                torch.cuda.synchronize()
                val_metrics["avg_sample_time"].append(time.time() - t_sample)

            # Decode to [0, 1] where 0=-1000, 1=1000
            if prof:
                torch.cuda.synchronize()
                t_decode = time.time()
            pred_ct_norm = self._decode(pred_latent)
            if prof:
                torch.cuda.synchronize()
                val_metrics["avg_decode_time"].append(time.time() - t_decode)

            # Convert GT back to HU (Original GT is in [0, 1] mapping to [-1000, 1000])
            gt_hu_raw = (ct * 2000.0) - 1000.0

            # For accurate metric comparison, we clamp GT to the same [-1000, 1000] range used by the model
            gt_hu = torch.clamp(gt_hu_raw, -1000.0, 1000.0)
            pred_hu = (pred_ct_norm * 2000.0) - 1000.0

            # Proxy diffusion loss (on resampled latent space)
            if getattr(self.cfg, "preencoded_latents_dir", None):
                ct_emb = torch.load(os.path.join(self.cfg.preencoded_latents_dir, f"{subj_id}_ct_latent.pt"), map_location=self.device, weights_only=False)
            else:
                ct_emb = self._encode_sliding_window(ct) * self.scale_factor
            noise = torch.randn_like(ct_emb)
            if hasattr(self.noise_scheduler, "sample_timesteps"):
                timesteps = self.noise_scheduler.sample_timesteps(ct_emb)
            else:
                timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (ct_emb.shape[0],), device=ct_emb.device).long()
            noisy_latent = self.noise_scheduler.add_noise(original_samples=ct_emb, noise=noise, timesteps=timesteps)
            with torch.amp.autocast("cuda"):
                class_labels = torch.ones(noisy_latent.shape[0], dtype=torch.long, device=self.device)
                down_res, mid_res = self.controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=mr, class_labels=class_labels)
                model_output = self.unet(
                    x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing, class_labels=class_labels, down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res
                )

            val_target = self._get_diffusion_target(ct_emb, noise, timesteps)
            val_metrics["loss"].append(F.l1_loss(model_output.float(), val_target.float()).item())

            # Evaluate standard metrics in matched [0, 1] range (referring to [-1000, 1000])
            pred_matched = pred_ct_norm
            gt_matched = (gt_hu + 1000.0) / 2000.0

            # Unpad back to original spatial shape for final metrics
            pred_unpad = unpad(pred_matched.float(), orig_shape)
            gt_unpad = unpad(gt_matched.float(), orig_shape)
            pred_hu_unpad = unpad(pred_hu.float(), orig_shape)
            gt_hu_unpad = unpad(gt_hu.float(), orig_shape)

            # compute_metrics expects (B, C, D, H, W). Use range 1.0 since data is normalized [0, 1]
            # prob_map (if loaded) stays at original resolution — same as orig_shape, no interpolation needed
            mask_unpad = None
            if getattr(self.cfg, "val_body_mask", False) and "prob_map" in subj:
                mask_unpad = unpad(subj["prob_map"]["data"].unsqueeze(0).to(self.device), orig_shape)

            met, body_met = self._compute_val_metrics(pred_unpad, gt_unpad, mask_unpad)

            # NVIDIA MAISI MAE HU Metric Logic (masked HU — separate from body mask)
            hu_mask = gt_hu_unpad > -900
            if hu_mask.any():
                mae_hu = torch.mean(torch.abs(pred_hu_unpad[hu_mask] - gt_hu_unpad[hu_mask])).item()
            else:
                mae_hu = 0.0

            met["mae_hu"] = mae_hu

            # Validation Dice
            if getattr(self.cfg, "validate_dice", False) and self.teacher_model is not None and "seg" in subj:
                from common.loss import get_class_dice
                seg = subj["seg"]["data"].unsqueeze(0).to(self.device)
                seg_unpad = unpad(seg, orig_shape)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    pred_probs = sliding_window_inference(
                        inputs=pred_unpad,
                        roi_size=(val_ps, val_ps, val_ps),
                        sw_batch_size=self.cfg.val_sw_batch_size,
                        predictor=self.teacher_model,
                        overlap=self.cfg.val_sw_overlap,
                        device=self.device,
                    )
                excl_bg = getattr(self.cfg, "dice_exclude_background", True)
                bone_idx = getattr(self.cfg, "dice_bone_idx", 5)

                class_dices, bone_dice = get_class_dice(pred_probs, seg_unpad, mask=None, bone_idx=bone_idx)
                met["dice_score_all"] = (class_dices[1:].mean() if excl_bg else class_dices.mean()).item()
                if bone_dice is not None:
                    met["dice_score_bone"] = bone_dice.item()

                if body_met is not None:
                    class_dices_body, bone_dice_body = get_class_dice(pred_probs, seg_unpad, mask=mask_unpad, bone_idx=bone_idx)
                    body_met["dice_score_all"] = (class_dices_body[1:].mean() if excl_bg else class_dices_body.mean()).item()
                    if bone_dice_body is not None:
                        body_met["dice_score_bone"] = bone_dice_body.item()

                del pred_probs, seg, seg_unpad

            for k, v in met.items():
                val_metrics[k].append(v)
            if body_met is not None:
                for k, v in body_met.items():
                    val_metrics[f"body_{k}"].append(v)
            val_subject_ids.append(subj_id)

            # Save predictions for ALL subjects (overwrite "last" each validation run)
            save_path = None
            if getattr(self.cfg, "save_val_volumes", True):
                base_dir = self.local_run_dir if (self.cfg.wandb and self.local_run_dir) else os.path.join(self.cfg.prediction_dir, self.run_name)

                pred_np = pred_hu_unpad.float().cpu().numpy().squeeze()
                affine = subj["ct"]["affine"]
                if hasattr(affine, "cpu"):
                    affine = affine.cpu().numpy()
                else:
                    affine = np.array(affine)
                nii = nib.Nifti1Image(pred_np, affine)

                last_dir = os.path.join(base_dir, "predictions", "last")
                os.makedirs(last_dir, exist_ok=True)
                save_path = os.path.join(last_dir, f"pred_{subj_id}.nii.gz")
                nib.save(nii, save_path)

                val_save_interval = getattr(self.cfg, "val_save_interval", 100)
                if val_save_interval > 0 and epoch % val_save_interval == 0:
                    epoch_dir = os.path.join(base_dir, "predictions", f"epoch_{epoch}")
                    os.makedirs(epoch_dir, exist_ok=True)
                    nib.save(nii, os.path.join(epoch_dir, f"pred_{subj_id}.nii.gz"))

            # Viz (only for selected subjects)
            if i in self.val_viz_indices and self.cfg.wandb:
                viz_metrics = {k: met[k] for k in ("ssim", "psnr", "mae_hu", "dice_score_all", "dice_score_bone") if k in met}
                viz_body = {k: body_met[k] for k in ("ssim", "psnr", "mae_hu", "dice_score_all", "dice_score_bone") if body_met and k in body_met} or None
                visualize_lite(pred_matched, ct, mr, subj_id, orig_shape, self.global_step, epoch, log_name=f"viz/val_{i}", metrics=viz_metrics, body_metrics=viz_body)

            # Cleanup memory after each subject
            del mr, ct, pred_latent, pred_ct_norm, gt_hu_raw, gt_hu, pred_hu, pred_matched, gt_matched, pred_unpad, gt_unpad, pred_hu_unpad, gt_hu_unpad
            gc.collect()
            torch.cuda.empty_cache()

        n_val = len(self.val_indices_to_run)
        extra = {"avg_inference_time": (time.time() - t_inf_start) / n_val if n_val > 0 else 0.0}
        avg_met = self._log_val_metrics(val_metrics, extra=extra, subject_ids=val_subject_ids)
        return avg_met

    def _log_training_patch(self, mr, ct, decoded_ct, step, step_idx, subj_id=None):
        """Visualizes [MR, GT, Decoded GT Latent] for full-volume training."""
        mr_img = mr[0, 0].cpu().numpy()
        gt_ct = ct[0, 0].cpu().numpy()
        dec_ct = decoded_ct[0, 0].cpu().numpy()

        cx, cy, cz = np.array(mr_img.shape) // 2
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))

        def plot_row(row_idx, vol, title):
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f"{title} Ax")
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f"{title} Sag")
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f"{title} Cor")

        plot_row(0, mr_img, "MRI (Cond)")
        plot_row(1, gt_ct, "GT CT")
        plot_row(2, dec_ct, "Decoded Latent")

        for ax in axes.flatten():
            ax.axis("off")
        plt.tight_layout()
        if self.cfg.wandb:
            wandb.log({"train/full_vol_viz": wandb.Image(fig, caption=f"Subj: {subj_id}")}, step=step)
        plt.close(fig)

    def train(self):
        print(f"[{self.prefix}] 🏁 Starting Loop: {self.cfg.total_epochs} epochs")
        self.global_start_time = time.time()

        for epoch in range(self.start_epoch, self.cfg.total_epochs):
            ep_start = time.time()
            loss, gn = self.train_epoch(epoch)

            val_duration = 0.0
            if epoch % self.cfg.val_interval == 0:
                val_start = time.time()
                avg_met = self.validate(epoch)
                val_duration = time.time() - val_start
                print(f"Ep {epoch} | Val Loss: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | MAE HU: {avg_met.get('mae_hu', 0):.1f}")

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + (self.elapsed_time_at_resume if self.elapsed_time_at_resume else 0)

            if self.cfg.wandb:
                wandb.log(
                    {
                        "train/total": loss,
                        "info/grad_norm": gn,
                        "info/epoch_duration": ep_duration,
                        "info/val_duration": val_duration,
                        "info/cumulative_time": cumulative_time,
                        "info/lr": self.optimizer.param_groups[0]["lr"],
                        "info/global_step": self.global_step,
                        "info/epoch": epoch,
                        "info/samples_seen": self.samples_seen,
                    },
                    step=self.global_step,
                )

            self._save_maisi_checkpoint(epoch, is_last=True)
            if epoch % self.cfg.model_save_interval == 0:
                self._save_maisi_checkpoint(epoch)

        if self.cfg.wandb:
            wandb.finish()

    def _save_maisi_checkpoint(self, epoch, is_last=False):
        save_dir = self.local_run_dir if (self.cfg.wandb and self.local_run_dir) else os.path.join(self.gpfs_root, "results", "models", "maisi")
        filename = "checkpoint_last.pt" if is_last else f"maisi_epoch{epoch:05d}.pt"
        self.save_checkpoint(
            self.controlnet,
            self.optimizer,
            self.scheduler,
            self.scaler,
            epoch,
            os.path.join(save_dir, filename),
            extra_state={"scale_factor": self.scale_factor},
        )


if __name__ == "__main__":
    pass
