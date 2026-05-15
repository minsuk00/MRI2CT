import copy
import gc
import json
import os
import sys
import time
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from monai.bundle import ConfigParser
from monai.data import DataLoader, PersistentDataset
from monai.data.utils import pickle_hashing
from monai.inferers import SlidingWindowInferer, sliding_window_inference
from monai.networks.schedulers import RFlowScheduler
from monai.networks.schedulers.ddpm import DDPMPredictionType
from monai.networks.utils import copy_model_state
from tqdm import tqdm

import wandb

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.config import Config
from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_gpu_transforms,
    gpu_augment_batch,
)
from common.trainer_base import BaseTrainer, StepTimer
from common.utils import count_parameters, unpad, visualize_lite


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
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 7. Model Summary Logging
        models_to_log = {"ControlNet (Trainable)": self.controlnet, "UNet (Frozen)": self.unet, "Autoencoder (Frozen)": self.autoencoder}
        self._log_model_summary(models_to_log)

        self._load_resume(self.controlnet, self.optimizer, self.scheduler)

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
        self.unet.load_state_dict(unet_ckpt["unet_state_dict"], strict=True)

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

        # 4. ControlNet/UNet are NOT torch.compile'd: subjects have different padded
        # sizes, which would force constant recompiles. Only the VAE encoder is
        # compiled (above), since SW patches always have a fixed shape.

        tot, train = count_parameters(self.controlnet)
        print(f"[{self.prefix}] ControlNet Params: Total={tot:,} | Trainable={train:,}")

        # 5. Teacher Model for Dice Validation
        self.teacher_model = None
        if getattr(self.cfg, "validate_dice", False):
            self.teacher_model = self._setup_teacher_model(compile_model=False)

    def _setup_data(self):
        # Safeguard: Disable augmentation and enforce 1 step/epoch if running on a single subject (SSO)
        if len(self.train_subjects) == 1:
            if self.cfg.augment:
                print(f"[{self.prefix}] ℹ️ Only 1 subject found. Forcefully disabling augmentations for SSO.")
                self.cfg.augment = False
            if self.cfg.steps_per_epoch > 1:
                print(f"[{self.prefix}] ℹ️ SSO Mode detected: Setting steps_per_epoch=1.")
                self.cfg.steps_per_epoch = 1

        # Preencoded latent mode: skip on-the-fly VAE encode at train time by
        # pulling pre-computed `{subj_id}_ct_latent.pt` (already × scale_factor)
        # from GPFS. Static latents are incompatible with random MR augmentation
        # — RandAffine/RandFlip on MR would mis-register against frozen CT latents.
        self.preencoded_dir = getattr(self.cfg, "preencoded_latents_dir", None)
        if self.preencoded_dir:
            if not os.path.isdir(self.preencoded_dir):
                raise FileNotFoundError(
                    f"[MAISI] preencoded_latents_dir not found: {self.preencoded_dir}. "
                    f"Run `python src/maisi_baseline/encode_all_volumes.py --output_dir {self.preencoded_dir}` first."
                )
            if self.cfg.augment:
                raise ValueError(
                    "[MAISI] preencoded_latents_dir is set but augment=True. "
                    "Static latents require augment=False (MR augmentations would mis-register against frozen CT latents)."
                )
            print(f"[{self.prefix}] 📦 Preencoded mode: loading CT latents from {self.preencoded_dir}")

        # MONAI pipeline (full-volume MAISI: cached preproc + GPU aug, NO crop).
        # MAISI norm presets: ct_range=(-1000, 1000), mri_norm="percentile" (0–99.5).
        # res_mult=32 to satisfy the VAE's 8x downsampling × 4 patch alignment.
        load_seg = getattr(self.cfg, "validate_dice", False)
        cache_dir = default_monai_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[{self.prefix}] 💾 MONAI cache dir: {cache_dir}")

        # Train side: skip CT image entirely in preencoded mode (replaced by LoadLatentd).
        train_cached_xform = get_cached_transforms(
            patch_size=self.cfg.patch_size,
            res_mult=32,
            enforce_ras=True,
            mri_norm="percentile",
            ct_range=(-1000, 1000),
            load_seg=False,
            load_ct_image=self.preencoded_dir is None,
            load_ct_latent_from=self.preencoded_dir,
        )
        # Val side: keep CT image (needed for metrics). In preencoded mode, also
        # append LoadLatentd so the proxy diffusion loss reuses the cached latent.
        val_cached_xform = get_cached_transforms(
            patch_size=self.cfg.patch_size,
            res_mult=32,
            enforce_ras=True,
            mri_norm="percentile",
            ct_range=(-1000, 1000),
            load_seg=load_seg,
            load_ct_image=True,
            load_ct_latent_from=self.preencoded_dir,
        )

        # Train: full padded volumes; batch_size=1 → no collation pad needed.
        train_dicts = build_data_dicts(self.cfg.root_dir, self.train_subjects, load_seg=False)
        if self.preencoded_dir is not None:
            # Drop "ct" path key — train side no longer loads the CT NIfTI. Also
            # differentiates the PersistentDataset cache key from on-the-fly mode.
            train_dicts = [{k: v for k, v in d.items() if k != "ct"} for d in train_dicts]
        # `hash_transform=pickle_hashing` makes PersistentDataset include the transform
        # spec in the cache key, so preencoded vs on-the-fly modes (and any future
        # MAISI preset change) cannot poison each other's cached tensors.
        train_ds = PersistentDataset(
            data=train_dicts,
            transform=train_cached_xform,
            cache_dir=cache_dir,
            hash_transform=pickle_hashing,
        )
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.dataloader_num_workers,
            persistent_workers=True,
            pin_memory=False,
        )
        self.train_iter = self._inf_gen(self.train_loader)

        # Full-volume aug (no crop): Affine/Flip/BiasField/Gamma/Noise/Scale.
        self.gpu_transforms = get_gpu_transforms(augment=self.cfg.augment, has_seg=False)

        # Val: same cached preproc, batch_size=1, no aug.
        self.val_loader = self._build_val_loader(val_cached_xform, load_seg, cache_dir, hash_transform=pickle_hashing)

        # Stratified Validation Sampling
        if self.cfg.full_val:
            self.val_indices_to_run = set(range(len(self.val_subjects)))
            self.val_viz_indices, _ = self._stratify_val_indices(self.cfg.viz_limit)
        else:
            # Reduced validation: 1 per region; visualize all of them.
            reduced, _ = self._stratify_val_indices(1)
            self.val_indices_to_run = reduced
            self.val_viz_indices = reduced

        print(f"[{self.prefix}] 📊 Validation strategy: full_val={self.cfg.full_val}, running on {len(self.val_indices_to_run)}/{len(self.val_subjects)} volumes.")

    def _setup_opt(self):
        self.optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=self.cfg.lr)

        # Standardize: Use fixed steps_per_epoch for comparability with other models
        total_steps = self.cfg.total_epochs * self.cfg.steps_per_epoch
        print(f"[{self.prefix}] 📉 Learning rate scheduler: total_steps={total_steps} (Epochs={self.cfg.total_epochs}, Steps/Epoch={self.cfg.steps_per_epoch})")

        # Original uses PolynomialLR with power 2.0
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_steps, power=2.0)

    def _sample_noise_and_timesteps(self, ct_emb):
        """Sample noise + timesteps and produce a corrupted latent (no autocast wrap)."""
        noise = torch.randn_like(ct_emb)
        if hasattr(self.noise_scheduler, "sample_timesteps"):
            timesteps = self.noise_scheduler.sample_timesteps(ct_emb)
        else:
            timesteps = torch.randint(0, self.noise_scheduler.num_train_timesteps, (ct_emb.shape[0],), device=ct_emb.device).long()
        noisy_latent = self.noise_scheduler.add_noise(original_samples=ct_emb, noise=noise, timesteps=timesteps)
        return noise, timesteps, noisy_latent

    def _controlnet_unet_forward(self, noisy_latent, mr, spacing, timesteps):
        """Run ControlNet (cond on MR) + frozen UNet to produce model_output. Caller controls autocast."""
        class_labels = torch.ones(noisy_latent.shape[0], dtype=torch.long, device=self.device)
        down_res, mid_res = self.controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=mr, class_labels=class_labels)
        model_output = self.unet(
            x=noisy_latent,
            timesteps=timesteps,
            spacing_tensor=spacing,
            class_labels=class_labels,
            down_block_additional_residuals=down_res,
            mid_block_additional_residual=mid_res,
        )
        return model_output, down_res, mid_res

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
        # ct_tensor is already in [0,1] mapping to [-1000,1000] HU from the cached pipeline.
        ct_norm = ct_tensor.clamp(0.0, 1.0)

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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                recon = self.autoencoder.decode_stage_2_outputs(latent)

        recon = torch.clamp(recon, 0.0, 1.0)
        # Cast to fp32 so downstream consumers (numpy viz, metrics) don't trip on bf16.
        return recon.float().to(self.device)

    @torch.no_grad()
    def _sample(self, mr, spacing, num_steps=10):
        """Iterative denoising to sample synthetic CT latent."""
        self.controlnet.eval()
        latent_shape = (1, 4, mr.shape[2] // 4, mr.shape[3] // 4, mr.shape[4] // 4)
        # fp32 mainline; autocast below casts ops to bf16. Mixing explicit .half() with
        # bf16 autocast silently promotes scheduler.step's `sample + v_pred*dt` to fp32
        # after iter 0, leaving the dtype contract incoherent.
        latents = torch.randn(latent_shape, device=self.device)

        try:
            num_voxels = int(torch.prod(torch.tensor(latent_shape[2:])).item())
            self.noise_scheduler.set_timesteps(num_inference_steps=num_steps, input_img_size_numel=num_voxels)
        except (TypeError, AttributeError):
            self.noise_scheduler.set_timesteps(num_inference_steps=num_steps)

        all_timesteps = self.noise_scheduler.timesteps.to(self.device)
        all_next_timesteps = torch.cat((all_timesteps[1:], torch.tensor([0], dtype=all_timesteps.dtype, device=self.device)))

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            for t, next_t in zip(all_timesteps, all_next_timesteps):
                t_tensor = torch.tensor([t], device=self.device).float()
                # Modality 1 corresponds to CT (refer to modality_mapping.json)
                class_labels = torch.ones(latents.shape[0], dtype=torch.long, device=self.device)
                down_res, mid_res = self.controlnet(x=latents, timesteps=t_tensor, controlnet_cond=mr, class_labels=class_labels)
                model_output = self.unet(x=latents, timesteps=t_tensor, spacing_tensor=spacing, class_labels=class_labels, down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res)
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

        monitor_every = max(1, getattr(self.cfg, "monitor_interval", 10))

        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            log_this_step = self.cfg.wandb and getattr(self.cfg, "monitor_resources", False) and step_idx % monitor_every == 0

            with StepTimer(log_this_step) as timer:
                for accum_idx in range(self.cfg.accum_steps):
                    with timer.cpu("data"):
                        batch = next(self.train_iter)
                        # `ct_spacing` is recorded into the cached pipeline as a plain tensor
                        # (PersistentDataset's weights_only=True save strips MetaTensor.affine).
                        spacing = batch["ct_spacing"].float().to(self.device) * 100.0

                    with timer.gpu("augment"):
                        # Apply batched GPU aug via batchaug to full padded volumes (no crop, B=1).
                        # In preencoded mode the batch has no "ct" key; gpu_augment_batch operates
                        # only on present keys so this is a no-op for absent keys.
                        batch = gpu_augment_batch(batch, self.gpu_transforms, self.device)

                    mr = batch["mri"]
                    ct = batch.get("ct", None)

                    # Modular Synchronized CutOut (via BaseTrainer). Skipped in preencoded
                    # mode (CT image absent + augment=False enforced upstream).
                    if ct is not None:
                        mr, ct, _ = self.apply_cutout(mr, ct)

                    with timer.gpu("encode"):
                        if self.preencoded_dir is not None:
                            # Latent already multiplied by scale_factor at encode time.
                            ct_emb = batch["ct_latent"].to(self.device)
                        else:
                            # 1. Encode CT to latent on-the-fly via sliding window
                            ct_emb = self._encode_sliding_window(ct) * self.scale_factor

                    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                        # 2. Diffusion forward: sample timestep and corrupt latent
                        noise, timesteps, noisy_latent = self._sample_noise_and_timesteps(ct_emb)

                        # 3. Predict velocity (ControlNet conditions the frozen UNet)
                        with timer.gpu("predict"):
                            model_output, down_res, mid_res = self._controlnet_unet_forward(noisy_latent, mr, spacing, timesteps)

                        target = self._get_diffusion_target(ct_emb, noise, timesteps)
                        loss = F.l1_loss(model_output.float(), target.float())
                        loss = loss / self.cfg.accum_steps

                    with timer.gpu("backward"):
                        loss.backward()
                    step_loss += loss.item()

                    if step_idx == 0 and accum_idx == 0 and self.cfg.wandb:
                        subj_id = batch["subj_id"][0] if "subj_id" in batch else None

                        # FREE MEMORY for visualization decode
                        # model_output, down_res, mid_res are the largest consumers
                        del model_output, down_res, mid_res, noisy_latent, target, noise
                        torch.cuda.empty_cache()

                        # decoded output is in [0,1] mapping to [-1000,1000] HU — same space as `ct`,
                        # so display them with identical normalization for honest side-by-side viz.
                        decoded_ct = self._decode(ct_emb[0:1])
                        self._log_training_patch(mr, ct, decoded_ct, self.global_step, step_idx, subj_id=subj_id)

                with timer.gpu("optimizer"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
                    self.optimizer.step()
            self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            if log_this_step:
                self._log_monitoring(
                    timer.timings_ms(),
                    throughput=(self.cfg.batch_size * self.cfg.accum_steps) / timer.elapsed_s(),
                )

            if self.cfg.wandb and self.global_step % 5 == 0:
                total_steps = self.cfg.steps_per_epoch * self.cfg.total_epochs
                wandb.log(
                    {
                        "train/loss": step_loss,
                        "train/grad_norm": grad_norm.item(),
                        "info/lr": self.optimizer.param_groups[0]["lr"],
                        "info/train_pct": self.global_step / total_steps,
                    },
                    step=self.global_step,
                )

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

        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            if i not in self.val_indices_to_run:
                continue

            mr = batch["mri"].to(self.device)
            ct = batch["ct"].to(self.device)
            # `ct_spacing` is recorded into the cached pipeline as a plain tensor
            # (PersistentDataset's weights_only=True save strips MetaTensor.affine).
            spacing = batch["ct_spacing"].float().to(self.device) * 100.0
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]

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

            # Proxy diffusion loss (on resampled latent space). In preencoded mode
            # the latent is already cached alongside the val batch (× scale_factor).
            if self.preencoded_dir is not None:
                ct_emb = batch["ct_latent"].to(self.device)
            else:
                ct_emb = self._encode_sliding_window(ct) * self.scale_factor
            noise, timesteps, noisy_latent = self._sample_noise_and_timesteps(ct_emb)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                model_output, _, _ = self._controlnet_unet_forward(noisy_latent, mr, spacing, timesteps)

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
            mr_unpad = unpad(mr.float(), orig_shape)

            # compute_metrics expects (B, C, D, H, W). Use range 1.0 since data is normalized [0, 1].
            # body_mask (if val_body_mask) is at the same padded resolution as ct, so unpad with orig_shape.
            mask_unpad = self._get_body_mask_unpad(batch, orig_shape)

            # MAISI CT was clipped to [-1000, 1000] HU → 2000-HU range (vs amix/unet's 2048).
            met, body_met = self._compute_val_metrics(pred_unpad, gt_unpad, mask_unpad, hu_range=2000)

            # NVIDIA MAISI's air-excluded MAE HU (gt_hu > -900). Logged under a distinct
            # name so it does NOT overwrite the standard `mae_hu` (which IS apples-to-apples
            # comparable with amix/unet). Skip when the volume is all-air (no body voxels) —
            # otherwise a synthetic 0 enters the mean and drags it down.
            hu_mask = gt_hu_unpad > -900
            if hu_mask.any():
                met["mae_hu_air_excluded"] = torch.mean(torch.abs(pred_hu_unpad[hu_mask] - gt_hu_unpad[hu_mask])).item()

            # Validation Dice
            if getattr(self.cfg, "validate_dice", False) and self.teacher_model is not None and "seg" in batch:
                seg = batch["seg"].to(self.device)
                seg_unpad = unpad(seg, orig_shape)
                pred_probs = self._run_teacher_sw(pred_unpad, val_ps)
                # Whole-volume dice (already at orig_shape, no further unpadding needed)
                self._compute_dice_metrics(pred_probs, seg_unpad, orig_shape=None, mask_unpad=None, target_met=met)
                if body_met is not None:
                    self._compute_dice_metrics(pred_probs, seg_unpad, orig_shape=None, mask_unpad=mask_unpad, target_met=body_met)
                del pred_probs, seg, seg_unpad

            for k, v in met.items():
                val_metrics[k].append(v)
            if body_met is not None:
                for k, v in body_met.items():
                    val_metrics[f"body_{k}"].append(v)
            val_subject_ids.append(subj_id)

            # Save predictions (overwrite "last" each val run; epoch_<N> snapshot every val_save_interval)
            save_path = self._save_val_pred(pred_hu_unpad, batch, subj_id, epoch, already_hu=True)

            # Viz (only for selected subjects). Use unpadded volumes so slice picker doesn't
            # walk into the zero-padded tail (pad-end can be >100 voxels in z).
            if i in self.val_viz_indices and self.cfg.wandb:
                viz_metrics, viz_body = self._select_viz_metrics(met, body_met)
                visualize_lite(pred_unpad, gt_unpad, mr_unpad, subj_id, orig_shape, self.global_step, epoch, log_name=f"viz/val_{i}", metrics=viz_metrics, body_metrics=viz_body)

            # Cleanup memory after each subject
            del mr, ct, pred_latent, pred_ct_norm, gt_hu_raw, gt_hu, pred_hu, pred_matched, gt_matched, pred_unpad, gt_unpad, pred_hu_unpad, gt_hu_unpad, mr_unpad
            gc.collect()
            torch.cuda.empty_cache()

        n_val = len(self.val_indices_to_run)
        extra = {"avg_inference_time": (time.time() - t_inf_start) / n_val if n_val > 0 else 0.0}
        avg_met = self._log_val_metrics(val_metrics, extra=extra, subject_ids=val_subject_ids)
        return avg_met

    def _log_training_patch(self, mr, ct, decoded_ct, step, step_idx, subj_id=None):
        """Visualizes [MR, (optional) GT, Decoded GT Latent] for full-volume training.

        In preencoded mode `ct` is None — the CT image isn't loaded — so the GT
        row is omitted and the figure shows MR + decoded(latent) only.
        """
        mr_img = mr[0, 0].cpu().numpy()
        dec_ct = decoded_ct[0, 0].cpu().numpy()
        gt_ct = ct[0, 0].cpu().numpy() if ct is not None else None

        cx, cy, cz = np.array(mr_img.shape) // 2
        n_rows = 3 if gt_ct is not None else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(10, 3.3 * n_rows))

        def plot_row(row_idx, vol, title):
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 0].set_title(f"{title} Ax")
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 1].set_title(f"{title} Sag")
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap="gray", vmin=0, vmax=1)
            axes[row_idx, 2].set_title(f"{title} Cor")

        plot_row(0, mr_img, "MRI (Cond)")
        if gt_ct is not None:
            plot_row(1, gt_ct, "GT CT")
            plot_row(2, dec_ct, "Decoded Latent")
        else:
            plot_row(1, dec_ct, "Decoded Latent")

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
                # info/lr is logged in the per-5-steps block in train_epoch; skip here to
                # avoid overwriting mid-epoch values at the epoch-boundary global_step.
                wandb.log(
                    {
                        "train/total": loss,
                        "info/grad_norm": gn,
                        "info/epoch_duration": ep_duration,
                        "info/val_duration": val_duration,
                        "info/cumulative_time": cumulative_time,
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
        filename = "checkpoint_last.pt" if is_last else f"maisi_epoch{epoch:05d}.pt"
        self.save_checkpoint(
            self.controlnet,
            self.optimizer,
            self.scheduler,
            epoch,
            os.path.join(self._default_save_dir(), filename),
            extra_state={"scale_factor": self.scale_factor},
        )


if __name__ == "__main__":
    pass
