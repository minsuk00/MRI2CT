"""MC-IDDPM Trainer for SynthRAD MRI->CT.

Subclasses `BaseTrainer` purely for seed / wandb / resume / save plumbing. The
training step itself is byte-equivalent to the notebook's `train()` body
(`MC-IDDPM main.ipynb`), just wrapped in the project's lifecycle methods so we
inherit chained-resume across SLURM 48-h cuts.

Paper-faithful choices kept verbatim:
  - SwinVITModel kwargs (image_size, channel_mult, window_size, sample_kernel, ...)
  - 1000 training diffusion steps, learn_sigma=True, predict noise eps
  - AdamW lr=2e-5, weight_decay=1e-4
  - Uniform random crop, no augmentation

Project-side additions (not in the notebook):
  - WandB logging of loss/total, loss/mse, loss/vb, info/*, model/*
  - In-training validation = visualization-only on a fixed subject (no metrics).
  - Two-tier checkpointing via BaseTrainer.save_checkpoint.
"""
import gc
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from monai.data import DataLoader
from tqdm import tqdm

# Project shared utilities. We import BaseTrainer for plumbing; the training math
# below is paper-native.
from common.config import Config
from common.trainer_base import BaseTrainer
from common.utils import clean_state_dict, count_parameters

# Cloned MC-IDDPM modules (imported via baselines.mc_ddpm.__init__ sys.path shim).
from baselines.mc_ddpm.diffusion.Create_diffusion import create_gaussian_diffusion
from baselines.mc_ddpm.diffusion.resampler import UniformSampler
from baselines.mc_ddpm.network.Diffusion_model_transformer import SwinVITModel

from baselines.mc_ddpm.data import PATCH, build_cached_xform, build_datasets


class Trainer(BaseTrainer):
    def __init__(self, config_dict):
        cfg = Config(config_dict)
        super().__init__(cfg, prefix="MCDDPM")

        self._setup_models()
        self._find_subjects()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()
        self._log_model_summary({"SwinVITModel": self.model})
        self._load_resume()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup_models(self):
        print(f"[{self.prefix}] 🏗️ Building SwinVITModel @ image_size={PATCH}")
        # All kwargs lifted verbatim from `MC-IDDPM main.ipynb`'s model construction.
        self.model = SwinVITModel(
            image_size=PATCH,
            in_channels=2,             # 1 (noisy CT) + 1 (MR condition)
            model_channels=64,
            out_channels=2,            # learn_sigma=True -> [eps, var_interp]
            dims=3,
            # Upstream quirk: SwinVITModel does `self.sample_kernel = sample_kernel[0]`
            # at line 359 of Diffusion_model_transformer.py, so this arg must be
            # wrapped in an outer 1-tuple — exactly what the notebook achieves with
            # `sample_kernel=([2,2,2],[2,2,1],[2,2,1],[2,2,1]),` (trailing comma).
            sample_kernel=(([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),),
            num_res_blocks=[2, 2, 2, 2],
            attention_resolutions=(32, 16, 8),
            dropout=0.0,
            channel_mult=(1, 2, 3, 4),
            num_classes=None,
            use_checkpoint=getattr(self.cfg, "use_checkpoint", False),
            use_fp16=False,
            num_heads=[4, 4, 8, 16],
            window_size=[[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]],
            num_head_channels=64,
            num_heads_upsample=-1,
            use_scale_shift_norm=True,
            resblock_updown=False,
            use_new_attention_order=False,
        ).to(self.device)

        # Training diffusion: full 1000-step schedule (paper).
        print(f"[{self.prefix}] 🌀 Building train diffusion (steps={self.cfg.diffusion_steps}, learn_sigma=True)")
        self.diffusion_train = create_gaussian_diffusion(
            steps=self.cfg.diffusion_steps,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule="linear",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            timestep_respacing="",
        )
        # Val viz diffusion: respaced down to a few steps for cheap visualization.
        print(f"[{self.prefix}] 🎨 Building viz diffusion (timestep_respacing=[{self.cfg.val_steps}])")
        self.diffusion_val = create_gaussian_diffusion(
            steps=self.cfg.diffusion_steps,
            learn_sigma=True,
            sigma_small=False,
            noise_schedule="linear",
            use_kl=False,
            predict_xstart=False,
            rescale_timesteps=True,
            rescale_learned_sigmas=True,
            timestep_respacing=[self.cfg.val_steps],
        )
        self.schedule_sampler = UniformSampler(self.diffusion_train)

        tot, train = count_parameters(self.model)
        print(f"[{self.prefix}] Model Params: Total={tot:,} | Trainable={train:,}")

    def _setup_data(self):
        train_ds, val_ds = build_datasets(self.cfg)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            persistent_workers=self.cfg.num_workers > 0,
            pin_memory=False,
        )
        self.train_iter = self._inf_gen(self.train_loader)
        # val_ds is kept around but in-training val uses a manually-loaded fixed
        # subject (see `validate()`); we don't iterate val_ds here.
        self.val_ds = val_ds

    def _setup_opt(self):
        # Paper: AdamW lr=2e-5, wd=1e-4.
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.lr,
            weight_decay=self.cfg.weight_decay,
        )
        if getattr(self.cfg, "lr_anneal_steps", 0) > 0:
            print(f"[{self.prefix}] 📉 CosineAnnealingLR over {self.cfg.lr_anneal_steps} steps")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.cfg.lr_anneal_steps, eta_min=0.0,
            )
        else:
            self.scheduler = None
        # No GradScaler — we use bf16 autocast (paper-style AMP), which has the
        # same exponent range as fp32 and doesn't need loss scaling.

    def _load_resume(self):
        super()._load_resume(self.model, self.optimizer, self.scheduler)

    def save_checkpoint(self, epoch, is_last=False):
        filename = "checkpoint_last.pt" if is_last else f"mcddpm_epoch{epoch:05d}.pt"
        path = os.path.join(self._default_save_dir(), filename)
        super().save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, path)

    # ------------------------------------------------------------------
    # Core
    # ------------------------------------------------------------------
    def _train_step(self, batch):
        """Byte-equivalent to the notebook's train() body."""
        mri = batch["mri"].to(self.device, non_blocking=True)   # [B', 1, 128, 128, 4] in [-1, 1]
        ct  = batch["ct"].to(self.device, non_blocking=True)
        # B' = batch_size * patches_per_volume (RandSpatialCropSamplesd flattens).
        t, weights = self.schedule_sampler.sample(mri.shape[0], self.device)

        # Paper notebook wraps training_losses in `torch.cuda.amp.autocast()` (fp16).
        # We use bf16 instead — same speed/memory benefit on A40 Ampere tensor cores,
        # full fp32 exponent range so no NaN/underflow risk (and no GradScaler needed).
        # validate.py uses the same bf16 autocast around p_sample_loop.
        if self.cfg.use_amp:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                losses = self.diffusion_train.training_losses(
                    self.model, x_start=ct, condition_start=mri, t=t,
                )
                loss = (losses["loss"] * weights).mean()
        else:
            losses = self.diffusion_train.training_losses(
                self.model, x_start=ct, condition_start=mri, t=t,
            )
            loss = (losses["loss"] * weights).mean()
        return loss, losses

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(range(self.cfg.steps_per_epoch),
                    desc=f"Train Ep {epoch}", leave=False, dynamic_ncols=True)
        running = {"total": 0.0, "mse": 0.0, "vb": 0.0}

        for step_idx in pbar:
            t_load_start = time.perf_counter()
            batch = next(self.train_iter)
            t_load = time.perf_counter() - t_load_start

            t_fwd_start = time.perf_counter()
            self.optimizer.zero_grad(set_to_none=True)
            loss, losses = self._train_step(batch)

            # bf16 autocast → no GradScaler needed; same backward/step path either way.
            loss.backward()
            self.optimizer.step()
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            t_fwd = time.perf_counter() - t_fwd_start

            if self.scheduler is not None:
                self.scheduler.step()

            l_total = loss.item()
            l_mse = losses["mse"].mean().item()
            l_vb  = losses["vb"].mean().item()
            running["total"] += l_total
            running["mse"]   += l_mse
            running["vb"]    += l_vb

            self.global_step += 1
            self.samples_seen += int(self.cfg.batch_size * self.cfg.patches_per_volume)

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix({"loss": f"{l_total:.4f}", "lr": f"{current_lr:.2e}"})

            if self.cfg.wandb and (step_idx % max(1, getattr(self.cfg, "log_every", 50)) == 0):
                cumulative_time = (
                    (time.time() - self.global_start_time) + self.elapsed_time_at_resume
                    if self.global_start_time else self.elapsed_time_at_resume
                )
                log = {
                    "loss/total": l_total,
                    "loss/mse":   l_mse,
                    "loss/vb":    l_vb,
                    "time/load":  t_load * 1000.0,
                    "time/total": (t_load + t_fwd) * 1000.0,
                    "info/lr":            current_lr,
                    "info/global_step":   self.global_step,
                    "info/samples_seen":  self.samples_seen,
                    "info/epoch":         epoch,
                    "info/cumulative_time": cumulative_time,
                }
                if getattr(self.cfg, "lr_anneal_steps", 0) > 0:
                    log["info/train_pct"] = self.global_step / max(1, self.cfg.lr_anneal_steps)
                wandb.log(log, step=self.global_step)

        n = max(1, self.cfg.steps_per_epoch)
        return {k: v / n for k, v in running.items()}

    @torch.inference_mode()
    def _log_sample_grid(self, mri_patch, ct_patch, epoch, log_prefix, label):
        """Run p_sample_loop on `mri_patch`, then log a 4-panel mid-slice figure
        (MRI / GT CT / Pred / |diff|) under `<log_prefix>/sample_image`.
        """
        shape = (1, 1, mri_patch.shape[2], mri_patch.shape[3], mri_patch.shape[4])
        pred = self.diffusion_val.p_sample_loop(
            self.model, shape, condition=mri_patch, clip_denoised=True,
            progress=False, device=self.device,
        )

        # Rescale [-1, 1] -> [0, 1] for visualization.
        mri01  = (mri_patch[0, 0].float().cpu().numpy() + 1.0) / 2.0
        ct01   = (ct_patch [0, 0].float().cpu().numpy() + 1.0) / 2.0
        pred01 = (pred     [0, 0].float().cpu().numpy() + 1.0) / 2.0
        diff   = np.abs(pred01 - ct01)
        mid_z  = pred01.shape[-1] // 2

        fig, axes = plt.subplots(1, 4, figsize=(16, 4))
        for ax, img, title, cmap, vmin, vmax in [
            (axes[0], mri01[..., mid_z],  "MRI",   "gray", 0, 1),
            (axes[1], ct01 [..., mid_z],  "GT CT", "gray", 0, 1),
            (axes[2], pred01[..., mid_z], "Pred",  "gray", 0, 1),
            (axes[3], diff [..., mid_z],  "|Pred-GT|", "magma", 0, 0.5),
        ]:
            ax.imshow(np.rot90(img), cmap=cmap, vmin=vmin, vmax=vmax)
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(f"[{log_prefix}] {label} | epoch {epoch} | steps={self.cfg.val_steps}")
        plt.tight_layout()
        wandb.log({
            f"{log_prefix}/sample_image":   wandb.Image(fig),
            f"{log_prefix}/subj_id":        label,
            f"{log_prefix}/sampling_steps": self.cfg.val_steps,
        }, step=self.global_step)
        plt.close(fig)

    @torch.inference_mode()
    def validate(self, epoch):
        """val viz: fixed subject center-patch sample (comparable over time).
        train viz: a random in-flight training patch (overfitting tell).
        Viz-only — full metrics live in scripts/validate.py.
        """
        from common.data import build_data_dicts

        if not self.cfg.wandb:
            return

        try:
            self.model.eval()

            # ---- val: fixed subject, center patch ----
            subj_id = getattr(self.cfg, "val_subj_id", None)
            if subj_id is None:
                print(f"[{self.prefix}] 🚫 val_subj_id unset; skipping val viz")
            else:
                dicts = build_data_dicts(self.cfg.root_dir, [subj_id], load_body_mask=True)
                if not dicts:
                    print(f"[{self.prefix}] ⚠️ val subject {subj_id} not found; skip viz")
                else:
                    xform = build_cached_xform(load_body_mask=True)
                    item = xform(dicts[0])
                    mri_full = item["mri"]   # (1, D, H, W) in [-1, 1]
                    ct_full  = item["ct"]
                    D, H, W  = mri_full.shape[1:]
                    ps_d, ps_h, ps_w = PATCH
                    d0 = max(0, (D - ps_d) // 2)
                    h0 = max(0, (H - ps_h) // 2)
                    w0 = max(0, (W - ps_w) // 2)
                    mri_patch = mri_full[:, d0:d0 + ps_d, h0:h0 + ps_h, w0:w0 + ps_w].unsqueeze(0).to(self.device)
                    ct_patch  = ct_full [:, d0:d0 + ps_d, h0:h0 + ps_h, w0:w0 + ps_w].unsqueeze(0).to(self.device)
                    self._log_sample_grid(mri_patch, ct_patch, epoch, "val", subj_id)

            # ---- train: one random in-flight patch from train_iter ----
            batch = next(self.train_iter)
            mri_patch = batch["mri"][:1].to(self.device)
            ct_patch  = batch["ct" ][:1].to(self.device)
            label = batch.get("subj_id", ["?"])[0] if isinstance(batch.get("subj_id"), (list, tuple)) else "random"
            self._log_sample_grid(mri_patch, ct_patch, epoch, "train", label)
        except Exception as e:
            print(f"[{self.prefix}] [WARNING] viz failed: {e}")
        finally:
            self.model.train()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def train(self):
        print(f"[{self.prefix}] 🏁 Starting Loop: Ep {self.start_epoch} -> {self.cfg.total_epochs}")
        self.global_start_time = time.time()

        global_pbar = tqdm(
            range(self.start_epoch, self.cfg.total_epochs),
            desc="🚀 Total Progress",
            initial=self.start_epoch, total=self.cfg.total_epochs,
            dynamic_ncols=True, unit="ep",
        )

        for epoch in global_pbar:
            ep_start = time.time()
            avg = self.train_epoch(epoch)
            ep_duration = time.time() - ep_start

            val_duration = 0.0
            if (epoch % self.cfg.val_interval == 0) or ((epoch + 1) == self.cfg.total_epochs):
                val_start = time.time()
                self.validate(epoch)
                val_duration = time.time() - val_start

            cumulative_time = (
                (time.time() - self.global_start_time) + self.elapsed_time_at_resume
                if self.global_start_time else self.elapsed_time_at_resume
            )
            tqdm.write(
                f"Ep {epoch} | loss/total={avg['total']:.4f} mse={avg['mse']:.4f} vb={avg['vb']:.4f} "
                f"| {ep_duration:.1f}s train + {val_duration:.1f}s val"
            )

            if self.cfg.wandb:
                wandb.log({
                    "train/loss_total":     avg["total"],
                    "train/loss_mse":       avg["mse"],
                    "train/loss_vb":        avg["vb"],
                    "info/epoch_duration":  ep_duration,
                    "info/val_duration":    val_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/global_step":     self.global_step,
                    "info/epoch":           epoch,
                    "info/samples_seen":    self.samples_seen,
                }, step=self.global_step)

            # Rolling-last + milestone saves.
            self.save_checkpoint(epoch, is_last=True)
            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        if self.cfg.wandb:
            wandb.log({"info/samples_seen_total": self.samples_seen})
            wandb.finish()
        print(f"[{self.prefix}] ✅ Done. Total Time: {time.time() - self.global_start_time:.1f}s")
