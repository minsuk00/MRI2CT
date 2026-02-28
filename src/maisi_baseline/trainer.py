import copy
import datetime
import gc
import json
import math
import os
import random
import sys
import time
import traceback
import warnings
from collections import defaultdict
from glob import glob

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
from tqdm import tqdm
import wandb
from scipy.stats import pearsonr

from monai.bundle import ConfigParser
from monai.networks.utils import copy_model_state
from monai.networks.schedulers import RFlowScheduler
from monai.data import DataLoader, Dataset, MetaTensor
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    ScaleIntensityRangePercentilesd,
    ScaleIntensityRanged,
    EnsureTyped,
)
from monai.inferers import SlidingWindowInferer

# Suppress noisy warnings
warnings.filterwarnings("ignore", message=".*Orientationd.__init__:labels.*")
warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence for multidimensional indexing.*")

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.maisi_baseline.config import DEFAULT_CONFIG, AUTOENCODER_PATH, DIFFUSION_PATH, NETWORK_CONFIG_PATH
from src.mri2ct.utils import cleanup_gpu, count_parameters, set_seed, compute_metrics

class MAISITrainer:
    def __init__(self, config_dict):
        # 1. Config Setup
        self.cfg = copy.deepcopy(DEFAULT_CONFIG)
        self.cfg.update(config_dict)
        self.gpfs_root = self.cfg['maisi_data_root']
        set_seed(self.cfg['seed'])
        self.device = torch.device(self.cfg['device'])
        print(f"[MAISI] 🚀 Initializing on {self.device}")

        # Default run name logic
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.run_name = f"{timestamp}_{self.cfg['run_name_prefix']}"
        if self.cfg.get('subjects'):
            self.run_name += f"_SingleSubj_{len(self.cfg['subjects'])}"

        # 0. Data Staging
        self._stage_data_local()

        # 2. Setup Components
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 3. State Tracking
        self.global_step = 0
        self.samples_seen = 0
        self.start_epoch = 0
        self.global_start_time = None
        self.elapsed_time_at_resume = 0
        self.train_start_time = None

        self._load_resume()

    def _stage_data_local(self):
        """Copies dataset to local NVMe RAID for blazing fast I/O."""
        user_id = os.environ.get("USER", "default")
        local_root = os.path.join("/tmp_data", f"maisi_{user_id}")

        if not os.path.exists("/tmp_data") or not os.access("/tmp_data", os.W_OK):
            print("[MAISI] ⚠️ Local storage not available. Staying on GPFS.")
            return

        if os.path.exists(local_root):
            print(f"[MAISI] ♻️  Local cache found at {local_root}. Syncing updates from GPFS...")
        else:
            print(f"[MAISI] 🚚 Staging data to local NVMe: {local_root}")
            os.makedirs(local_root, exist_ok=True)

        inc_str = "--include='*/' --include='*.nii.gz' --include='datalist.json' --exclude='*'"
        os.system(f"rsync -am --delete {inc_str} {self.gpfs_root}/ {local_root}/")
        
        self.cfg['maisi_data_root'] = local_root
        print(f"[MAISI] ✅ Data synchronized. New root: {self.cfg['maisi_data_root']}")

    def _setup_wandb(self):
        if not self.cfg['wandb']:
            return

        # Use the symlinked 'wandb' folder in project root
        wandb_dir = os.path.join(project_root, "wandb")
        os.makedirs(wandb_dir, exist_ok=True)

        wandb_id = self.cfg.get('resume_wandb_id')

        print(f"[MAISI] 📡 Initializing WandB in: {wandb_dir}")
        wandb.init(
            project=self.cfg['project_name'],
            name=self.run_name,
            config=self.cfg,
            reinit=True,
            dir=wandb_dir,
            id=wandb_id,
            resume="allow" if wandb_id else None,
        )
        wandb.run.log_code(".")

    def _load_resume(self):
        resume_path = self.cfg.get('resume_checkpoint')
        if not resume_path and self.cfg.get('resume_wandb_id'):
            gpfs_run_dir = os.path.join(self.cfg['model_save_root'], self.cfg['resume_wandb_id'])
            if os.path.exists(gpfs_run_dir):
                ckpts = sorted(glob(os.path.join(gpfs_run_dir, "*.pt")))
                if ckpts: resume_path = ckpts[-1]
            if not resume_path:
                wandb_dir = os.path.join(project_root, "wandb")
                run_folders = glob(os.path.join(wandb_dir, f"run-*-{self.cfg['resume_wandb_id']}"))
                for f in run_folders:
                    ckpts = glob(os.path.join(f, "files", "*.pt"))
                    if ckpts: resume_path = max(ckpts, key=os.path.getmtime)

        if not resume_path: return

        print(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device, weights_only=False)
        self.controlnet.load_state_dict(checkpoint['controlnet_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.start_epoch = checkpoint.get('epoch', 0) + 1
        self.global_step = checkpoint.get('global_step', 0)
        self.samples_seen = checkpoint.get('samples_seen', 0)
        self.elapsed_time_at_resume = checkpoint.get('elapsed_time', 0)

    def _setup_models(self):
        with open(NETWORK_CONFIG_PATH, 'r') as f:
            model_def = json.load(f)
        parser = ConfigParser()
        parser.update(model_def)
        
        # 1. VAE
        self.autoencoder = parser.get_parsed_content('autoencoder_def', instantiate=True).to(self.device)
        ae_ckpt = torch.load(AUTOENCODER_PATH, map_location=self.device, weights_only=False)
        self.autoencoder.load_state_dict(ae_ckpt['unet_state_dict'] if 'unet_state_dict' in ae_ckpt else ae_ckpt)
        self.autoencoder.eval()
        for p in self.autoencoder.parameters(): p.requires_grad = False
            
        # 2. Denoising UNet
        self.unet = parser.get_parsed_content('diffusion_unet_def', instantiate=True).to(self.device)
        unet_ckpt = torch.load(DIFFUSION_PATH, map_location=self.device, weights_only=False)
        self.unet.load_state_dict(unet_ckpt['unet_state_dict'], strict=False)
        
        # Automatic Scale Factor
        if 'scale_factor' in unet_ckpt:
            self.scale_factor = unet_ckpt['scale_factor']
            if isinstance(self.scale_factor, torch.Tensor): self.scale_factor = self.scale_factor.to(self.device)
            else: self.scale_factor = torch.tensor(self.scale_factor, device=self.device)
        else:
            self.scale_factor = torch.tensor(1.0, device=self.device)
            
        print(f"[MAISI] 📈 Scale Factor (Automatic): {self.scale_factor.item():.6f}")
        self.unet.eval()
        for p in self.unet.parameters(): p.requires_grad = False
            
        # 3. ControlNet
        self.controlnet = parser.get_parsed_content('controlnet_def', instantiate=True).to(self.device)
        copy_model_state(self.controlnet, self.unet.state_dict())
        self.noise_scheduler = parser.get_parsed_content('noise_scheduler', instantiate=True)

        tot, train = count_parameters(self.controlnet)
        print(f"[MAISI] ControlNet Params: Total={tot:,} | Trainable={train:,}")

    def _setup_data(self):
        datalist_path = os.path.join(self.cfg['maisi_data_root'], 'datalist.json')
        with open(datalist_path, 'r') as f:
            raw_data = json.load(f)['training']
            
        # Add root dir to paths
        data = []
        for d in raw_data:
            item = d.copy()
            item['mr_image'] = os.path.join(self.cfg['maisi_data_root'], item['mr_image'])
            item['ct_image'] = os.path.join(self.cfg['maisi_data_root'], item['ct_image'])
            item['ct_emb'] = os.path.join(self.cfg['maisi_data_root'], item['ct_emb'])
            data.append(item)

        if self.cfg.get('subjects'):
            # Single subject optimization: train and val on same filtered subjects
            data = [d for d in data if any(s in d['mr_image'] for s in self.cfg['subjects'])]
            train_files, val_files = data, data
        else:
            train_files = [d for d in data if d['fold'] == 0]
            val_files = [d for d in data if d['fold'] == 1]
            
        transforms = Compose([
            LoadImaged(keys=['mr_image', 'ct_image', 'ct_emb'], image_only=True, ensure_channel_first=True),
            Orientationd(keys=['mr_image', 'ct_image', 'ct_emb'], axcodes='RAS'), # STRICT RAS
            # MR is already normalized [0, 1] by prepare_data.py
            # CT is loaded raw for accurate MAE HU calculation in validate()
            EnsureTyped(keys=['mr_image', 'ct_image', 'ct_emb'], dtype=torch.float32),
        ])
        
        self.train_ds = Dataset(data=train_files, transform=transforms)
        self.train_loader = DataLoader(self.train_ds, batch_size=self.cfg['batch_size'], shuffle=True, num_workers=self.cfg['num_workers'])
        
        self.val_ds = Dataset(data=val_files, transform=transforms)
        self.val_loader = DataLoader(self.val_ds, batch_size=1, shuffle=False)
        
        print(f"[MAISI] Train: {len(train_files)} | Val: {len(val_files)}")

    def _setup_opt(self):
        self.optimizer = torch.optim.AdamW(self.controlnet.parameters(), lr=self.cfg['lr'])
        total_steps = self.cfg['total_epochs'] * len(self.train_loader)
        # STRICT PARITY: Original uses PolynomialLR with power 2.0
        self.scheduler = torch.optim.lr_scheduler.PolynomialLR(self.optimizer, total_iters=total_steps, power=2.0)
        self.scaler = torch.amp.GradScaler('cuda')

    @torch.no_grad()
    def _decode(self, z):
        """Decode latent back to image space with strict MAISI parity (dynamic infer)."""
        # Ensure input is z / scale_factor
        latent = z / self.scale_factor
        
        # Original MAISI logic: if smaller than roi_size, run directly
        roi_size = [64, 64, 64]
        with torch.amp.autocast("cuda"):
            if all(s <= r for s, r in zip(latent.shape[2:], roi_size)):
                recon = self.autoencoder.decode_stage_2_outputs(latent)
            else:
                inferer = SlidingWindowInferer(
                    roi_size=roi_size, 
                    sw_batch_size=1, 
                    overlap=0.4, 
                    mode="gaussian",
                    sw_device=self.device,
                    device=torch.device("cpu")
                )
                recon = inferer(latent, self.autoencoder.decode_stage_2_outputs)
            
        recon = torch.clamp(recon, 0.0, 1.0)
        return recon.to(self.device)

    @torch.no_grad()
    def _sample(self, mr, spacing, num_steps=10): 
        """Iterative denoising to sample synthetic CT latent."""
        self.controlnet.eval()
        latent_shape = (1, 4, mr.shape[2]//4, mr.shape[3]//4, mr.shape[4]//4)
        # STRICT PARITY: Sampling in half precision
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
                down_res, mid_res = self.controlnet(x=latents, timesteps=t_tensor, controlnet_cond=mr_h)
                # Unet must be in eval() per original script during controlnet training
                model_output = self.unet(x=latents, timesteps=t_tensor, spacing_tensor=sp_h,
                                         down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res)
                latents, _ = self.noise_scheduler.step(model_output, t, latents, next_t)
            
        return latents.float()

    @torch.no_grad()
    def _log_training_patch(self, mr, ct_emb, gt_ct_vol, step):
        """Visualizes [MR, Original GT, Decoded GT Latent]."""
        decoded_gt = self._decode(ct_emb[0:1]).cpu().float().numpy().squeeze()
        mr_img = mr[0, 0].cpu().numpy()
        gt_ct = gt_ct_vol[0, 0].cpu().numpy()
        
        cx, cy, cz = np.array(mr_img.shape) // 2
        fig, axes = plt.subplots(3, 3, figsize=(10, 10))
        
        def plot_row(row_idx, vol, title):
            vmin, vmax = vol.min(), vol.max()
            if abs(vmax - vmin) < 1e-5: vmax = vmin + 1.0
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{title} Ax ({vmin:.2f}/{vmax:.2f})")
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"{title} Sag")
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap='gray', vmin=vmin, vmax=vmax)
            axes[row_idx, 2].set_title(f"{title} Cor")

        plot_row(0, mr_img, "MR")
        plot_row(1, gt_ct, "Original GT CT")
        plot_row(2, decoded_gt, "Decoded GT Latent")
        
        for ax in axes.flatten(): ax.axis('off')
        plt.tight_layout()
        wandb.log({"train/patch_viz": wandb.Image(fig)}, step=step)
        plt.close(fig)

    @torch.no_grad()
    def _visualize_lite(self, pred, ct, mri, subj_id, step, epoch, idx):
        """Standard 4-column view: MRI, GT, Pred, Residual. Matching unet_baseline."""
        gt_mri = mri.squeeze().cpu().numpy()
        gt_ct = ct.squeeze().cpu().numpy()
        pred_ct = pred.squeeze().cpu().numpy()

        # Strict CT range for parity
        ct_vmin, ct_vmax = -1000, 1000
        pred_hu = (pred_ct * 2000.0) - 1000.0

        items = [
            (gt_mri, "GT MRI", "gray", (gt_mri.min(), gt_mri.max())),
            (gt_ct, "GT CT (HU)", "gray", (ct_vmin, ct_vmax)),
            (pred_hu, "Pred CT (HU)", "gray", (ct_vmin, ct_vmax)),
            (pred_hu - gt_ct, "Residual (HU)", "seismic", (-200, 200)),
        ]

        D_dim = gt_ct.shape[-1]
        num_cols = len(items)
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)

        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)

        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                if title == "Residual": res_im = im
                if i == 0: ax.set_title(title)
                ax.axis("off")

        if "res_im" in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")

        title_str = f"Subject: {subj_id} | Epoch {epoch} | Step {step}"
        fig.suptitle(title_str, fontsize=16, y=0.99)
        wandb.log({f"viz/val_{idx}": wandb.Image(fig, caption=f"Subject: {subj_id}")}, step=step)
        plt.close(fig)

    def train_epoch(self, epoch):
        self.controlnet.train()
        self.unet.eval() # STRICT PARITY
        total_loss = 0
        total_grad = 0
        pbar = tqdm(self.train_loader, desc=f"Ep {epoch}", leave=False, dynamic_ncols=True)
        
        for batch in pbar:
            mr = batch['mr_image'].to(self.device)
            # Use the actual latent embedding (ct_emb) for training
            ct_emb = batch['ct_emb'].to(self.device) * self.scale_factor
            spacing = torch.stack(batch['spacing'], dim=1).float().to(self.device) * 100
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                noise = torch.randn_like(ct_emb).to(self.device)
                timesteps = self.noise_scheduler.sample_timesteps(ct_emb)
                
                # Create noisy CT latent
                noisy_latent = self.noise_scheduler.add_noise(original_samples=ct_emb, noise=noise, timesteps=timesteps)
                
                # Forward pass
                down_res, mid_res = self.controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=mr)
                model_output = self.unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing,
                                         down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res)
                
                # Flow matching target (v = x1 - x0)
                target = ct_emb - noise
                loss = F.l1_loss(model_output.float(), target.float())
                
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.controlnet.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            total_loss += loss.item()
            total_grad += grad_norm.item()
            self.global_step += 1
            self.samples_seen += mr.shape[0]
            
            if self.cfg['wandb'] and self.global_step % 100 == 0:
                wandb.log({
                    'train/loss': loss.item(), 
                    'train/grad_norm': grad_norm.item(), 
                    'info/lr': self.optimizer.param_groups[0]['lr']
                }, step=self.global_step)
                
            if self.global_step % len(self.train_loader) == 1:
                if self.cfg['wandb']: self._log_training_patch(mr, ct_emb, batch['ct_image'], self.global_step)
                
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'gn': f"{grad_norm.item():.2f}"})

        return total_loss / len(self.train_loader), total_grad / len(self.train_loader)

    @torch.no_grad()
    def validate(self, epoch):
        self.controlnet.eval()
        val_metrics = defaultdict(list)
        
        t_inf_start = time.time()
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mr = batch['mr_image'].to(self.device)
            spacing = torch.stack(batch['spacing'], dim=1).float().to(self.device) * 100
            
            # Use RAW CT for metrics (unclipped)
            gt_ct_raw = batch['ct_image'].to(self.device) 
            
            # Robust subject ID extraction
            meta = batch['mr_image'].meta
            fname = meta['filename_or_obj'][0] if isinstance(meta['filename_or_obj'], (list, tuple)) else meta['filename_or_obj']
            subj_id = os.path.basename(os.path.dirname(fname))

            pred_latent = self._sample(mr, spacing, num_steps=10)
            pred_ct = self._decode(pred_latent).to(self.device) 
            
            # --- FINAL ALIGNED METRICS (HU Benchmark) ---
            pred_hu = (pred_ct * 2000.0) - 1000.0
            
            # NVIDIA MAISI Metric Logic: GT is NOT normalized during comparison
            gt_hu = gt_ct_raw 
            
            # NVIDIA MAISI Air Mask (Strict Parity: GT > -900)
            mask = gt_hu > -900
            
            if mask.any():
                mae_hu = torch.mean(torch.abs(pred_hu[mask] - gt_hu[mask])).item()
                p_corr, _ = pearsonr(pred_hu[mask].cpu().numpy().flatten(), gt_hu[mask].cpu().numpy().flatten())
            else:
                mae_hu, p_corr = 0.0, 0.0
                
            met = compute_metrics(pred_ct, (gt_hu + 1000.0) / 2000.0) # For standard metrics, use [0,1]
            met["mae_hu"] = mae_hu
            met["pearson"] = p_corr
            
            # Proxy diffusion loss
            ct_emb = batch['ct_emb'].to(self.device) * self.scale_factor
            noise = torch.randn_like(ct_emb)
            timesteps = self.noise_scheduler.sample_timesteps(ct_emb)
            noisy_latent = self.noise_scheduler.add_noise(original_samples=ct_emb, noise=noise, timesteps=timesteps)
            down_res, mid_res = self.controlnet(x=noisy_latent, timesteps=timesteps, controlnet_cond=mr)
            model_output = self.unet(x=noisy_latent, timesteps=timesteps, spacing_tensor=spacing,
                                     down_block_additional_residuals=down_res, mid_block_additional_residual=mid_res)
            met["loss"] = F.l1_loss(model_output.float(), (ct_emb - noise).float()).item()

            for k, v in met.items(): val_metrics[k].append(v)
            if i < 2 and self.cfg['wandb']:
                self._visualize_lite(pred_ct, gt_hu, mr, subj_id, self.global_step, epoch, i)

        avg_met = {k: np.mean(v) for k, v in val_metrics.items()}
        avg_met["inference_time"] = (time.time() - t_inf_start) / len(self.val_loader)
        if self.cfg['wandb']: wandb.log({f'val/{k}': v for k, v in avg_met.items()}, step=self.global_step)
        return avg_met

    def train(self):
        print(f"[MAISI] 🏁 Starting Loop: {self.cfg['total_epochs']} epochs")
        self.global_start_time = time.time()
        self.train_start_time = time.time()

        for epoch in range(self.start_epoch, self.cfg['total_epochs']):
            ep_start = time.time()
            loss, gn = self.train_epoch(epoch)
            if epoch % self.cfg['val_interval'] == 0:
                avg_met = self.validate(epoch)
                print(f"Ep {epoch} | Val Loss: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | Pearson: {avg_met.get('pearson', 0):.3f} | MAE HU: {avg_met.get('mae_hu', 0):.1f}")

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + (self.elapsed_time_at_resume if self.elapsed_time_at_resume else 0)
            if self.cfg['wandb']:
                wandb.log({
                    'train/total': loss, 'info/grad_norm': gn,
                    'info/epoch_duration': ep_duration, 'info/cumulative_time': cumulative_time,
                    'info/lr': self.optimizer.param_groups[0]['lr'], 'info/global_step': self.global_step,
                    'info/epoch': epoch, 'info/samples_seen': self.samples_seen,
                }, step=self.global_step)
            if epoch % self.cfg['model_save_interval'] == 0: self.save_model(epoch)

    def save_model(self, epoch):
        run_id = wandb.run.id if (self.cfg['wandb'] and wandb.run) else "standalone"
        save_dir = os.path.join(self.cfg['model_save_root'], run_id)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"maisi_controlnet_epoch{epoch:05d}.pt")
        torch.save({
            'epoch': epoch, 'global_step': self.global_step, 'samples_seen': self.samples_seen,
            'elapsed_time': (time.time() - self.global_start_time) + (self.elapsed_time_at_resume if self.elapsed_time_at_resume else 0),
            'controlnet_state_dict': self.controlnet.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scale_factor': self.scale_factor, 'config': self.cfg
        }, path)
        print(f"[Save] {path}")
        if self.cfg['wandb'] and wandb.run:
            local_path = os.path.join(wandb.run.dir, os.path.basename(path))
            if not os.path.exists(local_path): os.symlink(path, local_path)

if __name__ == "__main__":
    pass
