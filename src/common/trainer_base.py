import datetime
import os
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio

import wandb
from common.data import DataPreprocessing, get_augmentations, get_split_subjects, get_subject_paths, stage_data_to_local
from common.utils import clean_state_dict, send_notification, set_seed


class BaseTrainer:
    def __init__(self, cfg, prefix="Trainer"):
        self.cfg = cfg
        self.prefix = prefix
        self.gpfs_root = self.cfg.root_dir

        set_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"[{self.prefix}] 🚀 Initializing on {self.device}")

        # State Tracking
        self.start_epoch = 0
        self.global_step = 0
        self.samples_seen = 0
        self.global_start_time = None
        self.elapsed_time_at_resume = 0

        # Consistent run name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.run_name = f"{timestamp}_{self.cfg.run_name_prefix}" if hasattr(self.cfg, "run_name_prefix") else f"{timestamp}_Train"

    def _find_subjects(self):
        """Discovers subjects from split file, or uses explicit subjects list for SSO."""
        if not hasattr(self, "train_subjects"):
            if hasattr(self.cfg, "subjects") and self.cfg.subjects:
                self.train_subjects = list(self.cfg.subjects)
                self.val_subjects = list(self.cfg.subjects)
                print(f"[{self.prefix}] 🔬 SSO Mode: subjects={self.train_subjects}")
            else:
                print(f"[{self.prefix}] 📂 Using split file: {self.cfg.split_file}")
                self.train_subjects = get_split_subjects(self.cfg.split_file, "train")
                self.val_subjects = get_split_subjects(self.cfg.split_file, "val")
                print(f"[{self.prefix}] Data Split - Train: {len(self.train_subjects)} | Val: {len(self.val_subjects)}")

    def _stage_data_local(self):
        """Copies dataset to local NVMe RAID for blazing fast I/O."""
        all_to_sync = sorted(list(set(self.train_subjects) | set(self.val_subjects)))
        self.cfg.root_dir = stage_data_to_local(self.gpfs_root, all_to_sync, self.cfg, prefix=self.prefix)
        print(f"[{self.prefix}] ✅ Data staged. New root: {self.cfg.root_dir}")

    def _setup_wandb(self):
        if not self.cfg.wandb:
            return

        os.makedirs(self.cfg.log_dir, exist_ok=True)
        wandb_id = None if self.cfg.diverge_wandb_branch else self.cfg.resume_wandb_id

        wandb.init(
            project=self.cfg.project_name,
            name=self.run_name,
            config=vars(self.cfg),
            notes=self.cfg.wandb_note,
            reinit=True,
            dir=self.cfg.log_dir,
            id=wandb_id,
            resume="allow" if not self.cfg.diverge_wandb_branch else None,
        )
        # Save code state for reproducibility
        if wandb.run:
            wandb.run.log_code(".")

    def _load_resume(self, model, optimizer=None, scheduler=None, scaler=None, extra_modules=None):
        if not self.cfg.resume_wandb_id:
            return

        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        run_folders = glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}"))
        if not run_folders:
            print(f"[RESUME] ❌ Run folder not found for ID: {self.cfg.resume_wandb_id}")
            return

        all_ckpts = []
        for rf in run_folders:
            all_ckpts.extend(glob(os.path.join(rf, "files", "*.pt")))
            all_ckpts.extend(glob(os.path.join(rf, "files", "*.pth")))

        if not all_ckpts:
            print("[RESUME] ⚠️ No checkpoints found in run folder.")
            return

        # Sort by epoch if possible
        def get_epoch(path):
            try:
                name = os.path.basename(path)
                return int(name.split("epoch")[-1].split(".")[0].strip("_"))
            except Exception:
                return -1

        all_ckpts.sort(key=get_epoch)

        target_ckpt = all_ckpts[-1]
        if self.cfg.resume_epoch is not None:
            for ckpt in all_ckpts:
                if get_epoch(ckpt) == self.cfg.resume_epoch:
                    target_ckpt = ckpt
                    break

        print(f"[RESUME] 🔄 Loading checkpoint: {target_ckpt}")
        checkpoint = torch.load(target_ckpt, map_location=self.device)

        # Helper to load into potentially compiled model
        def load_state(m, state):
            state = clean_state_dict(state)
            getattr(m, "_orig_mod", m).load_state_dict(state)

        # 1. Restore Training State (so step info is available for scheduler calculation)
        if not self.cfg.diverge_wandb_branch:
            self.start_epoch = checkpoint.get("epoch", -1) + 1
            self.global_step = checkpoint.get("global_step", (self.start_epoch) * self.cfg.steps_per_epoch)
            self.samples_seen = checkpoint.get("samples_seen", 0)
            self.elapsed_time_at_resume = checkpoint.get("elapsed_time", 0)

        # 2. Restore Model Weights
        load_state(model, checkpoint["model_state_dict"])
        if extra_modules:
            for key, mod in extra_modules.items():
                state_key = f"{key}_state_dict"
                if state_key in checkpoint:
                    print(f"[RESUME] 📥 Loading {key} state...")
                    load_state(mod, checkpoint[state_key])

        # 3. Restore Optimizer/Scheduler/Scaler
        if not self.cfg.diverge_wandb_branch:
            # Capture the fresh LRs from config (which were just set in _setup_opt)
            if optimizer:
                intended_lrs = [group["lr"] for group in optimizer.param_groups]

                if "optimizer_state_dict" in checkpoint:
                    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

                # If override_lr is set, immediately force the optimizer back to the intended LRs
                if getattr(self.cfg, "override_lr", False):
                    print("[RESUME] 🔧 override_lr=True: Forcing new Learning Rates on optimizer.")
                    for group, lr in zip(optimizer.param_groups, intended_lrs):
                        group["lr"] = lr

            if scheduler and "scheduler_state_dict" in checkpoint:
                if getattr(self.cfg, "override_lr", False):
                    # Logic: Recalculate curve based on current step, skipping old state load
                    print("[RESUME] 🔧 Override LR enabled: Recalculating LR curve.")
                    scheduler.base_lrs = intended_lrs

                    new_t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
                    if hasattr(scheduler, "T_max"):
                        scheduler.T_max = new_t_max

                    scheduler.step(self.global_step)
                else:
                    # Standard logic: Load state, update T_max if needed, and step once
                    print("[RESUME] 📥 Loading Scheduler state...")
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    new_t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
                    if hasattr(scheduler, "T_max") and new_t_max != scheduler.T_max:
                        print(f"[RESUME] 🔧 Updating Scheduler T_max: {scheduler.T_max} -> {new_t_max}")
                        scheduler.T_max = new_t_max

                    restored_epoch = scheduler.last_epoch
                    scheduler.last_epoch = restored_epoch - 1
                    scheduler.step()

            if scaler and "scaler_state_dict" in checkpoint:
                scaler.load_state_dict(checkpoint["scaler_state_dict"])

            print(f"[RESUME] ✅ Resumed from Epoch {self.start_epoch}")
        else:
            print("[RESUME] 🌿 Diverging. Weights loaded, but state (epoch, step) reset.")

    def save_checkpoint(self, model, optimizer, scheduler, scaler, epoch, path, extra_state=None):
        """Universal checkpoint saver."""
        save_dir = os.path.dirname(path)
        os.makedirs(save_dir, exist_ok=True)

        target_model = getattr(model, "_orig_mod", model)
        save_dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "model_state_dict": target_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "elapsed_time": self.elapsed_time_at_resume + (time.time() - self.global_start_time) if self.global_start_time else self.elapsed_time_at_resume,
            "config": vars(self.cfg),
        }

        if extra_state:
            save_dict.update(extra_state)

        torch.save(save_dict, path)
        print(f"[{self.prefix}] [Save] {path}")

    def _inf_gen(self, loader):
        """Infinite generator for tio.SubjectsLoader."""
        while True:
            iterator = iter(loader)
            for batch in iterator:
                yield batch
            del iterator
            import gc

            gc.collect()

    def _log_aug_viz(self, step):
        """Visualizes augmentations."""
        try:
            subj_id = self.val_subjects[0]
            paths = get_subject_paths(self.cfg.root_dir, subj_id)
            subj = tio.Subject(mri=tio.ScalarImage(paths["mri"]), ct=tio.ScalarImage(paths["ct"]))
            prep = DataPreprocessing(patch_size=self.cfg.patch_size, res_mult=self.cfg.res_mult)
            subj = prep(subj)
            aug = get_augmentations()(subj)
            hist_str = " | ".join([t.name for t in aug.history])
            z = subj["mri"].shape[-1] // 2
            orig_sl = subj["mri"].data[0, ..., z].numpy()
            aug_sl = aug["mri"].data[0, ..., z].numpy()
            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(np.rot90(orig_sl), cmap="gray", vmin=0, vmax=1)
            ax[0].set_title("Original")
            ax[1].imshow(np.rot90(aug_sl), cmap="gray", vmin=0, vmax=1)
            ax[1].set_title(f"Augmented\n{hist_str}")
            ax[2].imshow(np.rot90(aug_sl - orig_sl), cmap="seismic", vmin=-0.5, vmax=0.5)
            ax[2].set_title("Diff")
            wandb.log({"val/aug_viz": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"[{self.prefix}] [WARNING] Aug Viz failed: {e}")

    def _setup_loss_and_scaler(self):
        from common.loss import CompositeLoss

        self.loss_fn = CompositeLoss(
            weights={
                "l1": self.cfg.l1_w,
                "l2": self.cfg.l2_w,
                "ssim": self.cfg.ssim_w,
                "dice_w": getattr(self.cfg, "dice_w", 0.0),
                "dice_bone_w": getattr(self.cfg, "dice_bone_w", 0.0),
                "dice_bone_idx": getattr(self.cfg, "dice_bone_idx", 5),
                "dice_exclude_background": getattr(self.cfg, "dice_exclude_background", True),
            }
        ).to(self.device)
        self.scaler = torch.amp.GradScaler("cuda")

    @torch.inference_mode()
    def _log_training_patch(self, mri, ct, pred, step, batch_idx, seg=None, pred_probs=None, subj_id=None):
        """
        Visualizes MRI, Prediction, and CT (GT) for the first patch in the batch.
        Optionally overlays Segmentation (GT) and Predicted Probs (Argmax).
        """
        img_in = mri[0, 0].detach().cpu().float().numpy()
        img_gt = ct[0, 0].detach().cpu().float().numpy()
        img_pred = pred[0, 0].detach().cpu().float().numpy()

        img_seg = None
        img_pred_seg = None

        if seg is not None:
            img_seg = seg[0, 0].detach().cpu().float().numpy()

        if pred_probs is not None:
            img_pred_seg = torch.argmax(pred_probs[0], dim=0).detach().cpu().float().numpy()

        cx, cy, cz = np.array(img_in.shape) // 2

        nrows = 6 if img_seg is not None else 3
        fig, axes = plt.subplots(nrows, 3, figsize=(10, 3.5 * nrows))

        def plot_row(row_idx, vol, title_prefix, vmin=None, vmax=None, cmap="gray", interpolation=None, alpha=None):
            # Axial
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, alpha=alpha)
            axes[row_idx, 0].set_title(f"{title_prefix} Ax")
            # Sagittal
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, alpha=alpha)
            axes[row_idx, 1].set_title(f"{title_prefix} Sag")
            # Coronal
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap=cmap, vmin=vmin, vmax=vmax, interpolation=interpolation, alpha=alpha)
            axes[row_idx, 2].set_title(f"{title_prefix} Cor")

            if alpha is None:  # Only show min/max text for base layers
                axes[row_idx, 0].text(-5, 10, f"Min: {vol.min():.2f}\nMax: {vol.max():.2f}", fontsize=8, color="white", backgroundcolor="black")

        plot_row(0, img_in, "MRI", vmin=0, vmax=1)
        plot_row(1, img_pred, "Pred", vmin=0, vmax=1)
        plot_row(2, img_gt, "GT CT", vmin=0, vmax=1)

        if img_seg is not None:
            seg_vmin = 0
            seg_vmax = img_seg.max()
            if img_pred_seg is not None:
                seg_vmax = max(seg_vmax, img_pred_seg.max())

            plot_row(3, img_seg, "GT Seg", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest")

            if img_pred_seg is not None:
                plot_row(4, img_pred_seg, "Pred Seg", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest")

                # Overlay Row: Show Pred CT first, then overlay Pred Seg
                plot_row(5, img_pred, "Overlay", vmin=0, vmax=1, cmap="gray")
                plot_row(5, img_pred_seg, "Overlay", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest", alpha=0.3)
            else:
                for r in range(4, 6):
                    for c in range(3):
                        axes[r, c].axis("off")

        for ax in axes.flatten():
            ax.axis("off")

        plt.tight_layout()
        caption = f"Subject: {subj_id}" if subj_id else None
        if self.cfg.wandb:
            wandb.log({"train/patch_viz": wandb.Image(fig, caption=caption)}, step=step)
        plt.close(fig)

    @torch.inference_mode()
    def _visualize_lite(self, pred, ct, mri, subj_id, shape, step, epoch, idx, offset=0, save_path=None):
        """Lightweight visualization: MRI, GT, Pred, Residual."""
        from common.utils import unpad

        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()
        items = [(gt_mri, "GT MRI", "gray", (0, 1)), (gt_ct, "GT CT", "gray", (0, 1)), (pred_ct, "Pred CT", "gray", (0, 1)), (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5))]
        D_dim = gt_ct.shape[-1]
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)
        fig, axes = plt.subplots(len(slice_indices), len(items), figsize=(3 * len(items), 3.5 * len(slice_indices)))
        plt.subplots_adjust(wspace=0.05, hspace=0.15)
        if len(slice_indices) == 1:
            axes = axes.reshape(1, -1)
        for i, z_slice in enumerate(slice_indices):
            for j, (data, title, cmap, clim) in enumerate(items):
                ax = axes[i, j]
                im = ax.imshow(data[:, :, z_slice], cmap=cmap, vmin=clim[0], vmax=clim[1])
                if title == "Residual":
                    res_im = im
                if i == 0:
                    ax.set_title(title)
                ax.axis("off")
        if "res_im" in locals():
            cbar = fig.colorbar(res_im, ax=axes[:, 3], fraction=0.04, pad=0.01)
            cbar.set_label("Residual Error")
        fig.suptitle(f"Subject: {subj_id} | Ep {epoch} | Step {step}", fontsize=16, y=0.99)
        if self.cfg.wandb:
            wandb.log({f"viz/{('val_' + str(idx))}": wandb.Image(fig)}, step=step)
        plt.close(fig)
