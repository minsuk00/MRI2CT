import datetime
import os
import random
import time
from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio
from monai.transforms import CutOut

import wandb
from common.data import DataPreprocessing, get_augmentations, get_split_subjects, get_subject_paths, stage_data_to_local
from common.utils import apply_synchronized_cutout, clean_state_dict, log_model_summary, send_notification, set_seed


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
        self.local_run_dir = None  # set in _setup_wandb once run ID is known
        self.session_dir = None    # per-resume subdir: <local_run_dir>/sessions/<run_name>/

        # Modular Augmentations
        self.cutout_transform = None
        if getattr(self.cfg, "use_cutout", False):
            print(f"[{self.prefix}] ✂️ Initializing Synchronized CutOut (Prob={self.cfg.cutout_prob}, Alpha={self.cfg.cutout_alpha})")
            self.cutout_transform = CutOut(batch_size=self.cfg.batch_size, alpha=self.cfg.cutout_alpha)

    def apply_cutout(self, mri, ct, seg=None):
        """
        Helper method for subclasses to apply synchronized cutout in their training loops.
        Handles the probability check and calls the centralized utility.
        """
        if self.cutout_transform is not None and random.random() < self.cfg.cutout_prob:
            mri_aug, ct_aug, seg_aug = apply_synchronized_cutout(mri, ct, self.cutout_transform, seg=seg)
            # If seg was None, seg_aug will be None
            return mri_aug, ct_aug, seg_aug
        return mri, ct, seg

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
            tags=getattr(self.cfg, "wandb_tags", []) or [],
            reinit=True,
            dir=self.cfg.log_dir,
            id=wandb_id,
            resume="allow" if not self.cfg.diverge_wandb_branch else None,
        )
        # Stable local dir — <timestamp>_<run_id> on first create, found via glob on resume
        if wandb.run:
            existing = sorted(glob(os.path.join(self.cfg.log_dir, "runs", f"*_{wandb.run.id}")), reverse=True)
            if existing:
                self.local_run_dir = existing[0]
            else:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
                self.local_run_dir = os.path.join(self.cfg.log_dir, "runs", f"{timestamp}_{wandb.run.id}")
            os.makedirs(self.local_run_dir, exist_ok=True)
            self.session_dir = os.path.join(self.local_run_dir, "sessions", self.run_name)
            os.makedirs(self.session_dir, exist_ok=True)

        # Save code state and session-specific config
        if wandb.run:
            wandb.run.log_code(".")
            import yaml

            config_path = os.path.join(self.session_dir, "config_final.yaml")
            with open(config_path, "w") as f:
                yaml.dump(vars(self.cfg), f, default_flow_style=False, sort_keys=True)
            wandb.save(config_path, base_path=self.local_run_dir)

            # Symlink SLURM log into session_dir and sync live to WandB
            slurm_job_id = os.environ.get("SLURM_JOB_ID")
            if slurm_job_id:
                # Search for logs that start with the job ID (standard) or end with it (some custom formats)
                slurm_logs = glob(os.path.join(os.path.dirname(__file__), "../../slurm_logs", f"{slurm_job_id}_*.log"))
                if not slurm_logs:
                    slurm_logs = glob(os.path.join(os.path.dirname(__file__), "../../slurm_logs", f"*_{slurm_job_id}.log"))
                
                if slurm_logs:
                    link_path = os.path.join(self.session_dir, "slurm.log")
                    if not os.path.exists(link_path):
                        os.symlink(os.path.abspath(slurm_logs[0]), link_path)
                    wandb.save(link_path, base_path=self.local_run_dir, policy="live")

    def _log_model_summary(self, model_dict):
        """Standardized model summary logging across all trainers."""
        if not self.cfg.wandb or not wandb.run:
            return

        summary_path = os.path.join(self.local_run_dir, "model_summary.txt")
        tot, train = log_model_summary(model_dict, summary_path)

        wandb.save(summary_path, base_path=self.local_run_dir)
        wandb.config.update({"total_params": tot, "trainable_params": train}, allow_val_change=True)
        return tot, train

    def _load_resume(self, model, optimizer=None, scheduler=None, scaler=None, extra_modules=None):
        if not self.cfg.resume_wandb_id:
            return

        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        # New-style: <timestamp>_<run_id> dir (preferred)
        new_style = sorted(glob(os.path.join(self.cfg.log_dir, "runs", f"*_{self.cfg.resume_wandb_id}")), reverse=True)
        # Old-style: timestamped wandb run folders (backward compat), newest first
        old_run_folders = sorted(glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}")), reverse=True)

        files_dirs = []
        if new_style:
            files_dirs.append(new_style[0])
        files_dirs.extend(os.path.join(rf, "files") for rf in old_run_folders)

        if not files_dirs:
            print(f"[RESUME] ❌ Run folder not found for ID: {self.cfg.resume_wandb_id}")
            return

        if self.cfg.resume_epoch is not None:
            # Load a specific epoch snapshot
            target_ckpt = None
            for fd in files_dirs:
                matches = glob(os.path.join(fd, f"*epoch{self.cfg.resume_epoch:05d}.pt"))
                if matches:
                    target_ckpt = matches[0]
                    break
            if target_ckpt is None:
                print(f"[RESUME] ❌ No checkpoint found for epoch {self.cfg.resume_epoch}.")
                return
        else:
            # Default: look for checkpoint_last.pt across ALL candidate folders
            target_ckpt = None
            candidates = []
            for fd in files_dirs:
                last = os.path.join(fd, "checkpoint_last.pt")
                if os.path.exists(last):
                    candidates.append(last)
            
            if candidates:
                # Pick the one with the highest epoch stored inside
                def get_ckpt_epoch(p):
                    try:
                        c = torch.load(p, map_location='cpu', weights_only=False)
                        return c.get('epoch', -1)
                    except:
                        return -1
                target_ckpt = max(candidates, key=get_ckpt_epoch)
            
            if target_ckpt is None:
                # Fallback for runs that predate checkpoint_last.pt
                all_ckpts = []
                for fd in files_dirs:
                    all_ckpts.extend(glob(os.path.join(fd, "*.pt")))
                    all_ckpts.extend(glob(os.path.join(fd, "*.pth")))
                if not all_ckpts:
                    print("[RESUME] ⚠️ No checkpoints found in run folder.")
                    return

                def get_epoch(path):
                    try:
                        name = os.path.basename(path)
                        return int(name.split("epoch")[-1].split(".")[0].strip("_"))
                    except Exception:
                        return -1

                all_ckpts.sort(key=get_epoch)
                target_ckpt = all_ckpts[-1]

        print(f"[RESUME] 🔄 Loading checkpoint: {target_ckpt}")
        checkpoint = torch.load(target_ckpt, map_location=self.device, weights_only=False)

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
            prep = DataPreprocessing(patch_size=self.cfg.patch_size, res_mult=self.cfg.res_mult, enforce_ras=getattr(self.cfg, "enforce_ras", False))
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

    def _log_val_metrics(self, val_metrics, exclude=None, extra=None, subject_ids=None):
        """Compute mean over val subjects and log to WandB. Returns avg_met."""
        exclude = exclude or set()
        avg_met = {k: np.mean(v) for k, v in val_metrics.items() if not k.startswith("body_")}
        min_met = {k: np.min(v)  for k, v in val_metrics.items() if not k.startswith("body_")}
        max_met = {k: np.max(v)  for k, v in val_metrics.items() if not k.startswith("body_")}
        avg_body = {k[5:]: np.mean(v) for k, v in val_metrics.items() if k.startswith("body_")}
        min_body = {k[5:]: np.min(v)  for k, v in val_metrics.items() if k.startswith("body_")}
        max_body = {k[5:]: np.max(v)  for k, v in val_metrics.items() if k.startswith("body_")}

        if extra:
            avg_met.update(extra)

        if self.cfg.wandb:
            wandb.log(
                {f"val/{k}": v for k, v in avg_met.items() if k not in exclude},
                # | {f"val/{k}_min": v for k, v in min_met.items() if k not in exclude}
                # | {f"val/{k}_max": v for k, v in max_met.items() if k not in exclude},
                step=self.global_step,
            )
            if avg_body:
                wandb.log(
                    {f"val_body/{k}": v for k, v in avg_body.items()},
                    # | {f"val_body/{k}_min": v for k, v in min_body.items()}
                    # | {f"val_body/{k}_max": v for k, v in max_body.items()},
                    step=self.global_step,
                )

        if subject_ids is not None and "mae_hu" in val_metrics:
            self._save_val_ranking(subject_ids, val_metrics)

        return avg_met

    def _save_val_ranking(self, subject_ids, val_metrics):
        """Write per-subject val metrics sorted by MAE (best→worst). Overwrites each val run."""
        mae_vals = val_metrics["mae_hu"]
        extra_keys = [k for k in ("ssim", "psnr", "dice_score_all", "dice_score_bone") if k in val_metrics]
        rows = sorted(zip(subject_ids, mae_vals, *[val_metrics[k] for k in extra_keys]), key=lambda x: x[1])

        if self.local_run_dir:
            path = os.path.join(self.local_run_dir, "val_rankings.txt")
        else:
            path = os.path.join(self.cfg.prediction_dir, self.run_name, "val_rankings.txt")
        os.makedirs(os.path.dirname(path), exist_ok=True)

        header_extra = "".join(f"  {k:<16}" for k in extra_keys)
        with open(path, "w") as f:
            f.write(f"# Val ranking by MAE | Step {self.global_step}\n")
            f.write(f"{'rank':<6}{'subject_id':<20}{'mae_hu':<12}{header_extra}\n")
            for rank, row in enumerate(rows, 1):
                sid, mae = row[0], row[1]
                extra_vals = "".join(f"  {v:<16.4f}" for v in row[2:])
                f.write(f"{rank:<6}{sid:<20}{mae:<12.2f}{extra_vals}\n")

    def _compute_val_metrics(self, pred_unpad, ct_unpad, mask_unpad=None):
        """Returns (standard_met, body_met_or_None). Log under val/ and val_body/ respectively."""
        from common.utils import compute_metrics, compute_metrics_body

        met = compute_metrics(pred_unpad, ct_unpad)
        body_met = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad) if mask_unpad is not None else None
        return met, body_met

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

    def _precompute_gt_drrs(self):
        """Render GT CT DRRs once at init for all viz subjects. Cached in self.gt_drrs."""
        from evaluate.drr import render_drr

        self.gt_drrs = {}
        self.drr_thetas = None
        viz_subj_ids = [self.val_subjects[i] for i in sorted(self.val_viz_indices)]
        print(f"[{self.prefix}] Pre-computing GT DRRs for {len(viz_subj_ids)} subjects...")
        for subj_id in viz_subj_ids:
            gt_path = get_subject_paths(self.cfg.root_dir, subj_id)["ct"]
            img, thetas = render_drr(
                gt_path, self.device,
                num_angles=self.cfg.val_drr_angles,
                height=self.cfg.val_drr_res,
                width=self.cfg.val_drr_res,
            )
            self.gt_drrs[subj_id] = img[:, 0].cpu().numpy()  # [N, H, W]
            if self.drr_thetas is None:
                self.drr_thetas = thetas
            del img
            torch.cuda.empty_cache()
        print(f"[{self.prefix}] GT DRRs cached for: {list(self.gt_drrs.keys())}")

    @torch.inference_mode()
    def _log_drr_comparison(self, subj_id, save_path):
        """Render sCT DRR from save_path, compare with cached GT DRR, log to WandB."""
        from evaluate.drr import make_drr_figure, render_drr

        try:
            img_pred, _ = render_drr(
                save_path, self.device,
                num_angles=self.cfg.val_drr_angles,
                height=self.cfg.val_drr_res,
                width=self.cfg.val_drr_res,
            )
            pred_np = img_pred[:, 0].cpu().numpy()
            del img_pred
            torch.cuda.empty_cache()  # free subject + img_pred before CPU-only figure work
            from evaluate.drr import DELX, ORIENTATION, SDD, TRANSLATION
            res = self.cfg.val_drr_res
            cam_info = f"SDD={SDD}mm  delx=dely={DELX}mm  res={res}×{res}  translation={TRANSLATION}mm  orientation={ORIENTATION}"
            caption = f"pred: {save_path}\n{cam_info}"
            fig = make_drr_figure(self.gt_drrs[subj_id], pred_np, self.drr_thetas, title=f"DRR: {subj_id}", caption=caption)
            wandb.log({f"val_drr/{subj_id}": wandb.Image(fig)}, step=self.global_step)
            plt.close(fig)
        except Exception as e:
            print(f"[{self.prefix}] [WARNING] DRR comparison failed for {subj_id}: {e}")
