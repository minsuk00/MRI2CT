import copy
import gc
import os
import sys
import time
import traceback
from collections import defaultdict

import torch
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import wandb

# Enables TF32 and cuDNN benchmark for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 64

# Add 'src' directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anatomix.model.network import Unet

from common.config import DEFAULT_CONFIG, Config
from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_gpu_transforms,
    get_random_crop,
    gpu_augment_batch,
)
from common.trainer_base import BaseTrainer, StepTimer
from common.utils import cleanup_gpu, count_parameters, send_notification, unpad, visualize_lite

# ==========================================
# CONFIGURATION
# ==========================================
BASELINE_CONFIG = {
    "split_file": "splits/center_wise_split.txt",
    "stage_data": False,
    "batch_size": 8,
    "lr": 3e-4,
    "total_epochs": 1000,
    "steps_per_epoch": 500,  # halved from 1000 since batch_size doubled (4→8); keeps total samples_seen
    "val_interval": 20,
    "model_save_interval": 100,
    "use_weighted_sampler": True,
    "resume_wandb_id": None,
    "resume_epoch": None,  # Optional: specify epoch number
    "diverge_wandb_branch": False,  # Create new run instead of resuming existing
    "dice_w": 0.1,  # Optional Dice loss
    "validate_dice": True,  # Optional Dice validation
    # "enable_profiling": True,
    "enable_profiling": False,
    # "patches_per_volume": 15,
    "data_queue_max_length": 150,
    "data_queue_num_workers": 4,
    # ----------------------
    # Experiment Basics
    "project_name": "mri2ct",
    "run_name_prefix": "unet",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    "wandb_tags": ["unet"],
    "wandb_note": "Baseline U-Net (Input 1 -> Output 1). L1+SSIM.",
    # Data
    "patch_size": 128,  # Same as main
    "res_mult": 16,  # Divisibility requirement
    # Model Architecture (Baseline U-Net)
    "input_nc": 1,
    "output_nc": 1,
    "ngf": 16,
    "num_downs": 4,  # Standard depth
    "model_type": "unet_baseline",
    "norm": "batch",  # "batch", "instance", or "none"
    # Optimization
    "compile_mode": "model",  # "model" compiles UNet+teacher separately; "full" breaks on fused_ssim3d
    "scheduler_min_lr": 0.0,
    "accum_steps": 1,
    "viz_limit": 2,
    # Loss Weights (Matches DEFAULT_CONFIG)
    "l1_w": 1.0,
    "ssim_w": 0.1,
    "l2_w": 0.0,
    "perceptual_w": 0.0,  # Anatomix v1_4 perceptual loss weight (0 = off)
    "perceptual_layers": None,  # comma-separated decoder layer indices; None -> [38,45,52,65]
    "perceptual_metric": "ncc",  # "ncc" (normalized cross-correlation, default) or "l1"
    # Resuming
    "override_lr": False,  # If True, uses 'lr' from config instead of saved state
    # Validation
    "val_sw_batch_size": 1,  # 256^3 val patches: reduce 8x vs 128^3
    "val_sw_overlap": 0.25,
    "save_val_volumes": True,
    # Mode
    "sanity_check": False,
}


# ==========================================
# TRAINER CLASS
# ==========================================
class BaselineTrainer(BaseTrainer):
    def __init__(self, config_dict):
        # Merge with default to ensure all keys exist
        full_conf = copy.deepcopy(DEFAULT_CONFIG)
        full_conf.update(config_dict)
        cfg = Config(full_conf)
        super().__init__(cfg, prefix="UNet_Baseline")

        # Setup
        self._setup_models()
        self._find_subjects()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 8. Model Summary Logging
        self._log_model_summary({"UNet_Baseline": self.model})

        self._load_resume(self.model, self.optimizer, self.scheduler)

        if getattr(self.cfg, "val_drr", False):
            self._precompute_gt_drrs()

    def _setup_data(self):
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)
        cache_dir = default_monai_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Baseline] 💾 MONAI cache dir: {cache_dir}")

        cached_xform = get_cached_transforms(
            patch_size=self.cfg.patch_size,
            res_mult=self.cfg.res_mult,
            enforce_ras=getattr(self.cfg, "enforce_ras", False),
            mri_norm=getattr(self.cfg, "mri_norm", "minmax"),
            load_seg=load_seg,
            use_float16_storage=getattr(self.cfg, "use_float16_storage", False),
        )

        # Train: cached full volumes -> CPU random crop (workers) -> default collate (uniform patches) -> GPU aug.
        train_dicts = build_data_dicts(self.cfg.root_dir, self.train_subjects, load_seg=load_seg)
        base_train = PersistentDataset(data=train_dicts, transform=cached_xform, cache_dir=cache_dir)
        train_random_crop = get_random_crop(
            patch_size=self.cfg.patch_size,
            use_weighted_sampler=self.cfg.use_weighted_sampler,
            has_seg=load_seg,
            num_samples=self.cfg.patches_per_volume,
        )
        train_ds = Dataset(data=base_train, transform=train_random_crop)
        self.train_loader = DataLoader(
            train_ds,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.data_queue_num_workers,
            persistent_workers=True,
            pin_memory=False,
        )
        self.train_iter = self._inf_gen(self.train_loader)

        self.gpu_transforms = get_gpu_transforms(
            augment=self.cfg.augment,
            has_seg=load_seg,
        )

        # Val: same cached transforms, no augmentation (sliding-window inference on full volumes).
        self.val_loader = self._build_val_loader(cached_xform, load_seg, cache_dir)

        # Stratified Validation Sampling (viz_limit per region)
        self.val_viz_indices, _ = self._stratify_val_indices(self.cfg.viz_limit)

    def _setup_models(self):
        print(f"[Baseline] 🏗️ Building Simple U-Net (In: {self.cfg.input_nc}, Out: {self.cfg.output_nc})")
        # Direct MRI -> CT translation
        model = Unet(
            dimension=3,
            input_nc=self.cfg.input_nc,
            output_nc=self.cfg.output_nc,
            num_downs=self.cfg.num_downs,
            ngf=self.cfg.ngf,
            norm=getattr(self.cfg, "norm", "batch"),
            final_act="sigmoid",
        ).to(self.device)

        # 1. Model-level compile (only if specifically requested)
        if self.cfg.compile_mode == "model":
            print("[Baseline] 🚀 Compiling model (default)...")
            self.model = torch.compile(model, mode="default")
        else:
            self.model = model

        # 2. Teacher Model (Baby U-Net) for Dice Loss / Validation
        self.teacher_model = None
        if getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False):
            self.teacher_model = self._setup_teacher_model(compile_model=(self.cfg.compile_mode == "model"))

        # 3. Step-level compile (only if specifically requested)
        # Note: If compile_mode is "full", self.model is the RAW model,
        # so we avoid nested compilation.
        if self.cfg.compile_mode == "full":
            print("[Baseline] 🚀 Compiling training step (default)...")
            self.train_step = torch.compile(self._train_step, mode="default")
        else:
            self.train_step = self._train_step

        tot, train = count_parameters(self.model)
        print(f"[Baseline] Model Params: Total={tot:,} | Trainable={train:,}")

    def _train_step(self, mri, ct, seg=None):
        """Isolated training step for torch.compile"""
        # A40 Optimization: Use bfloat16 AMP
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            pred = self.model(mri)

            # Dice Loss Calculation
            pred_probs = None
            if (getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "dice_bone_w", 0) > 0) and self.teacher_model is not None and seg is not None:
                pred_probs = self.teacher_model(pred)

            loss, comps = self.loss_fn(pred, ct, pred_probs=pred_probs, target_mask=seg)
            loss = loss / self.cfg.accum_steps

        loss.backward()
        return pred, loss, comps, pred_probs

    def _setup_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
        print(f"[Baseline] 📉 Initializing Scheduler (CosineAnnealingLR) T_max={t_max}, min_lr={self.cfg.scheduler_min_lr}")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=self.cfg.scheduler_min_lr)

        self._setup_loss()

    # ==========================================
    # TRAINING LOOPS
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_grad = 0.0
        comp_accum = {}

        pbar = tqdm(range(self.cfg.steps_per_epoch), desc=f"Train Ep {epoch}", leave=False, dynamic_ncols=True)

        monitor_every = max(1, getattr(self.cfg, "monitor_interval", 10))

        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            log_this_step = self.cfg.wandb and getattr(self.cfg, "monitor_resources", False) and step_idx % monitor_every == 0

            with StepTimer(log_this_step) as timer:
                for _ in range(self.cfg.accum_steps):
                    with timer.cpu("data"):
                        batch = next(self.train_iter)
                    with timer.gpu("augment"):
                        # Move-to-device + batched GPU augment via batchaug (one fused grid_sample for spatial ops).
                        batch = gpu_augment_batch(batch, self.gpu_transforms, self.device)

                    mri = batch["mri"]
                    ct = batch["ct"]
                    seg = batch["seg"] if "seg" in batch else None

                    # Modular Synchronized CutOut (via BaseTrainer)
                    mri, ct, seg = self.apply_cutout(mri, ct, seg=seg)

                    with timer.gpu("compute"):
                        # Call the training step (modular)
                        pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                    if self.cfg.wandb and step_idx == 0 and epoch % self.cfg.viz_interval == 0:
                        subj_id = batch["subj_id"][0] if "subj_id" in batch else None
                        self._log_training_patch(mri, ct, pred, self.global_step, step_idx, seg=seg, pred_probs=pred_probs, subj_id=subj_id)

                    for k, v in comps.items():
                        val = v.item() if hasattr(v, "item") else v
                        comp_accum[k] = comp_accum.get(k, 0.0) + (val / self.cfg.accum_steps)

                    step_loss += loss.item()

                    if pred_probs is not None:
                        del pred_probs

                with timer.gpu("optimizer"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

            if self.scheduler is not None:
                if self.scheduler.last_epoch < self.scheduler.T_max:
                    self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.patches_per_volume * self.cfg.accum_steps

            pbar_dict = {"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"}

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar_dict["lr"] = f"{current_lr:.2e}"

            if log_this_step:
                samples_per_step = self.cfg.batch_size * self.cfg.patches_per_volume * self.cfg.accum_steps
                timings = timer.timings_ms()
                self._log_monitoring(timings, throughput=samples_per_step / timer.elapsed_s())
                pbar_dict.update(
                    {
                        "dt": f"{timings['data']:.0f}ms",
                        "cmp": f"{timings['compute']:.0f}ms",
                        "tot": f"{timings['step_total'] / 1000:.2f}s",
                    }
                )

            pbar.set_postfix(pbar_dict)

            # Log step-level info to WandB
            if self.cfg.wandb and step_idx % 200 == 0:
                cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume if self.global_start_time else self.elapsed_time_at_resume
                step_log = {
                    "info/lr": current_lr,
                    "info/grad_norm": grad_norm.item(),
                    "info/samples_seen": self.samples_seen,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/cumulative_time": cumulative_time,
                }
                wandb.log(step_log, step=self.global_step)

        return total_loss / self.cfg.steps_per_epoch, {k: v / self.cfg.steps_per_epoch for k, v in comp_accum.items()}, total_grad / self.cfg.steps_per_epoch

    @torch.inference_mode()
    def validate(self, epoch):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        val_metrics = defaultdict(list)
        val_ps = getattr(self.cfg, "val_patch_size", self.cfg.patch_size)

        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch["mri"].to(self.device)
            ct = batch["ct"].to(self.device)
            seg = batch["seg"].to(self.device) if "seg" in batch else None
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]

            # fp32 inference (no bf16 autocast): bf16's 7-bit mantissa quantizes a
            # single-pass regressor's output to ~4-8 HU steps, biasing the logged
            # SSIM/MAE (and posterizing soft tissue in narrow-window regions).
            pred = sliding_window_inference(
                inputs=mri,
                roi_size=(val_ps, val_ps, val_ps),
                sw_batch_size=self.cfg.val_sw_batch_size,
                predictor=self.model,
                overlap=self.cfg.val_sw_overlap,
                device=self.device,
            )

            # Metrics
            pred_unpad = unpad(pred, orig_shape)
            ct_unpad = unpad(ct, orig_shape)
            mri_unpad = unpad(mri, orig_shape)

            mask_unpad = self._get_body_mask_unpad(batch, orig_shape)

            met, body_met = self._compute_val_metrics(pred_unpad, ct_unpad, mask_unpad)

            # Validation Dice & Probabilities
            pred_probs = None
            if getattr(self.cfg, "validate_dice", False) and self.teacher_model is not None and seg is not None:
                pred_probs = self._run_teacher_sw(pred, val_ps)

            # Total Composite Loss for Validation
            # Cast ct (fp16 from cached storage) to fp32 to match pred:
            # nn.L1Loss on mixed fp16/fp32 inputs would otherwise rely on implicit type promotion.
            l_val, l_comps = self.loss_fn(pred.float(), ct.float(), pred_probs=pred_probs, target_mask=seg, compute_perceptual=False)
            met["loss"] = l_val.item()
            for k, v in l_comps.items():
                met[k] = v.item() if hasattr(v, "item") else v

            # Body-masked dice (only when val_body_mask=True and teacher available)
            if body_met is not None and pred_probs is not None and seg is not None:
                self._compute_dice_metrics(pred_probs, seg, orig_shape, mask_unpad=mask_unpad, target_met=body_met)

            if pred_probs is not None:
                del pred_probs  # Fix VRAM leak

            for k, v in met.items():
                val_metrics[k].append(v)
            if body_met is not None:
                for k, v in body_met.items():
                    val_metrics[f"body_{k}"].append(v)

            # Save predictions (overwrite "last" each val run; epoch_<N> snapshot every val_save_interval)
            save_path = self._save_val_pred(pred_unpad, batch, subj_id, epoch)

            # Viz & DRR (only for selected subjects)
            if i in self.val_viz_indices:
                if self.cfg.wandb:
                    viz_metrics, viz_body = self._select_viz_metrics(met, body_met)
                    visualize_lite(pred_unpad, ct_unpad, mri_unpad, subj_id, orig_shape, self.global_step, epoch, log_name=f"viz/val_{i}", metrics=viz_metrics, body_metrics=viz_body)

                if self.cfg.val_drr and save_path and subj_id in self.gt_drrs:
                    self._log_drr_comparison(subj_id, save_path)

            del mri, ct, pred, pred_unpad, ct_unpad, mri_unpad

        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
            self._log_aug_viz(self.global_step)

        avg_met = self._log_val_metrics(val_metrics, subject_ids=self.val_subjects)

        return avg_met

    def train(self):
        print(f"[Baseline] 🏁 Starting Loop: {self.cfg.total_epochs} epochs")
        self.global_start_time = time.time()

        do_val = bool(self.val_subjects) and not getattr(self.cfg, "no_val", False)
        if do_val and getattr(self.cfg, "sanity_check", False) and not self.cfg.resume_wandb_id:
            print("[Baseline] Running sanity check...")
            self.validate(0)

        global_pbar = tqdm(range(self.start_epoch, self.cfg.total_epochs), desc="🚀 Total Progress", initial=self.start_epoch, total=self.cfg.total_epochs, dynamic_ncols=True, unit="ep")

        for epoch in global_pbar:
            ep_start = time.time()
            loss, comps, gn = self.train_epoch(epoch)

            # Raw (unweighted) per-term magnitudes -> stdout, for picking loss weights.
            raw = {k.replace("loss_", ""): (v.item() if hasattr(v, "item") else v)
                   for k, v in comps.items() if "score" not in k}
            tqdm.write(f"[raw-comps] Ep {epoch} | " + " | ".join(f"{k}={v:.5f}" for k, v in raw.items()))

            # Validation interval
            val_duration = 0.0
            if do_val and ((epoch % self.cfg.val_interval == 0) or (epoch + 1) == self.cfg.total_epochs):
                val_start = time.time()
                avg_met = self.validate(epoch)
                val_duration = time.time() - val_start
                print(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Dice: {avg_met.get('dice_score_all', 0):.4f} | Bone: {avg_met.get('dice_score_bone', 0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume

            if self.cfg.wandb:
                # info/lr is logged in the per-200-steps block above; skip here to avoid
                # overwriting mid-epoch values at the epoch-boundary global_step.
                total_steps = self.cfg.steps_per_epoch * self.cfg.total_epochs
                log = {
                    "train/total": loss,
                    "info/grad_norm": gn,
                    "info/epoch_duration": ep_duration,
                    "info/val_duration": val_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/samples_seen": self.samples_seen,
                    "info/train_pct": self.global_step / total_steps,
                }
                for k, v in comps.items():
                    if "score" in k:
                        continue
                    log[k.replace("loss_", "train/")] = v
                wandb.log(log, step=self.global_step)

            self.save_checkpoint(epoch, is_last=True)
            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)

            # Explicit cleanup after each epoch
            gc.collect()
            torch.cuda.empty_cache()

        if self.cfg.wandb:
            wandb.finish()

    def save_checkpoint(self, epoch, is_last=False):
        filename = "checkpoint_last.pt" if is_last else f"{self.cfg.model_type}_epoch{epoch:05d}.pt"
        path = os.path.join(self._default_save_dir(), filename)
        super().save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, path)


# ==========================================
# MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file", type=str, help="Path to split mapping file (e.g., splits/original_splits.txt)")
    parser.add_argument("--wandb", type=str, default="True", choices=["True", "False"], help="Enable/disable wandb (True/False)")
    parser.add_argument("--dice_w", type=float, help="Dice loss weight")
    parser.add_argument("--resume_id", type=str, help="WandB run ID to resume")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--run_name", type=str, help="WandB run name prefix")
    parser.add_argument("--augment", type=str, choices=["True", "False"], help="Enable/disable data augmentation (True/False)")
    parser.add_argument("--weighted_sampler", type=str, choices=["True", "False"], help="Enable/disable weighted sampler (True/False)")
    parser.add_argument("--epochs", type=int, help="Total epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, help="Number of steps per epoch")
    parser.add_argument("--num_workers", type=int, help="Number of workers for the data queue")
    parser.add_argument("--norm", type=str, choices=["batch", "instance", "none"], help="Normalization type for U-Net (batch, instance, or none)")
    parser.add_argument("--tags", type=str, help="Comma-separated extra WandB tags (e.g. 'thorax,high bone dice')")
    parser.add_argument("--dice_bone_w", type=float, help="Bone-specific dice loss weight")
    parser.add_argument("--ssim_w", type=float, help="SSIM loss weight (0=off)")
    parser.add_argument("--use_cutout", type=str, choices=["True", "False"], help="Enable/disable cutout augmentation (True/False)")
    parser.add_argument("--cutout_alpha", type=float, help="Beta(alpha, alpha) parameter controlling cutout box size distribution")
    parser.add_argument("--validate_dice", type=str, choices=["True", "False"], help="Enable/disable dice validation (True/False)")
    parser.add_argument("--stage_data", type=str, choices=["True", "False"], help="Stage data to local NVMe (True/False)")
    parser.add_argument("--val_interval", type=int, help="Run validation every N epochs")
    parser.add_argument("--val_patch_size", type=int, help="Sliding-window val ROI size (default 256; lower to cut val VRAM)")
    parser.add_argument("--no_val", action="store_true", help="Skip all validation (fast loss-magnitude probes)")
    parser.add_argument("--perceptual_w", type=float, help="Anatomix v1_4 perceptual loss weight (0=off)")
    parser.add_argument("--perceptual_layers", type=str, help="Comma-separated decoder layer indices (default: 38,45,52,65)")
    parser.add_argument("--perceptual_metric", type=str, choices=["l1", "ncc"], help="Perceptual feature distance: 'l1' (default) or 'ncc'")
    parser.add_argument("--perceptual_separable", type=str, choices=["True", "False"], help="LNCC box-sum via separable 1-D convs (default True, exact & faster); False = dense conv")
    args = parser.parse_args()

    # Convert wandb arg to boolean
    use_wandb = args.wandb == "True"
    BASELINE_CONFIG["wandb"] = use_wandb

    # Override with CLI args if provided
    if args.use_cutout is not None:
        BASELINE_CONFIG["use_cutout"] = args.use_cutout == "True"
    if args.cutout_alpha is not None:
        BASELINE_CONFIG["cutout_alpha"] = args.cutout_alpha
    if args.validate_dice is not None:
        BASELINE_CONFIG["validate_dice"] = args.validate_dice == "True"
    if args.dice_w is not None:
        BASELINE_CONFIG["dice_w"] = args.dice_w
    if args.dice_bone_w is not None:
        BASELINE_CONFIG["dice_bone_w"] = args.dice_bone_w
    if args.ssim_w is not None:
        BASELINE_CONFIG["ssim_w"] = args.ssim_w
    if args.resume_id is not None:
        BASELINE_CONFIG["resume_wandb_id"] = args.resume_id
    if args.batch_size is not None:
        BASELINE_CONFIG["batch_size"] = args.batch_size
    if args.run_name is not None:
        BASELINE_CONFIG["run_name_prefix"] = args.run_name
    if args.augment is not None:
        BASELINE_CONFIG["augment"] = args.augment == "True"
    if args.weighted_sampler is not None:
        BASELINE_CONFIG["use_weighted_sampler"] = args.weighted_sampler == "True"
    if args.split_file is not None:
        BASELINE_CONFIG["split_file"] = args.split_file
    if args.epochs is not None:
        BASELINE_CONFIG["total_epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        BASELINE_CONFIG["steps_per_epoch"] = args.steps_per_epoch
    if args.num_workers is not None:
        BASELINE_CONFIG["data_queue_num_workers"] = args.num_workers
    if args.norm is not None:
        BASELINE_CONFIG["norm"] = args.norm
    if args.tags is not None:
        BASELINE_CONFIG.setdefault("wandb_tags", [])
        BASELINE_CONFIG["wandb_tags"] = BASELINE_CONFIG["wandb_tags"] + [t.strip(' "') for t in args.tags.split(",") if t.strip()]
    if args.stage_data is not None:
        BASELINE_CONFIG["stage_data"] = args.stage_data == "True"
    if args.val_interval is not None:
        BASELINE_CONFIG["val_interval"] = args.val_interval
    if args.val_patch_size is not None:
        BASELINE_CONFIG["val_patch_size"] = args.val_patch_size
    if args.no_val:
        BASELINE_CONFIG["no_val"] = True
    if args.perceptual_w is not None:
        BASELINE_CONFIG["perceptual_w"] = args.perceptual_w
    if args.perceptual_layers is not None:
        BASELINE_CONFIG["perceptual_layers"] = args.perceptual_layers
    if args.perceptual_metric is not None:
        BASELINE_CONFIG["perceptual_metric"] = args.perceptual_metric
    if args.perceptual_separable is not None:
        BASELINE_CONFIG["perceptual_separable"] = args.perceptual_separable == "True"

    try:
        trainer = BaselineTrainer(BASELINE_CONFIG)
        trainer.train()
        cleanup_gpu()
    except KeyboardInterrupt:
        print("Interrupted.")
        cleanup_gpu()
    except Exception as e:
        send_notification(f"❌ UNet Baseline Failed: {e}")
        traceback.print_exc()
        cleanup_gpu()
