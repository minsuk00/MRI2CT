import copy
import gc
import os
import random
import sys
import time
import traceback
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import wandb

# Enables TF32 and cuDNN benchmark for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 64

import warnings

warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence for multidimensional indexing is deprecated.*")

# Add 'src' directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anatomix.model.network import Unet

from common.config import DEFAULT_CONFIG, Config
from common.data import DataPreprocessing, build_tio_subjects, get_augmentations, get_region_key, get_subject_paths
from common.trainer_base import BaseTrainer
from common.utils import cleanup_gpu, compute_metrics, count_parameters, get_ram_info, send_notification, set_seed, unpad

# ==========================================
# CONFIGURATION
# ==========================================
BASELINE_CONFIG = {
    "split_file": "splits/original_splits.txt",
    "stage_data": True,
    "batch_size": 4,
    "lr": 3e-4,
    "total_epochs": 1000,
    "steps_per_epoch": 1000,
    "val_interval": 2,
    "model_save_interval": 1,
    "use_weighted_sampler": True,
    "resume_wandb_id": None,
    "resume_epoch": None,  # Optional: specify epoch number
    "diverge_wandb_branch": False,  # Create new run instead of resuming existing
    "dice_w": 0.1,  # Optional Dice loss
    "validate_dice": True,  # Optional Dice validation
    # "enable_profiling": True,
    "enable_profiling": False,
    "patches_per_volume": 15,
    "data_queue_max_length": 150,
    "data_queue_num_workers": 4,
    # ----------------------
    # Experiment Basics
    "project_name": "MRI2CT_UNet_Baseline",
    "run_name_prefix": "UNet_Baseline",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
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
    # Optimization
    "compile_mode": "full",  # "full", "model", or None
    "scheduler_min_lr": 0.0,
    "accum_steps": 1,
    "viz_limit": 10,
    # Loss Weights (Matches DEFAULT_CONFIG)
    "l1_w": 1.0,
    "ssim_w": 0.1,
    "l2_w": 0.0,
    # Resuming
    "override_lr": False,  # If True, uses 'lr' from config instead of saved state
    # Validation
    "val_sw_batch_size": 8,
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
        self._stage_data_local()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        self._load_resume(self.model, self.optimizer, self.scheduler, self.scaler)

    def _setup_data(self, seed=None):
        if seed is not None:
            # Seed-based worker rotation for correct data coverage
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 1. Subjects already found in _find_subjects()
        # 2. Build Datasets
        # Load seg if Dice weight > 0 or Dice validation is enabled
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)

        # Train Queue
        train_objs = build_tio_subjects(self.cfg.root_dir, self.train_subjects, use_weighted_sampler=self.cfg.use_weighted_sampler, load_seg=load_seg)
        preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult, use_weighted_sampler=self.cfg.use_weighted_sampler)
        transforms = tio.Compose([preprocess, get_augmentations()]) if self.cfg.augment else preprocess
        train_ds = tio.SubjectsDataset(train_objs, transform=transforms)

        if self.cfg.use_weighted_sampler:
            print("[Baseline] ⚖️ Initializing Weighted Sampler (using body_mask.nii.gz)")
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

        # Create infinite iterator (Must re-bind every time loader is created)
        self.train_iter = self._inf_gen(self.train_loader)

        # Val Loader
        val_objs = build_tio_subjects(self.cfg.root_dir, self.val_subjects, load_seg=load_seg)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess)
        # Use SubjectsLoader to avoid warnings and ensure correct collation
        self.val_loader = tio.SubjectsLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Stratified Validation Sampling (2 per region)
        total_val = len(self.val_subjects)
        rng = random.Random(self.cfg.seed)

        region_to_indices = defaultdict(list)
        for idx, subj_id in enumerate(self.val_subjects):
            region = get_region_key(subj_id)
            region_to_indices[region].append(idx)

        viz_indices = []
        for region, indices in region_to_indices.items():
            # Pick up to 2 subjects per region
            num_to_pick = min(len(indices), 2)
            viz_indices.extend(rng.sample(indices, num_to_pick))

        self.val_viz_indices = set(viz_indices)

    def _setup_models(self):
        print(f"[Baseline] 🏗️ Building Simple U-Net (In: {self.cfg.input_nc}, Out: {self.cfg.output_nc})")
        # Direct MRI -> CT translation
        model = Unet(
            dimension=3,
            input_nc=self.cfg.input_nc,
            output_nc=self.cfg.output_nc,
            num_downs=self.cfg.num_downs,
            ngf=self.cfg.ngf,
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
            from anatomix.segmentation.segmentation_utils import load_model_v1_2

            print("[Baseline] 👨‍🏫 Initializing Baby U-Net Teacher for Dice Loss...")
            try:
                # Load Baby U-Net (12 classes: 11 organs + Brain)
                self.teacher_model = load_model_v1_2(pretrained_ckpt=self.cfg.teacher_weights_path, n_classes=self.cfg.n_classes - 1, device=self.device, compile_model=False)

                # Freeze Teacher
                self.teacher_model.to(device=self.device, dtype=torch.bfloat16)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                if self.cfg.compile_mode == "model":
                    print("[Baseline] 🚀 Compiling Teacher with mode: default")
                    self.teacher_model = torch.compile(self.teacher_model, mode="default")

                tot, train = count_parameters(self.teacher_model)
                print(f"[Baseline] Teacher Params: Total={tot:,} | Trainable={train:,} | Dtype=BFloat16")
            except Exception as e:
                print(f"[Baseline] ❌ Failed to init Teacher Model: {e}")
                if self.cfg.dice_w > 0:
                    raise e

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

        self.scaler.scale(loss).backward()
        return pred, loss, comps, pred_probs

    def _setup_opt(self):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)

        t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
        print(f"[Baseline] 📉 Initializing Scheduler (CosineAnnealingLR) T_max={t_max}, min_lr={self.cfg.scheduler_min_lr}")
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=self.cfg.scheduler_min_lr)

        self._setup_loss_and_scaler()

    # ==========================================
    # TRAINING LOOPS
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_grad = 0.0
        comp_accum = {}

        pbar = tqdm(range(self.cfg.steps_per_epoch), desc=f"Train Ep {epoch}", leave=False, dynamic_ncols=True)

        for step_idx in pbar:
            self.optimizer.zero_grad(set_to_none=True)
            step_loss = 0.0

            if self.cfg.enable_profiling:
                step_t_data = 0.0
                step_t_fwd = 0.0
                t_step_start = time.time()

            for _ in range(self.cfg.accum_steps):
                if self.cfg.enable_profiling:
                    t0 = time.time()
                batch = next(self.train_iter)
                if self.cfg.enable_profiling:
                    t1 = time.time()
                    step_t_data += t1 - t0

                # Convert to plain tensors for compiler stability
                mri = batch["mri"][tio.DATA].to(self.device, non_blocking=True)
                ct = batch["ct"][tio.DATA].to(self.device, non_blocking=True)
                seg = batch["seg"][tio.DATA].to(self.device, non_blocking=True) if "seg" in batch else None

                if self.cfg.enable_profiling:
                    torch.cuda.synchronize()
                    t2 = time.time()

                # Call the training step (modular)
                pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                if self.cfg.enable_profiling:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    step_t_fwd += t3 - t2

                if self.cfg.wandb and step_idx == 0:
                    subj_id = batch["subj_id"][0] if "subj_id" in batch else None
                    self._log_training_patch(mri, ct, pred, self.global_step, step_idx, seg=seg, pred_probs=pred_probs, subj_id=subj_id)

                for k, v in comps.items():
                    # Handle both tensors (from compiled step) and scalars
                    val = v.item() if hasattr(v, "item") else v
                    comp_accum[k] = comp_accum.get(k, 0.0) + (val / self.cfg.accum_steps)

                step_loss += loss.item()

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                if self.scheduler.last_epoch < self.scheduler.T_max:
                    self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()
            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            pbar_dict = {"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"}

            # Add LR to tqdm
            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar_dict["lr"] = f"{current_lr:.2e}"

            if self.cfg.enable_profiling:
                t_step_end = time.time()
                step_t_total = t_step_end - t_step_start

                avg_data = (step_t_data / self.cfg.accum_steps) * 1000
                avg_fwd = (step_t_fwd / self.cfg.accum_steps) * 1000

                pbar_dict.update({"dt": f"{avg_data:.1f}ms", "fwd": f"{avg_fwd:.1f}ms", "tot": f"{step_t_total:.2f}s", "lr": f"{current_lr:.2e}"})

                # Log to WandB EVERY step for accurate profiling
                if self.cfg.wandb:
                    wandb.log(
                        {
                            "info/time_data_ms": avg_data,
                            "info/time_forward_ms": avg_fwd,
                            "info/time_step_total_s": step_t_total,
                        },
                        step=self.global_step,
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

            # ---------------------------------------------------------
            # RAM 모니터링 (Optional)
            # ---------------------------------------------------------
            if self.cfg.wandb and self.cfg.enable_profiling and step_idx % 200 == 0:
                ram_info = get_ram_info()

                queue = self.train_loader.dataset
                curr_patches = len(queue.patches_list)

                wandb.log(
                    {
                        "perf/ram_system_percent": ram_info["percent"],
                        "perf/ram_app_total_gb": ram_info["total_gb"],
                        "perf/ram_main_rss_gb": ram_info["main_rss_gb"],
                        "perf/num_workers": ram_info["num_children"],
                        "perf/queue_curr_patches": curr_patches,
                    },
                    step=self.global_step,
                )

                if ram_info["percent"] > 90:
                    tqdm.write(f"[WARNING] ⚠️ RAM usage critical: {ram_info['percent']:.1f}% (Total: {ram_info['total_gb']:.2f} GB)")
                    gc.collect()
                    torch.cuda.empty_cache()
            # ---------------------------------------------------------

        return total_loss / self.cfg.steps_per_epoch, {k: v / self.cfg.steps_per_epoch for k, v in comp_accum.items()}, total_grad / self.cfg.steps_per_epoch

    @torch.inference_mode()
    def validate(self, epoch):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        val_metrics = defaultdict(list)

        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch["mri"][tio.DATA].to(self.device)
            ct = batch["ct"][tio.DATA].to(self.device)
            seg = batch["seg"][tio.DATA].to(self.device) if "seg" in batch else None
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]

            # Safe pad_offset extraction
            po = batch.get("pad_offset", 0)
            if torch.is_tensor(po):
                pad_offset = int(po[0])
            elif isinstance(po, (list, tuple)):
                pad_offset = int(po[0])
            else:
                pad_offset = int(po)

            # Inference
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = sliding_window_inference(
                    inputs=mri,
                    roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size),
                    sw_batch_size=self.cfg.val_sw_batch_size,
                    predictor=self.model,
                    overlap=self.cfg.val_sw_overlap,
                    device=self.device,
                )

            # Metrics
            pred_unpad = unpad(pred, orig_shape, pad_offset)
            ct_unpad = unpad(ct, orig_shape, pad_offset)
            met = compute_metrics(pred_unpad, ct_unpad)

            # Validation Dice & Probabilities
            pred_probs = None
            if getattr(self.cfg, "validate_dice", False) and self.teacher_model is not None and seg is not None:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    # Always use sliding window for teacher on full volumes to prevent OOM
                    pred_probs = sliding_window_inference(
                        inputs=pred,
                        roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size),
                        sw_batch_size=self.cfg.val_sw_batch_size,
                        predictor=self.teacher_model,
                        overlap=self.cfg.val_sw_overlap,
                        device=self.device,
                    )

            # Total Composite Loss for Validation
            l_val, l_comps = self.loss_fn(pred, ct, pred_probs=pred_probs, target_mask=seg)
            met["loss"] = l_val.item()
            for k, v in l_comps.items():
                met[k] = v.item() if hasattr(v, "item") else v

            if pred_probs is not None:
                del pred_probs  # Fix VRAM leak

            for k, v in met.items():
                val_metrics[k].append(v)

            # Save Volumes (Optional - only visualized to save time)
            if self.cfg.save_val_volumes and i in self.val_viz_indices:
                save_dir = os.path.join(self.cfg.prediction_dir, self.run_name, f"epoch_{epoch}")
                os.makedirs(save_dir, exist_ok=True)
                pred_np = pred_unpad.float().cpu().numpy().squeeze()
                pred_hu = (pred_np * 2048.0) - 1024.0

                # Safe affine extraction (handles both Tensors and numpy arrays)
                affine = batch["ct"]["affine"][0]
                if hasattr(affine, "cpu"):
                    affine = affine.cpu().numpy()
                else:
                    affine = np.array(affine)

                nii = nib.Nifti1Image(pred_hu, affine)
                save_path = os.path.join(save_dir, f"pred_{subj_id}.nii.gz")
                nib.save(nii, save_path)
            else:
                save_path = None

            # Viz
            if i in self.val_viz_indices:
                if self.cfg.wandb:
                    self._visualize_lite(pred, ct, mri, subj_id, orig_shape, self.global_step, epoch, idx=i, offset=pad_offset, save_path=save_path)

            del mri, ct, pred, pred_unpad, ct_unpad

        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
            self._log_aug_viz(self.global_step)

        avg_met = {k: np.mean(v) for k, v in val_metrics.items()}
        if self.cfg.wandb:
            wandb.log({f"val/{k}": v for k, v in avg_met.items()}, step=self.global_step)

        return avg_met

    def train(self):
        print(f"[Baseline] 🏁 Starting Loop: {self.cfg.total_epochs} epochs")
        self.global_start_time = time.time()

        if getattr(self.cfg, "sanity_check", False) and not self.cfg.resume_wandb_id:
            print("[Baseline] Running sanity check...")
            self.validate(0)

        global_pbar = tqdm(range(self.start_epoch, self.cfg.total_epochs), desc="🚀 Total Progress", initial=self.start_epoch, total=self.cfg.total_epochs, dynamic_ncols=True, unit="ep")

        for epoch in global_pbar:
            ep_start = time.time()
            loss, comps, gn = self.train_epoch(epoch)

            # Validation interval
            val_duration = 0.0
            if (epoch % self.cfg.val_interval == 0) or (epoch + 1) == self.cfg.total_epochs:
                val_start = time.time()
                avg_met = self.validate(epoch)
                val_duration = time.time() - val_start
                print(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Dice: {avg_met.get('dice_score_all', 0):.4f} | Bone: {avg_met.get('dice_score_bone', 0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume

            if self.cfg.wandb:
                current_lr = self.optimizer.param_groups[0]["lr"]
                log = {
                    "train/total": loss,
                    "info/grad_norm": gn,
                    "info/epoch_duration": ep_duration,
                    "info/val_duration": val_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/lr": current_lr,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/samples_seen": self.samples_seen,
                }
                for k, v in comps.items():
                    if "score" in k:
                        continue
                    log[k.replace("loss_", "train/")] = v
                wandb.log(log, step=self.global_step)

            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)

            # Explicit cleanup after each epoch
            gc.collect()
            torch.cuda.empty_cache()

        self.save_checkpoint(self.cfg.total_epochs)
        if self.cfg.wandb:
            wandb.finish()

    def save_checkpoint(self, epoch):
        # Determine save directory: use wandb dir if active, else fallback to results/models/baseline
        if self.cfg.wandb and wandb.run and wandb.run.dir:
            save_dir = wandb.run.dir
        else:
            save_dir = os.path.join(self.gpfs_root, "results", "models", "baseline")

        filename = f"{self.cfg.model_type}_epoch{epoch:05d}.pt"
        path = os.path.join(save_dir, filename)

        super().save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler, epoch, path)


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
    parser.add_argument("--augment", type=str, choices=["True", "False"], help="Enable/disable data augmentation (True/False)")
    parser.add_argument("--epochs", type=int, help="Total epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, help="Number of steps per epoch")
    parser.add_argument("--num_workers", type=int, help="Number of workers for the data queue")
    args = parser.parse_args()

    # Convert wandb arg to boolean
    use_wandb = args.wandb == "True"
    BASELINE_CONFIG["wandb"] = use_wandb

    # Override with CLI args if provided
    if args.dice_w is not None:
        BASELINE_CONFIG["dice_w"] = args.dice_w
    if args.resume_id is not None:
        BASELINE_CONFIG["resume_wandb_id"] = args.resume_id
    if args.augment is not None:
        BASELINE_CONFIG["augment"] = args.augment == "True"
    if args.split_file is not None:
        BASELINE_CONFIG["split_file"] = args.split_file
    if args.epochs is not None:
        BASELINE_CONFIG["total_epochs"] = args.epochs
    if args.steps_per_epoch is not None:
        BASELINE_CONFIG["steps_per_epoch"] = args.steps_per_epoch
    if args.num_workers is not None:
        BASELINE_CONFIG["data_queue_num_workers"] = args.num_workers

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
