import copy
import datetime
import gc
import math
import os
import random
import sys
import time
import traceback
from collections import defaultdict
from glob import glob

import numpy as np
import torch

# Enables TF32 and cuDNN benchmark for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 64

import matplotlib

matplotlib.use("Agg")
import warnings

import matplotlib.pyplot as plt
import nibabel as nib
import torchio as tio
from monai.inferers import sliding_window_inference
from tqdm import tqdm

import wandb

warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence for multidimensional indexing is deprecated.*")

# Add 'src' directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from anatomix.model.network import Unet

from mri2ct.config import DEFAULT_CONFIG, Config
from mri2ct.data import DataPreprocessing, get_augmentations, get_region_key, get_subject_paths
from mri2ct.loss import CompositeLoss
from mri2ct.utils import cleanup_gpu, compute_metrics, count_parameters, get_ram_info, set_seed, unpad


def clean_state_dict(state_dict):
    """Removes '_orig_mod.' prefix from keys if present (added by torch.compile)."""
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[10:] if k.startswith("_orig_mod.") else k
        new_state_dict[name] = v
    return new_state_dict


# ==========================================
# CONFIGURATION
# ==========================================
BASELINE_CONFIG = {
    "batch_size": 4,
    "lr": 3e-4,
    "total_epochs": 1000,
    "steps_per_epoch": 1000,
    "val_interval": 1,
    "model_save_interval": 1,
    "use_weighted_sampler": True,
    "use_registered_data": False,
    # "resume_wandb_id": None,
    "resume_wandb_id": "wq11cuvy",
    "resume_epoch": None,  # Optional: specify epoch number
    "diverge_wandb_branch": False,  # Create new run instead of resuming existing
    "dice_w": 0.00,  # Optional Dice loss
    "validate_dice": True,  # Optional Dice validation
    # "validate_dice": False,
    "enable_profiling": True,
    # "enable_profiling": False,
    "patches_per_volume": 200,
    "data_queue_max_length": 400,
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
    "data_queue_num_workers": 4,
    "augment": True,
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
    "dice_bone_only": False,
    # Resuming
    "override_lr": False,  # If True, uses 'lr' from config instead of saved state
    # Validation
    "val_sw_batch_size": 8,
    "val_sw_overlap": 0.25,
    "save_val_volumes": True,
    "prediction_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/predictions",
    # Mode
    "sanity_check": False,
    # "subjects": ["1ABA005"], # Uncomment for single subject overfitting test
}


# ==========================================
# TRAINER CLASS
# ==========================================
class BaselineTrainer:
    def __init__(self, config_dict):
        # Merge with default to ensure all keys exist
        full_conf = copy.deepcopy(DEFAULT_CONFIG)
        full_conf.update(config_dict)
        self.cfg = Config(full_conf)
        self.gpfs_root = self.cfg.root_dir  # Preserve permanent storage path

        set_seed(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        print(f"[Baseline] 🚀 Initializing on {self.device}")

        # Default run name logic
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
        self.run_name = f"{timestamp}_{self.cfg.run_name_prefix}"
        if self.cfg.subjects:
            self.run_name += f"_SingleSubj_{len(self.cfg.subjects)}"

        # 0. Data Staging (to local NVMe RAID)
        self._stage_data_local()

        # Setup
        self._setup_models()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        self.global_step = 0
        self.samples_seen = 0
        self.start_epoch = 0
        self.global_start_time = None
        self.elapsed_time_at_resume = 0

        self._load_resume()

    def _stage_data_local(self):
        """Copies dataset to local NVMe RAID for blazing fast I/O."""
        user_id = os.environ.get("USER", "default")
        # Extract resolution string (e.g., '1.5x1.5x1.5mm') from GPFS path to differentiate cache
        res_str = os.path.basename(self.gpfs_root.rstrip("/"))
        local_root = os.path.join("/tmp_data", f"mri2ct_{user_id}_{res_str}")

        if not os.path.exists("/tmp_data") or not os.access("/tmp_data", os.W_OK):
            print("[Baseline] ⚠️ Local storage not available. Staying on GPFS.")
            return

        if os.path.exists(local_root):
            print(f"[Baseline] ♻️  Local cache found at {local_root}. Syncing updates...")
        else:
            print(f"[Baseline] 🚚 Staging data to local NVMe: {local_root}")
            os.makedirs(local_root, exist_ok=True)

        # Construct rsync includes dynamically
        includes = ["--include='*/'", "--include='ct.nii.gz'", "--include='mr.nii.gz'"]
        if self.cfg.use_registered_data:
            includes.extend(["--include='moved_mr*.nii*'"])

        # Only sync masks/segs if needed
        if getattr(self.cfg, "use_weighted_sampler", False):
            includes.append("--include='body_mask.nii.gz'")
        if getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False):
            includes.append("--include='cads_ct_seg.nii.gz'")

        includes.append("--exclude='*'")
        inc_str = " ".join(includes)

        # Use gpfs_root as source to prevent recursive/broken syncing on resume
        for split in ["train", "val"]:
            src = os.path.join(self.gpfs_root, split)
            dst = os.path.join(local_root, split)
            if os.path.exists(src):
                print(f"  - Syncing {split}...")
                os.system(f"rsync -am {inc_str} {src}/ {dst}/")

        self.cfg.root_dir = local_root
        print(f"[Baseline] ✅ Data staged. New root: {self.cfg.root_dir}")

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
        wandb.run.log_code(".")

    def _load_resume(self):
        if not self.cfg.resume_wandb_id:
            return

        print(f"[RESUME] 🕵️ Searching for Run ID: {self.cfg.resume_wandb_id}")
        run_folders = glob(os.path.join(self.cfg.log_dir, "wandb", f"run-*-{self.cfg.resume_wandb_id}"))
        if not run_folders:
            print(f"[RESUME] ❌ Run folder not found for ID: {self.cfg.resume_wandb_id}")
            return

        all_ckpts = []
        for f in run_folders:
            ckpts = glob(os.path.join(f, "files", "*.pt"))
            all_ckpts.extend(ckpts)

        if not all_ckpts:
            print("[RESUME] ⚠️ No checkpoints found inside run folder.")
            return

        if self.cfg.resume_epoch is not None:
            epoch_str = f"epoch{self.cfg.resume_epoch:05d}"
            target_ckpts = sorted([c for c in all_ckpts if epoch_str in os.path.basename(c)])
            if not target_ckpts:
                print(f"[RESUME] ❌ Could not find checkpoint for epoch {self.cfg.resume_epoch}")
                return
            resume_path = target_ckpts[-1]
        else:
            resume_path = max(all_ckpts, key=os.path.getmtime)

        print(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)

        # Load Model
        if hasattr(self.model, "_orig_mod"):
            self.model._orig_mod.load_state_dict(checkpoint["model_state_dict"])
        else:
            self.model.load_state_dict(checkpoint["model_state_dict"])

        # Load Optimizer
        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.cfg.override_lr:
                print(f"[RESUME] 🔧 Forcing new Learning Rate: {self.cfg.lr}")
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.cfg.lr

                # Update scheduler base_lrs to match new optimizer LRs
                if self.scheduler is not None:
                    self.scheduler.base_lrs = [group["lr"] for group in self.optimizer.param_groups]

        # Load Scheduler
        if self.cfg.override_lr and self.scheduler is not None:
            print("[RESUME] 🔧 Override LR enabled: Skipping scheduler state load and recalculating LR.")

            # 1. Restore the step count
            # Use global_step if available, otherwise estimate from epoch
            restored_step = checkpoint.get("global_step", (self.start_epoch) * self.cfg.steps_per_epoch)
            self.scheduler.last_epoch = restored_step

            # 2. Calculate Closed-Form LR
            # lr = eta_min + 0.5 * (base - eta_min) * (1 + cos(pi * t / T))
            t = restored_step
            T = self.cfg.steps_per_epoch * self.cfg.total_epochs
            eta_min = self.cfg.scheduler_min_lr
            base_lr = self.cfg.lr  # This is the NEW base LR

            import math

            new_lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + math.cos(math.pi * t / T))

            # 3. Apply to Optimizer
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = new_lr

            # 4. Update Scheduler Base LRs (so it knows the peak)
            self.scheduler.base_lrs = [base_lr] * len(self.optimizer.param_groups)
            self.scheduler.T_max = T

            print(f"[RESUME] 🧮 Recalculated LR for step {t}/{T}: {new_lr:.2e} (Base: {base_lr})")

        elif checkpoint.get("scheduler_state_dict") is not None and self.scheduler is not None:
            print("[RESUME] 📥 Loading Scheduler state...")
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            print(f"[DEBUG] Loaded Scheduler: last_epoch={self.scheduler.last_epoch}, T_max={self.scheduler.T_max}")

            # Automatically update T_max from config
            new_t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
            if new_t_max != self.scheduler.T_max:
                print(f"[RESUME] 🔧 Updating Scheduler T_max: {self.scheduler.T_max} -> {new_t_max}")
                self.scheduler.T_max = new_t_max

            # Step the scheduler to the restored epoch to update optimizer params
            restored_epoch = self.scheduler.last_epoch
            self.scheduler.last_epoch = restored_epoch - 1

            self.scheduler.step()

            print(f"[RESUME] 🔄 Scheduler stepped to epoch {self.scheduler.last_epoch}. LR: {self.optimizer.param_groups[0]['lr']:.2e}")

        # Load Scaler
        if "scaler_state_dict" in checkpoint:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        # Load Epoch/Step
        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]

        if "samples_seen" in checkpoint:
            self.samples_seen = checkpoint["samples_seen"]
            print(f"[RESUME] 📈 Restored samples_seen: {self.samples_seen}")

        if "elapsed_time" in checkpoint:
            self.elapsed_time_at_resume = checkpoint["elapsed_time"]
            print(f"[RESUME] ⏱️ Restored elapsed_time: {self.elapsed_time_at_resume:.1f}s")
        else:
            self.elapsed_time_at_resume = 0

    def _setup_data(self):
        # 1. Reuse existing logic to find subjects
        print(f"[Baseline] 📂 Searching data in: {self.cfg.root_dir}")

        # Load seg if Dice weight > 0 or Dice validation is enabled
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)

        def scan_split(split_name):
            split_dir = os.path.join(self.cfg.root_dir, split_name)
            if not os.path.exists(split_dir):
                return []
            valid_subjs = []
            for d in sorted(os.listdir(split_dir)):
                subj_path = os.path.join(split_dir, d)
                if not os.path.isdir(subj_path):
                    continue
                has_ct = os.path.exists(os.path.join(subj_path, "ct.nii.gz"))
                if self.cfg.use_registered_data:
                    has_mr = len(glob(os.path.join(subj_path, "moved_mr*.nii*"))) > 0
                else:
                    has_mr = os.path.exists(os.path.join(subj_path, "mr.nii.gz"))

                # Check for segmentation if Dice is needed
                has_seg = True
                if load_seg:
                    has_seg = os.path.exists(os.path.join(subj_path, "cads_ct_seg.nii.gz"))

                if has_ct and has_mr and has_seg:
                    valid_subjs.append(os.path.join(split_name, d))
                elif not has_seg and load_seg:
                    print(f"  [Skip] {os.path.join(split_name, d)}: Missing segmentation.")
            return valid_subjs

        train_candidates = scan_split("train")
        val_candidates = scan_split("val")

        if self.cfg.subjects:
            print(f"[Baseline] 🎯 Filtering subjects: {self.cfg.subjects}")
            self.train_subjects = [c for c in train_candidates + val_candidates if os.path.basename(c) in self.cfg.subjects]
            self.val_subjects = self.train_subjects
        else:
            self.train_subjects = train_candidates
            self.val_subjects = val_candidates

        print(f"[Baseline] Train: {len(self.train_subjects)} | Val: {len(self.val_subjects)}")

        # 2. Build Datasets
        # Load seg if Dice weight > 0 or Dice validation is enabled
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)

        def _make_subj_list(subjs, load_seg=False):
            subj_list = []
            for s in subjs:
                paths = get_subject_paths(self.cfg.root_dir, s, use_registered=self.cfg.use_registered_data)
                kwargs = {
                    "mri": tio.ScalarImage(paths["mri"]),
                    "ct": tio.ScalarImage(paths["ct"]),
                    "subj_id": os.path.basename(s),
                }
                if self.cfg.use_weighted_sampler and "body_mask" in paths:
                    kwargs["prob_map"] = tio.LabelMap(paths["body_mask"])

                if load_seg:
                    seg_path = os.path.join(self.cfg.root_dir, s, "cads_ct_seg.nii.gz")
                    if os.path.exists(seg_path):
                        kwargs["seg"] = tio.LabelMap(seg_path)
                    else:
                        raise FileNotFoundError(f"Segmentation missing for {s} but Dice evaluation is active.")

                subj_list.append(tio.Subject(**kwargs))
            return subj_list

        # Train Queue
        train_objs = _make_subj_list(self.train_subjects, load_seg=load_seg)
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

        def _inf_gen(loader):
            while True:
                for batch in loader:
                    yield batch

        self.train_iter = _inf_gen(self.train_loader)

        # Val Loader
        val_objs = _make_subj_list(self.val_subjects, load_seg=load_seg)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess)
        # Use SubjectsLoader to avoid warnings and ensure correct collation
        self.val_loader = tio.SubjectsLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

        # Stratified Validation Sampling (2 per region) - Matching src/trainer.py
        total_val = len(self.val_subjects)
        rng = random.Random(self.cfg.seed)

        region_to_indices = defaultdict(list)
        for idx, subj_path in enumerate(self.val_subjects):
            subj_id = os.path.basename(subj_path)
            region = get_region_key(subj_id)
            region_to_indices[region].append(idx)

        viz_indices = []
        for region, indices in region_to_indices.items():
            # Pick up to 2 subjects per region
            num_to_pick = min(len(indices), 2)
            viz_indices.extend(rng.sample(indices, num_to_pick))

        self.val_viz_indices = set(viz_indices)

    def _log_aug_viz(self, step):
        try:
            # 1. Get Data
            subj_id = self.val_subjects[0]
            paths = get_subject_paths(self.cfg.root_dir, subj_id, use_registered=self.cfg.use_registered_data)

            subj = tio.Subject(mri=tio.ScalarImage(paths["mri"]), ct=tio.ScalarImage(paths["ct"]))
            prep = DataPreprocessing(patch_size=self.cfg.patch_size, res_mult=self.cfg.res_mult)
            subj = prep(subj)

            # 2. Augment
            aug = get_augmentations()(subj)
            hist_str = " | ".join([t.name for t in aug.history])

            # 3. Slice & Plot
            z = subj["mri"].shape[-1] // 2

            # NOTE: If aug changes shape, this line will crash.
            orig_sl = subj["mri"].data[0, ..., z].numpy()
            aug_sl = aug["mri"].data[0, ..., z].numpy()

            fig, ax = plt.subplots(1, 3, figsize=(10, 4))
            ax[0].imshow(np.rot90(orig_sl), cmap="gray", vmin=0, vmax=1)
            ax[0].set_title("Original")
            ax[1].imshow(np.rot90(aug_sl), cmap="gray", vmin=0, vmax=1)
            ax[1].set_title(f"Augmented\n{hist_str}")

            # The simple Diff you wanted
            ax[2].imshow(np.rot90(aug_sl - orig_sl), cmap="seismic", vmin=-0.5, vmax=0.5)
            ax[2].set_title("Diff")

            wandb.log({"val/aug_viz": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"[WARNING] Aug Viz failed: {e}")

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
            if getattr(self.cfg, "dice_w", 0) > 0 and self.teacher_model is not None and seg is not None:
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

        # NOTE: Loss initialization
        self.loss_fn = CompositeLoss(
            weights={
                "l1": self.cfg.l1_w,
                "l2": self.cfg.l2_w,
                "ssim": self.cfg.ssim_w,
                "dice_w": getattr(self.cfg, "dice_w", 0.0),
                "dice_bone_only": getattr(self.cfg, "dice_bone_only", False),
                "dice_exclude_background": getattr(self.cfg, "dice_exclude_background", True),
            }
        ).to(self.device)

        self.scaler = torch.amp.GradScaler("cuda")

    # ==========================================
    # VISUALIZATION (Mirrors src/trainer.py)
    # ==========================================
    @torch.inference_mode()
    def _visualize_lite(self, pred, ct, mri, subj_id, shape, step, epoch, idx, offset=0, save_path=None):
        """
        Lightweight visualization: MRI, GT, Pred, Residual.
        """
        # 1. Unpad Volumes
        w, h, d = shape
        gt_ct = unpad(ct, shape, offset).cpu().numpy().squeeze()
        gt_mri = unpad(mri, shape, offset).cpu().numpy().squeeze()
        pred_ct = unpad(pred, shape, offset).cpu().numpy().squeeze()

        # 2. Define Items (Standard 4-column view)
        items = [
            (gt_mri, "GT MRI", "gray", (0, 1)),
            (gt_ct, "GT CT", "gray", (0, 1)),
            (pred_ct, "Pred CT", "gray", (0, 1)),
            (pred_ct - gt_ct, "Residual", "seismic", (-0.5, 0.5)),
        ]

        # 3. Plotting
        D_dim = gt_ct.shape[-1]
        num_cols = len(items)
        # Select 5 equidistant slices
        slice_indices = np.linspace(0.1 * D_dim, 0.9 * D_dim, 5, dtype=int)

        fig, axes = plt.subplots(len(slice_indices), num_cols, figsize=(3 * num_cols, 3.5 * len(slice_indices)))
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

        title_str = f"Subject: {subj_id} | Epoch {epoch} | Step {step}"
        caption = f"Subject: {subj_id}"
        if save_path:
            caption += f"\nSaved to: {save_path}"
        fig.suptitle(title_str, fontsize=16, y=0.99)

        if self.cfg.wandb:
            wandb.log({f"viz/{('val_' + str(idx))}": wandb.Image(fig, caption=caption)}, step=step)
        plt.close(fig)

    @torch.inference_mode()
    def _log_training_patch(self, mri, ct, pred, step, seg=None, pred_probs=None):
        """Visualizes the first patch of the batch."""
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
        nrows = 4 if img_seg is not None else 3
        fig, axes = plt.subplots(nrows, 3, figsize=(10, 3.5 * nrows))

        def plot_row(row_idx, vol, title_prefix, vmin=None, vmax=None, cmap="gray"):
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 0].set_title(f"{title_prefix} Ax")
            axes[row_idx, 1].imshow(np.rot90(vol[cx, :, :]), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 1].set_title(f"{title_prefix} Sag")
            axes[row_idx, 2].imshow(np.rot90(vol[:, cy, :]), cmap=cmap, vmin=vmin, vmax=vmax)
            axes[row_idx, 2].set_title(f"{title_prefix} Cor")

            axes[row_idx, 0].text(-5, 10, f"Min: {vol.min():.2f}\nMax: {vol.max():.2f}", fontsize=8, color="white", backgroundcolor="black")

        plot_row(0, img_in, "MRI", vmin=0, vmax=1)
        plot_row(1, img_pred, "Pred", vmin=0, vmax=1)
        plot_row(2, img_gt, "GT CT", vmin=0, vmax=1)

        if img_seg is not None:
            seg_vmin = 0
            seg_vmax = img_seg.max()
            if img_pred_seg is not None:
                seg_vmax = max(seg_vmax, img_pred_seg.max())

            axes[3, 0].imshow(np.rot90(img_seg[:, :, cz]), cmap="tab20", vmin=seg_vmin, vmax=seg_vmax, interpolation="nearest")
            axes[3, 0].set_title("GT Seg Ax")
            if img_pred_seg is not None:
                axes[3, 1].imshow(np.rot90(img_pred_seg[:, :, cz]), cmap="tab20", vmin=seg_vmin, vmax=seg_vmax, interpolation="nearest")
                axes[3, 1].set_title("Pred Seg Ax")
                axes[3, 2].imshow(np.rot90(img_pred[:, :, cz]), cmap="gray", vmin=0, vmax=1)
                axes[3, 2].imshow(np.rot90(img_pred_seg[:, :, cz]), cmap="tab20", vmin=seg_vmin, vmax=seg_vmax, alpha=0.3, interpolation="nearest")
                axes[3, 2].set_title("Pred Overlay")
            else:
                axes[3, 1].axis("off")
                axes[3, 2].axis("off")

        for ax in axes.flatten():
            ax.axis("off")
        plt.tight_layout()
        wandb.log({f"train/patch_viz": wandb.Image(fig)}, step=step)
        plt.close(fig)

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
                t_step_start = time.time()

            for _ in range(self.cfg.accum_steps):
                batch = next(self.train_iter)
                # Convert to plain tensors for compiler stability
                mri = batch["mri"][tio.DATA].to(self.device, non_blocking=True)
                ct = batch["ct"][tio.DATA].to(self.device, non_blocking=True)
                seg = batch["seg"][tio.DATA].to(self.device, non_blocking=True) if "seg" in batch else None

                pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                if self.cfg.wandb and step_idx == 0:
                    self._log_training_patch(mri, ct, pred, self.global_step, seg, pred_probs)

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
                step_t_total = time.time() - t_step_start
                pbar_dict["t(s)"] = f"{step_t_total:.3f}"

                # Log only every 100 steps to avoid IO bottleneck
                if self.cfg.wandb and step_idx % 100 == 0:
                    wandb.log({"info/time_step_total(s)": step_t_total}, step=self.global_step)

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
                sys_percent, app_gb = get_ram_info()

                queue = self.train_loader.dataset
                curr_patches = len(queue.patches_list)

                wandb.log({"perf/ram_system_percent": sys_percent, "perf/ram_app_total_gb": app_gb, "perf/queue_curr_patches": curr_patches}, step=self.global_step)

                if sys_percent > 90:
                    tqdm.write(f"[WARNING] ⚠️ RAM usage critical: {sys_percent}% (App: {app_gb:.2f} GB)")
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

            # Loss (Approx for Val)
            l_val, _ = self.loss_fn(pred, ct)
            met["loss"] = l_val.item()

            # Validation Dice (Semantic Consistency)
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

                val_dice_loss = self.loss_fn.soft_dice_loss(pred_probs, seg)
                met["dice"] = 1.0 - val_dice_loss.item()
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
            if (epoch % self.cfg.val_interval == 0) or (epoch + 1) == self.cfg.total_epochs:
                avg_met = self.validate(epoch)
                print(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Bone: {avg_met.get('bone_dice', 0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume

            if self.cfg.wandb:
                current_lr = self.optimizer.param_groups[0]["lr"]
                log = {
                    "train/total": loss,
                    "info/grad_norm": gn,
                    "info/epoch_duration": ep_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/lr": current_lr,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/samples_seen": self.samples_seen,
                }
                for k, v in comps.items():
                    log[k.replace("loss_", "train/")] = v
                wandb.log(log, step=self.global_step)

            if epoch % self.cfg.model_save_interval == 0:
                self.save_model(epoch)

        self.save_model(self.cfg.total_epochs, is_final=True)
        if self.cfg.wandb:
            wandb.finish()

    def save_model(self, epoch, is_final=False):
        # Determine save directory: use wandb dir if active, else fallback to results/models/baseline
        if self.cfg.wandb and wandb.run and wandb.run.dir:
            save_dir = wandb.run.dir
        else:
            save_dir = os.path.join(self.gpfs_root, "results", "models", "baseline")

        os.makedirs(save_dir, exist_ok=True)

        filename = f"{self.cfg.model_type}_epoch{epoch:05d}.pt"
        path = os.path.join(save_dir, filename)

        # Handle torch.compile wrapper if present
        model_state = self.model.state_dict()
        if hasattr(self.model, "_orig_mod"):
            model_state = self.model._orig_mod.state_dict()

        # Full checkpoint for resuming
        save_dict = {
            "epoch": epoch,
            "global_step": self.global_step,
            "samples_seen": self.samples_seen,
            "elapsed_time": (time.time() - self.global_start_time) + self.elapsed_time_at_resume if self.global_start_time else self.elapsed_time_at_resume,
            "model_state_dict": model_state,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict(),
            "config": vars(self.cfg),
        }

        torch.save(save_dict, path)
        print(f"[Save] {path}")

        # Explicitly tell wandb to sync this file immediately if desired,
        # though saving to wandb.run.dir usually handles it.
        if self.cfg.wandb:
            wandb.save(path, base_path=save_dir)


# ==========================================
# MAIN ENTRY
# ==========================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--single_subject", type=str, help="Run overfitting test on specific subject ID (e.g., 1ABA005)")
    parser.add_argument("--wandb", type=str, default="True", choices=["True", "False"], help="Enable/disable wandb (True/False)")
    args = parser.parse_args()

    # Convert wandb arg to boolean
    use_wandb = args.wandb == "True"
    BASELINE_CONFIG["wandb"] = use_wandb

    # Modify config based on args
    if args.single_subject:
        print(f"🔬 RUNNING SINGLE SUBJECT TEST: {args.single_subject}")
        BASELINE_CONFIG.update(
            {
                "subjects": [args.single_subject],
                "wandb_note": f"Baseline Overfitting Test - {args.single_subject}",
                "augment": False,  # Disable aug for pure overfitting check
                "steps_per_epoch": 50,  # Reduced steps for faster feedback
                "val_interval": 5,
                # Speed up startup but ensure enough buffering
                "data_queue_max_length": 200,
                "data_queue_num_workers": 4,
            }
        )
        # Disable compile for quick test
        torch.compile = None

    try:
        trainer = BaselineTrainer(BASELINE_CONFIG)
        trainer.train()
        cleanup_gpu()
    except KeyboardInterrupt:
        print("Interrupted.")
        cleanup_gpu()
    except Exception as _:
        traceback.print_exc()
        cleanup_gpu()
