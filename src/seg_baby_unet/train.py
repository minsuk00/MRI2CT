import datetime
import glob
import os
import random
import sys
import time
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import monai
import nibabel as nib
import numpy as np
import torch
from monai.data import CacheDataset, DataLoader, list_data_collate
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    RandAdjustContrastd,
    RandAffined,
    RandGaussianNoised,
    RandGaussianSharpend,
    RandGaussianSmoothd,
    RandSimulateLowResolutiond,
    RandSpatialCropd,
    ScaleIntensityd,
    ScaleIntensityRanged,
    SpatialPadd,
    ToTensord,
)
from tqdm import tqdm

import wandb

# A40 Optimization: Enable TensorFloat-32
torch.set_float32_matmul_precision("high")
torch._dynamo.config.cache_size_limit = 64

# Suppress annoying warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch._dynamo")
warnings.filterwarnings("ignore", message=".*Using a non-tuple sequence for multidimensional indexing.*")
warnings.filterwarnings("ignore", message=".*Dynamo detected a call to a `functools.lru_cache`.*")

# Add 'src' directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from anatomix.segmentation.segmentation_utils import load_model_v1_1, worker_init_fn

from mri2ct.data import get_region_key
from mri2ct.utils import count_parameters, get_ram_info

# ==========================================
# INLINE CONFIGURATION
# ==========================================
SEG_CONFIG = {
    ##### System
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm",
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "project_name": "MRI2CT_Seg_Baby_Unet",
    "wandb": True,
    "wandb_note": "Finetuning Anatomix on CADS Segmentation (11 classes) - CT Input - 1.5mm - Full Logic Parity",
    ##### Classes
    "n_classes": 12,
    "class_names": [
        "Background",
        "Subcutaneous tissue",
        "Muscle",
        "Abdominal cavity",
        "Thoracic cavity",
        "Bones",
        "Gland structure",
        "Pericardium",
        "Prosthetic breast implant",
        "Mediastinum",
        "Spinal cord",
        "Brain",
    ],
    ##### Data
    "patch_size": 128,
    "batch_size": 8,
    "lr": 3e-4,
    "n_epochs": 500,
    "iters_per_epoch": 250,
    "val_interval": 5,
    "model_save_interval": 10,
    ##### Model
    # "anatomix_weights_path": "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth",
    "anatomix_weights_path": "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G.pth",
    "finetune": True,
    "compile_mode": "model",  # Options: None, "model", "full"
    "few_shot": False,
    "few_shot_amount": 1,
    "few_shot_region": None,
    # "resume_wandb_id": None,
    # "resume_epoch": None,
    "resume_wandb_id": "p0i9chz3",
    "resume_epoch": 469,
    "diverge_wandb_branch": False,
    "override_lr": False,
    ##### Validation / Viz
    "save_val_volumes": True,
    "viz_limit": 2,
    "val_sw_overlap": 0.7,
    "augment": True,
    "enable_profiling": False,
    ##### DataLoader
    "cache_rate": 1.0,
    "num_workers": 4,
    # "num_workers": 1,
    ##### few shot
    # "lr": 2e-4,
    # "batch_size": 2,
    # "few_shot": True,
    # "few_shot_region": "abdomen",
    # "n_epochs": 30,
    # "iters_per_epoch": 250,
    # "val_interval": 1,
    # "few_shot_amount": 3,
    # "viz_limit": 5,
}


# ==========================================
# TRAINER CLASS
# ==========================================
class SegTrainer:
    def __init__(self, config):
        self.cfg = config
        self._set_seed(self.cfg["seed"])
        self.device = torch.device(self.cfg["device"])
        tqdm.write(f"[SegTrainer] 🚀 Initializing on {self.device}")
        tqdm.write(f"[DEBUG] n_classes={self.cfg['n_classes']}")

        # State Tracking
        self.start_epoch = 0
        self.global_step = 0

        self._setup_wandb()
        self._setup_data()
        self._setup_model()
        self._load_resume()

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        # Optimization: Enable cuDNN benchmark for fixed input sizes
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

    def _setup_wandb(self):
        if self.cfg["wandb"]:
            os.makedirs(self.cfg["log_dir"], exist_ok=True)
            run_name = f"{datetime.datetime.now():%Y%m%d_%H%M}_Seg_Baby_UNet_CT_1.5mm"

            wandb_id = None if self.cfg.get("diverge_wandb_branch", False) else self.cfg.get("resume_wandb_id")

            wandb.init(
                project=self.cfg["project_name"],
                name=run_name,
                config=self.cfg,
                dir=self.cfg["log_dir"],
                notes=self.cfg["wandb_note"],
                id=wandb_id,
                resume="allow" if not self.cfg.get("diverge_wandb_branch", False) else None,
            )
            tqdm.write(f"[SegTrainer] 📡 WandB Initialized. Run ID: {wandb.run.id}")
            # Log Code Parity
            wandb.run.log_code(".")

    def _load_resume(self):
        resume_id = self.cfg.get("resume_wandb_id")
        if not resume_id:
            return

        tqdm.write(f"[RESUME] 🕵️ Searching for Run ID: {resume_id}")
        run_folders = glob.glob(os.path.join(self.cfg["log_dir"], "wandb", f"run-*-{resume_id}"))
        if not run_folders:
            tqdm.write(f"[RESUME] ❌ Run folder not found for ID: {resume_id}")
            return

        all_ckpts = []
        for f in run_folders:
            # Check standard models folder and wandb files folder
            ckpts = glob.glob(os.path.join(f, "files", "*.pth"))
            all_ckpts.extend(ckpts)

        # Also check project models dir if not in wandb
        proj_models = glob.glob(os.path.join(self.cfg["log_dir"], "models", "*.pth"))
        all_ckpts.extend(proj_models)

        if not all_ckpts:
            tqdm.write("[RESUME] ⚠️ No checkpoints found.")
            return

        if self.cfg.get("resume_epoch") is not None:
            epoch_str = f"epoch_{self.cfg['resume_epoch']}.pth"
            target_ckpts = sorted([c for c in all_ckpts if epoch_str in os.path.basename(c)])
            if not target_ckpts:
                tqdm.write(f"[RESUME] ❌ Could not find checkpoint for epoch {self.cfg['resume_epoch']}")
                return
            resume_path = target_ckpts[-1]
        else:
            # Try to find 'best' or latest
            best_ckpts = [c for c in all_ckpts if "best" in os.path.basename(c)]
            if best_ckpts:
                resume_path = max(best_ckpts, key=os.path.getmtime)
            else:
                resume_path = max(all_ckpts, key=os.path.getmtime)

        tqdm.write(f"[RESUME] 📥 Loading: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=self.device)

        # Load state dict (handle potential compile)
        def load_state(m, state):
            getattr(m, "_orig_mod", m).load_state_dict(state)

        load_state(self.model, checkpoint["model_state_dict"])

        if "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.cfg.get("override_lr", False):
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.cfg["lr"]
                if self.scheduler is not None:
                    self.scheduler.base_lrs = [self.cfg["lr"]]

        if "scheduler_state_dict" in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if "epoch" in checkpoint:
            self.start_epoch = checkpoint["epoch"] + 1

        if "global_step" in checkpoint:
            self.global_step = checkpoint["global_step"]
        else:
            self.global_step = self.start_epoch * self.cfg["iters_per_epoch"]

    def _setup_data(self):
        print(f"[SegTrainer] 📂 Scanning data in {self.cfg['root_dir']}...")

        def get_file_list(split_name):
            split_dir = os.path.join(self.cfg["root_dir"], split_name)
            if not os.path.exists(split_dir):
                return []

            data_dicts = []
            for subj in sorted(os.listdir(split_dir)):
                subj_path = os.path.join(split_dir, subj)
                if not os.path.isdir(subj_path):
                    continue

                img_path = os.path.join(subj_path, "ct.nii.gz")
                seg_path = os.path.join(subj_path, "cads_ct_seg.nii.gz")

                if os.path.exists(img_path) and os.path.exists(seg_path):
                    data_dicts.append({"image": img_path, "label": seg_path, "subj_id": subj})
            return data_dicts

        train_files = get_file_list("train")
        val_files = get_file_list("val")

        if self.cfg.get("few_shot", False):
            # 1. Filter by desired region if specified
            target_region = self.cfg.get("few_shot_region")
            if target_region:
                train_files = [f for f in train_files if get_region_key(f["subj_id"]) == target_region]
                if not train_files:
                    raise ValueError(f"No training subjects found for region: {target_region}")

            # 2. Deterministic selection
            train_files.sort(key=lambda x: x["image"])
            random.Random(self.cfg["seed"]).shuffle(train_files)
            train_files = train_files[: self.cfg["few_shot_amount"]]

            # 3. Filter val to same region and limit to 10 for speed
            train_region = get_region_key(train_files[0]["subj_id"])
            val_files = [v for v in val_files if get_region_key(v["subj_id"]) == train_region]
            val_files = val_files[:10]

            tqdm.write(f"[SegTrainer] 🤏 Few-shot mode: {len(train_files)} train volumes (Region: {train_region}).")
            tqdm.write(f"[SegTrainer] ⏱️ Limited val to {len(val_files)} volumes in same region.")

        print(f"[SegTrainer] Found {len(train_files)} train, {len(val_files)} val subjects.")

        # Stratified Validation Visualization logic (Parity with trainer.py)
        self.val_subjects = val_files
        self.val_viz_indices = set()
        if len(val_files) > 0:
            region_to_indices = defaultdict(list)
            for idx, item in enumerate(val_files):
                region = get_region_key(item["subj_id"])
                region_to_indices[region].append(idx)

            rng = random.Random(self.cfg["seed"])
            viz_indices = []
            viz_limit_per_region = self.cfg.get("viz_limit", 2)
            for region, indices in region_to_indices.items():
                num_to_pick = min(len(indices), viz_limit_per_region)  # Pick up to viz_limit per region
                viz_indices.extend(rng.sample(indices, num_to_pick))
            self.val_viz_indices = set(viz_indices)
            tqdm.write(f"[SegTrainer] 🖼️  Stratified Val Viz Indices: {self.val_viz_indices}")

        crop_size = self.cfg["patch_size"]
        aug_prob = 0.33

        self.train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                # Ensure image is large enough for cropping
                SpatialPadd(keys=["image", "label"], spatial_size=[crop_size] * 3),
                RandSpatialCropd(keys=["image", "label"], roi_size=[crop_size] * 3, random_size=False),
                ScaleIntensityRanged(keys="image", a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
                RandSimulateLowResolutiond(keys="image", prob=aug_prob, zoom_range=(0.5, 1.0)),
                RandGaussianNoised(keys="image", prob=aug_prob),
                RandAdjustContrastd(keys="image", prob=aug_prob),
                RandGaussianSmoothd(keys="image", prob=aug_prob, sigma_x=(0.0, 0.1), sigma_y=(0.0, 0.1), sigma_z=(0.0, 0.1)),
                RandGaussianSharpend(keys="image", prob=aug_prob),
                RandAffined(
                    keys=["image", "label"],
                    prob=aug_prob,
                    mode=("bilinear", "nearest"),
                    rotate_range=(np.pi / 4, np.pi / 4, np.pi / 4),
                    scale_range=(0.2, 0.2, 0.2),
                    shear_range=(0.2, 0.2, 0.2),
                    spatial_size=[crop_size] * 3,
                    padding_mode="zeros",
                ),
                ScaleIntensityd(keys="image"),
                ToTensord(keys=["image", "label"], track_meta=True),
            ]
        )

        self.val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                EnsureChannelFirstd(keys=["image", "label"]),
                EnsureTyped(keys=["image", "label"]),
                ScaleIntensityRanged(keys="image", a_min=-1024, a_max=1024, b_min=0.0, b_max=1.0, clip=True),
                ToTensord(keys=["image", "label"], track_meta=True),
            ]
        )

        samples_needed = self.cfg["iters_per_epoch"] * self.cfg["batch_size"]
        num_unique_train = len(train_files)

        # Adaptive Caching Strategy for Training
        # Use RAM cache for few-shot, Disk cache for large datasets (600+ volumes)
        if self.cfg.get("few_shot", False) or num_unique_train < 50:
            print(f"[SegTrainer] 🚀 Using RAM (CacheDataset) for {num_unique_train} training volumes.")
            train_ds_base = CacheDataset(
                data=train_files,
                transform=self.train_transforms,
                cache_rate=self.cfg["cache_rate"],
                num_workers=2,
            )
        else:
            # High-Performance Caching: Use Local NVMe RAID (/tmp_data) to eliminate GPFS spikes.
            user_id = os.environ.get("USER", "default")
            local_nvme = "/tmp_data"

            if os.path.exists(local_nvme) and os.access(local_nvme, os.W_OK):
                cache_dir = os.path.join(local_nvme, f"mri2ct_cache_{user_id}")
                storage_type = "LOCAL NVMe"
            else:
                cache_dir = os.path.join(os.path.dirname(self.cfg["root_dir"]), "persistent_cache_seg_v2")
                storage_type = "GPFS (Networked)"

            os.makedirs(cache_dir, exist_ok=True)
            print(f"[SegTrainer] 💾 Using {storage_type} (PersistentDataset) for {num_unique_train} training volumes.")
            print(f"[SegTrainer] Cache Directory: {cache_dir}")

            train_ds_base = monai.data.PersistentDataset(data=train_files, transform=self.train_transforms, cache_dir=cache_dir)

        if num_unique_train > 0:
            # Decouple Dataset Size from Epoch Length using Subset
            # This handles 'iters_per_epoch' correctly without duplicating cache memory
            repeats = max(1, (samples_needed + num_unique_train - 1) // num_unique_train)
            indices = np.tile(np.arange(num_unique_train), repeats)[:samples_needed]
            self.train_ds = torch.utils.data.Subset(train_ds_base, indices)

            total_iters = self.cfg["n_epochs"] * self.cfg["iters_per_epoch"]
            print("[SegTrainer] 🔁 Training Summary:")
            print(f"   - Unique subjects: {num_unique_train}")
            print(f"   - Samples per epoch: {len(self.train_ds)} ({self.cfg['iters_per_epoch']} iterations @ BS={self.cfg['batch_size']})")
            print(f"   - Total planned: {self.cfg['n_epochs']} epochs ({total_iters:,} iterations)")
        else:
            self.train_ds = train_ds_base

        self.train_loader = DataLoader(
            self.train_ds,
            batch_size=self.cfg["batch_size"],
            shuffle=True,
            num_workers=self.cfg["num_workers"],
            collate_fn=list_data_collate,
            worker_init_fn=worker_init_fn,
            pin_memory=False,
        )

        # Validation Dataset - Keep as CacheDataset per request
        self.val_ds = CacheDataset(data=val_files, transform=self.val_transforms, cache_rate=1.0, num_workers=2)

        self.val_loader = DataLoader(
            self.val_ds,
            batch_size=1,
            num_workers=0,
            collate_fn=list_data_collate,
            worker_init_fn=worker_init_fn,
            shuffle=False,
            pin_memory=False,
        )

    def _setup_model(self):
        tqdm.write("[SegTrainer] 🏗️ Building Model (Anatomix)...")
        # compile_model=True only if "model" mode is selected.
        # If "full", we skip model-level compile and compile the entire step.
        load_compile = self.cfg.get("compile_mode") == "model"
        self.model = load_model_v1_1(
            pretrained_ckpt=self.cfg["anatomix_weights_path"],
            n_classes=self.cfg["n_classes"] - 1,
            device=self.device,
            compile_model=load_compile,
        )

        tot, train = count_parameters(self.model)
        tqdm.write(f"[Model] Segmentation Unet Params: Total={tot:,} | Trainable={train:,}")

        # Optimization: to_onehot_y=False because we pre-process it in transforms
        self.loss_function = monai.losses.DiceCELoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
        )
        self.val_loss_function = monai.losses.DiceLoss(
            softmax=True,
            to_onehot_y=True,
            include_background=False,
        )
        # Per-class Dice Metric (excludes background class 0 from Mean)
        self.dice_metric = monai.metrics.DiceMetric(
            include_background=False,
            reduction="mean",
            ignore_empty=True,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            self.cfg["lr"],
            weight_decay=1e-5,
            fused=True if torch.cuda.is_available() else False,  # Fused optimizer speedup
        )
        # Use n_epochs from config for the scheduler cycle
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.cfg["n_epochs"])

    @torch.no_grad()
    def _log_training_patch(self, image, label, pred, step):
        """Visualizes a 3D patch (middle slices) to WandB."""
        # [B, 1, H, W, D]
        img_np = image[0, 0].cpu().numpy()
        lab_np = label[0, 0].cpu().numpy()
        # [B, C, H, W, D] -> Argmax -> [H, W, D]
        pred_np = torch.argmax(pred[0], dim=0).cpu().numpy()

        # Slices
        cx, cy, cz = np.array(img_np.shape) // 2

        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        def plot_row(row_idx, vol, title, cmap="gray", is_mask=False):
            kwargs = {"cmap": cmap}
            if is_mask:
                kwargs.update({"vmin": 0, "vmax": self.cfg["n_classes"], "interpolation": "nearest"})
            else:
                # CT Image: Lock to 0-1 range (prevents gray background from neg noise)
                kwargs.update({"vmin": 0.0, "vmax": 1.0})

            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), **kwargs)
            axes[row_idx, 0].set_title(f"{title} Ax")
            axes[row_idx, 1].imshow(np.rot90(vol[:, cy, :]), **kwargs)
            axes[row_idx, 1].set_title(f"{title} Cor")
            axes[row_idx, 2].imshow(np.rot90(vol[cx, :, :]), **kwargs)
            axes[row_idx, 2].set_title(f"{title} Sag")

        plot_row(0, img_np, "Image")
        plot_row(1, lab_np, "GT Seg", cmap="tab20", is_mask=True)
        plot_row(2, pred_np, "Pred Seg", cmap="tab20", is_mask=True)

        for ax in axes.flatten():
            ax.axis("off")

        plt.tight_layout()
        wandb.log({"train/patch_viz": wandb.Image(fig)}, step=step)
        plt.close(fig)

    @torch.no_grad()
    def _visualize_val(self, image, label, pred, step, subj_id):
        """Full volume visualization slices parity with trainer.py."""
        img_np = image[0, 0].cpu().numpy()
        lab_np = label[0, 0].cpu().numpy()
        pred_np = torch.argmax(pred[0], dim=0).cpu().numpy()
        cx, cy, cz = np.array(img_np.shape) // 2
        fig, axes = plt.subplots(3, 3, figsize=(12, 10))

        def plot_row(row_idx, vol, title, cmap="gray", is_mask=False):
            kwargs = {"cmap": cmap}
            if is_mask:
                kwargs.update({"vmin": 0, "vmax": self.cfg["n_classes"], "interpolation": "nearest"})
            axes[row_idx, 0].imshow(np.rot90(vol[:, :, cz]), **kwargs)
            axes[row_idx, 0].set_title(f"{title} Ax")
            axes[row_idx, 1].imshow(np.rot90(vol[:, cy, :]), **kwargs)
            axes[row_idx, 1].set_title(f"{title} Cor")
            axes[row_idx, 2].imshow(np.rot90(vol[cx, :, :]), **kwargs)
            axes[row_idx, 2].set_title(f"{title} Sag")

        plot_row(0, img_np, "Image")
        plot_row(1, lab_np, "GT Seg", cmap="tab20", is_mask=True)
        plot_row(2, pred_np, "Pred Seg", cmap="tab20", is_mask=True)
        for ax in axes.flatten():
            ax.axis("off")
        plt.tight_layout()
        wandb.log({f"val_viz/viz_{subj_id}": wandb.Image(fig)}, step=step)
        plt.close(fig)

    def train(self):
        tqdm.write(f"[SegTrainer] 🏁 Starting Loop: Ep {self.start_epoch} -> {self.cfg['n_epochs']}")
        best_val_loss = float("inf")

        epoch_range = range(self.start_epoch, self.cfg["n_epochs"])
        epoch_pbar = tqdm(epoch_range, desc="Overall Training")
        for epoch in epoch_pbar:
            self.model.train()
            epoch_loss = 0
            step = 0

            train_pbar = tqdm(self.train_loader, desc=f"Train Ep {epoch + 1}", leave=False)
            for batch_data in train_pbar:
                step += 1
                inputs = batch_data["image"].to(self.device)
                labels = batch_data["label"].to(self.device)

                self.optimizer.zero_grad()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = self.model(inputs)
                    loss = self.loss_function(outputs, labels)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                current_lr = self.optimizer.param_groups[0]["lr"]

                pbar_dict = {"loss": f"{loss.item():.4f}", "gn": f"{grad_norm.item():.2f}", "lr": f"{current_lr:.2e}"}
                train_pbar.set_postfix(pbar_dict)

                if self.cfg["wandb"] and step == 1:
                    self._log_training_patch(inputs, labels, outputs, self.global_step)

                # RAM Monitoring (Optional Parity)
                if self.cfg["wandb"] and self.global_step % 100 == 0:
                    sys_percent, app_gb = get_ram_info()
                    wandb.log({"perf/ram_system_percent": sys_percent, "perf/ram_app_total_gb": app_gb}, step=self.global_step)

                self.global_step += 1

            epoch_loss /= step
            self.scheduler.step()

            if self.cfg["wandb"]:
                wandb.log({"train/loss": epoch_loss, "info/epoch": epoch + 1, "info/lr": self.optimizer.param_groups[0]["lr"], "info/grad_norm": grad_norm.item()}, step=self.global_step)

            if (epoch + 1) % self.cfg["val_interval"] == 0:
                val_loss = self.validate(epoch)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, best=True)
                    tqdm.write(f"✅ Epoch {epoch + 1} - New Best Dice Loss: {val_loss:.4f}")
                else:
                    tqdm.write(f"✅ Epoch {epoch + 1} - Validation Dice Loss: {val_loss:.4f}")

            if (epoch + 1) % self.cfg["model_save_interval"] == 0:
                self.save_checkpoint(epoch)

    def validate(self, epoch):
        self.model.eval()
        val_loss = 0.0
        val_steps = 0
        crop_size = self.cfg["patch_size"]
        self.dice_metric.reset()

        with torch.no_grad():
            for i, val_data in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
                val_images = val_data["image"].to(self.device)
                val_labels = val_data["label"].to(self.device)
                subj_id = val_data["subj_id"][0]

                roi_size = (crop_size, crop_size, crop_size)
                sw_batch_size = 16
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    val_outputs = sliding_window_inference(
                        val_images,
                        roi_size,
                        sw_batch_size,
                        self.model,
                        overlap=self.cfg["val_sw_overlap"],
                    )

                # Update Per-class Metric
                # 1. Convert logits to discrete labels
                val_outputs_discrete = monai.networks.utils.one_hot(torch.argmax(val_outputs, dim=1, keepdim=True), num_classes=self.cfg["n_classes"])
                val_labels_discrete = monai.networks.utils.one_hot(val_labels, num_classes=self.cfg["n_classes"])
                self.dice_metric(y_pred=val_outputs_discrete, y=val_labels_discrete)

                loss = self.val_loss_function(val_outputs, val_labels)
                val_loss += loss.item()
                val_steps += 1

                if self.cfg["wandb"] and i in self.val_viz_indices:
                    self._visualize_val(val_images, val_labels, val_outputs, self.global_step, subj_id)

                if self.cfg["save_val_volumes"] and i in self.val_viz_indices:
                    save_dir = os.path.join(self.cfg["log_dir"], "predictions", f"epoch_{epoch + 1}")
                    os.makedirs(save_dir, exist_ok=True)
                    pred_np = torch.argmax(val_outputs[0], dim=0).cpu().numpy().astype(np.uint8)
                    # Newer MONAI stores affine in the image tensor itself or .affine
                    affine = val_images.affine[0].cpu().numpy()
                    nib.save(nib.Nifti1Image(pred_np, affine), os.path.join(save_dir, f"{subj_id}_pred.nii.gz"))

        avg_val_loss = val_loss / val_steps if val_steps > 0 else 0

        # Aggregate Per-class scores (excludes class 0 because of DiceMetric config)
        # aggregate(reduction="mean_batch") returns a tensor of shape [C-1]
        all_class_means = self.dice_metric.aggregate(reduction="mean_batch")

        class_scores = {}
        for c_idx in range(all_class_means.shape[0]):
            class_id = c_idx + 1  # Indices are shifted
            name = self.cfg["class_names"][class_id]
            score = all_class_means[c_idx].item()
            class_scores[f"val_dice/class_{class_id}_{name.replace(' ', '_')}"] = score

        # Mean Hard Dice (excluding background)
        # Use nanmean to ignore classes that are not present in the validation set
        avg_hard_dice = torch.nanmean(all_class_means).item()

        if self.cfg["wandb"]:
            log_dict = {"val/dice_loss": avg_val_loss, "val/dice_score_soft": 1.0 - avg_val_loss, "val/dice_score_hard": avg_hard_dice}
            log_dict.update(class_scores)
            wandb.log(log_dict, step=self.global_step)

        return avg_val_loss

    def save_checkpoint(self, epoch, best=False):
        save_dir = wandb.run.dir if self.cfg["wandb"] else os.path.join(self.cfg["log_dir"], "models")
        os.makedirs(save_dir, exist_ok=True)
        filename = "seg_baby_unet_best.pth" if best else f"seg_baby_unet_epoch_{epoch}.pth"
        save_path = os.path.join(save_dir, filename)

        # Save non-compiled weights if applicable
        model_state = self.model.state_dict()
        if hasattr(self.model, "_orig_mod"):
            model_state = self.model._orig_mod.state_dict()

        torch.save(
            {
                "epoch": epoch,
                "global_step": self.global_step,
                "model_state_dict": model_state,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "config": self.cfg,
            },
            save_path,
        )
        tqdm.write(f"[SAVE] 💾 Checkpoint saved: {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--n_epochs", type=int, default=SEG_CONFIG["n_epochs"])
    parser.add_argument("--iters_per_epoch", type=int, default=SEG_CONFIG["iters_per_epoch"])
    parser.add_argument("--num_workers", type=int, default=SEG_CONFIG["num_workers"])
    args = parser.parse_args()

    SEG_CONFIG["n_epochs"] = args.n_epochs
    SEG_CONFIG["iters_per_epoch"] = args.iters_per_epoch
    SEG_CONFIG["num_workers"] = args.num_workers
    try:
        trainer = SegTrainer(SEG_CONFIG)
        trainer.train()
    except Exception as e:
        print(f"Training Failed: {e}")
        import traceback

        traceback.print_exc()
