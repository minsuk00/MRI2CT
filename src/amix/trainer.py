import datetime
import gc
import os
import random
import sys
import time
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
import torchio as tio
from anatomix.model.network import Unet
from anatomix.segmentation.segmentation_utils import load_model_v1_2
from monai.inferers import sliding_window_inference

# from sklearn.decomposition import PCA
from tqdm import tqdm

import wandb

# Add CADS to path (Optional now, but kept for legacy)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "CADS")))

from common.config import Config
from common.data import DataPreprocessing, build_tio_subjects, get_augmentations, get_region_key, get_subject_paths
from common.trainer_base import BaseTrainer
from common.utils import clean_state_dict, compute_metrics, count_parameters, get_ram_info, set_seed, unpad


class Trainer(BaseTrainer):
    def __init__(self, config_dict):
        # 1. Config Setup
        cfg = Config(config_dict)
        super().__init__(cfg, prefix="Trainer")

        # Setup
        self._setup_models()
        self._find_subjects()
        self._stage_data_local()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        self._load_resume()

    def _setup_data(self, seed=None):
        if seed is not None:
            # Re-seed to ensure different shuffling after worker restart
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

        # 1. Subjects already found in _find_subjects()
        if self.cfg.analyze_shapes:
            shapes = []
            for s in tqdm(self.train_subjects[:30], desc="Analyzing Shapes (Sample)"):
                try:
                    p = get_subject_paths(self.cfg.root_dir, s)
                    sh = nib.load(p["mri"]).header.get_data_shape()
                    shapes.append(sh[:3])  # Only take X, Y, Z
                except Exception:
                    print(f"  [WARNING] Failed to load {s} for shape analysis.")
                    pass
            if shapes:
                avg_shape = np.mean(shapes, axis=0).astype(int)
                print(f"📊 Mean Volume Shape: {tuple(int(x) for x in avg_shape)}")

        # 5. Train Loader (Queue)
        # Load seg if Dice weight > 0 or Dice validation is enabled
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)
        train_objs = build_tio_subjects(self.cfg.root_dir, self.train_subjects, use_weighted_sampler=self.cfg.use_weighted_sampler, load_seg=load_seg)
        # Main aligned with baseline: Disable safety padding by default
        use_safety = False

        # from common.data import Float16Storage

        preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=use_safety, res_mult=self.cfg.res_mult, use_weighted_sampler=self.cfg.use_weighted_sampler)

        transform_list = [preprocess]
        if self.cfg.augment:
            transform_list.append(get_augmentations())

        # Cast to float16 at the VERY END for RAM storage optimization
        # transform_list.append(Float16Storage())

        transforms = tio.Compose(transform_list)

        train_ds = tio.SubjectsDataset(train_objs, transform=transforms)

        if self.cfg.use_weighted_sampler:
            print("[DEBUG] ⚖️ Initializing Weighted Sampler (using body_mask.nii.gz)")
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

        self.train_iter = self._inf_gen(self.train_loader)

        # 6. Val Loader (Full Volume)
        # Load seg for validation if we are using Dice loss (to measure semantic consistency)
        val_objs = build_tio_subjects(self.cfg.root_dir, self.val_subjects, load_seg=load_seg)
        val_preprocess = DataPreprocessing(patch_size=self.cfg.patch_size, enable_safety_padding=False, res_mult=self.cfg.res_mult)
        val_ds = tio.SubjectsDataset(val_objs, transform=val_preprocess)
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

        # print(f"[DEBUG] 🖼️  Stratified Val Viz Indices: {self.val_viz_indices}")
        for region, indices in region_to_indices.items():
            picked = [i for i in viz_indices if i in indices]
            print(f"   - {region:10}: picked {len(picked)}/{len(indices)} (indices: {picked})")

    def _setup_models(self):
        # 1. Anatomix (Feature Extractor)
        print(f"[DEBUG] 🏗️ Building Anatomix ({self.cfg.anatomix_weights})...")
        if self.cfg.anatomix_weights == "v1":
            self.cfg.res_mult = 16
            self.feat_extractor = Unet(3, 1, 16, 4, 16).to(self.device)
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
        elif self.cfg.anatomix_weights == "v2":
            self.cfg.res_mult = 32
            self.feat_extractor = Unet(3, 1, 16, 5, 20, norm="instance", interp="trilinear", pooling="Avg").to(self.device)
            # Optimize inference speed - Only compile if we aren't compiling the full step later
            if self.cfg.compile_mode != "full":
                print("[DEBUG] 🚀 Compiling Anatomix Feature Extractor...")
                self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v2.pth"
        else:
            raise ValueError("Invalid anatomix_weights")

        if os.path.exists(ckpt):
            # Strip _orig_mod prefix if present
            state_dict = clean_state_dict(torch.load(ckpt, map_location=self.device))
            # Load into the underlying module if compiled
            target = getattr(self.feat_extractor, "_orig_mod", self.feat_extractor)
            target.load_state_dict(state_dict, strict=True)
            print(f"[DEBUG] Loaded Anatomix weights from {ckpt}")
        else:
            print(f"[WARNING] ⚠️ Anatomix weights NOT FOUND at {ckpt}")

        if self.cfg.finetune_feat_extractor:
            print("[DEBUG] 🔓 Unfreezing Anatomix for Fine-Tuning...")
            for p in self.feat_extractor.parameters():
                p.requires_grad = True
            self.feat_extractor.train()
        else:
            for p in self.feat_extractor.parameters():
                p.requires_grad = False
            self.feat_extractor.eval()

        tot, train = count_parameters(self.feat_extractor)
        print(f"[Model] Anatomix Feat Extractor Params: Total={tot:,} | Trainable={train:,}")

        # 2. Unet Translator (Generator)
        print("[DEBUG] 🏗️ Building Unet Translator...")
        translator_input_nc = 16
        if getattr(self.cfg, "pass_mri_to_translator", False):
            print("[DEBUG] 🧬 Passing Original MRI to Translator (Channels: 16 -> 17)")
            translator_input_nc = 17

        model = Unet(
            dimension=3,
            input_nc=translator_input_nc,
            output_nc=1,
            num_downs=4,
            ngf=16,
            final_act="sigmoid",
        ).to(self.device)

        # Non-recursive compilation logic
        # If 'full' mode, we compile the step, so we leave the models raw.
        # If 'model' mode, we compile the models individually.
        should_compile_models = self.cfg.compile_mode == "model"

        if should_compile_models:
            print("[DEBUG] 🚀 Compiling Generator (mode=default)...")
            self.model = torch.compile(model, mode="default")
        else:
            self.model = model

        tot, train = count_parameters(self.model)
        print(f"[Model] Unet Translator Params: Total={tot:,} | Trainable={train:,}")

        # 3. Teacher Model (Baby U-Net) for Dice Loss / Validation
        self.teacher_model = None
        if getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False):
            print("[DEBUG] 👨‍🏫 Initializing Baby U-Net Teacher for Dice Loss / Validation...")
            try:
                # Load Baby U-Net (12 classes: 11 organs + Brain)
                self.teacher_model = load_model_v1_2(pretrained_ckpt=self.cfg.teacher_weights_path, n_classes=self.cfg.n_classes - 1, device=self.device, compile_model=False)

                # Freeze Teacher
                self.teacher_model.to(device=self.device, dtype=torch.bfloat16)
                self.teacher_model.eval()
                for p in self.teacher_model.parameters():
                    p.requires_grad = False

                if should_compile_models:
                    print("[DEBUG] 🚀 Compiling Teacher with mode: default")
                    self.teacher_model = torch.compile(self.teacher_model, mode="default")

                tot, train = count_parameters(self.teacher_model)
                print(f"[Model] Teacher Params: Total={tot:,} | Trainable={train:,} | Dtype=BFloat16")
                print("[DEBUG] ✅ Teacher initialized.")
            except Exception as e:
                print(f"[ERROR] ❌ Failed to init Teacher Model: {e}")
                if self.cfg.dice_w > 0:
                    raise e

        # 4. Step Compilation
        if self.cfg.compile_mode == "full":
            print("[SegTrainer] 🚀 Compiling Training Step (mode=default)...")
            self.train_step = torch.compile(self._train_step, mode="default")
        else:
            self.train_step = self._train_step

    def _train_step(self, mri, ct, seg=None):
        # Note: zero_grad is handled in the outer loop for accumulation
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.cfg.finetune_feat_extractor:
                features = self.feat_extractor(mri)
            else:
                # Feat extractor is usually frozen, so we can wrap in no_grad even inside compile
                with torch.no_grad():
                    features = self.feat_extractor(mri)

            # A) Optional Instance Normalization on Amix Features
            if getattr(self.cfg, "feat_instance_norm", False):
                features = torch.nn.functional.instance_norm(features)

            if getattr(self.cfg, "pass_mri_to_translator", False):
                translator_input = torch.cat([features, mri], dim=1)
                
                # B) Optional Input Dropout (3D) - Only if MRI is passed/concatenated
                drop_p = getattr(self.cfg, "input_dropout_p", 0.0)
                if drop_p > 0:
                    translator_input = torch.nn.functional.dropout3d(translator_input, p=drop_p, training=True)
            else:
                translator_input = features

            pred = self.model(translator_input)

            # Dice Loss Calculation
            pred_probs = None
            if (getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "dice_bone_w", 0) > 0) and self.teacher_model is not None and seg is not None:
                pred_probs = self.teacher_model(pred)

            loss, comps = self.loss_fn(pred, ct, pred_probs=pred_probs, target_mask=seg)

            loss = loss / self.cfg.accum_steps

        self.scaler.scale(loss).backward()
        return pred, loss, comps, pred_probs

    def _setup_opt(self):
        params = [{"params": self.model.parameters(), "lr": self.cfg.lr}]
        if self.cfg.finetune_feat_extractor:
            params.append({"params": self.feat_extractor.parameters(), "lr": self.cfg.lr_feat_extractor})

        self.optimizer = torch.optim.Adam(params)

        # 3. Scheduler Setup
        if self.cfg.scheduler_type == "cosine":
            t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
            print(f"[DEBUG] 📉 Initializing Scheduler (CosineAnnealingLR) T_max={t_max}, min_lr={self.cfg.scheduler_min_lr}")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=self.cfg.scheduler_min_lr)
        else:
            print("[DEBUG] 🛑 Scheduler DISABLED. Using fixed LR.")
            self.scheduler = None

        self._setup_loss_and_scaler()

    def _load_resume(self):
        super()._load_resume(self.model, self.optimizer, self.scheduler, self.scaler, extra_modules={"feat_extractor": self.feat_extractor})

    def save_checkpoint(self, epoch, is_final=False):
        filename = f"{self.cfg.model_type}_epoch{epoch:05d}.pt"
        save_dir = wandb.run.dir if self.cfg.wandb else os.path.join(self.gpfs_root, "results", "models")
        path = os.path.join(save_dir, filename)

        extra_state = {}
        if self.cfg.finetune_feat_extractor:
            # Handle compiled model logic correctly
            target_feat = getattr(self.feat_extractor, "_orig_mod", self.feat_extractor)
            extra_state["feat_extractor_state_dict"] = target_feat.state_dict()

        super().save_checkpoint(self.model, self.optimizer, self.scheduler, self.scaler, epoch, path, extra_state=extra_state)

    # ==========================================
    # CORE LOGIC
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        if self.cfg.finetune_feat_extractor:
            self.feat_extractor.train()
        else:
            self.feat_extractor.eval()

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
                step_t_bwd = 0.0
                t_step_start = time.time()

            for _ in range(self.cfg.accum_steps):
                if self.cfg.enable_profiling:
                    t0 = time.time()
                batch = next(self.train_iter)
                if self.cfg.enable_profiling:
                    t1 = time.time()
                    step_t_data += t1 - t0

                # Extract tensors (torchio uses [tio.DATA])
                mri = batch["mri"][tio.DATA].to(self.device, non_blocking=True)
                ct = batch["ct"][tio.DATA].to(self.device, non_blocking=True)
                seg = batch["seg"][tio.DATA].to(self.device, non_blocking=True) if "seg" in batch else None

                if self.cfg.enable_profiling:
                    torch.cuda.synchronize()
                    t2 = time.time()

                # Call the training step (compiled or eager)
                # Note: self.train_step handles mixed precision context internally
                pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                if self.cfg.enable_profiling:
                    torch.cuda.synchronize()
                    t3 = time.time()
                    step_t_fwd += t3 - t2
                    step_t_bwd += 0  # Backward is inside train_step

                step_loss += loss.item()

                # Log patch (using the returned predictions)
                if self.cfg.wandb and step_idx == 0:
                    subj_id = batch["subj_id"][0] if "subj_id" in batch else None
                    self._log_training_patch(mri, ct, pred, self.global_step, step_idx, seg=seg, pred_probs=pred_probs, subj_id=subj_id)

                for k, v in comps.items():
                    # Handle both tensors (from compiled step) and scalars
                    val = v.item() if hasattr(v, "item") else v
                    comp_accum[k] = comp_accum.get(k, 0.0) + (val / self.cfg.accum_steps)

            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + (list(self.feat_extractor.parameters()) if self.cfg.finetune_feat_extractor else []), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # Step scheduler per iteration for cosine
            if self.scheduler is not None and self.cfg.scheduler_type == "cosine":
                if self.scheduler.last_epoch < self.scheduler.T_max:
                    self.scheduler.step()

            total_loss += step_loss
            total_grad += grad_norm.item()

            self.global_step += 1
            self.samples_seen += self.cfg.batch_size * self.cfg.accum_steps

            pbar_dict = {"loss": f"{step_loss:.4f}", "gn": f"{grad_norm.item():.2f}"}

            current_lr = self.optimizer.param_groups[0]["lr"]
            if self.cfg.enable_profiling:
                t_step_end = time.time()
                step_t_total = t_step_end - t_step_start
                avg_batch_data = (step_t_data / self.cfg.accum_steps) * 1000
                avg_batch_fwd = (step_t_fwd / self.cfg.accum_steps) * 1000
                pbar_dict.update({"dt": f"{avg_batch_data:.1f}ms", "fwd": f"{avg_batch_fwd:.1f}ms", "tot": f"{step_t_total:.2f}s", "lr": f"{current_lr:.2e}"})

                if self.cfg.wandb:
                    wandb.log(
                        {
                            "info/time_data_ms": avg_batch_data,
                            "info/time_forward_ms": avg_batch_fwd,
                            "info/time_step_total_s": step_t_total,
                        },
                        step=self.global_step,
                    )

            pbar.set_postfix(pbar_dict)

            # Log step-level info to WandB
            if self.cfg.wandb and step_idx % 200 == 0:
                cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume if self.global_start_time else self.elapsed_time_at_resume
                log_dict = {
                    "info/lr": current_lr,
                    "info/grad_norm": grad_norm.item(),
                    "info/samples_seen": self.samples_seen,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/cumulative_time": cumulative_time,
                }
                wandb.log(log_dict, step=self.global_step)

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
        self.feat_extractor.eval()
        val_metrics = defaultdict(list)

        # 1. Validation Loop
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch["mri"][tio.DATA].to(self.device)
            ct = batch["ct"][tio.DATA].to(self.device)
            seg = batch["seg"][tio.DATA].to(self.device) if "seg" in batch else None
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]
            pad_offset = int(batch["pad_offset"][0]) if "pad_offset" in batch else 0

            # Always use Sliding Window for Full Volumes to prevent OOM
            def combined_forward(x):
                # 1. Features
                f = self.feat_extractor(x)
                
                # 2. Instance Norm
                if getattr(self.cfg, "feat_instance_norm", False):
                    f = torch.nn.functional.instance_norm(f)

                # 3. Concatenation
                if getattr(self.cfg, "pass_mri_to_translator", False):
                    f = torch.cat([f, x], dim=1)
                    
                    # 4. Input Dropout (Conditional on pass_mri)
                    drop_p = getattr(self.cfg, "input_dropout_p", 0.0)
                    if drop_p > 0:
                        f = torch.nn.functional.dropout3d(f, p=drop_p, training=False)
                
                return self.model(f)

            # Optimization: AMP for faster inference
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                pred = sliding_window_inference(
                    inputs=mri,
                    roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size),
                    sw_batch_size=self.cfg.val_sw_batch_size,
                    predictor=combined_forward,
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
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    # Always use sliding window for teacher on full volumes to prevent OOM
                    pred_probs = sliding_window_inference(
                        inputs=pred,
                        roi_size=(self.cfg.patch_size, self.cfg.patch_size, self.cfg.patch_size),
                        sw_batch_size=self.cfg.val_sw_batch_size,
                        predictor=self.teacher_model,
                        overlap=self.cfg.val_sw_overlap,
                        device=self.device,
                    )

            # Loss (Composite) - Now includes Dice if pred_probs is available
            l_val, l_comps = self.loss_fn(pred, ct, pred_probs=pred_probs, target_mask=seg)
            met["loss"] = l_val.item()
            for k, v in l_comps.items():
                met[k] = v.item() if hasattr(v, "item") else v

            if pred_probs is not None:
                del pred_probs  # Fix memory leak

            for k, v in met.items():
                val_metrics[k].append(v)

            # Viz & Save
            if i in self.val_viz_indices:
                save_path = None
                # Optimization: Only save volumes if visualized
                if self.cfg.save_val_volumes:
                    save_dir = os.path.join(self.cfg.prediction_dir, self.run_name, f"epoch_{epoch}")
                    os.makedirs(save_dir, exist_ok=True)

                    # Denormalize [0, 1] -> [-1024, 1024]
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

                if self.cfg.wandb:
                    self._visualize_lite(pred, ct, mri, subj_id, orig_shape, self.global_step, epoch, idx=i, offset=pad_offset, save_path=save_path)

            del mri, ct, pred, pred_unpad, ct_unpad
            if "feats" in locals():
                del feats

        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
            self._log_aug_viz(self.global_step)

        # 3. Log
        avg_met = {k: np.mean(v) for k, v in val_metrics.items()}
        if self.cfg.wandb:
            wandb.log({f"val/{k}": v for k, v in avg_met.items()}, step=self.global_step)

        gc.collect()
        torch.cuda.empty_cache()

        return avg_met

    def train(self):
        print(f"[DEBUG] 🏁 Starting Loop: Ep {self.start_epoch} -> {self.cfg.total_epochs}")
        self.global_start_time = time.time()

        if self.cfg.sanity_check and not self.cfg.resume_wandb_id:
            print("[DEBUG] running sanity check...")
            avg_met = self.validate(0)
            tqdm.write(
                f"Ep -1 | Train: 0.0000 | Val: {avg_met.get('loss', 0):.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Dice: {avg_met.get('dice_score_all', 0):.4f}"
            )

        global_pbar = tqdm(range(self.start_epoch, self.cfg.total_epochs), desc="🚀 Total Progress", initial=self.start_epoch, total=self.cfg.total_epochs, dynamic_ncols=True, unit="ep")

        for epoch in global_pbar:
            ep_start = time.time()
            loss, comps, gn = self.train_epoch(epoch)

            if (epoch % self.cfg.val_interval == 0) or (epoch + 1) == self.cfg.total_epochs:
                avg_met = self.validate(epoch)
                val_loss = avg_met.get("loss", 0)

                tqdm.write(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {val_loss:.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Dice: {avg_met.get('dice_score_all', 0):.4f} | Bone: {avg_met.get('dice_score_bone', 0):.4f}"
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
                    if "score" in k: # Skip all non-loss scores from training charts
                        continue
                    log[k.replace("loss_", "train/")] = v
                wandb.log(log, step=self.global_step)

            if epoch % self.cfg.model_save_interval == 0:
                self.save_checkpoint(epoch)

            # Explicit cleanup after each epoch
            gc.collect()
            torch.cuda.empty_cache()

        self.save_checkpoint(self.cfg.total_epochs, is_final=True)
        if self.cfg.wandb:
            wandb.log({"info/samples_seen_total": self.samples_seen})  # Log total samples at the end
        if self.cfg.wandb:
            wandb.finish()
        print(f"✅ Training Complete. Total Time: {time.time() - self.global_start_time:.1f}s")
