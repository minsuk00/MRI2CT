import gc
import os
import random
import sys
import time
from collections import defaultdict

import nibabel as nib
import numpy as np
import torch
from anatomix.model.network import Unet
from monai.data import DataLoader, Dataset, PersistentDataset
from monai.inferers import sliding_window_inference

# from sklearn.decomposition import PCA
from tqdm import tqdm

import wandb

# Add CADS to path (Optional now, but kept for legacy)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "CADS")))

from common.config import Config
from common.data import (
    build_data_dicts,
    default_monai_cache_dir,
    get_cached_transforms,
    get_gpu_transforms,
    get_random_crop,
    get_subject_paths,
    gpu_augment_batch,
)
from common.trainer_base import BaseTrainer, StepTimer
from common.utils import (
    clean_state_dict,
    count_parameters,
    log_model_summary,
    set_seed,
    unpad,
    visualize_lite,
)


class Trainer(BaseTrainer):
    def __init__(self, config_dict):
        # 1. Config Setup
        cfg = Config(config_dict)
        super().__init__(cfg, prefix="Trainer")

        # Setup
        self._setup_models()
        self._find_subjects()
        self._setup_data()
        self._setup_opt()
        self._setup_wandb()

        # 8. Model Summary Logging
        models_to_log = {
            "Unet_Translator": self.model,
            "Anatomix_Feat_Extractor": self.feat_extractor,
        }
        if self.teacher_model is not None:
            models_to_log["Teacher_Model"] = self.teacher_model

        self._log_model_summary(models_to_log)

        self._load_resume()

        if getattr(self.cfg, "val_drr", False):
            self._precompute_gt_drrs()

    def _setup_data(self):
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

        # MONAI pipeline: cached deterministic transforms (CPU, PersistentDataset)
        # + per-step random GPU transforms applied in train_epoch via gpu_augment_batch.
        load_seg = getattr(self.cfg, "dice_w", 0) > 0 or getattr(self.cfg, "validate_dice", False)
        cache_dir = default_monai_cache_dir()
        os.makedirs(cache_dir, exist_ok=True)
        print(f"[Trainer] 💾 MONAI cache dir: {cache_dir}")

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
            persistent_workers=False,
            pin_memory=False,
        )
        self.train_iter = self._inf_gen(self.train_loader)

        self.gpu_transforms = get_gpu_transforms(
            augment=self.cfg.augment,
            has_seg=load_seg,
        )

        # Val: same cached transforms, no augmentation (full volumes via sliding-window inference).
        self.val_loader = self._build_val_loader(cached_xform, load_seg, cache_dir)

        # Stratified Validation Sampling (viz_limit per region)
        self.val_viz_indices, region_to_indices = self._stratify_val_indices(self.cfg.viz_limit)
        for region, indices in region_to_indices.items():
            picked = [i for i in indices if i in self.val_viz_indices]
            print(f"   - {region:10}: picked {len(picked)}/{len(indices)} (indices: {picked})")

    def _setup_models(self):
        # 1. Anatomix (Feature Extractor)
        print(f"[DEBUG] 🏗️ Building Anatomix ({self.cfg.anatomix_weights})...")
        if self.cfg.anatomix_weights == "v1":
            self.cfg.res_mult = 16
            self.feat_extractor = Unet(3, 1, 16, 4, 16).to(self.device)
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/anatomix.pth"
        elif self.cfg.anatomix_weights in ["v2", "v1_2", "v1_3"]:
            self.cfg.res_mult = 32
            feat_norm = getattr(self.cfg, "feat_norm", "instance")
            # Always use bias=True so conv biases from the pretrained checkpoint transfer correctly
            self.feat_extractor = Unet(3, 1, 16, 5, 20, norm=feat_norm, interp="trilinear", pooling="Avg", use_bias=True).to(self.device)
            # Optimize inference speed - Only compile if we aren't compiling the full step later
            if self.cfg.compile_mode != "full":
                print("[DEBUG] 🚀 Compiling Anatomix Feature Extractor...")
                self.feat_extractor = torch.compile(self.feat_extractor, mode="default")

            if self.cfg.anatomix_weights in ["v2", "v1_2"]:
                ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_v1_2.pth"
            else:  # v1_3
                ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_real_v1_3.pth"
        elif self.cfg.anatomix_weights == "v1_4":
            self.cfg.res_mult = 16
            # v1_4: same depth as v1 (ndowns=4) but 2x channels (ngf=32), trained with BatchNorm
            self.feat_extractor = Unet(3, 1, 16, 4, 32, norm="batch", interp="nearest", pooling="Max").to(self.device)
            if self.cfg.compile_mode != "full":
                print("[DEBUG] 🚀 Compiling Anatomix Feature Extractor...")
                self.feat_extractor = torch.compile(self.feat_extractor, mode="default")
            ckpt = "/home/minsukc/MRI2CT/anatomix/model-weights/best_val_net_G_BN_v1_4.pth"
        else:
            raise ValueError(f"Invalid anatomix_weights: {self.cfg.anatomix_weights}")

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
            depth = getattr(self.cfg, "finetune_depth", 0)
            self._feat_train_modules = None
            if depth == -1:
                print("[DEBUG] 🔓 Unfreezing Anatomix (Full Fine-Tuning)...")
                for p in self.feat_extractor.parameters():
                    p.requires_grad = True
                self.feat_extractor.train()
            elif depth > 0:
                # Access the underlying nn.Sequential model
                target_model = getattr(self.feat_extractor, "_orig_mod", self.feat_extractor)
                if hasattr(target_model, "model"):
                    modules = list(target_model.model.children())
                    unfreeze_start = max(0, len(modules) - depth)
                    print(f"[DEBUG] 🔓 Unfreezing Anatomix from module {unfreeze_start}/{len(modules)} (depth={depth})...")
                    # Freeze everything first, then selectively unfreeze
                    for p in self.feat_extractor.parameters():
                        p.requires_grad = False
                    self.feat_extractor.eval()
                    self._feat_train_modules = []
                    for i in range(unfreeze_start, len(modules)):
                        for p in modules[i].parameters():
                            p.requires_grad = True
                        modules[i].train()
                        self._feat_train_modules.append(modules[i])
                else:
                    raise RuntimeError(
                        f"finetune_feat_extractor=True with depth={depth} requires the feature extractor "
                        f"to expose a `.model` attribute (nn.Sequential of stages). Got: {type(target_model).__name__}."
                    )
            else:
                # Fallback for depth=0 or unset
                for p in self.feat_extractor.parameters():
                    p.requires_grad = False
                self.feat_extractor.eval()
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
            self.teacher_model = self._setup_teacher_model(compile_model=should_compile_models)

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

            # A2) Optional feature scale-down
            feat_scale = getattr(self.cfg, "feat_scale_down", 1)
            if feat_scale != 1:
                features = features / feat_scale

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

        loss.backward()
        return pred, loss, comps, pred_probs

    def _setup_opt(self):
        # We only pass parameters that actually require gradients to the optimizer
        params = [{"params": [p for p in self.model.parameters() if p.requires_grad], "lr": self.cfg.lr}]

        if getattr(self.cfg, "finetune_feat_extractor", False):
            feat_params = [p for p in self.feat_extractor.parameters() if p.requires_grad]
            if feat_params:
                print(f"[DEBUG] 💡 Adding {len(feat_params)} feat_extractor parameters to optimizer (LR={self.cfg.lr_feat_extractor})")
                params.append({"params": feat_params, "lr": self.cfg.lr_feat_extractor})
            else:
                print("[WARNING] ⚠️ finetune_feat_extractor is True, but no parameters require gradients.")

        self.optimizer = torch.optim.Adam(params)

        # 3. Scheduler Setup
        if self.cfg.scheduler_type == "cosine":
            t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
            print(f"[DEBUG] 📉 Initializing Scheduler (CosineAnnealingLR) T_max={t_max}, min_lr={self.cfg.scheduler_min_lr}")
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=t_max, eta_min=self.cfg.scheduler_min_lr)
        else:
            print("[DEBUG] 🛑 Scheduler DISABLED. Using fixed LR.")
            self.scheduler = None

        self._setup_loss()

    def _load_resume(self):
        super()._load_resume(self.model, self.optimizer, self.scheduler, extra_modules={"feat_extractor": self.feat_extractor})

    def save_checkpoint(self, epoch, is_last=False):
        filename = "checkpoint_last.pt" if is_last else f"{self.cfg.model_type}_epoch{epoch:05d}.pt"
        path = os.path.join(self._default_save_dir(), filename)

        extra_state = {}
        if self.cfg.finetune_feat_extractor:
            # Handle compiled model logic correctly
            target_feat = getattr(self.feat_extractor, "_orig_mod", self.feat_extractor)
            extra_state["feat_extractor_state_dict"] = target_feat.state_dict()

        super().save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, path, extra_state=extra_state)

    # ==========================================
    # CORE LOGIC
    # ==========================================
    def train_epoch(self, epoch):
        self.model.train()
        if self.cfg.finetune_feat_extractor:
            if self._feat_train_modules is not None:
                # Partial finetuning: only set unfrozen modules to train
                for m in self._feat_train_modules:
                    m.train()
            else:
                self.feat_extractor.train()
        else:
            self.feat_extractor.eval()

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
                        # Call the training step (compiled or eager)
                        # Note: self.train_step handles mixed precision context internally
                        pred, loss, comps, pred_probs = self.train_step(mri, ct, seg)

                    step_loss += loss.item()

                    # Log patch (using the returned predictions)
                    if self.cfg.wandb and step_idx == 0:
                        subj_id = batch["subj_id"][0] if "subj_id" in batch else None
                        self._log_training_patch(mri, ct, pred, self.global_step, step_idx, seg=seg, pred_probs=pred_probs, subj_id=subj_id)

                    for k, v in comps.items():
                        # Handle both tensors (from compiled step) and scalars
                        val = v.item() if hasattr(v, "item") else v
                        comp_accum[k] = comp_accum.get(k, 0.0) + (val / self.cfg.accum_steps)

                    if pred_probs is not None:
                        del pred_probs

                with timer.gpu("optimizer"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(list(self.model.parameters()) + (list(self.feat_extractor.parameters()) if self.cfg.finetune_feat_extractor else []), max_norm=1.0)
                    self.optimizer.step()

            # Step scheduler per iteration for cosine
            if self.scheduler is not None and self.cfg.scheduler_type == "cosine":
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
                pbar_dict.update({
                    "dt": f"{timings['data']:.0f}ms",
                    "cmp": f"{timings['compute']:.0f}ms",
                    "tot": f"{timings['step_total'] / 1000:.2f}s",
                })

            pbar.set_postfix(pbar_dict)

            # Log step-level info to WandB
            if self.cfg.wandb and step_idx % 200 == 0:
                cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume if self.global_start_time else self.elapsed_time_at_resume
                total_steps = self.cfg.steps_per_epoch * self.cfg.total_epochs
                log_dict = {
                    "info/lr": current_lr,
                    "info/grad_norm": grad_norm.item(),
                    "info/samples_seen": self.samples_seen,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/cumulative_time": cumulative_time,
                    "info/train_pct": self.global_step / total_steps,
                }
                wandb.log(log_dict, step=self.global_step)

        return total_loss / self.cfg.steps_per_epoch, {k: v / self.cfg.steps_per_epoch for k, v in comp_accum.items()}, total_grad / self.cfg.steps_per_epoch

    @torch.inference_mode()
    def validate(self, epoch):
        gc.collect()
        torch.cuda.empty_cache()

        self.model.eval()
        self.feat_extractor.eval()
        val_metrics = defaultdict(list)
        val_ps = getattr(self.cfg, "val_patch_size", self.cfg.patch_size)

        # 1. Validation Loop
        for i, batch in enumerate(tqdm(self.val_loader, desc="Validating", leave=False)):
            mri = batch["mri"].to(self.device)
            ct = batch["ct"].to(self.device)
            seg = batch["seg"].to(self.device) if "seg" in batch else None
            orig_shape = batch["original_shape"][0].tolist()
            subj_id = batch["subj_id"][0]
            # Always use Sliding Window for Full Volumes to prevent OOM
            def combined_forward(x):
                # 1. Features
                f = self.feat_extractor(x)

                # 2. Instance Norm
                if getattr(self.cfg, "feat_instance_norm", False):
                    f = torch.nn.functional.instance_norm(f)

                # 2b. Feature scale-down
                feat_scale = getattr(self.cfg, "feat_scale_down", 1)
                if feat_scale != 1:
                    f = f / feat_scale

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
                    roi_size=(val_ps, val_ps, val_ps),
                    sw_batch_size=self.cfg.val_sw_batch_size,
                    predictor=combined_forward,
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

            # Loss (Composite) - Now includes Dice if pred_probs is available.
            # Cast pred (bf16 from autocast SW inference) and ct (fp16 from cached storage) to fp32:
            # nn.L1Loss on mixed bf16/fp16 inputs would otherwise rely on implicit type promotion.
            l_val, l_comps = self.loss_fn(pred.float(), ct.float(), pred_probs=pred_probs, target_mask=seg)
            met["loss"] = l_val.item()
            for k, v in l_comps.items():
                met[k] = v.item() if hasattr(v, "item") else v

            # Body-masked dice (only when val_body_mask=True and teacher available)
            if body_met is not None and pred_probs is not None and seg is not None:
                self._compute_dice_metrics(pred_probs, seg, orig_shape, mask_unpad=mask_unpad, target_met=body_met)

            if pred_probs is not None:
                del pred_probs  # Fix memory leak

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
                    visualize_lite(
                        pred_unpad, ct_unpad, mri_unpad, subj_id, orig_shape, self.global_step, epoch, log_name=f"viz/val_{i}", metrics=viz_metrics, body_metrics=viz_body
                    )

                if self.cfg.val_drr and save_path and subj_id in self.gt_drrs:
                    self._log_drr_comparison(subj_id, save_path)

            del mri, ct, pred, pred_unpad, ct_unpad, mri_unpad
            # if "feats" in locals():
            #     del feats

        # 2. Augmentation Viz
        if self.cfg.wandb and self.cfg.augment:
            self._log_aug_viz(self.global_step)

        # 3. Log
        avg_met = self._log_val_metrics(val_metrics, subject_ids=self.val_subjects)

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

            val_duration = 0.0
            if (epoch % self.cfg.val_interval == 0) or (epoch + 1) == self.cfg.total_epochs:
                val_start = time.time()
                avg_met = self.validate(epoch)
                val_duration = time.time() - val_start
                val_loss = avg_met.get("loss", 0)

                tqdm.write(
                    f"Ep {epoch} | Train: {loss:.4f} | Val: {val_loss:.4f} | SSIM: {avg_met.get('ssim', 0):.4f} | PSNR: {avg_met.get('psnr', 0):.2f} | Dice: {avg_met.get('dice_score_all', 0):.4f} | Bone: {avg_met.get('dice_score_bone', 0):.4f}"
                )

            ep_duration = time.time() - ep_start
            cumulative_time = (time.time() - self.global_start_time) + self.elapsed_time_at_resume

            if self.cfg.wandb:
                # info/lr is logged in the per-200-steps block above; skip here to avoid
                # overwriting mid-epoch values at the epoch-boundary global_step.
                log = {
                    "train/total": loss,
                    "info/grad_norm": gn,
                    "info/epoch_duration": ep_duration,
                    "info/val_duration": val_duration,
                    "info/cumulative_time": cumulative_time,
                    "info/global_step": self.global_step,
                    "info/epoch": epoch,
                    "info/samples_seen": self.samples_seen,
                }
                for k, v in comps.items():
                    if "score" in k:  # Skip all non-loss scores from training charts
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
            wandb.log({"info/samples_seen_total": self.samples_seen})  # Log total samples at the end
            wandb.finish()
        print(f"✅ Training Complete. Total Time: {time.time() - self.global_start_time:.1f}s")
