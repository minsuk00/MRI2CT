import datetime
import gc
import os
import random
import textwrap
import time
from collections import defaultdict
from contextlib import contextmanager
from glob import glob

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from monai.transforms import CutOut
from tqdm import tqdm

import wandb
from common.data import (
    build_data_dicts,
    get_cached_transforms,
    get_gpu_transforms,
    get_random_crop,
    get_region_key,
    get_split_subjects,
    get_subject_paths,
)
from common.utils import apply_synchronized_cutout, clean_state_dict, get_ram_info, log_model_summary, set_seed, unpad

VIZ_METRIC_KEYS = ("ssim", "psnr", "mae_hu", "dice_score_all", "dice_score_bone")


class StepTimer:
    """Per-step timing aggregator for monitoring. Use as a `with` block; wrap
    sub-sections with `timer.cpu(name)` (perf_counter) or `timer.gpu(name)`
    (cuda.Event for accurate GPU-stream timing). Multiple records of the same
    name are averaged on `timings_ms()` (suits gradient-accumulation).

    Disabled mode (`enabled=False`) makes every section a zero-overhead no-op,
    so trainers can wrap critical paths unconditionally.
    """

    def __init__(self, enabled: bool):
        self.enabled = enabled
        self._cpu_total = {}
        self._cpu_count = {}
        self._gpu_evts = {}
        self._t_step_start = None
        self._t_step_end = None

    def __enter__(self):
        if self.enabled:
            torch.cuda.synchronize()
            self._t_step_start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.enabled:
            torch.cuda.synchronize()
            self._t_step_end = time.perf_counter()
        return False

    @contextmanager
    def cpu(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._cpu_total[name] = self._cpu_total.get(name, 0.0) + (time.perf_counter() - t0)
            self._cpu_count[name] = self._cpu_count.get(name, 0) + 1

    @contextmanager
    def gpu(self, name: str):
        if not self.enabled:
            yield
            return
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        try:
            yield
        finally:
            e.record()
            self._gpu_evts.setdefault(name, []).append((s, e))

    def elapsed_s(self) -> float:
        if not self.enabled or self._t_step_start is None or self._t_step_end is None:
            return 0.0
        return self._t_step_end - self._t_step_start

    def timings_ms(self) -> dict:
        """Per-section averages in milliseconds, plus 'step_total'. Empty when disabled."""
        if not self.enabled:
            return {}
        out = {}
        for name, total in self._cpu_total.items():
            out[name] = (total / self._cpu_count[name]) * 1000.0
        for name, evts in self._gpu_evts.items():
            out[name] = sum(s.elapsed_time(e) for s, e in evts) / max(1, len(evts))
        out["step_total"] = self.elapsed_s() * 1000.0
        return out


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

    def _load_resume(self, model, optimizer=None, scheduler=None, extra_modules=None):
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

        # 3. Restore Optimizer/Scheduler
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
                    # Standard logic: load_state_dict fully restores last_epoch + base_lrs.
                    # Optimizer.lr was restored by optimizer.load_state_dict above; the next
                    # scheduler.step() in the train loop advances correctly from there.
                    # NOTE: do NOT call scheduler.step() here — the recurrence formula in
                    # CosineAnnealingLR / PolynomialLR uses optimizer.lr as "previous lr",
                    # which is already at last_epoch=N, so a manual extra step would apply
                    # one extra round of decay (~1% drift per resume).
                    print("[RESUME] 📥 Loading Scheduler state...")
                    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

                    new_t_max = self.cfg.steps_per_epoch * self.cfg.total_epochs
                    if hasattr(scheduler, "T_max") and new_t_max != scheduler.T_max:
                        print(f"[RESUME] 🔧 Updating Scheduler T_max: {scheduler.T_max} -> {new_t_max}")
                        scheduler.T_max = new_t_max
                    elif hasattr(scheduler, "total_iters") and new_t_max != scheduler.total_iters:
                        print(f"[RESUME] 🔧 Updating Scheduler total_iters: {scheduler.total_iters} -> {new_t_max}")
                        scheduler.total_iters = new_t_max

            print(f"[RESUME] ✅ Resumed from Epoch {self.start_epoch}")
        else:
            print("[RESUME] 🌿 Diverging. Weights loaded, but state (epoch, step) reset.")

    def save_checkpoint(self, model, optimizer, scheduler, epoch, path, extra_state=None):
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
            "elapsed_time": self.elapsed_time_at_resume + (time.time() - self.global_start_time) if self.global_start_time else self.elapsed_time_at_resume,
            "config": vars(self.cfg),
        }

        if extra_state:
            save_dict.update(extra_state)

        torch.save(save_dict, path)
        print(f"[{self.prefix}] [Save] {path}")

    def _inf_gen(self, loader):
        """Infinite generator over a torch DataLoader."""
        while True:
            iterator = iter(loader)
            for batch in iterator:
                yield batch
            del iterator
            import gc

            gc.collect()

    def _log_monitoring(self, timings_ms: dict, throughput: float = None):
        """Log RAM (parent + DataLoader workers via PSS), VRAM (alloc + interval peak),
        per-section timings and throughput to wandb under `monitoring/`.

        Trainers should call this every `cfg.monitor_interval` steps after a
        `torch.cuda.synchronize()` so cuda.Event timings are valid.
        """
        if not self.cfg.wandb or not getattr(self.cfg, "monitor_resources", False):
            return

        payload = {f"monitoring/time_{k}_ms": v for k, v in timings_ms.items()}

        ram = get_ram_info()
        payload.update({
            "monitoring/ram_total_gb": ram["total_gb"],
            "monitoring/ram_main_rss_gb": ram["main_rss_gb"],
            "monitoring/ram_workers_count": ram["num_children"],
            "monitoring/ram_percent": ram["percent"],
        })

        if torch.cuda.is_available():
            payload.update({
                "monitoring/vram_alloc_gb": torch.cuda.memory_allocated() / (1024**3),
                "monitoring/vram_peak_gb": torch.cuda.max_memory_allocated() / (1024**3),
            })
            torch.cuda.reset_peak_memory_stats()

        if throughput is not None:
            payload["monitoring/samples_per_sec"] = throughput

        wandb.log(payload, step=self.global_step)

        if ram["percent"] > 90:
            tqdm.write(f"[WARNING] ⚠️ RAM {ram['percent']:.1f}% (Total: {ram['total_gb']:.2f} GB) — gc + cache flush.")
            gc.collect()
            torch.cuda.empty_cache()

    def _log_aug_viz(self, step):
        """Visualize one cropped patch before/after random augmentation (batchaug pipeline).

        Records the per-transform fire mask by wrapping each Rand*d transform's
        `sample_params` for this single call, so the figure shows what *actually*
        fired this step — not just the static pipeline manifest.
        """
        try:
            subj_id = self.val_subjects[0]
            dicts = build_data_dicts(self.cfg.root_dir, [subj_id], load_seg=False)
            cached = get_cached_transforms(
                patch_size=self.cfg.patch_size,
                res_mult=self.cfg.res_mult,
                enforce_ras=getattr(self.cfg, "enforce_ras", False),
                mri_norm=getattr(self.cfg, "mri_norm", "minmax"),
            )
            crop = get_random_crop(
                patch_size=self.cfg.patch_size,
                use_weighted_sampler=True, has_seg=False, num_samples=1,
            )
            full_aug = get_gpu_transforms(augment=True, has_seg=False)

            base = cached(dicts[0])
            patch = crop(base)[0]

            # Move to GPU as 5D batched dict (B=1) for batchaug.
            def _prep(t):
                t = t.as_tensor() if hasattr(t, "as_tensor") else t
                return t.unsqueeze(0).to(self.device).float()

            orig5 = {"mri": _prep(patch["mri"]), "ct": _prep(patch["ct"])}

            # Patch sample_params on each Rand*d transform to capture its fire mask
            # for batch element 0. batchaug stores the dict-wrapped transform under
            # `t.transform`; in lazy mode Compose calls that directly.
            fire_log: dict[str, bool] = {}
            patches: list[tuple[object, str, callable]] = []
            order: list[str] = []
            for t in full_aug.transforms:
                cls_name = type(t).__name__
                if not cls_name.startswith("Rand"):
                    continue
                inner = getattr(t, "transform", None)
                if inner is None or not hasattr(inner, "sample_params"):
                    continue
                # Use repeated-key handling: append index if duplicate (shouldn't occur).
                label = cls_name
                if label in fire_log:
                    label = f"{cls_name}#{order.count(cls_name)}"
                order.append(label)
                fire_log[label] = False
                orig_fn = inner.sample_params

                def make_wrapper(orig, key):
                    def wrapped(batch_size, shape, device):
                        params = orig(batch_size, shape, device)
                        m = params.get("mask") if isinstance(params, dict) else None
                        if m is not None and m.numel() > 0:
                            fire_log[key] = bool(m.flatten()[0].item())
                        return params
                    return wrapped

                inner.sample_params = make_wrapper(orig_fn, label)
                patches.append((inner, "sample_params", orig_fn))

            try:
                aug5 = full_aug({k: v.clone() for k, v in orig5.items()})
            finally:
                for inner, attr, orig_fn in patches:
                    setattr(inner, attr, orig_fn)

            fired = [k for k in order if fire_log[k]]
            skipped = [k for k in order if not fire_log[k]]
            n_f, n_total = len(fired), len(order)

            def _wrap(items, width=70):
                if not items:
                    return "(none)"
                return "\n  ".join(textwrap.wrap(", ".join(items), width=width))

            caption = (
                f"Fired ({n_f}/{n_total}):\n  {_wrap(fired)}\n"
                f"Skipped ({n_total - n_f}/{n_total}):\n  {_wrap(skipped)}"
            )

            z = orig5["mri"].shape[-1] // 2
            orig_sl = orig5["mri"][0, 0, ..., z].cpu().numpy()
            aug_sl = aug5["mri"][0, 0, ..., z].float().cpu().numpy()
            fig, ax = plt.subplots(1, 3, figsize=(12, 5.0))
            ax[0].imshow(np.rot90(orig_sl), cmap="gray", vmin=0, vmax=1)
            ax[0].set_title("Original")
            ax[1].imshow(np.rot90(aug_sl), cmap="gray", vmin=0, vmax=1)
            ax[1].set_title(f"Augmented ({n_f}/{n_total} fired)")
            ax[2].imshow(np.rot90(aug_sl - orig_sl), cmap="seismic", vmin=-0.5, vmax=0.5)
            ax[2].set_title("Diff")
            for a in ax:
                a.set_xticks([]); a.set_yticks([])
            fig.text(0.01, 0.01, caption, fontsize=7, family="monospace",
                     ha="left", va="bottom")
            fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.30)
            wandb.log({"val/aug_viz": wandb.Image(fig)}, step=step)
            plt.close(fig)
        except Exception as e:
            print(f"[{self.prefix}] [WARNING] Aug Viz failed: {e}")

    _DEFAULT_VAL_EXCLUDE = {"loss_l1", "loss_ssim", "loss_dice", "grad_diff"}

    def _log_val_metrics(self, val_metrics, exclude=None, extra=None, subject_ids=None):
        """Mean-reduce over val subjects and log to WandB under three namespaces:
          val/        — image metrics (mae_hu, psnr, ssim, dice_score_*)
          val_loss/   — composite loss + components ("loss" → total; "loss_X" → X)
          val_body/   — body-masked metrics

        Defaults exclude redundant components (loss_l1≡mae_hu, loss_ssim≡ssim,
        loss_dice≡dice_score_all, grad_diff). Returns the unsplit avg_met dict
        for downstream callers (train printing, etc.).
        """
        exclude = exclude if exclude is not None else self._DEFAULT_VAL_EXCLUDE
        avg_met = {k: np.mean(v) for k, v in val_metrics.items() if not k.startswith("body_")}
        avg_body = {k[5:]: np.mean(v) for k, v in val_metrics.items() if k.startswith("body_")}

        if extra:
            avg_met.update(extra)

        if self.cfg.wandb:
            metric_log, loss_log = {}, {}
            for k, v in avg_met.items():
                if k in exclude:
                    continue
                if k == "loss":
                    loss_log["total"] = v
                elif k.startswith("loss_"):
                    loss_log[k[5:]] = v
                else:
                    metric_log[k] = v

            if metric_log:
                wandb.log({f"val/{k}": v for k, v in metric_log.items()}, step=self.global_step)
            if loss_log:
                wandb.log({f"val_loss/{k}": v for k, v in loss_log.items()}, step=self.global_step)
            if avg_body:
                wandb.log({f"val_body/{k}": v for k, v in avg_body.items() if k not in exclude}, step=self.global_step)

        if self.cfg.wandb and subject_ids is not None and "psnr" in val_metrics:
            region_psnr = defaultdict(list)
            for subj_id, v in zip(subject_ids, val_metrics["psnr"]):
                region_psnr[get_region_key(subj_id)].append(v)
            wandb.log({f"val_region/psnr/{r}": np.mean(vs) for r, vs in region_psnr.items()}, step=self.global_step)

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

    def _compute_val_metrics(self, pred_unpad, ct_unpad, mask_unpad=None, hu_range=2048):
        """Returns (standard_met, body_met_or_None). Log under val/ and val_body/ respectively.

        `hu_range`: data-space HU width to scale the [0,1] MAE into HU.
          - amix/unet: 2048 (CT clipped to [-1024, 1024])
          - maisi:     2000 (CT clipped to [-1000, 1000])
        """
        from common.utils import compute_metrics, compute_metrics_body

        met = compute_metrics(pred_unpad, ct_unpad, hu_range=hu_range)
        body_met = compute_metrics_body(pred_unpad, ct_unpad, mask_unpad, hu_range=hu_range) if mask_unpad is not None else None
        return met, body_met

    def _get_body_mask_unpad(self, batch, orig_shape):
        """Unpad body_mask from batch when cfg.val_body_mask is True; else None."""
        if getattr(self.cfg, "val_body_mask", False) and "body_mask" in batch:
            return unpad(batch["body_mask"].to(self.device), orig_shape)
        return None

    def _select_viz_metrics(self, met, body_met):
        """Build the (viz_metrics, viz_body) dict pair from full metric dicts."""
        viz_metrics = {k: met[k] for k in VIZ_METRIC_KEYS if k in met}
        viz_body = {k: body_met[k] for k in VIZ_METRIC_KEYS if body_met and k in body_met} or None
        return viz_metrics, viz_body

    def _compute_dice_metrics(self, pred_probs, seg, orig_shape, mask_unpad=None, target_met=None):
        """Compute dice_score_all / dice_score_bone from teacher pred_probs + seg.

        If pred_probs/seg are pre-unpadded (passed already at orig_shape), pass orig_shape=None.
        Returns the updated `target_met` dict (or a fresh dict). When mask_unpad is given, dices
        are body-masked via get_class_dice's mask kwarg.
        """
        from common.loss import get_class_dice

        if orig_shape is not None:
            pred_probs = unpad(pred_probs, orig_shape)
            seg = unpad(seg, orig_shape)
        bone_idx = getattr(self.cfg, "dice_bone_idx", 5)
        class_dices, bone_dice = get_class_dice(pred_probs, seg, mask=mask_unpad, bone_idx=bone_idx)
        out = target_met if target_met is not None else {}
        out["dice_score_all"] = class_dices.mean().item()
        if bone_dice is not None:
            out["dice_score_bone"] = bone_dice.item()
        return out

    def _run_teacher_sw(self, inputs, val_ps):
        """Run sliding-window inference with self.teacher_model under bf16 autocast."""
        from monai.inferers import sliding_window_inference

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return sliding_window_inference(
                inputs=inputs,
                roi_size=(val_ps, val_ps, val_ps),
                sw_batch_size=self.cfg.val_sw_batch_size,
                predictor=self.teacher_model,
                overlap=self.cfg.val_sw_overlap,
                device=self.device,
            )

    def _setup_teacher_model(self, compile_model=False, dtype=torch.bfloat16):
        """Load + freeze + (optionally compile) the Baby U-Net teacher.

        Caller is responsible for the decision to call this (i.e., for the
        enable predicate). Returns the loaded teacher, or None on failure
        when cfg.dice_w==0 (preserving original behavior of swallowing the
        exception unless dice loss is required).
        """
        from anatomix.segmentation.segmentation_utils import load_model_v1_2

        print(f"[{self.prefix}] 👨‍🏫 Initializing Baby U-Net Teacher...")
        try:
            teacher = load_model_v1_2(
                pretrained_ckpt=self.cfg.teacher_weights_path,
                n_classes=self.cfg.n_classes - 1,
                device=self.device,
                compile_model=False,
            )
            teacher.to(device=self.device, dtype=dtype)
            teacher.eval()
            for p in teacher.parameters():
                p.requires_grad = False
            if compile_model:
                print(f"[{self.prefix}] 🚀 Compiling Teacher (mode=default)")
                teacher = torch.compile(teacher, mode="default")
            from common.utils import count_parameters
            tot, trn = count_parameters(teacher)
            print(f"[{self.prefix}] Teacher Params: Total={tot:,} | Trainable={trn:,} | Dtype={dtype}")
            return teacher
        except Exception as e:
            print(f"[{self.prefix}] ❌ Failed to init Teacher Model: {e}")
            if self.cfg.dice_w > 0:
                raise
            return None

    def _default_save_dir(self):
        """Resolve the canonical checkpoint directory: local_run_dir if wandb, else gpfs fallback."""
        if self.cfg.wandb and self.local_run_dir:
            return self.local_run_dir
        return os.path.join(self.gpfs_root, "results", "models")

    def _build_val_loader(self, cached_xform, load_seg, cache_dir, hash_transform=None):
        """Build the standard val DataLoader: PersistentDataset, batch=1, no workers.

        `hash_transform`: optional callable forwarded to PersistentDataset so the
        cache key includes the transform spec. Pass `pickle_hashing` when the
        same data dict can be paired with semantically-different transforms
        (e.g. MAISI preencoded vs on-the-fly), to prevent cache poisoning.
        """
        from monai.data import DataLoader, PersistentDataset

        val_dicts = build_data_dicts(self.cfg.root_dir, self.val_subjects, load_seg=load_seg)
        kwargs = {"hash_transform": hash_transform} if hash_transform is not None else {}
        val_ds = PersistentDataset(data=val_dicts, transform=cached_xform, cache_dir=cache_dir, **kwargs)
        return DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    def _stratify_val_indices(self, n_per_region: int, seed: int = None):
        """Pick `n_per_region` validation subject indices per anatomical region.

        Returns (chosen_set, region_to_indices). Uses cfg.seed if seed is None,
        so the choice is reproducible across resumes. Subject IDs listed in
        cfg.viz_force_include are always added to the chosen set (if present
        in val_subjects).
        """
        rng = random.Random(seed if seed is not None else self.cfg.seed)
        region_to_indices = defaultdict(list)
        for idx, subj_id in enumerate(self.val_subjects):
            region_to_indices[get_region_key(subj_id)].append(idx)
        chosen = []
        for region, indices in region_to_indices.items():
            chosen.extend(rng.sample(indices, min(n_per_region, len(indices))))
        chosen_set = set(chosen)
        for forced_id in getattr(self.cfg, "viz_force_include", []) or []:
            for idx, subj_id in enumerate(self.val_subjects):
                if subj_id == forced_id:
                    chosen_set.add(idx)
                    break
        return chosen_set, region_to_indices

    def _save_val_pred(self, pred_unpad, batch, subj_id, epoch, *, already_hu: bool = False):
        """Save a validation prediction as NIfTI under `<run_dir>/predictions/last/`,
        plus a copy under `predictions/epoch_<N>/` every cfg.val_save_interval epochs.

        already_hu=False expects pred in [0,1] -> rescaled to HU via *2048-1024 (amix/unet).
        already_hu=True expects pred already in HU (MAISI).

        Returns the 'last' save path, or None if cfg.save_val_volumes is False.
        """
        if not getattr(self.cfg, "save_val_volumes", True):
            return None

        base_dir = self.local_run_dir if (self.cfg.wandb and self.local_run_dir) \
            else os.path.join(self.cfg.prediction_dir, self.run_name)

        pred_np = pred_unpad.float().cpu().numpy().squeeze()
        if not already_hu:
            pred_np = (pred_np * 2048.0) - 1024.0

        # Pulled from the cached pipeline as a plain tensor under "ct_affine"
        # (PersistentDataset's weights_only=True save strips MetaTensor.affine).
        affine = batch["ct_affine"][0] if "ct_affine" in batch else batch["ct"].affine[0]
        affine = affine.cpu().numpy() if hasattr(affine, "cpu") else np.array(affine)

        nii = nib.Nifti1Image(pred_np, affine)

        last_dir = os.path.join(base_dir, "predictions", "last")
        os.makedirs(last_dir, exist_ok=True)
        save_path = os.path.join(last_dir, f"pred_{subj_id}.nii.gz")
        nib.save(nii, save_path)

        val_save_interval = getattr(self.cfg, "val_save_interval", 0)
        if val_save_interval > 0 and epoch % val_save_interval == 0:
            epoch_dir = os.path.join(base_dir, "predictions", f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            nib.save(nii, os.path.join(epoch_dir, f"pred_{subj_id}.nii.gz"))

        return save_path

    def _setup_loss(self):
        from common.loss import AnatomixPerceptualLoss, CompositeLoss

        perceptual_w = getattr(self.cfg, "perceptual_w", 0.0)
        perceptual = None
        if perceptual_w > 0:
            layers = getattr(self.cfg, "perceptual_layers", None)
            if isinstance(layers, str):
                layers = [int(x) for x in layers.split(",") if x.strip()]
            metric = getattr(self.cfg, "perceptual_metric", "ncc")
            separable = getattr(self.cfg, "perceptual_separable", True)
            # Compile the LNCC kernel under the same condition we compile the extractor
            # ("model" mode); in "full" mode the whole step is already compiled.
            compile_lncc = getattr(self.cfg, "compile_mode", None) == "model"
            perceptual = AnatomixPerceptualLoss(
                layers=layers, device=self.device, metric=metric,
                separable=separable, compile_lncc=compile_lncc,
            )
            # The perceptual extractor lives inside CompositeLoss, so the model-level compile in
            # _setup_models never reaches it. Compile it here in "model" mode (benchmark: ~6% faster,
            # ~3GB less; in "full" mode it's already inside the compiled step). See _reports/compile_mode_benchmark.html
            if getattr(self.cfg, "compile_mode", None) == "model":
                print(f"[{self.prefix}] 🚀 Compiling Perceptual Extractor (mode=default)")
                perceptual.extractor = torch.compile(perceptual.extractor, mode="default")

        self.loss_fn = CompositeLoss(
            weights={
                "l1": self.cfg.l1_w,
                "l2": self.cfg.l2_w,
                "ssim": self.cfg.ssim_w,
                "perceptual": perceptual_w,
                "dice_w": getattr(self.cfg, "dice_w", 0.0),
                "dice_bone_w": getattr(self.cfg, "dice_bone_w", 0.0),
                "dice_bone_idx": getattr(self.cfg, "dice_bone_idx", 5),
            },
            perceptual=perceptual,
        ).to(self.device)

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

                # Pred → GT alternation matches the CT rows above (Pred CT → GT CT).
                plot_row(3, img_pred_seg, "Pred Seg", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest")
                plot_row(4, img_seg, "GT Seg", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest")

                # Overlay Row: Show Pred CT first, then overlay Pred Seg
                plot_row(5, img_pred, "Overlay", vmin=0, vmax=1, cmap="gray")
                plot_row(5, img_pred_seg, "Overlay", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest", alpha=0.3)
            else:
                plot_row(3, img_seg, "GT Seg", vmin=seg_vmin, vmax=seg_vmax, cmap="tab20", interpolation="nearest")
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
