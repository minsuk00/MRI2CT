"""
End-to-end training/validation test for all 4 trainers.

Tests:
  - Train + val loop (all 4 trainers)
  - val/ and val_body/ both logged to wandb (captured by patching wandb.log AFTER wandb.init)
  - Checkpoint save, resume, LR resume correctness
  - Finetune feat extractor (AMIX only)

Usage:
    cd /home/minsukc/MRI2CT
    micromamba run -n mrct python tests/test_all_trainers.py [--trainers amix unet mcddpm maisi]
"""

import copy
import os
import sys
import traceback
import glob
from unittest.mock import patch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")
MC_DDPM_DIR = os.path.join(REPO_ROOT, "MC-DDPM")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, MC_DDPM_DIR)

import torch
import wandb

from common.config import DEFAULT_CONFIG
from common.utils import cleanup_gpu

# ─────────────────────────────────────────────
# Shared minimal config overlay
# ─────────────────────────────────────────────
FAST_OVERRIDES = {
    "split_file": "splits/single_subject_split.txt",   # 1ABA005 train+val
    "stage_data": False,
    "wandb": True,
    "project_name": "mri2ct-tests",
    "steps_per_epoch": 5,
    "total_epochs": 2,
    "val_interval": 1,
    "sanity_check": False,
    "model_save_interval": 1,
    "augment": False,
    "compile_mode": None,
    "analyze_shapes": False,
    "viz_limit": 1,
    "save_val_volumes": False,
    "patches_per_volume": 3,
    "data_queue_max_length": 10,
    "data_queue_num_workers": 0,
    "val_sw_batch_size": 2,
    "val_body_mask": True,
}

results = {}


def check(name, condition, detail=""):
    status = "✅ PASS" if condition else "❌ FAIL"
    results[name] = condition
    msg = f"  {status}  {name}"
    if detail:
        msg += f": {detail}"
    print(msg)
    return condition


def run_trainer(trainer_cls, cfg_dict, label, extra_checks_fn=None):
    """
    Initialize trainer, run train(), verify wandb keys, return (trainer, run_id, saved_lr, ok).

    wandb.log is patched AFTER trainer.__init__() so it captures calls made during train().
    (wandb.init() is called inside __init__, replacing wandb.log; patching before init
    would be overwritten. wandb.run.summary is also empty until wandb.finish() in v0.25+.)
    """
    trainer = None
    run_id = None
    saved_lr = None
    ok = False
    log_calls = []

    try:
        # Init trainer (this calls wandb.init() internally)
        trainer = trainer_cls(cfg_dict)
        run_id = wandb.run.id if wandb.run else None

        # Patch wandb.log NOW (after init) to capture all training/val log calls
        original_log = wandb.log

        def capturing_log(data, *args, **kwargs):
            log_calls.append(copy.deepcopy(data))
            return original_log(data, *args, **kwargs)

        with patch.object(wandb, "log", side_effect=capturing_log):
            trainer.train()

        # Flatten all logged keys across all calls
        all_logged_keys = set()
        for d in log_calls:
            all_logged_keys.update(d.keys())

        # ── Wandb logging checks ────────────────────────────
        val_keys = [k for k in all_logged_keys if k.startswith("val/")]
        body_keys = [k for k in all_logged_keys if k.startswith("val_body/")]

        check(f"{label} val/ logged to wandb", len(val_keys) > 0, str(sorted(val_keys)[:5]))
        check(f"{label} val_body/ logged to wandb", len(body_keys) >= 3, str(sorted(body_keys)))
        check(f"{label} val_body/mae_hu present", "val_body/mae_hu" in all_logged_keys)
        check(f"{label} val_body/ssim present", "val_body/ssim" in all_logged_keys)
        check(f"{label} val_body/psnr present", "val_body/psnr" in all_logged_keys)

        # ── Checkpoint saved ─────────────────────────────────
        if wandb.run:
            ckpts = (glob.glob(os.path.join(wandb.run.dir, "*.pt")) +
                     glob.glob(os.path.join(wandb.run.dir, "*.pth")))
            check(f"{label} checkpoint saved", len(ckpts) > 0, f"found {len(ckpts)}")

        if extra_checks_fn:
            extra_checks_fn(trainer, all_logged_keys)

        saved_lr = trainer.optimizer.param_groups[0]["lr"]
        ok = True

    except Exception as e:
        check(f"{label} training completed without error", False, str(e))
        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
        cleanup_gpu()

    return trainer, run_id, saved_lr, ok


def run_resume(trainer_cls, cfg_dict, label, run_id, saved_lr, expected_start_epoch=3):
    """Initialize a resumed trainer and verify epoch + LR are restored correctly."""
    print(f"\n  [{label} Resume Test]")
    if not run_id:
        print("  ⚠️  no run_id — skipping resume test")
        return

    cfg_r = copy.deepcopy(cfg_dict)
    cfg_r["resume_wandb_id"] = run_id
    cfg_r["diverge_wandb_branch"] = False

    try:
        trainer2 = trainer_cls(cfg_r)
        check(f"{label} resume: start_epoch == {expected_start_epoch}",
              trainer2.start_epoch == expected_start_epoch,
              f"got {trainer2.start_epoch}")
        resumed_lr = trainer2.optimizer.param_groups[0]["lr"]
        check(f"{label} resume: LR restored correctly",
              abs(saved_lr - resumed_lr) < 1e-9,
              f"saved={saved_lr:.3e} resumed={resumed_lr:.3e}")
    except Exception as e:
        check(f"{label} resume: initialization succeeded", False, str(e))
        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
        cleanup_gpu()


# ─────────────────────────────────────────────
# TEST 1: AMIX
# ─────────────────────────────────────────────
def test_amix():
    print("\n" + "=" * 60)
    print("TEST: AMIX Trainer")
    print("=" * 60)

    from amix.trainer import Trainer

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(FAST_OVERRIDES)
    cfg.update({
        "run_name_prefix": "test_amix",
        "wandb_note": "test_amix",
        "patch_size": 64,
        "batch_size": 2,
        "accum_steps": 1,
        "validate_dice": True,
        "dice_w": 0.1,
        "dice_bone_w": 0.05,
        "finetune_feat_extractor": False,
        "log_dir": os.path.join(REPO_ROOT, "tests/test_logs"),
        "prediction_dir": os.path.join(REPO_ROOT, "tests/test_preds"),
        "diverge_wandb_branch": False,
        "resume_wandb_id": None,
    })

    def amix_extra(trainer, logged_keys):
        check("AMIX val_body/dice_score_all present", "val_body/dice_score_all" in logged_keys)
        check("AMIX val_body/dice_score_bone present", "val_body/dice_score_bone" in logged_keys)

    trainer, run_id, saved_lr, ok = run_trainer(Trainer, cfg, "AMIX", amix_extra)

    if ok:
        run_resume(Trainer, cfg, "AMIX", run_id, saved_lr)

    # ── Finetune feat extractor ──────────────────────────────
    print("\n  [AMIX Finetune Feat Extractor Test]")
    cfg_ft = copy.deepcopy(cfg)
    cfg_ft.update({
        "finetune_feat_extractor": True,
        "finetune_depth": -1,
        "resume_wandb_id": None,
        "total_epochs": 1,
    })
    try:
        trainer_ft = Trainer(cfg_ft)
        trainable = sum(p.numel() for p in trainer_ft.feat_extractor.parameters() if p.requires_grad)
        check("AMIX feat extractor is trainable", trainable > 0, f"{trainable:,} params")
        n_groups = len(trainer_ft.optimizer.param_groups)
        check("AMIX optimizer has 2 param groups (translator + feat)", n_groups == 2,
              f"got {n_groups}")
        trainer_ft.train()
        check("AMIX finetune train completed without error", True)
    except Exception as e:
        check("AMIX finetune train completed without error", False, str(e))
        traceback.print_exc()
    finally:
        if wandb.run:
            wandb.finish()
        cleanup_gpu()


# ─────────────────────────────────────────────
# TEST 2: UNet
# ─────────────────────────────────────────────
def test_unet():
    print("\n" + "=" * 60)
    print("TEST: UNet Baseline Trainer")
    print("=" * 60)

    from unet_baseline.train import BaselineTrainer

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(FAST_OVERRIDES)
    cfg.update({
        "run_name_prefix": "test_unet",
        "wandb_note": "test_unet",
        "patch_size": 64,
        "batch_size": 2,
        "accum_steps": 1,
        "validate_dice": True,
        "dice_w": 0.1,
        "dice_bone_w": 0.05,
        "ngf": 16,
        "num_downs": 3,
        "input_nc": 1,
        "output_nc": 1,
        "model_type": "unet_baseline",
        "log_dir": os.path.join(REPO_ROOT, "tests/test_logs"),
        "prediction_dir": os.path.join(REPO_ROOT, "tests/test_preds"),
        "diverge_wandb_branch": False,
        "resume_wandb_id": None,
        "weight_decay": 1e-4,
    })

    def unet_extra(trainer, logged_keys):
        check("UNet val_body/dice_score_all present", "val_body/dice_score_all" in logged_keys)
        check("UNet val_body/dice_score_bone present", "val_body/dice_score_bone" in logged_keys)

    trainer, run_id, saved_lr, ok = run_trainer(BaselineTrainer, cfg, "UNet", unet_extra)

    if ok:
        run_resume(BaselineTrainer, cfg, "UNet", run_id, saved_lr)


# ─────────────────────────────────────────────
# TEST 3: MC-DDPM
# ─────────────────────────────────────────────
def test_mcddpm():
    print("\n" + "=" * 60)
    print("TEST: MC-DDPM Trainer")
    print("=" * 60)

    from mc_ddpm_baseline.trainer import MCDDPMTrainer

    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg.update(FAST_OVERRIDES)
    cfg.update({
        "run_name_prefix": "test_mcddpm",
        "wandb_note": "test_mcddpm",
        "patch_size": (64, 64, 4),
        "batch_size": 2,
        "accum_steps": 1,
        "diffusion_steps": 100,
        "learn_sigma": True,
        "timestep_respacing": [2],
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": True,
        "rescale_timesteps": True,
        "rescale_learned_sigmas": True,
        "num_channels": 32,
        "attention_resolutions": (32, 16, 8),
        "channel_mult": (1, 2, 3, 4),
        "num_heads": [4, 4, 8, 16],
        "window_size": [[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]],
        "num_res_blocks": [1, 1, 1, 1],
        "sample_kernel": (([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),),
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": False,
        "weight_decay": 1e-4,
        "val_sw_overlap": 0.0,
        "val_sw_batch_size": 4,
        "log_dir": os.path.join(REPO_ROOT, "tests/test_logs"),
        "prediction_dir": os.path.join(REPO_ROOT, "tests/test_preds"),
        "diverge_wandb_branch": False,
        "resume_wandb_id": None,
        "scheduler_type": None,
    })

    def mcddpm_extra(trainer, logged_keys):
        # MC-DDPM has no teacher model so no dice in val_body/
        body_keys = [k for k in logged_keys if k.startswith("val_body/")]
        no_dice = not any("dice" in k for k in body_keys)
        check("MC-DDPM val_body/ has no dice (no teacher)", no_dice,
              f"body_keys={sorted(body_keys)}")

    trainer, run_id, saved_lr, ok = run_trainer(MCDDPMTrainer, cfg, "MC-DDPM", mcddpm_extra)

    if ok:
        run_resume(MCDDPMTrainer, cfg, "MC-DDPM", run_id, saved_lr)


# ─────────────────────────────────────────────
# TEST 4: MAISI
# ─────────────────────────────────────────────
def test_maisi():
    print("\n" + "=" * 60)
    print("TEST: MAISI Trainer")
    print("=" * 60)

    AUTOENCODER_PATH = os.path.join(REPO_ROOT, "maisi-mr-to-ct", "models", "autoencoder_v1.pt")
    DIFFUSION_PATH = os.path.join(REPO_ROOT, "maisi-mr-to-ct", "models", "diff_unet_3d_rflow-ct.pt")
    NETWORK_CONFIG_PATH = os.path.join(REPO_ROOT, "maisi-mr-to-ct", "configs", "config_network.json")

    if not all(os.path.exists(p) for p in [AUTOENCODER_PATH, DIFFUSION_PATH, NETWORK_CONFIG_PATH]):
        print("  ⚠️  MAISI weights not found — skipping")
        results["MAISI (skipped — weights missing)"] = None
        return

    from maisi_baseline.trainer import MAISITrainer

    cfg = {
        "split_file": "splits/single_subject_split.txt",
        "stage_data": False,
        "wandb": True,
        "project_name": "mri2ct-tests",
        "run_name_prefix": "test_maisi",
        "wandb_note": "test_maisi",
        "steps_per_epoch": 5,
        "total_epochs": 2,
        "val_interval": 1,
        "sanity_check": False,
        "model_save_interval": 1,
        "augment": False,
        "compile_mode": None,
        "analyze_shapes": False,
        "viz_limit": 1,
        "save_val_volumes": False,
        "val_body_mask": True,
        "batch_size": 1,
        "accum_steps": 1,
        "lr": 5e-4,
        "weight_decay": 1e-4,
        "seed": 42,
        "device": "cuda",
        "patch_size": 128,
        "res_mult": 16,
        "val_sw_batch_size": 2,
        "val_sw_overlap": 0.25,
        "num_inference_steps": 5,
        "validate_dice": False,
        "log_dir": os.path.join(REPO_ROOT, "tests/test_logs"),
        "prediction_dir": os.path.join(REPO_ROOT, "tests/test_preds"),
        "diverge_wandb_branch": False,
        "resume_wandb_id": None,
        "autoencoder_path": AUTOENCODER_PATH,
        "diffusion_path": DIFFUSION_PATH,
        "network_config_path": NETWORK_CONFIG_PATH,
        "scheduler_type": "cosine",
        "scheduler_min_lr": 0.0,
        "preencoded_latents_dir": None,
        "dataloader_num_workers": 0,
        "use_weighted_sampler": True,
        "model_type": "maisi",
    }

    trainer, run_id, saved_lr, ok = run_trainer(MAISITrainer, cfg, "MAISI")

    if ok:
        run_resume(MAISITrainer, cfg, "MAISI", run_id, saved_lr)


# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)
    for name, val in results.items():
        if val is True:
            print(f"  ✅  {name}")
        elif val is False:
            print(f"  ❌  {name}")
        else:
            print(f"  ⚠️   {name}")
    print(f"\nPassed: {passed} | Failed: {failed} | Skipped: {skipped}")
    return failed == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainers", nargs="+",
                        choices=["amix", "unet", "mcddpm", "maisi", "all"],
                        default=["all"])
    args = parser.parse_args()

    run_all = "all" in args.trainers

    os.makedirs(os.path.join(REPO_ROOT, "tests/test_logs"), exist_ok=True)
    os.makedirs(os.path.join(REPO_ROOT, "tests/test_preds"), exist_ok=True)

    if run_all or "amix" in args.trainers:
        test_amix()
    if run_all or "unet" in args.trainers:
        test_unet()
    if run_all or "mcddpm" in args.trainers:
        test_mcddpm()
    if run_all or "maisi" in args.trainers:
        test_maisi()

    success = print_summary()
    sys.exit(0 if success else 1)
