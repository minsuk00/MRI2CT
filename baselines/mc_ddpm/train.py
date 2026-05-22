"""MC-IDDPM training entry. Argparse + EXPERIMENT_CONFIG -> Trainer(conf).train()."""
import argparse
import copy
import os
import sys
import traceback
import warnings

import torch

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

import matplotlib
matplotlib.use("Agg")

# Make `common.*` import-able (mirrors src/amix/train.py).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from baselines.mc_ddpm.trainer import Trainer
from common.config import DEFAULT_CONFIG
from common.utils import cleanup_gpu

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")

# Paper-faithful base config for MC-IDDPM. Merged onto DEFAULT_CONFIG below; CLI
# args (if provided) take final precedence.
MCDDPM_CONFIG = {
    "run_name_prefix": "mcddpm",
    # Data
    "patches_per_volume": 2,        # paper notebook patch_num
    "batch_size": 4,                # paper notebook batch_size; smoke #1 may force drop
    "num_workers": 4,
    # Optim (paper text, page 6: "AdamW lr=3e-5, wd=1e-5, 500 epochs")
    "lr": 3e-5,
    "weight_decay": 1e-5,
    "lr_anneal_steps": 0,           # paper uses constant LR (no scheduler)
    "use_amp": True,                # bf16 autocast (paper notebook uses fp16 AMP)
    "use_checkpoint": False,        # SwinVITModel gradient checkpointing
    # Diffusion
    "diffusion_steps": 1000,
    "val_steps": 25,                # cheap viz sampler (full eval uses 50 via validate.py)
    # Loop
    "total_epochs": 500,
    "steps_per_epoch": 500,
    "val_interval": 5,              # epochs
    "model_save_interval": 5,       # epochs — milestone cadence
    "log_every": 50,                # steps — per-step wandb log cadence
    # Val viz
    "val_subj_id": "1THB011",
    # WandB
    "wandb_tags": ["mc-iddpm"],
    "wandb_note": "mc-iddpm initial run",
}

EXPERIMENT_CONFIG = [MCDDPM_CONFIG]


def _bool(s):
    return s == "True"


def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--wandb", type=str, choices=["True", "False"])
    p.add_argument("--resume_id", type=str, help="WandB run ID to resume")
    p.add_argument("--run_name", type=str)
    p.add_argument("--split_file", type=str)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--patches_per_volume", type=int)
    p.add_argument("--num_workers", type=int)
    p.add_argument("--lr", type=float)
    p.add_argument("--weight_decay", type=float)
    p.add_argument("--lr_anneal_steps", type=int)
    p.add_argument("--use_amp", type=str, choices=["True", "False"])
    p.add_argument("--use_checkpoint", type=str, choices=["True", "False"])
    p.add_argument("--diffusion_steps", type=int)
    p.add_argument("--val_steps", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--steps_per_epoch", type=int)
    p.add_argument("--val_interval", type=int)
    p.add_argument("--save_interval", type=int)
    p.add_argument("--val_subj_id", type=str)
    p.add_argument("--tags", type=str, help="Comma-separated extra wandb tags")
    p.add_argument("--diverge_branch", type=str, choices=["True", "False"])
    return p


def apply_args(exp, args):
    if args.wandb is not None:               exp["wandb"] = _bool(args.wandb)
    if args.resume_id is not None:           exp["resume_wandb_id"] = args.resume_id
    if args.run_name is not None:            exp["run_name_prefix"] = args.run_name
    if args.split_file is not None:          exp["split_file"] = args.split_file
    if args.batch_size is not None:          exp["batch_size"] = args.batch_size
    if args.patches_per_volume is not None:  exp["patches_per_volume"] = args.patches_per_volume
    if args.num_workers is not None:         exp["num_workers"] = args.num_workers
    if args.lr is not None:                  exp["lr"] = args.lr
    if args.weight_decay is not None:        exp["weight_decay"] = args.weight_decay
    if args.lr_anneal_steps is not None:     exp["lr_anneal_steps"] = args.lr_anneal_steps
    if args.use_amp is not None:             exp["use_amp"] = _bool(args.use_amp)
    if args.use_checkpoint is not None:      exp["use_checkpoint"] = _bool(args.use_checkpoint)
    if args.diffusion_steps is not None:     exp["diffusion_steps"] = args.diffusion_steps
    if args.val_steps is not None:           exp["val_steps"] = args.val_steps
    if args.epochs is not None:              exp["total_epochs"] = args.epochs
    if args.steps_per_epoch is not None:     exp["steps_per_epoch"] = args.steps_per_epoch
    if args.val_interval is not None:        exp["val_interval"] = args.val_interval
    if args.save_interval is not None:       exp["model_save_interval"] = args.save_interval
    if args.val_subj_id is not None:         exp["val_subj_id"] = args.val_subj_id
    if args.diverge_branch is not None:      exp["diverge_wandb_branch"] = _bool(args.diverge_branch)
    if args.tags is not None:
        exp.setdefault("wandb_tags", [])
        exp["wandb_tags"] = list(exp["wandb_tags"]) + [
            t.strip(' "') for t in args.tags.split(",") if t.strip()
        ]


if __name__ == "__main__":
    args = build_arg_parser().parse_args()

    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")
    for i, exp in enumerate(EXPERIMENT_CONFIG):
        apply_args(exp, args)
        print(f"\n{'=' * 40}\nSTARTING EXPERIMENT {i + 1}/{len(EXPERIMENT_CONFIG)}\nConfig overrides: {exp}\n{'=' * 40}\n")

        try:
            conf = copy.deepcopy(DEFAULT_CONFIG)
            conf.update(exp)
            trainer = Trainer(conf)
            trainer.train()
            del trainer
            cleanup_gpu()
        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
            cleanup_gpu()
            break
        except Exception as e:
            print(f"❌ Experiment {i + 1} Failed: {e}")
            traceback.print_exc()
