import argparse
import os
import sys

# os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from common.utils import cleanup_gpu
from maisi_baseline.trainer import MAISITrainer

# ==========================================
# PATHS & PRE-TRAINED WEIGHTS
# ==========================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
AUTOENCODER_PATH = os.path.join(PROJECT_ROOT, "maisi-mr-to-ct", "models", "autoencoder_v1.pt")
DIFFUSION_PATH = os.path.join(PROJECT_ROOT, "maisi-mr-to-ct", "models", "diff_unet_3d_rflow-ct.pt")
NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "maisi-mr-to-ct", "configs", "config_network.json")

# ==========================================
# CONFIGURATION
# ==========================================
MAISI_CONFIG = {
    "split_file": "splits/original_splits.txt",
    "stage_data": True,
    "batch_size": 1,
    "accum_steps": 1,  # Match effective batch size of other models (4)
    "lr": 5e-4,
    "total_epochs": 1000,
    "steps_per_epoch": 1000,
    "val_interval": 2,
    "model_save_interval": 200,
    "use_weighted_sampler": True,
    "resume_wandb_id": None,
    "resume_epoch": None,
    "diverge_wandb_branch": False,
    "validate_dice": False,
    "dataloader_num_workers": 4,
    # ----------------------
    # Experiment Basics
    "project_name": "mri2ct",
    "run_name_prefix": "maisi",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    "wandb_tags": ["maisi"],
    "wandb_note": "MAISI Baseline ControlNet (On-the-fly VAE).",
    "preencoded_latents_dir": None,  # If set, pre-encode all CT volumes once and cache latents here
    # Data
    "patch_size": 128,  # Must be divisible by 4 (VAE requirement)
    "res_mult": 16,  # Standard alignment
    "augment": True,  # Enable standard augmentations
    # Validation
    "val_sw_batch_size": 1,
    "val_sw_overlap": 0.4,
    "num_inference_steps": 10,
    # MAISI Paths
    "autoencoder_path": AUTOENCODER_PATH,
    "diffusion_path": DIFFUSION_PATH,
    "network_config_path": NETWORK_CONFIG_PATH,
}


def main():
    parser = argparse.ArgumentParser(description="Train MAISI ControlNet Baseline")
    parser.add_argument("--root_dir", type=str, help="Root directory containing subject folders")
    parser.add_argument("--split_file", type=str, help="Path to split mapping file")
    parser.add_argument("--subjects", nargs="*", help="Specific subjects for single image optimization (e.g., 1ABA005)")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--accum_steps", type=int, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--epochs", type=int, help="Total epochs")
    parser.add_argument("--steps_per_epoch", type=int, help="Steps per epoch")
    parser.add_argument("--val_interval", type=int, help="Validation interval")
    parser.add_argument("--model_save_interval", type=int, help="Model save interval")
    parser.add_argument("--wandb", type=str, choices=["True", "False"], help="Enable/disable wandb")
    parser.add_argument("--resume_wandb_id", type=str, help="WandB Run ID to resume from")
    parser.add_argument("--preencoded_latents_dir", type=str, help="Directory to cache pre-encoded CT latents")
    parser.add_argument("--tags", type=str, help="Comma-separated extra WandB tags (e.g. 'thorax,high bone dice')")
    parser.add_argument("--use_cutout", type=str, choices=["True", "False"], help="Enable/disable cutout augmentation (True/False)")

    args = parser.parse_args()

    if args.subjects:
        print(f"🔬 RUNNING SINGLE SUBJECT TEST: {args.subjects}")
        # Default SSO overrides
        MAISI_CONFIG.update(
            {
                "subjects": args.subjects,
                "total_epochs": 1000,
                "lr": 5e-4,
                "batch_size": 1,
                "accum_steps": 1,
                "wandb_note": f"MAISI Overfitting Test - {args.subjects}",
                "val_interval": 50,
                "model_save_interval": 200,
                "augment": False,
            }
        )

    # CLI Overrides (Must come last to have priority)
    if args.root_dir:
        MAISI_CONFIG["root_dir"] = args.root_dir
    if args.split_file:
        MAISI_CONFIG["split_file"] = args.split_file
    if args.batch_size:
        MAISI_CONFIG["batch_size"] = args.batch_size
    if args.accum_steps:
        MAISI_CONFIG["accum_steps"] = args.accum_steps
    if args.lr:
        MAISI_CONFIG["lr"] = args.lr
    if args.epochs:
        MAISI_CONFIG["total_epochs"] = args.epochs
    if args.steps_per_epoch:
        MAISI_CONFIG["steps_per_epoch"] = args.steps_per_epoch
    if args.val_interval:
        MAISI_CONFIG["val_interval"] = args.val_interval
    if args.model_save_interval:
        MAISI_CONFIG["model_save_interval"] = args.model_save_interval
    if args.wandb:
        MAISI_CONFIG["wandb"] = args.wandb == "True"
    if args.resume_wandb_id:
        MAISI_CONFIG["resume_wandb_id"] = args.resume_wandb_id
    if args.preencoded_latents_dir:
        MAISI_CONFIG["preencoded_latents_dir"] = args.preencoded_latents_dir
    if args.use_cutout is not None:
        MAISI_CONFIG["use_cutout"] = args.use_cutout == "True"
    if args.tags:
        MAISI_CONFIG.setdefault("wandb_tags", [])
        MAISI_CONFIG["wandb_tags"] = MAISI_CONFIG["wandb_tags"] + [t.strip(' "') for t in args.tags.split(",") if t.strip()]

    try:
        trainer = MAISITrainer(MAISI_CONFIG)
        trainer.train()
        cleanup_gpu()
    except KeyboardInterrupt:
        print("Interrupted.")
        cleanup_gpu()
    except Exception as _:
        import traceback

        traceback.print_exc()
        cleanup_gpu()


if __name__ == "__main__":
    main()
