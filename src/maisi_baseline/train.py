import argparse
import os
import sys

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
    "lr": 3e-4,
    "total_epochs": 1000,
    "steps_per_epoch": 1000,
    "val_interval": 2,
    "model_save_interval": 1,
    "use_weighted_sampler": True,
    "resume_wandb_id": None,
    "resume_epoch": None,
    "diverge_wandb_branch": False,
    "validate_dice": False,
    "dataloader_num_workers": 4,
    # ----------------------
    # Experiment Basics
    "project_name": "MRI2CT_MAISI_Baseline",
    "run_name_prefix": "MAISI_ControlNet",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    "wandb_note": "MAISI Baseline ControlNet (On-the-fly VAE).",
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

    args = parser.parse_args()

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

    if args.subjects:
        print(f"🔬 RUNNING SINGLE SUBJECT TEST: {args.subjects}")
        MAISI_CONFIG.update(
            {
                "subjects": args.subjects,
                "total_epochs": 1000,
                "lr": 5e-4,
                "batch_size": 1,
                "accum_steps": 1,
                "wandb_note": f"MAISI Overfitting Test - {args.subjects}",
                "val_interval": 50,
                "model_save_interval": 100,
                "augment": False,  # usually disable augment for overfitting test
            }
        )

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
