import os
import sys
import torch
import copy
import warnings

# Enables TF32 and cuDNN benchmark for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 64

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
# os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from mc_ddpm_baseline.trainer import MCDDPMTrainer
from common.config import DEFAULT_CONFIG

EXPERIMENT_CONFIG = [
    {
        # Base
        "patch_size": (64, 64, 4), # (X, Y, Z)
        "batch_size": 4,
        "accum_steps": 1,
        "steps_per_epoch": 100, # 1000
        "total_epochs": 500,
        "val_interval": 1,
        "model_save_interval": 200,
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "sanity_check": False,
        "run_name_prefix": "mcddpm",
        "wandb_note": "mc-ddpm-baseline",
        
        # Diffusion params
        "diffusion_steps": 1000,
        "learn_sigma": True,
        "timestep_respacing": [10],
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": True,
        "rescale_timesteps": True,
        "rescale_learned_sigmas": True,
        
        # SwinViT params
        "num_channels": 64,
        "attention_resolutions": (32, 16, 8),
        "channel_mult": (1, 2, 3, 4),
        "num_heads": [4, 4, 8, 16],
        "window_size": [[4, 4, 4], [4, 4, 4], [4, 4, 2], [4, 4, 2]],
        "num_res_blocks": [2, 2, 2, 2],
        "sample_kernel": (([2, 2, 2], [2, 2, 1], [2, 2, 1], [2, 2, 1]),),
        "dropout": 0.0,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        
        # Data loader configs
        "patches_per_volume": 4,
        "data_queue_max_length": 150,
        "data_queue_num_workers": 4,
        "use_weighted_sampler": False,
        "val_sw_batch_size": 4,
        "val_sw_overlap": 0.5,
        "viz_limit": 4,
    }
]

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, help="Total epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, help="Number of steps per epoch")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--wandb", type=str, choices=["True", "False"], help="Enable/disable wandb")
    parser.add_argument("--run_name", type=str, help="WandB run name prefix")
    parser.add_argument("--split_file", type=str, help="Path to split mapping file (e.g., splits/original_splits.txt)")
    parser.add_argument("--resume_id", type=str, help="WandB run ID to resume")
    parser.add_argument("--val_interval", type=int, help="Validation interval (epochs)")
    parser.add_argument("--model_save_interval", type=int, help="Checkpoint save interval (epochs)")
    parser.add_argument("--full_val", type=str, choices=["True", "False"], help="Validate on full val set or 1 per region")
    parser.add_argument("--use_cutout", type=str, choices=["True", "False"], help="Enable/disable cutout augmentation")
    parser.add_argument("--tags", type=str, help="Comma-separated extra WandB tags")
    args = parser.parse_args()

    for i, exp in enumerate(EXPERIMENT_CONFIG):
        print(f"STARTING EXPERIMENT {i + 1}/{len(EXPERIMENT_CONFIG)}")
        if args.tags is not None:
            exp.setdefault("wandb_tags", [])
            exp["wandb_tags"] = exp["wandb_tags"] + [t.strip(' "') for t in args.tags.split(",") if t.strip()]
        if args.use_cutout is not None:
            exp["use_cutout"] = args.use_cutout == "True"
        if args.epochs is not None:
            exp["total_epochs"] = args.epochs
        if args.steps_per_epoch is not None:
            exp["steps_per_epoch"] = args.steps_per_epoch
        if args.batch_size is not None:
            exp["batch_size"] = args.batch_size
        if args.wandb is not None:
            exp["wandb"] = args.wandb == "True"
        if args.run_name is not None:
            exp["run_name_prefix"] = args.run_name
        if args.split_file is not None:
            exp["split_file"] = args.split_file
        if args.resume_id is not None:
            exp["resume_wandb_id"] = args.resume_id
        if args.val_interval is not None:
            exp["val_interval"] = args.val_interval
        if args.model_save_interval is not None:
            exp["model_save_interval"] = args.model_save_interval
        if args.full_val is not None:
            exp["full_val"] = args.full_val == "True"
        conf = copy.deepcopy(DEFAULT_CONFIG)
        conf.update(exp)
        trainer = MCDDPMTrainer(conf)
        trainer.train()
