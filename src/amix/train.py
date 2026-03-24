import copy
import os
import sys
import traceback
import warnings

import torch

# Enables TF32 and cuDNN benchmark for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True
torch._dynamo.config.cache_size_limit = 64

import matplotlib

matplotlib.use("Agg")

# Add 'src' directory to sys.path so we can import 'mri2ct' as a package
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from amix.trainer import Trainer
from common.config import DEFAULT_CONFIG
from common.utils import cleanup_gpu, send_notification

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

EXPERIMENT_CONFIG = [
    {
        "val_interval": 2,
        "steps_per_epoch": 1000,
        "model_save_interval": 1,
        "total_epochs": 1000,
        "batch_size": 4,
        "accum_steps": 1,
        "sanity_check": False,
        "use_weighted_sampler": True,
        "resume_wandb_id": None,
        "resume_epoch": None,  # Optional: specify epoch number
        "diverge_wandb_branch": False,  # Create new run instead of resuming existing
        "dice_w": 0.1,
        "validate_dice": True,
        # "enable_profiling": True,
        "enable_profiling": False,
        "patches_per_volume": 15,
        "data_queue_max_length": 150,
        "data_queue_num_workers": 4,
        # --------------------
        # "subjects": ["1ABA011"],
        "viz_limit": 10,
        "compile_mode": "full",
        # "compile_mode": "None",
        "lr": 3e-4,
        "scheduler_min_lr": 0.0,
        "wandb_note": "long_run_anatomix_v2_baby_unet_teacher",
        "patch_size": 128,
        "dice_bone_only": False,
        "val_sw_batch_size": 8,
        "val_sw_overlap": 0.25,
        "feat_instance_norm": False,
        "input_dropout_p": 0.0,
    },
]

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dice_w", type=float, help="Dice loss weight")
    parser.add_argument("--resume_id", type=str, help="WandB run ID to resume")
    parser.add_argument("--split_file", type=str, help="Path to split mapping file (e.g., splits/original_splits.txt)")
    parser.add_argument("--augment", type=str, choices=["True", "False"], help="Enable/disable data augmentation (True/False)")
    parser.add_argument("--pass_mri", type=str, choices=["True", "False"], help="Pass original MRI to the translator (True/False)")
    args = parser.parse_args()

    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")

    for i, exp in enumerate(EXPERIMENT_CONFIG):
        # Override with CLI args if provided
        if args.dice_w is not None:
            exp["dice_w"] = args.dice_w
        if args.resume_id is not None:
            exp["resume_wandb_id"] = args.resume_id
        if args.augment is not None:
            exp["augment"] = args.augment == "True"
        if args.pass_mri is not None:
            exp["pass_mri_to_translator"] = args.pass_mri == "True"
        if args.split_file is not None:
            exp["split_file"] = args.split_file

        print(f"\n{'=' * 40}")
        print(f"STARTING EXPERIMENT {i + 1}/{len(EXPERIMENT_CONFIG)}")
        print(f"Config: {exp}")
        print(f"{'=' * 40}\n")

        try:
            # Merge Configs
            conf = copy.deepcopy(DEFAULT_CONFIG)
            conf.update(exp)

            # Execute
            trainer = Trainer(conf)
            trainer.train()

            # Clean up
            del trainer
            cleanup_gpu()

        except KeyboardInterrupt:
            print("\n⛔ Interrupted by user.")
            cleanup_gpu()
            break
        except Exception as e:
            print(f"❌ Experiment {i + 1} Failed: {e}")
            send_notification(f"❌ Experiment {i + 1} Failed: {e}")
            traceback.print_exc()
