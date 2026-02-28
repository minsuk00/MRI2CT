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

from mri2ct.config import DEFAULT_CONFIG
from mri2ct.trainer import Trainer
from mri2ct.utils import cleanup_gpu

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

EXPERIMENT_CONFIG = [
    {
        "val_interval": 1,
        "steps_per_epoch": 1000,
        "model_save_interval": 1,
        "total_epochs": 1000,
        "batch_size": 4,
        "accum_steps": 1,
        "sanity_check": False,
        "use_weighted_sampler": True,
        "use_registered_data": False,
        # "resume_wandb_id": "a8beg6v1",  # no dice
        "resume_wandb_id": "b5we7kh6",  # yes dice
        # "resume_wandb_id": None,
        "resume_epoch": None,  # Optional: specify epoch number
        "diverge_wandb_branch": False,  # Create new run instead of resuming existing
        "dice_w": 0.05,
        # "dice_w": 0.00,
        # "validate_dice": True,
        "validate_dice": True,
        "enable_profiling": True,
        # "enable_profiling": False,
        "patches_per_volume": 200,
        "data_queue_max_length": 400,
        # --------------------
        "viz_limit": 10,
        "compile_mode": "full",
        # "compile_mode": "None",
        "data_queue_num_workers": 4,
        "lr": 3e-4,
        "scheduler_min_lr": 0.0,
        "wandb_note": "long_run_anatomix_v2_baby_unet_teacher",
        "patch_size": 128,
        "dice_bone_only": False,
        "val_sw_batch_size": 8,
        "val_sw_overlap": 0.25,
    },
]

if __name__ == "__main__":
    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")

    for i, exp in enumerate(EXPERIMENT_CONFIG):
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
            traceback.print_exc()
