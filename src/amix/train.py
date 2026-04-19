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
# os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

EXPERIMENT_CONFIG = [
    {
        "val_interval": 2,
        "steps_per_epoch": 1000,
        "model_save_interval": 200,
        "total_epochs": 1000,
        "batch_size": 4,
        "accum_steps": 1,
        "sanity_check": False,
        "run_name_prefix": "amix",
        "use_weighted_sampler": True,
        "finetune_feat_extractor": False,
        "finetune_depth": 0,  # 0: Frozen, -1: All, >0: Last N
        "lr_feat_extractor": 1e-5,
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
        "viz_limit": 4,
        "compile_mode": "full",
        # "compile_mode": "None",
        "lr": 3e-4,
        "scheduler_min_lr": 0.0,
        "wandb_tags": ["amix"],
        "wandb_note": "long_run_anatomix_v2_baby_unet_teacher",
        "patch_size": 128,
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
    parser.add_argument("--dice_bone_w", type=float, help="Bone-specific dice loss weight")
    parser.add_argument("--wandb", type=str, choices=["True", "False"], help="Enable/disable wandb (True/False)")
    parser.add_argument("--resume_id", type=str, help="WandB run ID to resume")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--run_name", type=str, help="WandB run name prefix")
    parser.add_argument("--split_file", type=str, help="Path to split mapping file (e.g., splits/original_splits.txt)")
    parser.add_argument("--augment", type=str, choices=["True", "False"], help="Enable/disable data augmentation (True/False)")
    parser.add_argument("--pass_mri", type=str, choices=["True", "False"], help="Pass original MRI to the translator (True/False)")
    parser.add_argument("--feat_instance_norm", type=str, choices=["True", "False"], help="Enable instance norm for features (True/False)")
    parser.add_argument("--use_zero_mask", type=str, choices=["True", "False"], help="Enable zero-masking for background (True/False)")
    parser.add_argument("--weighted_sampler", type=str, choices=["True", "False"], help="Enable/disable weighted sampler (True/False)")
    parser.add_argument("--finetune_feat", type=str, choices=["True", "False"], help="Finetune feature extractor (True/False)")
    parser.add_argument("--finetune_depth", type=int, help="Number of final layers/modules to finetune (-1 for all)")
    parser.add_argument("--lr_feat", type=float, help="Learning rate for feature extractor")
    parser.add_argument("--input_dropout_p", type=float, help="Input dropout probability")
    parser.add_argument("--amix_weights", type=str, choices=["v1", "v1_2", "v1_3"], help="Anatomix weights version (v1, v1_2, v1_3)")
    parser.add_argument("--feat_norm", type=str, choices=["instance", "batch"], help="Norm layer for feat extractor (instance, batch)")
    parser.add_argument("--epochs", type=int, help="Total epochs to train")
    parser.add_argument("--steps_per_epoch", type=int, help="Number of steps per epoch")
    parser.add_argument("--num_workers", type=int, help="Number of workers for the data queue")
    parser.add_argument("--tags", type=str, help="Comma-separated extra WandB tags (e.g. 'thorax,high bone dice')")
    parser.add_argument("--use_cutout", type=str, choices=["True", "False"], help="Enable/disable cutout augmentation (True/False)")
    parser.add_argument("--cutout_alpha", type=float, help="Beta(alpha, alpha) parameter controlling cutout box size distribution")
    parser.add_argument("--mask_body_input", type=str, choices=["True", "False"], help="Zero out MRI voxels outside body mask before sampling (True/False)")
    parser.add_argument("--validate_dice", type=str, choices=["True", "False"], help="Enable/disable dice validation (True/False)")
    parser.add_argument("--compile_mode", type=str, help="torch.compile mode: 'full', 'model', or 'none'")
    parser.add_argument("--feat_scale_down", type=float, help="Divide features by this value (e.g. 100)")
    args = parser.parse_args()

    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")

    for i, exp in enumerate(EXPERIMENT_CONFIG):
        # Override with CLI args if provided
        if args.dice_w is not None:
            exp["dice_w"] = args.dice_w
        if args.dice_bone_w is not None:
            exp["dice_bone_w"] = args.dice_bone_w
        if args.wandb is not None:
            exp["wandb"] = args.wandb == "True"
        if args.batch_size is not None:
            exp["batch_size"] = args.batch_size
        if args.run_name is not None:
            exp["run_name_prefix"] = args.run_name
        if args.resume_id is not None:
            exp["resume_wandb_id"] = args.resume_id
        if args.augment is not None:
            exp["augment"] = args.augment == "True"
        if args.pass_mri is not None:
            exp["pass_mri_to_translator"] = args.pass_mri == "True"
        if args.split_file is not None:
            exp["split_file"] = args.split_file
        if args.feat_instance_norm is not None:
            exp["feat_instance_norm"] = args.feat_instance_norm == "True"
        if args.use_zero_mask is not None:
            exp["use_zero_mask"] = args.use_zero_mask == "True"
        if args.weighted_sampler is not None:
            exp["use_weighted_sampler"] = args.weighted_sampler == "True"
        if args.finetune_feat is not None:
            exp["finetune_feat_extractor"] = args.finetune_feat == "True"
        if args.finetune_depth is not None:
            exp["finetune_depth"] = args.finetune_depth
        if args.lr_feat is not None:
            exp["lr_feat_extractor"] = args.lr_feat
        if args.input_dropout_p is not None:
            exp["input_dropout_p"] = args.input_dropout_p
        if args.amix_weights is not None:
            exp["anatomix_weights"] = args.amix_weights
        if args.feat_norm is not None:
            exp["feat_norm"] = args.feat_norm
        if args.epochs is not None:
            exp["total_epochs"] = args.epochs
        if args.steps_per_epoch is not None:
            exp["steps_per_epoch"] = args.steps_per_epoch
        if args.num_workers is not None:
            exp["data_queue_num_workers"] = args.num_workers
        if args.tags is not None:
            exp.setdefault("wandb_tags", [])
            exp["wandb_tags"] = exp["wandb_tags"] + [t.strip(' "') for t in args.tags.split(",") if t.strip()]

        # Cutout Override
        if args.use_cutout is not None:
            exp["use_cutout"] = args.use_cutout == "True"
        if args.cutout_alpha is not None:
            exp["cutout_alpha"] = args.cutout_alpha
        if args.mask_body_input is not None:
            exp["mask_body_input"] = args.mask_body_input == "True"

        # Dice Validation Override
        if args.validate_dice is not None:
            exp["validate_dice"] = args.validate_dice == "True"
        if args.compile_mode is not None:
            exp["compile_mode"] = None if args.compile_mode.lower() == "none" else args.compile_mode
        if args.feat_scale_down is not None:
            exp["feat_scale_down"] = args.feat_scale_down

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
