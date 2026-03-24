from types import SimpleNamespace

import torch


class Config(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)


DEFAULT_CONFIG = {
    # System
    "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat",
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "prediction_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/predictions/1.5x1.5x1.5mm_registered",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    "project_name": "mri2ct",
    # Data
    "split_file": "splits/original_splits.txt",
    "stage_data": True,
    "augment": True,
    "patch_size": 128,
    "patches_per_volume": 10,
    "data_queue_max_length": 100,
    "data_queue_num_workers": 4,
    "anatomix_weights": "v2",  # "v1", "v2"
    "teacher_weights_path": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/wandb/run-20260222_160128-p0i9chz3/files/seg_baby_unet_epoch_499.pth",
    "res_mult": 32,
    "analyze_shapes": True,
    "enable_profiling": False,
    # Training
    "lr": 3e-4,
    "scheduler_type": "cosine",  # "plateau", "cosine", None
    "scheduler_min_lr": 0.0,
    "val_interval": 1,
    "sanity_check": True,
    "accum_steps": 2,
    "model_save_interval": 10,
    "viz_limit": 6,
    "viz_pca": False,
    "steps_per_epoch": 1000,
    "finetune_feat_extractor": False,
    "lr_feat_extractor": 1e-5,
    "override_lr": False,
    # Model Choice
    "model_type": "anatomix_translator",
    "compile_mode": "model",  # Options: None, "model", "full"
    "total_epochs": 5001,
    "dropout": 0,
    # CNN Specifics
    "batch_size": 4,
    "final_activation": "sigmoid",
    "use_weighted_sampler": True,
    "pass_mri_to_translator": False,
    "n_classes": 12,  # 11 Organs + Brain
    # Sliding Window & Viz Options
    "val_sw_batch_size": 4,
    "val_sw_overlap": 0.25,
    "validate_dice": False,
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 0.1,
    "dice_w": 0.0,
    "dice_bone_w": 0.0,
    "dice_bone_idx": 5,
    "dice_exclude_background": True,
    "dice_bone_only": False,
    "wandb_note": "test_run",
    "resume_wandb_id": None,
    "resume_epoch": None,
    "diverge_wandb_branch": False,
    # Validation Saving
    "save_val_volumes": True,
}
