import os
from types import SimpleNamespace
import torch

BASE_DIR = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined"

class Config(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)
        
        # Dynamic Path Construction based on dataset_spacing
        if hasattr(self, "dataset_spacing"):
            sp = self.dataset_spacing
            sp_str = f"{sp[0]:.1f}x{sp[1]:.1f}x{sp[2]:.1f}mm"
            
            # Root directory for data
            self.root_dir = os.path.join(BASE_DIR, sp_str)
            
            # Directory for predictions
            self.prediction_dir = os.path.join(BASE_DIR, "predictions", sp_str)
            
            # Ensure log dir exists or is correctly relative
            # (Leaving log_dir as provided in dict for now)

DEFAULT_CONFIG = {
    # System
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    "project_name": "mri2ct",
    
    # Data
    "subjects": None,
    "region": None, # "AB", "TH", "HN"
    "dataset_spacing": [3.0, 3.0, 3.0],
    "val_split": 0.1, # deprecated
    "augment": True,
    "patch_size": 96,
    "patches_per_volume": 40,
    "data_queue_max_length": 400,
    "data_queue_num_workers": 6,
    "anatomix_weights": "v2", # "v1", "v2"
    "teacher_weights_path": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/models/seg_baby_unet_best.pth",
    "res_mult": 32 ,
    "analyze_shapes": True, 
    "enable_profiling": False,
    
    # Training
    "lr": 3e-4,
    "scheduler_type": "cosine", # "plateau", "cosine", None
    "scheduler_t_max": 1000000,
    "scheduler_min_lr": 1e-6,
    "val_interval": 1,
    "sanity_check": True,
    "accum_steps": 2,
    "model_save_interval": 10,
    "viz_limit": 6,
    "viz_pca": False,
    "steps_per_epoch": 25,
    "finetune_feat_extractor": False,
    "lr_feat_extractor": 1e-5,
    "use_scheduler": False,
    "override_lr": False,
    
    # Model Choice
    "model_type": "cnn", # deprecated
    "model_compile_mode": "default", # "default", "reduce-overhead", None
    "compile_mode": "model", # New standard: None, "model", "full"
    "total_epochs": 5001,
    "dropout": 0,
    
    # CNN Specifics
    "batch_size": 4,
    "cnn_depth": 9, # deprecated
    "cnn_hidden": 128, # deprecated
    "final_activation": "sigmoid",
    "enable_safety_padding": True,
    "use_weighted_sampler": True,
    "n_classes": 12, # 11 Organs + Brain

    # Sliding Window & Viz Options
    "val_sliding_window": True, 
    "val_sw_batch_size": 8, 
    "val_sw_overlap": 0.7,
    "validate_dice": False,
    
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 0.1,
    "perceptual_w": 0.0,
    "dice_w": 0.0,
    "dice_exclude_background": True,
    "dice_bone_only": False,

    "wandb_note": "test_run",
    "resume_wandb_id": None, 
    "resume_epoch": None,
    "diverge_wandb_branch": False,
    
    # Validation Saving
    "save_val_volumes": True, 
}
