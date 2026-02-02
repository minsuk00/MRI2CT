from types import SimpleNamespace
import torch

class Config(SimpleNamespace):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        for key, value in dictionary.items():
            setattr(self, key, value)

DEFAULT_CONFIG = {
    # System
    # "root_dir": "/home/minsukc/MRI2CT",
    "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/3.0x3.0x3.0mm", 
    "log_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs",
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "wandb": True,
    "project_name": "mri2ct",
    
    # Data
    "subjects": None,
    "region": None, # "AB", "TH", "HN"
    "val_split": 0.1,
    "augment": True,
    "patch_size": 96,
    "patches_per_volume": 100,
    "data_queue_max_length": 1000,
    "data_queue_num_workers": 6,
    "anatomix_weights": "v2", # "v1", "v2"
    "res_mult": 32 ,
    "analyze_shapes": True, 
    
    # Training
    "lr": 3e-4,
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
    "model_type": "cnn",
    "model_compile_mode": None, # "default", "reduce-overhead", None
    "total_epochs": 5001,
    "dropout": 0,
    
    # CNN Specifics
    "batch_size": 4,
    "cnn_depth": 9,
    "cnn_hidden": 128,
    "final_activation": "sigmoid",
    "enable_safety_padding": True,
    "use_weighted_sampler": True,

    # Sliding Window & Viz Options
    "val_sliding_window": True, 
    "val_sw_batch_size": 4, 
    "val_sw_overlap": 0.25, 
    
    # Loss Weights
    "l1_w": 1.0,
    "l2_w": 0.0,
    "ssim_w": 0.1,
    "perceptual_w": 0.0,

    "wandb_note": "test_run",
    "resume_wandb_id": None, 
    
    # Validation Saving
    "save_val_volumes": True, 
    "prediction_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/predictions", 
}
