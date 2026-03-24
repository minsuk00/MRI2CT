import copy
import os
import sys

import wandb

sys.path.append(os.path.abspath("src"))
from amix.trainer import Trainer
from common.config import DEFAULT_CONFIG

cfg = copy.deepcopy(DEFAULT_CONFIG)
cfg.update(
    {
        "root_dir": "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5mm_registered_flat",
        "project_name": "MRI2CT_Test",
        "run_name_prefix": "Amix_Thorax_SSO",
        "sanity_check": False,
        "stage_data": False,
        "wandb": True,
        "split_file": "splits/thorax_single_subject_split.txt",
        "total_epochs": 100,
        "steps_per_epoch": 20,
        "val_interval": 10,
        "model_save_interval": 20,
        "compile_mode": None,
        "augment": False,
        "l1_w": 1.0,
        "ssim_w": 0.1,
        "dice_w": 0.0,
        "dice_bone_w": 10.0,
        "validate_dice": True,
    }
)

if __name__ == "__main__":
    trainer = Trainer(cfg)
    trainer.train()
