import os
import sys
import copy
import warnings
import torch
import traceback

# Add project root to path to allow absolute imports (e.g., from src.trainer import ...)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.config import DEFAULT_CONFIG
from src.trainer import Trainer
from src.utils import cleanup_gpu

# ==========================================
# 0. GLOBAL SETUP & UTILS
# ==========================================
# Enables TF32 for significantly faster training on Ampere+ GPUs
torch.set_float32_matmul_precision('high')

warnings.filterwarnings("ignore", category=UserWarning, module="monai.utils.module")
warnings.filterwarnings("ignore", message=".*SubjectsLoader in PyTorch >= 2.3.*")
warnings.filterwarnings("ignore", message=".*non-tuple sequence for multidimensional indexing.*")
os.environ["WANDB_IGNORE_GLOBS"] = "*.pt;*.pth"

EXPERIMENT_CONFIG = [
    {
        "total_epochs": 5000,
        "sanity_check": False,
        
        "batch_size": 8,
        "steps_per_epoch": 100,
        "val_interval": 1,
        "viz_limit": 10,
        "model_save_interval": 1,

        "model_compile_mode": "reduce-overhead",
        "accum_steps": 1,
        "wandb_note": "unet",
        "enable_safety_padding": False,
        "use_weighted_sampler": False,

        # "override_lr": True,
    
        # "wandb_note": "long_run_anatomix_v2",
        # "resume_wandb_id": "gozzhvfn", 
        
        # "anatomix_weights": "v1",
        # "wandb_note": "long_run_anatomix_v1",
        # "resume_wandb_id": "msj4nmzy", 
    },
]

if __name__ == "__main__":
    print(f"📚 Found {len(EXPERIMENT_CONFIG)} experiments to run.")
    
    for i, exp in enumerate(EXPERIMENT_CONFIG):
        print(f"\n{'='*40}")
        print(f"STARTING EXPERIMENT {i+1}/{len(EXPERIMENT_CONFIG)}")
        print(f"Config: {exp}")
        print(f"{'='*40}\n")
        
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
            print(f"❌ Experiment {i+1} Failed: {e}")
            traceback.print_exc()
