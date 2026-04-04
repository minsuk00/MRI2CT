import sys
import os
import torch
import shutil

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from mc_ddpm_baseline.trainer import MCDDPMTrainer

def test_mc_ddpm_end_to_end():
    # Mini config for testing
    config = {
        "patch_size": (32, 32, 4), 
        "batch_size": 1,
        "accum_steps": 1,
        "steps_per_epoch": 1,
        "total_epochs": 1,
        "val_interval": 1,
        "model_save_interval": 1,
        "lr": 2e-5,
        "weight_decay": 1e-4,
        "sanity_check": False,
        "wandb": False,
        "stage_data": False,
        "run_name_prefix": "test_run",
        
        # Diffusion params (fast mode)
        "diffusion_steps": 1000,
        "learn_sigma": True,
        "timestep_respacing": [2], # Only 2 steps for inference
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": True,
        "rescale_timesteps": True,
        "rescale_learned_sigmas": True,
        
        # SwinViT params (very small)
        "num_channels": 32,
        "attention_resolutions": (16, 8),
        "channel_mult": (1, 2),
        "num_heads": [4, 4],
        "window_size": [[4, 4, 2], [4, 4, 2]],
        "num_res_blocks": [1, 1],
        "sample_kernel": (([2, 2, 1], [2, 2, 1]),),
        "dropout": 0.0,
        "use_checkpoint": False,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        
        # Data loader configs
        "subjects": ["1ABA011"], # Use a single subject for test
        "patches_per_volume": 1,
        "data_queue_max_length": 1,
        "data_queue_num_workers": 0,
        "use_weighted_sampler": False,
        "val_sw_batch_size": 1,
        "val_sw_overlap": 0.0,
        "viz_limit": 0,
    }
    
    trainer = MCDDPMTrainer(config)
    trainer.train()

if __name__ == "__main__":
    test_mc_ddpm_end_to_end()
    print("Test passed!")
