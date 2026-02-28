import os

# ==========================================
# PATHS & STORAGE
# ==========================================
# Original Data (Tutorial Sample)
# DATA_ROOT = "/home/minsukc/MRI2CT/maisi-mr-to-ct/data"
DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/1.5x1.5x1.5mm"

# Processed MAISI Data (Latents)
# MAISI_DATA_ROOT = "/home/minsukc/MRI2CT/maisi-mr-to-ct/data/training"
MAISI_DATA_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/maisi_data_1.5mm"

# Permanent Model Storage (GPFS)
MODEL_SAVE_ROOT = "/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/maisi_models"

# Pre-trained Weights
WEIGHTS_DIR = os.path.dirname(os.path.abspath(__file__))
AUTOENCODER_PATH = os.path.join(WEIGHTS_DIR, "models", "autoencoder_v1.pt")
DIFFUSION_PATH = os.path.join(WEIGHTS_DIR, "models", "diff_unet_3d_rflow-ct.pt")

# Network Config
PROJECT_ROOT = os.path.abspath(os.path.join(WEIGHTS_DIR, "..", ".."))
NETWORK_CONFIG_PATH = os.path.join(PROJECT_ROOT, "maisi-mr-to-ct", "configs", "config_network.json")

# ==========================================
# TRAINING DEFAULTS
# ==========================================
DEFAULT_CONFIG = {
    # Experiment
    "project_name": "MRI2CT_MAISI_1.5mm",
    "run_name_prefix": "MAISI_ControlNet_1.5mm",
    "seed": 42,
    "device": "cuda",
    "wandb": True,
    
    # Data
    "batch_size": 1,
    "patch_size": None, 
    "num_workers": 4,
    
    # Model
    "lr": 5e-4,
    "total_epochs": 1000,
    "val_interval": 10,
    "model_save_interval": 50,
    
    # Diffusion
    "num_inference_steps": 10,
    
    # Validation
    "val_sw_batch_size": 1,
    "val_sw_overlap": 0.4,
    
    # Paths
    "root_dir": DATA_ROOT,
    "maisi_data_root": MAISI_DATA_ROOT,
    "model_save_root": MODEL_SAVE_ROOT,
}
