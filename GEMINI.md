# MRI2CT Project Context

## Project Overview
This project, `MRI2CT`, aims to synthesize CT images from MRI scans using deep learning techniques. It utilizes PyTorch for model training, with `monai` and `torchio` for medical image data loading and augmentation. The core architecture involves a U-Net-based generator, leveraging a pre-trained feature extractor (`anatomix`).

## Directory Structure

*   **`src/`**: Contains the source code for training and preprocessing.
    *   `train.py`: The main entry point for training the synthesis model. Contains the `Trainer` class, configuration (`DEFAULT_CONFIG`), and the training loop.
    *   `preprocess/`: Scripts for data preparation.
        *   `registration_batch.py`: Likely for registering MRI to CT (or vice versa).
        *   `resample_batch.py`: For resampling volumes to a common resolution (e.g., 3.0mm isotropic).
        *   `totalsegmentator_batch.py`: Integration with TotalSegmentator for anatomical segmentation.
*   **`anatomix/`**: A local Python package containing model definitions (e.g., `Unet`) and pre-trained weights.
    *   `model/network.py`: Likely contains the `Unet` class definition used in `train.py`.
    *   `model-weights/`: Stores pre-trained weights for the feature extractor.
*   **`data/`**: Directory housing the medical imaging data. (deprecated. data is now stored in another directory: /gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/)
*   **`_deprecated/`**: Contains old notebooks and scripts. Useful for reference but obsolete.

## Key Components

### 1. Training (`src/train.py`)
The `train.py` script orchestrates the training process.
*   **Model**: Uses a U-Net translator. Optionally uses an `anatomix` U-Net as a frozen or fine-tuned feature extractor.
*   **Loss Function**: Composite loss including L1, L2, SSIM, and Perceptual Loss (using `anatomix` features).
*   **Data Loading**: Uses `torchio` for patch-based training (`Queue`, `SubjectsDataset`) with weighted sampling based on tissue probability.
*   **Logging**: Integrated with `wandb` for experiment tracking and visualization.

### 2. Configuration
Configuration is managed via the `DEFAULT_CONFIG` dictionary in `src/train.py`. Key parameters include:
*   `root_dir`: Path to the data directory.
*   `anatomix_weights`: Version of the feature extractor weights ("v1", "v2").
*   `patch_size`: Size of 3D patches for training (default: 96).
*   `loss_weights`: Weights for L1, SSIM, Perceptual, etc.

## Setup & Usage

### Dependencies
Key dependencies include:
*   `torch`
*   `torchio`
*   `monai`
*   `nibabel`
*   `wandb`
*   `numpy`, `scipy`, `matplotlib`
*   `fused_ssim` (a custom/installed module)

### Running Training
To start training, execute the `src/train.py` script. It will iterate through `EXPERIMENT_CONFIG` to run defined experiments.

```bash
python src/train.py
```

Ensure `root_dir` in `DEFAULT_CONFIG` points to the correct data location.

## Development Notes
*   **Conventions**: The project uses `black` or similar formatting (inferred). Type hinting is minimal.
*   **Paths**: Paths in `train.py` might be hardcoded to specific clusters/users (e.g., `/gpfs/accounts/...`). **Always verify and update paths before running.**
*   **GPU**: Code is optimized for CUDA GPUs (Ampere+) using `TF32` and `bfloat16` mixed precision.

## 🗺️ RESEARCH MAP & PATHS
- **Codebase (HOME):** Primary directory for edits.
- **Data (SCRATCH):** `/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/3.0x3.0x3.0mm`. (Read-only).
- **Environment:** Conda env `mri2ct`.
- The code is running on a school GPU server.

## 🛠️ GPU NODE COMMANDS
- **Check GPU:** `!nvidia-smi`
- **Run Training:** `!python src/train.py` (Run directly on node).

## 🔄 RESEARCH WORKFLOW
1. **Investigate:** Search codebase for dependencies before edits.
2. **Plan:** Propose a plan in chat.
3. **Verify:** Ask me for confirmation before EVERY file edit.
4. **Scratchpad:** Log hyperparameter shifts in `research_notes.md`. (Date | Action | Reason).

## 🚨 STANDARDS
- **Minimalism:** Do not refactor working code. Minimal, simple code is better.
- **Git:** Work on branches. Remind me to commit after successful edits.
