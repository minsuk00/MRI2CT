# MRI2CT Preprocessing Pipeline Setup
Order of preprocessing:
- resample_batch.py -> totalsegmentator_batch.py (or _sharded.py) -> registration_tune.py

- NOTE: All necessary changes you need to make are at the very bottom of the script under CONFIGURATION

## 1. Downloading Datasets
- SynthRAD2023: https://zenodo.org/records/7260705
1. wget -O synthRAD2023.zip https://zenodo.org/records/7260705/files/Task1.zip?download=1
2. unzip synthRAD2023.zip & rm synthRAD2023.zip
3. mv Task1 synthRAD2023

- SynthRAD2025: https://zenodo.org/records/15373853
1. wget -O synthRAD2025.zip https://zenodo.org/records/15373853/files/synthRAD2025_Task1_Train.zip?download=1
2. unzip synthRAD2025.zip & synthRAD2025.zip
3. mv synthRAD2025_Task1_Train synthRAD2025

(SynthRAD 2023 uses .nii.gz, but 2025 uses .mha. resample_batch.py will convert everything to .nii.gz)

## 2. Preprocessing

### Step 1: Resampling and Data Splitting
- **Script**: `resample_batch.py`
- **Output**: Creates a `synthRAD_combined/1.0x1.0x1.0mm/` directory with `train`, `val`, and `test` subfolders.
- (you can check visualizations of resampled slices under `synthRAD_combined/1.0x1.0x1.0mm/visualizations/{split}`)

- TODO: Change directory paths in INPUT_DIRS to downloaded dataset path. (for 2025 dataset, append /Task1 to end of path)
- TODO: Change OUTPUT_ROOT as well.

### Step 2: Anatomical Segmentation
- **Script**: `totalsegmentator_batch.py` or `totalsegmentator_batch_sharded.py`.
- **Cluster Usage**: Use `submit_seg.sh` to launch a Slurm array job for faster processing across multiple nodes. (Not sure what system MIT uses. I'll just leave my Slurm config as it is.)

- TODO: Change ROOT to the OUTPUT_ROOT from Step 1. (synthRAD_combind directory)

- You can view visualizations of the segmentation under each subject folder. If you used the sharded script, you will need to run the `totalsegmentator_batch.py` again for the visualization. (it will skip totalsegmentator if the segmentation file exists)

### Step 3: Registration Tuning
- **Script**: `registration_tune.py`

- The results will be saved to `tuning_results` directory

- TODO: put anatomix v1.1 weights under `anatomix/model-weights/`
- TODO: change ROOT and DATA_DIR
- TODO: change TARGET_SUBJECTS to names of volumes to use for tuning
- TODO: change GRID to hyperparameters to try out
- TODO: for registration tuning, I modified the anatomix script (`run_convex_adam_with_network_feats`) so I can pass the pre-loaded anatomix model to `convex_adam`. You can also not modify anatomix and simply pass the checkpoint path to `convex_adam`.

- NOTE: dice score will be calculated based on the regions in REGION_MAPS

