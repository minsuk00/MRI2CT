#!/bin/bash
# Submit all train_*.sh scripts.
# Use `bash` (not `sbatch`) so each script's self-submission block runs and
# constructs a descriptive JOB_NAME from its config (e.g. amix_v1_4_center_wise_…_bs-8_ep-800).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Brain diagnostic UNets (fresh runs).
bash "$SCRIPT_DIR/train_unet_brain_random.sh"
bash "$SCRIPT_DIR/train_unet_brain_center_wise.sh"

# Resumes of the all-region center-wise baselines.
bash "$SCRIPT_DIR/train_amix_v1_3.sh"
bash "$SCRIPT_DIR/train_amix_v1_4.sh"
bash "$SCRIPT_DIR/train_cwdm.sh"
bash "$SCRIPT_DIR/train_maisi_noaug.sh"
bash "$SCRIPT_DIR/train_mcddpm.sh"

