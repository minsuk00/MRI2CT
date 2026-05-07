#!/bin/bash
# Submit all four train_*.sh scripts.
# Use `bash` (not `sbatch`) so each script's self-submission block runs and
# constructs a descriptive JOB_NAME from its config (e.g. amix_v1_4_center_wise_…_bs-8_ep-800).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "$SCRIPT_DIR/train_amix_v1_3.sh"
bash "$SCRIPT_DIR/train_amix_v1_4.sh"
bash "$SCRIPT_DIR/train_unet.sh"
bash "$SCRIPT_DIR/train_maisi.sh"
