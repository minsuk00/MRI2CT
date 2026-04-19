#!/bin/bash
# Submit all train_* sbatch scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/train_amix_v1.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_featonly.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_featonly_noinorm.sh"
sbatch "$SCRIPT_DIR/train_unet.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_featscale.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_noinc.sh"


