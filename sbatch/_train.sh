#!/bin/bash
# Submit all train_* sbatch scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/train_maisi.sh"
sbatch "$SCRIPT_DIR/train_unet.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_4.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_3.sh"
sbatch "$SCRIPT_DIR/train_amix_v1.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_2.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_3_pctmri.sh"
