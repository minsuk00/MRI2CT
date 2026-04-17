#!/bin/bash
# Submit all train_* sbatch scripts

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

sbatch "$SCRIPT_DIR/train_amix_v1_3_batchnorm_finetune_14.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_3_batchnorm_finetune_full.sh"

sbatch "$SCRIPT_DIR/train_amix_v1.sh"
sbatch "$SCRIPT_DIR/train_unet.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_3_cutout_nodrop.sh"
sbatch "$SCRIPT_DIR/train_amix_v1_3_cutout.sh"
sbatch "$SCRIPT_DIR/train_maisi_nocut.sh"
sbatch "$SCRIPT_DIR/train_mcddpm_nocut.sh"


