#!/bin/bash
# Resume all 6 long runs after the June 15 maintenance.
# Each train_*.sh self-submits via sbatch (no SLURM_JOB_ID here) and resumes its
# wandb run from checkpoint. Safe to run from any cwd.
cd "$(dirname "$0")"

bash train_amix_v1_4.sh
bash train_unet_all_train.sh
bash train_unet_perc_dice.sh
bash train_unet.sh
bash train_cwdm.sh
bash train_maisi_fresh.sh
# bash train_seg_center.sh
# bash train_seg_all.sh