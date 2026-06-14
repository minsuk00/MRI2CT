#!/usr/bin/env bash
# Source this file in every koalAI SLURM script. Sets nnUNet + wandb env vars.

export NNSYN_WORKSPACE=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/nnsyn_workspace
export nnUNet_raw=$NNSYN_WORKSPACE/raw
export nnUNet_preprocessed=$NNSYN_WORKSPACE/preprocessed
export nnUNet_results=$NNSYN_WORKSPACE/results
export NNSYN_ORIGIN_ROOT=$NNSYN_WORKSPACE/origin

export WANDB_PROJECT=mri2ct
export WANDB_DIR=$NNSYN_WORKSPACE/wandb

mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" \
         "$NNSYN_ORIGIN_ROOT" "$WANDB_DIR"
