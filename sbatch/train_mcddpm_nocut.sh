#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# --- Configuration Area ---
PREFIX="mcddpm_nocut"
SPLIT_FILE="splits/thorax_center_wise_split.txt"
EPOCHS=5000
STEPS_PER_EPOCH=100
VAL_INTERVAL=5
SAVE_INTERVAL=1
FULL_VAL="False"
USE_CUTOUT="False"
RESUME_ID="9qjvw911"  
TAGS="thorax,mcddpm,no_cutout"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_ep-${EPOCHS}"
    if [ ! -z "$RESUME_ID" ]; then
        JOB_NAME="${JOB_NAME}_res-${RESUME_ID}"
    fi
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    echo "🚀 Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" --output="/home/minsukc/MRI2CT/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" "$0"
    exit
fi

# --- Training Logic ---
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
cd /home/minsukc/MRI2CT
SCRIPT="src/mc_ddpm_baseline/train.py"
CMD="python $SCRIPT --split_file $SPLIT_FILE --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --model_save_interval $SAVE_INTERVAL --full_val $FULL_VAL --use_cutout $USE_CUTOUT"
if [ ! -z "$RESUME_ID" ]; then CMD="$CMD --resume_id $RESUME_ID"; fi
if [ ! -z "$TAGS" ]; then CMD="$CMD --tags \"$TAGS\""; fi
echo "Running command: $CMD"
$CMD
