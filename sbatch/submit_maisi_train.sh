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
PREFIX="maisi"
SPLIT_FILE="splits/thorax_center_wise_split.txt" # "splits/center_wise_split.txt" for full
LR=5e-4
BATCH_SIZE=1
ACCUM_STEPS=4
EPOCHS=1000
STEPS_PER_EPOCH=500
VAL_INTERVAL=1
SAVE_INTERVAL=1
WANDB="True"
PREENCODED_LATENTS_DIR="" # Leave empty to encode on-the-fly (e.g., "/tmp/maisi_latents")
RESUME_ID="86funnoa" # Leave empty if not resuming (e.g., "uszjimzg")
TAGS="thorax" # Comma-separated extra WandB tags

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    # Construct descriptive Job Name
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_lr-${LR}_acc-${ACCUM_STEPS}"
    if [ ! -z "$RESUME_ID" ]; then
        JOB_NAME="${JOB_NAME}_res-${RESUME_ID}"
    fi

    # Ensure log directory exists
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/

    # Re-submit this script with the dynamic names
    echo "🚀 Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" \
           --output="/home/minsukc/MRI2CT/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" \
           "$0"
    exit
fi

# --- Training Logic (Runs only on the GPU Node) ---
# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

cd /home/minsukc/MRI2CT

SCRIPT="src/maisi_baseline/train.py"

# Construct the command
CMD="python $SCRIPT --split_file $SPLIT_FILE --lr $LR --batch_size $BATCH_SIZE --accum_steps $ACCUM_STEPS --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --model_save_interval $SAVE_INTERVAL --wandb $WANDB"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_wandb_id $RESUME_ID"
fi
if [ ! -z "$PREENCODED_LATENTS_DIR" ]; then
    CMD="$CMD --preencoded_latents_dir $PREENCODED_LATENTS_DIR"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
