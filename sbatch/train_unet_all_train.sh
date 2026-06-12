#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=14-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# --- Configuration Area ---
# Same config as the unet baseline (L1 + SSIM + Dice 0.1 + Bone Dice 0.4, background-included
# dice, no perceptual, bs8, norm batch, augment + weighted sampler, 800ep x 500 steps),
# but trained on ALL SynthRAD data (splits/all_train_split.txt) with NO held-out val.
# Validation is skipped automatically when the split has no `val` lines (so val_interval /
# validate_dice are no-ops here). FRESH run on the new dice impl (can't resume tz56kmux).
PREFIX="unet"
SPLIT_FILE="splits/all_train_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"  # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=0.4
BATCH_SIZE=8
EPOCHS=800
STEPS_PER_EPOCH=500     # matches 9xmodnhn: 800*500*8 = 3.2M samples seen
VAL_INTERVAL=20         # no-op here (no val set), kept for config parity
NUM_WORKERS=4
RESUME_ID="krdhs2k0"  # wandb id of the launched run (job 51684715). Resubmitting this script resumes it past walltime. Clear to "" for fresh.
TAGS="all_train,bs8" # Comma-separated extra WandB tags

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_dice-${DICE_W}_bdice-${DICE_BONE_W}_norm-${NORM}_bs-${BATCH_SIZE}_ep-${EPOCHS}"
    if [ ! -z "$RESUME_ID" ]; then
        JOB_NAME="${JOB_NAME}_res-${RESUME_ID}"
    fi

    mkdir -p /home/minsukc/MRI2CT/slurm_logs/

    echo "🚀 Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" \
           --output="/home/minsukc/MRI2CT/slurm_logs/${TIMESTAMP}_${JOB_NAME}_%j.log" \
           "$0"
    exit
fi

# --- Training Logic ---
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

cd /home/minsukc/MRI2CT

SCRIPT="src/unet_baseline/train.py"

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --augment $AUGMENT --weighted_sampler $WEIGHTED_SAMPLER --norm $NORM --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
