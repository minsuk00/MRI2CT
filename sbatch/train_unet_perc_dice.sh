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
# Same config as the unet baseline (splits/center_wise_split.txt, bs8, dice 0.1/0.4,
# l1 1.0, 800ep x 500steps) PLUS the Anatomix v1_4 perceptual loss, but with SSIM REMOVED:
# this arm REPLACES ssim with perceptual (baseline = l1+ssim+dice, this = l1+perceptual+dice).
# Perceptual is NCC on decoder features [38,45,52,65]. Background dice now always included.
# Measured raw magnitudes (thorax smoke): ncc-perceptual ~0.28, L1 ~0.08-0.12, so at w=0.1
# the perceptual term ~0.28x the L1 term early, ramping toward parity as L1 converges.
# Perceptual is skipped in validation (OOMs on full-volume); dice excluded from val (validate_dice=False).
PREFIX="unet_perc_dice"
SPLIT_FILE="splits/center_wise_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"  # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=0.4
SSIM_W=0                  # SSIM removed for this arm (replaced by perceptual loss)
PERCEPTUAL_W=0.1          # Anatomix v1_4 perceptual loss weight (was 0.5 for 06e850ny)
BATCH_SIZE=8
EPOCHS=800
STEPS_PER_EPOCH=500     # matches 9xmodnhn
VAL_INTERVAL=40
VALIDATE_DICE="False"   # exclude teacher dice from validation (val loss = L1 only; perceptual+ssim are not in val)
NUM_WORKERS=4
RESUME_ID="y5tqp2bt"  # wandb id of the launched run (job 51684713). Resubmitting this script resumes it
                      # from checkpoint_last.pt past the 3-day walltime. Clear to "" to start a fresh run.
TAGS="bs8,perceptual,dice" # Comma-separated extra WandB tags

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_dice-${DICE_W}_bdice-${DICE_BONE_W}_perc-${PERCEPTUAL_W}_norm-${NORM}_bs-${BATCH_SIZE}_ep-${EPOCHS}"
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

# Required when perceptual_w>0: the frozen extractor's compiled backward needs contiguous
# cuDNN workspace at bs=8; without this it dies with "FIND was unable to find an engine".
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/minsukc/MRI2CT

SCRIPT="src/unet_baseline/train.py"

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --ssim_w $SSIM_W --perceptual_w $PERCEPTUAL_W --augment $AUGMENT --weighted_sampler $WEIGHTED_SAMPLER --norm $NORM --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --validate_dice $VALIDATE_DICE --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
