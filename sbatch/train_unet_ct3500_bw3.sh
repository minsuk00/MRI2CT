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
# FRESH run: same as nbn71048 (center_wise unet, sigmoid, L1+SSIM+Dice, norm batch, bs8, 800ep)
# EXCEPT GT CT clipped to -1024..3500 HU (so dense bone isn't clipped at 1024) and dice_bone_w 3.0.
# Dice still on the new CADS 35-class teacher; pred is rescaled to the teacher's native
# -1024..1024 convention for the dice term only (L1/SSIM use the full 3500 range).
PREFIX="unet_ct3500"
SPLIT_FILE="splits/center_wise_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"  # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=3.0
CT_RANGE_HI=3500     # upper HU clip (lo stays -1024); widens the [0,1] target to capture dense bone
BATCH_SIZE=8
EPOCHS=800
STEPS_PER_EPOCH=500
VAL_INTERVAL=40
VALIDATE_DICE="False"
NUM_WORKERS=4
RESUME_ID=""          # fresh run; set to the minted wandb id to resume past walltime
TAGS="bs8,ct3500,bw3"

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

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --ct_range_hi $CT_RANGE_HI --augment $AUGMENT --weighted_sampler $WEIGHTED_SAMPLER --norm $NORM --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --validate_dice $VALIDATE_DICE --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
