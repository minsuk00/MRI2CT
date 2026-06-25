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
# FRESH all_train unet, from scratch: L1 + fused-LNCC perceptual + Dice (bone_w 3.0),
# new CADS 35-class dice (teacher/seg from DEFAULT_CONFIG). SSIM off (perceptual replaces it).
PREFIX="unet_perc_fused"
SPLIT_FILE="splits/all_train_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"             # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=3.0
SSIM_W=0                 # SSIM removed (replaced by perceptual loss)
PERCEPTUAL_W=0.1         # Anatomix v1_4 perceptual loss weight
SEPARABLE="True"         # ignored when fused=True
FUSED="True"             # use the fused_lncc CUDA kernel for the ncc metric
BATCH_SIZE=4             # perceptual loss VRAM -> bs4 (not bs8)
EPOCHS=800
STEPS_PER_EPOCH=1000     # bs4 -> 1000 steps keeps samples_seen at 3.2M
VAL_INTERVAL=40
VALIDATE_DICE="False"    # no teacher dice in validation (training dice still on)
NUM_WORKERS=4
RESUME_ID=""             # from scratch; set to the minted wandb id to resume past walltime
TAGS="bs4,perceptual,dice,lncc,fused,all_train,bw3" # Comma-separated extra WandB tags

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
# cuDNN workspace; without this it dies with "FIND was unable to find an engine".
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/minsukc/MRI2CT

SCRIPT="src/unet_baseline/train.py"

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --ssim_w $SSIM_W --perceptual_w $PERCEPTUAL_W --perceptual_separable $SEPARABLE --perceptual_fused $FUSED --augment $AUGMENT --weighted_sampler $WEIGHTED_SAMPLER --norm $NORM --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --validate_dice $VALIDATE_DICE --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
