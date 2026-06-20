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
# Twin of train_unet_perc_sep.sh, IDENTICAL params, but the LNCC perceptual loss is computed
# by the fused_lncc CUDA kernel (--perceptual_fused True) instead of the Python box-conv.
# Same squared-LNCC semantics (kernel_size=7, smooth 1e-5), so it's a drop-in: training curves
# should match the sep arm while the step is faster and lighter on VRAM. Fresh run (no resume).
PREFIX="unet_perc_fused"
SPLIT_FILE="splits/center_wise_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"             # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=0.4
SSIM_W=0                 # SSIM removed for this arm (replaced by perceptual loss)
PERCEPTUAL_W=0.1         # Anatomix v1_4 perceptual loss weight (same as the sep twin)
SEPARABLE="True"         # ignored when fused=True (kept identical to the sep twin's CMD)
FUSED="True"             # use the fused_lncc CUDA kernel for the ncc metric
BATCH_SIZE=4
EPOCHS=800
STEPS_PER_EPOCH=1000
VAL_INTERVAL=40
VALIDATE_DICE="False"    # exclude teacher dice from validation (val loss = L1 only)
NUM_WORKERS=4
RESUME_ID=""             # fresh run (the sep twin resumed mwrwxvvu; this is its fused comparison)
TAGS="bs4,perceptual,dice,lncc,fused" # Comma-separated extra WandB tags

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
