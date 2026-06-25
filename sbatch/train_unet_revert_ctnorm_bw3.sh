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
# REFERENCE run: byte-identical to train_unet_bw3.sh EXCEPT --aug_renorm_ct True, which
# restores the legacy CT aug (CT included in the per-patch ScaleIntensity renorm). Isolates
# the effect of the ct-normalization fix; every other knob (default ct_range, sigmoid,
# bw3.0, CADS 35-class dice) is the same as bw3.
PREFIX="unet_revertctnorm"
SPLIT_FILE="splits/center_wise_split.txt"
AUGMENT="True"
WEIGHTED_SAMPLER="True"
NORM="batch"  # "batch", "instance", or "none"
DICE_W=0.1
DICE_BONE_W=3.0
AUG_RENORM_CT="True"    # legacy CT aug (pre-fix behavior); the ONLY difference vs bw3
BATCH_SIZE=8
EPOCHS=800
STEPS_PER_EPOCH=500     # halved from 1000 since BATCH_SIZE doubled; keeps total samples_seen at 3.2M
VAL_INTERVAL=40
VALIDATE_DICE="False"   # no teacher dice in validation (manual held-out eval at the end); training dice still logged
NUM_WORKERS=4
RESUME_ID=""          # fresh run; set to the minted wandb id to resume past walltime
TAGS="bs8,bwsweep,revert_ctnorm,reference" # Comma-separated extra WandB tags

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

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --aug_renorm_ct $AUG_RENORM_CT --augment $AUGMENT --weighted_sampler $WEIGHTED_SAMPLER --norm $NORM --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --validate_dice $VALIDATE_DICE --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi

echo "Running command: $CMD"
$CMD
