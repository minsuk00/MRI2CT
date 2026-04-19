#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# --- Configuration Area ---
PREFIX="amix"
SPLIT_FILE="splits/thorax_center_wise_split.txt"
AUGMENT="True"
USE_CUTOUT="False"
CUTOUT_ALPHA=1.0
DICE_W=0.1
DICE_BONE_W=0.3
PASS_MRI="True"
FEAT_INST_NORM="False"
FEAT_SCALE_DOWN=100
INPUT_DROPOUT=0.5
USE_ZERO_MASK="False"
WEIGHTED_SAMPLER="True"
FINETUNE_FEAT="False"
FINETUNE_DEPTH=0
LR_FEAT=1e-5
AMIX_WEIGHTS="v1"
EPOCHS=1000
STEPS_PER_EPOCH=500
NUM_WORKERS=4
RESUME_ID="" # Leave empty if not resuming
TAGS="thorax,v1,featscale,noinorm" # Comma-separated extra WandB tags


# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_amix-${AMIX_WEIGHTS}_dice-${DICE_W}_bdice-${DICE_BONE_W}_aug-${AUGMENT}_mri-${PASS_MRI}_inorm-${FEAT_INST_NORM}_fscale-${FEAT_SCALE_DOWN}_zmask-${USE_ZERO_MASK}_wsmpl-${WEIGHTED_SAMPLER}_ftune-${FINETUNE_FEAT}_fdepth-${FINETUNE_DEPTH}_drop-${INPUT_DROPOUT}_cut-${USE_CUTOUT}_calpha-${CUTOUT_ALPHA}_ep-${EPOCHS}"
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

SCRIPT="src/amix/train.py"

CMD="python $SCRIPT --split_file $SPLIT_FILE --amix_weights $AMIX_WEIGHTS --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --augment $AUGMENT --pass_mri $PASS_MRI --feat_instance_norm $FEAT_INST_NORM --feat_scale_down $FEAT_SCALE_DOWN --use_zero_mask $USE_ZERO_MASK --weighted_sampler $WEIGHTED_SAMPLER --finetune_feat $FINETUNE_FEAT --finetune_depth $FINETUNE_DEPTH --lr_feat $LR_FEAT --input_dropout_p $INPUT_DROPOUT --use_cutout $USE_CUTOUT --cutout_alpha $CUTOUT_ALPHA --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi
echo "Running command: $CMD"
$CMD
