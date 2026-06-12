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
# v1.3 and v1.4 differ ONLY by AMIX_WEIGHTS — keep the rest identical for fair comparison.
PREFIX="amix_v1_4"
AMIX_WEIGHTS="v1_4"
SPLIT_FILE="splits/center_wise_split.txt"
AUGMENT="True"
PASS_MRI="True"
FEAT_INST_NORM="True"
INPUT_DROPOUT=0.5
WEIGHTED_SAMPLER="True"
DICE_W=0.1
DICE_BONE_W=0.4
FINETUNE_FEAT="False"
FINETUNE_DEPTH=0
LR_FEAT=1e-5
BATCH_SIZE=8
EPOCHS=800
STEPS_PER_EPOCH=500     # halved from 1000 since BATCH_SIZE doubled; keeps total samples_seen at 3.2M
VAL_INTERVAL=40
VALIDATE_DICE="False"   # no teacher dice in validation (manual held-out eval at the end); training dice still logged
NUM_WORKERS=4
RESUME_ID="827la6dp" # wandb id of the launched run (job 51684714). Resubmitting this script resumes it
                     # from checkpoint_last.pt past the 3-day walltime. Clear to "" to start a fresh run.
TAGS="v1_4,bs8" # Comma-separated extra WandB tags


# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_dice-${DICE_W}_bdice-${DICE_BONE_W}_mri-${PASS_MRI}_inorm-${FEAT_INST_NORM}_drop-${INPUT_DROPOUT}_bs-${BATCH_SIZE}_ep-${EPOCHS}"
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

CMD="python $SCRIPT --split_file $SPLIT_FILE --amix_weights $AMIX_WEIGHTS --batch_size $BATCH_SIZE --dice_w $DICE_W --dice_bone_w $DICE_BONE_W --augment $AUGMENT --pass_mri $PASS_MRI --feat_instance_norm $FEAT_INST_NORM --weighted_sampler $WEIGHTED_SAMPLER --finetune_feat $FINETUNE_FEAT --finetune_depth $FINETUNE_DEPTH --lr_feat $LR_FEAT --input_dropout_p $INPUT_DROPOUT --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --validate_dice $VALIDATE_DICE --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi
echo "Running command: $CMD"
$CMD
