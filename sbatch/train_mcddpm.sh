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
# MC-IDDPM paper-faithful run (Pan et al., Med Phys 2024).
# Prostate variant (128,128,4) patches — our thorax volumes are larger than both
# paper datasets, so the larger preset is the better fit.
PREFIX="mcddpm"
SPLIT_FILE="splits/center_wise_split.txt"
LR=3e-5         # paper page 6
WEIGHT_DECAY=1e-5  # paper page 6
BATCH_SIZE=4
PATCHES_PER_VOL=2
LR_ANNEAL_STEPS=0         # 0 = constant LR (scheduler is None); train as long as we keep submitting
USE_AMP="True"     # bf16 autocast — matches notebook AMP path, safer than fp16
USE_CHECKPOINT="False"
DIFFUSION_STEPS=1000
VAL_STEPS=25
EPOCHS=7000               # extend further; LR stays constant since LR_ANNEAL_STEPS=0
STEPS_PER_EPOCH=500
SAVE_INTERVAL=50          # bumped from 5: at SAVE_INTERVAL=5 a 657MB snapshot every 5 epochs filled the gpfs quota
VAL_INTERVAL=5
VAL_SUBJ_ID="1THB011"
NUM_WORKERS=4
RESUME_ID="a3g28rez"      # resume run; extend training
TAGS="extended"           # Comma-separated extra WandB tags


# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_bs-${BATCH_SIZE}_pv-${PATCHES_PER_VOL}_ep-${EPOCHS}"
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

# WandB on gpfs so checkpoints survive home-quota / SLURM cuts.
export WANDB_DIR=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs

cd /home/minsukc/MRI2CT

SCRIPT="baselines/mc_ddpm/train.py"

CMD="python $SCRIPT \
    --split_file $SPLIT_FILE \
    --lr $LR --weight_decay $WEIGHT_DECAY \
    --batch_size $BATCH_SIZE --patches_per_volume $PATCHES_PER_VOL \
    --lr_anneal_steps $LR_ANNEAL_STEPS \
    --use_amp $USE_AMP --use_checkpoint $USE_CHECKPOINT \
    --diffusion_steps $DIFFUSION_STEPS --val_steps $VAL_STEPS \
    --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH \
    --save_interval $SAVE_INTERVAL --val_interval $VAL_INTERVAL \
    --val_subj_id $VAL_SUBJ_ID \
    --num_workers $NUM_WORKERS"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    CMD="$CMD --tags \"$TAGS\""
fi
echo "Running command: $CMD"
$CMD
