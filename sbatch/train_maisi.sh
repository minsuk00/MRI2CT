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
# Shared knobs match amix/unet on the applicable ones (SPLIT_FILE, EPOCHS, AUGMENT,
# NUM_WORKERS via MAISI_CONFIG default). MAISI is full-volume so BATCH_SIZE=1 + ACCUM_STEPS=4
# = effective batch 4 (vs amix/unet bs=8 × accum=1). STEPS_PER_EPOCH=1000 keeps
# total samples_seen at 3.2M. VAL_INTERVAL=10 because full val on 207 subjects ≈ 60 min.
PREFIX="maisi"
SPLIT_FILE="splits/center_wise_split.txt"
LR="5e-4"
BATCH_SIZE=1                # full-volume; cannot increase
ACCUM_STEPS=4               # effective batch = 1 × 4 = 4 samples/optimizer step
EPOCHS=800
STEPS_PER_EPOCH=1000        # 1×4×1000×800 = 3.2M samples_seen, matches amix/unet
VAL_INTERVAL=10             # full val on 207 subjects ≈ 60 min
COMPILE_MODE="None"         # confirmed safe (full+vae triggered dynamo recompile thrash in val SW)
VAE_COMPILE="True"          # confirmed best — VAE input fixed-size, compiles once
PREENCODED_LATENTS_DIR=""   # Leave empty to encode on-the-fly
RESUME_ID="u5vad8vg"        # Leave empty if not resuming
TAGS="thorax"               # Comma-separated extra WandB tags

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_lr-${LR}_bs-${BATCH_SIZE}_accum-${ACCUM_STEPS}_ep-${EPOCHS}"
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

SCRIPT="src/maisi_baseline/train.py"

CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --accum_steps $ACCUM_STEPS --compile_mode $COMPILE_MODE --vae_compile $VAE_COMPILE"
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
