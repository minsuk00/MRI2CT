#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80g
#SBATCH --time=14-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# --- Configuration Area ---
# MAISI ControlNet training on the REGENERATED (dynamic_infer-fixed) pre-encoded latents.
# The original fresh run (wandb ahzkny1u) was launched with RESUME_ID="" + 2-23h walltime.
# RESUME_ID is now set to ahzkny1u so re-running this resumes that run; clear it for a brand-new run.
#   - LR=1e-5  (matches NV-Generate-CTMR config_maisi_controlnet_train_rflow-ct.json:6)
#   - augment off, constant LR (no scheduler), eff batch = 1 x accum 8 = 8
#   - 14-day walltime (= partition MaxTime) for the post-maintenance resume.
# Launch:  sbatch sbatch/train_maisi_fresh.sh
#   Only run while no other job is writing wandb run ahzkny1u (i.e. after 51687784 ends),
#   or chain it:  sbatch --dependency=afterany:51687784 sbatch/train_maisi_fresh.sh
PREFIX="maisi"
SPLIT_FILE="splits/center_wise_split.txt"
LR="1e-5"
BATCH_SIZE=1
ACCUM_STEPS=8
EPOCHS=8000
STEPS_PER_EPOCH=50
VAL_INTERVAL=50
FULL_VAL="False"
AUGMENT="False"
VAE_COMPILE="True"
PREENCODED_LATENTS_DIR="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked_maisi_latents"
NO_SCHEDULER="True"
OVERRIDE_LR="False"
RESUME_ID="ahzkny1u"               # resume the run on the same wandb dashboard; clear ("") for a brand-new run
TAGS="fresh,fixed-latents,lr1e-5"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_lr-${LR}_bs-${BATCH_SIZE}_accum-${ACCUM_STEPS}_ep-${EPOCHS}_fresh"
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
CMD="python $SCRIPT --split_file $SPLIT_FILE --batch_size $BATCH_SIZE --lr $LR --epochs $EPOCHS --steps_per_epoch $STEPS_PER_EPOCH --val_interval $VAL_INTERVAL --accum_steps $ACCUM_STEPS --vae_compile $VAE_COMPILE --augment $AUGMENT --no_scheduler $NO_SCHEDULER --override_lr $OVERRIDE_LR --full_val $FULL_VAL"
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
