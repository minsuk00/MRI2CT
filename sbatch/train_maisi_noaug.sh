#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=80g
#SBATCH --time=10-00:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# --- Configuration Area ---
# Matches maisi-mr-to-ct ablation recipe (NVIDIA's MR→CT extension of MAISI):
#   - Only ControlNet trained; VAE + Diffusion UNet frozen.
#   - lr=5e-4 (configs/config_train.json:6 in maisi-mr-to-ct).
#   - NO augmentation (scripts/train.py uses LoadImaged + Orientationd +
#     ScaleIntensityRangePercentilesd + Lambdad only; no Rand* transforms).
#   - Original NV-Generate-CTMR paper uses lr=1e-5 but on 58k volumes for
#     mask→image; ablation bumps to 5e-4 for small-data MR→CT.
# Single-A40 setup: BATCH_SIZE=1 × ACCUM_STEPS=8 = eff_bs 8 (matches paper 8-GPU DDP).
PREFIX="maisi"
SPLIT_FILE="splits/center_wise_split.txt"
LR="1e-5"                   # lowered from 1e-4 on resume at ~ep3268 (SSIM~0.81/MAE~60): finer fine-tune. OVERRIDE_LR resets the optimizer LR to this value.
BATCH_SIZE=1                # full-volume; cannot increase
ACCUM_STEPS=8               # effective batch = 1 × 8 = 8 samples/optimizer step
EPOCHS=8000                 # 8000*50=400k opt steps * eff_bs 8 = 3.2M volumes — matches amix/unet 3.2M-patch budget (was 1500=600k); scheduler disabled so LR stays constant
STEPS_PER_EPOCH=50
VAL_INTERVAL=50             # reduced val (1 subj/region = ~5) is cheap; fire often for a progress signal
FULL_VAL="False"            # 1-subject-per-region val (~5 subjects) instead of all 207 — like cWDM. Full eval lives in src/maisi_baseline/validate.py.
AUGMENT="False"             # match ablation: no random augmentation
VAE_COMPILE="True"          # confirmed best — VAE input fixed-size, compiles once
PREENCODED_LATENTS_DIR="/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked_maisi_latents"   # Leave empty to encode on-the-fly. Generate with src/maisi_baseline/encode_all_volumes.py.
NO_SCHEDULER="True"         # disable PolynomialLR -> constant LR for extended training past the original poly-decay horizon
OVERRIDE_LR="True"          # reset optimizer LR to $LR=1e-4 (checkpoint state has it decayed near 0 by the old scheduler)
RESUME_ID="5hprtpwl"                # resume run; extend past original 30k-step poly-decay horizon
TAGS="noaug,lr5e-4,extended"

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_lr-${LR}_bs-${BATCH_SIZE}_accum-${ACCUM_STEPS}_ep-${EPOCHS}_noaug"

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
