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
PREFIX="cwdm"
SPLIT_FILE="/home/minsukc/MRI2CT/splits/center_wise_split.txt"
DATASET="synthrad"
CONTR="ct"
LR=1e-5
SAVE_INTERVAL=5000           # preserved milestone checkpoints every N steps (~5.5h at ~4s/iter)
SAVE_LAST_INTERVAL=1000      # rolling <dataset>_last.pt overwrite every N steps (~1h apart); used by auto-resume
LR_ANNEAL_STEPS=0            # PAPER-FAITHFUL: constant LR, no auto-stop. Target 1.2M total iters
                             # — monitor `info/global_step` in wandb and stop submitting once it crosses 1.2M.
VAL_INTERVAL=5000            # in-training light val sample cadence (steps) — ~20s overhead per fire
VAL_SUBJ_ID="1THB011"        # fixed subject for divergence check
VAL_DDIM_STEPS=50            # reduced-step DDPM via timestep_respacing
RESUME_ID="smg8thkr"         # WandB run id to resume from (job 50243521, started 2026-05-15 04:38). Empty = fresh run.
RESUME_CHECKPOINT=""         # absolute path to synthrad_NNNNNN.pt (loads model + sibling optNNNNNN.pt + sets step counter)
TAGS=""                      # extra comma-separated wandb tags


# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    SPLIT_NAME=$(basename "$SPLIT_FILE" .txt)
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="${PREFIX}_${SPLIT_NAME}_lr-${LR}_iters-${LR_ANNEAL_STEPS}"
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

# Keep wandb run + checkpoint dirs under the same root as amix runs.
export WANDB_DIR=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs
mkdir -p "$WANDB_DIR"

# Reduce CUDA allocator fragmentation — necessary for the largest train subjects.
# Without this, run hits OOM in backward on 1HNA008 / 1HNA038 / 1THA235 even with grad checkpointing on.
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/minsukc/MRI2CT/baselines/cwdm

# Build the same arg layout as run.sh but in one place, so SLURM-side overrides
# (RESUME_ID, TAGS, LR_ANNEAL_STEPS) are easy.
DATA_DIR=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD/1.5mm_registered_flat_masked

COMMON="
--dataset=${DATASET}
--num_channels=64
--class_cond=False
--num_res_blocks=2
--num_heads=1
--learn_sigma=False
--use_scale_shift_norm=False
--attention_resolutions=
--channel_mult=1,2,2,4,4
--diffusion_steps=1000
--noise_schedule=linear
--rescale_learned_sigmas=False
--rescale_timesteps=False
--dims=3
--batch_size=1
--num_groups=32
--in_channels=16
--out_channels=8
--bottleneck_attention=False
--resample_2d=False
--renormalize=True
--additive_skips=False
--use_freq=False
--predict_xstart=True
--contr=${CONTR}
--use_checkpoint=True
"

TRAIN="
--data_dir=${DATA_DIR}
--split_file=${SPLIT_FILE}
--resume_checkpoint=${RESUME_CHECKPOINT}
--resume_step=0
--image_size=224
--use_fp16=False
--lr=${LR}
--lr_anneal_steps=${LR_ANNEAL_STEPS}
--save_interval=${SAVE_INTERVAL}
--save_last_interval=${SAVE_LAST_INTERVAL}
--num_workers=2
--devices=0
--val_subj_id=${VAL_SUBJ_ID}
--val_interval=${VAL_INTERVAL}
--val_ddim_steps=${VAL_DDIM_STEPS}
--use_wandb=True
--wandb_project=mri2ct
"
if [ ! -z "$RESUME_ID" ]; then
    TRAIN="$TRAIN --wandb_resume_id=$RESUME_ID"
fi
if [ ! -z "$TAGS" ]; then
    TRAIN="$TRAIN --wandb_extra_tags=$TAGS"
fi

CMD="python scripts/train.py $TRAIN $COMMON"
echo "Running command: $CMD"
$CMD
