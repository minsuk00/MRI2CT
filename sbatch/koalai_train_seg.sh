#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Train student TotalSegmentator model (Dataset<SEG_ID>) for one region.
# Runs from the koalAI-seg worktree (nnunetv2 branch).
# Usage:
#   REGION=thorax           sbatch sbatch/koalai_train_seg.sh   # fold 0 = center-wise (default)
#   REGION=thorax FOLD=1    sbatch sbatch/koalai_train_seg.sh   # fold 1 = random split

set -eo pipefail
REGION=${REGION:?REGION env var is required (abdomen|thorax|head_neck|brain|pelvis)}
FOLD=${FOLD:-0}

declare -A SYN_IDS=( [abdomen]=960 [thorax]=962 [head_neck]=964 [brain]=966 [pelvis]=968 )
declare -A SEG_IDS=( [abdomen]=961 [thorax]=963 [head_neck]=965 [brain]=967 [pelvis]=969 )

if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="koalai_seg_${REGION}_f${FOLD}" --export=ALL,REGION="$REGION",FOLD="$FOLD" "$0"
    exit
fi

cd /home/minsukc/MRI2CT/baselines/koalAI-seg
source /home/minsukc/MRI2CT/sbatch/koalai_env.sh

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate koalai

SYN_ID=${SYN_IDS[$REGION]}
SEG_ID=${SEG_IDS[$REGION]}

SPLIT_TAG=$([ "$FOLD" = "0" ] && echo centerwise || echo random)

export KOALAI_WANDB=1
export KOALAI_REGION="$REGION"
export KOALAI_STAGE="seg"
export KOALAI_FOLD="$FOLD"
export KOALAI_SPLIT="$SPLIT_TAG"
export KOALAI_RUN_NAME="$(date +%Y%m%d_%H%M)_koalai_${REGION}_seg_f${FOLD}_${SPLIT_TAG}"
export KOALAI_IMAGE_LOG_EVERY=${KOALAI_IMAGE_LOG_EVERY:-25}
# GPU-aug variant: heavy work runs on the GPU in train_step. Workers only do
# load+crop+tensorize, but per-batch GPFS read is the new bottleneck — measured
# at /tmp/deep-check/iter_gpu.log as median next(dl_tr)=670ms with 3 workers vs
# fwd_bwd=547ms. Use all available CPUs minus one for the main thread.
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-$((SLURM_CPUS_PER_TASK - 1))}

# Forward SIGUSR1 from the batch shell (set via --signal=B:USR1@120) to the python
# child. Without the explicit forward + wait, bash falls through `wait` after the
# trap fires and exits, causing SLURM to tear down the cgroup before python's
# `scontrol requeue` subprocess completes. Verified end-to-end on this cluster via
# /home/minsukc/MRI2CT/slurm_logs/requeue_test_50710694.log.
_forward_usr1() {
    echo "[sbatch] caught SIGUSR1, forwarding to python pid=$_TRAIN_PID"
    kill -USR1 "$_TRAIN_PID" 2>/dev/null
    wait "$_TRAIN_PID" 2>/dev/null
    echo "[sbatch] python exited cleanly after signal"
    exit 0
}
trap _forward_usr1 USR1

nnUNetv2_train "$SEG_ID" 3d_fullres "$FOLD" \
    -tr nnUNetTrainer_seg_gpu_aug \
    -p "nnUNetResEncUNetLPlans_Dataset${SYN_ID}" \
    --c &
_TRAIN_PID=$!
wait $_TRAIN_PID
