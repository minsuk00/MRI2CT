#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48g
#SBATCH --time=96:00:00
#SBATCH --requeue
#SBATCH --signal=B:USR1@120
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Train MR→CT synthesis with MAP loss (Dataset<SYN_ID>) for one region.
# Requires the seg model (Dataset<SEG_ID>) to be trained first for the same fold.
# Usage:
#   REGION=thorax           sbatch sbatch/koalai_train_synth.sh   # fold 0 = center-wise (default)
#   REGION=thorax FOLD=1    sbatch sbatch/koalai_train_synth.sh   # fold 1 = random split

set -eo pipefail
REGION=${REGION:?REGION env var is required (abdomen|thorax|head_neck|brain|pelvis)}
FOLD=${FOLD:-0}

declare -A SYN_IDS=( [abdomen]=960 [thorax]=962 [head_neck]=964 [brain]=966 [pelvis]=968 )

if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="koalai_synth_${REGION}_f${FOLD}" --export=ALL,REGION="$REGION",FOLD="$FOLD" "$0"
    exit
fi

cd /home/minsukc/MRI2CT/baselines/koalAI
source /home/minsukc/MRI2CT/sbatch/koalai_env.sh

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate koalai

SYN_ID=${SYN_IDS[$REGION]}

SPLIT_TAG=$([ "$FOLD" = "0" ] && echo centerwise || echo random)

export KOALAI_WANDB=1
export KOALAI_REGION="$REGION"
export KOALAI_STAGE="synth"
export KOALAI_FOLD="$FOLD"
export KOALAI_SPLIT="$SPLIT_TAG"
export KOALAI_RUN_NAME="$(date +%Y%m%d_%H%M)_koalai_${REGION}_synth_f${FOLD}_${SPLIT_TAG}"
# Cap nnUNet DA workers to match cgroup-allocated CPUs (os.cpu_count() reports node total, not allocation)
export nnUNet_n_proc_DA=${nnUNet_n_proc_DA:-$((SLURM_CPUS_PER_TASK - 1))}

# Forward SIGUSR1 from the batch shell (set via --signal=B:USR1@120) to the python
# child. Same pattern as koalai_train_seg.sh — verified end-to-end via
# /home/minsukc/MRI2CT/slurm_logs/requeue_test_50710694.log.
_forward_usr1() {
    echo "[sbatch] caught SIGUSR1, forwarding to python pid=$_TRAIN_PID"
    kill -USR1 "$_TRAIN_PID" 2>/dev/null
    wait "$_TRAIN_PID" 2>/dev/null
    echo "[sbatch] python exited cleanly after signal"
    exit 0
}
trap _forward_usr1 USR1

nnsyn_train "$SYN_ID" 3d_fullres "$FOLD" \
    -tr nnUNetTrainer_nnsyn_loss_map \
    -p nnUNetResEncUNetLPlans \
    --c &
_TRAIN_PID=$!
wait $_TRAIN_PID
