#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=14:00:00
#SBATCH --array=0-3
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%A_%a_%x.log

# CADS stage 2 (GPU): run all 9 tasks on preprocessed CTs, 4-way array (~8.5h/shard).
# Requires stage 1 to be complete.
#   sbatch sbatch/cads_stage2_inference.sh

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="cads_stage2" "$0"
    exit
fi

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
cd /home/minsukc/MRI2CT

python src/cads/stage2_inference.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --threads "${SLURM_CPUS_PER_TASK}"
