#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16g
#SBATCH --time=02:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%A_%a_%x.log

# CADS stage 3 (CPU): restore segmentations to original CT geometry, 8-way array.
# Requires stage 2 to be complete.
#   sbatch sbatch/cads_stage3_restore.sh

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="cads_stage3" "$0"
    exit
fi

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
cd /home/minsukc/MRI2CT

python src/cads/stage3_restore.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    --threads "${SLURM_CPUS_PER_TASK}"
