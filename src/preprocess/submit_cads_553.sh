#!/bin/bash
#SBATCH --job-name=cads_553_brain
#SBATCH --account=jjparkcv98
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/%u/logs/%x-%A_%a.log

# Ensure the logs directory exists
mkdir -p /home/minsukc/logs/

# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mri2ct_home

# Set CUDA_VISIBLE_DEVICES to the assigned GPU (usually handled by SLURM)
export CUDA_VISIBLE_DEVICES=$(echo $SLURM_JOB_GPUS | cut -d',' -f1)

echo "Running chunk ${SLURM_ARRAY_TASK_ID} of 10 on $(hostname) with GPU ${CUDA_VISIBLE_DEVICES}"

# Run the python script
python src/preprocess/run_cads_553_merge.py \
    --num_chunks 10 \
    --chunk_idx ${SLURM_ARRAY_TASK_ID} \
    --batch_size 4

echo "Job array task ${SLURM_ARRAY_TASK_ID} finished."
