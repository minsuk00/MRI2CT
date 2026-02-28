#!/bin/bash
#SBATCH --job-name=totalseg
#SBATCH --account=jjparkcv98
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/%u/logs/%x-%A_%a.log

# Ensure the directory /home/minsukc/logs/ exists
mkdir -p /home/minsukc/logs/

# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mri2ct_home

python src/preprocess/totalsegmentator_batch_sharded.py --part $SLURM_ARRAY_TASK_ID --total_parts 10

# ls -lt /home/minsukc/logs/ | head -n 10
# tail -f /home/minsukc/logs/totalseg-123456_0.log
# grep -i "error\|fail" /home/minsukc/logs/totalseg-123456_*.log