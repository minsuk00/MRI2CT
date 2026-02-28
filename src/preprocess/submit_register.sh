#!/bin/bash
#SBATCH --job-name=mri2ct_reg
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=3:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/%u/logs/%x-%A_%a.log

# Ensure the log directory exists
mkdir -p /home/minsukc/logs/

# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

# Run the sharded registration script (10 parts total)
python src/preprocess/register_batch_sharded_isolated.py --part $SLURM_ARRAY_TASK_ID --total_parts 10

# -----------------------------------------------------------------------------
# HELPER COMMANDS (Run these manually after all jobs finish):
#
# 1. Merge all failure logs:
#    cat register_failures_part_*.txt > all_failures.txt
#
#
# 2. Check for overall progress:
#    ls -1 /gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/SynthRAD_combined/native_masked/*/moved_mr_*.nii | wc -l
# -----------------------------------------------------------------------------

# 217 fails
# 134 fails

# 133 fails

# 740 succcess