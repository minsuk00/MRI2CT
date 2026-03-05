#!/bin/bash
#SBATCH --job-name=mri2ct_train
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%x-%j.log

# Ensure the log directory exists
mkdir -p /home/minsukc/MRI2CT/slurm_logs/


# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

SCRIPT="src/mri2ct/train.py"

# Arguments
DICE_W=0.0
# RESUME_ID="your_wandb_id"

# Construct the command
CMD="python $SCRIPT --dice_w $DICE_W"

if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi

echo "Running command: $CMD"
$CMD
