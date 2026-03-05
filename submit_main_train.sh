#!/bin/bash
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

# --- Configuration Area ---
PREFIX="main"
DICE_W=0.05
RESUME_ID=""  # Leave empty if not resuming

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    # Construct descriptive Job Name
    JOB_NAME="${PREFIX}_dice-${DICE_W}"
    if [ ! -z "$RESUME_ID" ]; then
        JOB_NAME="${JOB_NAME}_res-${RESUME_ID}"
    fi

    # Ensure log directory exists
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/

    # Re-submit this script with the dynamic names
    echo "🚀 Submitting: $JOB_NAME"
    sbatch --job-name="$JOB_NAME" \
           --output="/home/minsukc/MRI2CT/slurm_logs/${JOB_NAME}_%j.log" \
           "$0"
    exit
fi

# --- Training Logic (Runs only on the GPU Node) ---
# Load micromamba environment
export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

SCRIPT="src/mri2ct/train.py"

# Construct the command
CMD="python $SCRIPT --dice_w $DICE_W"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi

echo "Running command: $CMD"
$CMD
