#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=48:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL

# --- Configuration Area ---
PREFIX="main"
RESUME_ID=""  # Leave empty if not resuming

DICE_W=0.05
# AUGMENT="True" # "True" or "False"
AUGMENT="False"
# RESUME_ID="4lyodgtl"  # 0.05 dice

# DICE_W=0.0
# RESUME_ID="jxk30spy"  # 0.0 dice

# --- Self-Submission Logic ---
if [ -z "$SLURM_JOB_ID" ]; then
    # Construct descriptive Job Name
    JOB_NAME="${PREFIX}_dice-${DICE_W}_aug-${AUGMENT}"
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
CMD="python $SCRIPT --dice_w $DICE_W --augment $AUGMENT"
if [ ! -z "$RESUME_ID" ]; then
    CMD="$CMD --resume_id $RESUME_ID"
fi

echo "Running command: $CMD"
$CMD

# python src/mri2ct/train.py --dice_w 0.05 --resume_id "4lyodgtl"
# python src/mri2ct/train.py --dice_w 0.0 --resume_id "jxk30spy"