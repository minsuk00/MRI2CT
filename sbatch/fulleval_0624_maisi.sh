#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=04:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# full_eval_20260624: MAISI run ahzkny1u advanced to milestone maisi_epoch05600.pt
# (epoch 5600, samples_seen 2,240,000 = 0.70x of the 3.2M budget; still training).
# The 0617 report used the same run at ep1800 (0.22x), so this is a large advance.
PREFIX="fulleval_0624_maisi"
EV=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260624
CHECKPOINT="$EV/pinned_checkpoints/maisi_ahzkny1u_ep5600.pt"
OUT_DIR="$EV/raw/maisi"
SPLIT_FILE="splits/center_wise_split.txt"
SPLIT_NAME="val"
NUM_INFERENCE_STEPS=30
VAL_SW_OVERLAP=0.4

if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    echo "🚀 Submitting: $PREFIX"
    sbatch --job-name="$PREFIX" \
           --output="/home/minsukc/MRI2CT/slurm_logs/${TIMESTAMP}_${PREFIX}_%j.log" "$0"
    exit
fi

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
cd /home/minsukc/MRI2CT
mkdir -p "$OUT_DIR"
python src/maisi_baseline/validate.py \
    --checkpoint "$CHECKPOINT" \
    --split_file "$SPLIT_FILE" --split_name "$SPLIT_NAME" \
    --num_inference_steps "$NUM_INFERENCE_STEPS" \
    --val_sw_overlap "$VAL_SW_OVERLAP" \
    --out_dir "$OUT_DIR"
