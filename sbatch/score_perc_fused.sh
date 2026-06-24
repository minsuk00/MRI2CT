#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=6:00:00
#SBATCH --job-name=score_perc_fused
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Score the 4-way fused-LNCC perceptual ablation (epoch-300 parity).
# Track-A MAE/PSNR/SSIM (full + body) + Hard Bone/all Dice (Baby-UNet teacher on
# each prediction). Reads <RAW>/<tag>/<subj>/sample.nii.gz, GT from the dataset.
# Emits metrics/{per_subject,by_region,overall}.csv.

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/minsukc/MRI2CT

EVAL_ROOT=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/perc_ablation_fused_20260624

python src/evaluate/score_perc_ablation.py \
    --raw_dir "$EVAL_ROOT/raw" \
    --tags nbn71048_ep300 827la6dp_ep300 mwrwxvvu_ep300 fxudaqcp_ep300 \
    --out_dir "$EVAL_ROOT/metrics" \
    --split_file splits/center_wise_split.txt \
    --split_name val

echo "[score] ✅ done -> $EVAL_ROOT/metrics"
