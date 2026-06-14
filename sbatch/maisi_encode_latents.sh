#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=06:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_maisi_encode_latents.log

# Regenerate MAISI CT latents with the dynamic_infer fix (no oversized-ROI padding).
# Writes to the default output dir (= the path the trainer reads):
#   .../1.5mm_registered_flat_masked_maisi_latents
# Encodes train+val of center_wise_split.txt.

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/minsukc/MRI2CT
python src/maisi_baseline/encode_all_volumes.py \
    --split_file splits/center_wise_split.txt \
    --splits train val
