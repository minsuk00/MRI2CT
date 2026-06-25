#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%A_%a_%x.log

# TotalSegmentator body task on the masked flat CTs -> <subj>/totalseg_body_mask.nii.gz
# (1=body_trunc, 2=body_extremities). Native 1.5 mm (fast=False), sharded x8.
# Use FORCE=1 to overwrite existing body masks.

if [ -z "$SLURM_JOB_ID" ]; then
    JOB_NAME="body_mask_flat"
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="$JOB_NAME" "$0"
    exit
fi

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
cd /home/minsukc/MRI2CT

# Sanity-check: TotalSegmentator only produces valid labels with UPSTREAM nnunetv2.
python -c "import nnunetv2; \
  assert 'baselines/koalAI' not in nnunetv2.__file__, \
  'nnsyn fork is active. Run: pip install nnunetv2==2.5.2 --force-reinstall  (then re-submit)'"

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

python baselines/koalAI/mri2ct_scripts/body_mask_for_flat.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    $FORCE_FLAG
