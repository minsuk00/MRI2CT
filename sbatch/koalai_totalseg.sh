#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=12:00:00
#SBATCH --array=0-7
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%A_%a_%x.log

# Run TotalSegmentator on all 844 subjects (sharded across 8 array tasks).
# Use FORCE=1 to overwrite the 33 subjects with existing labels.

if [ -z "$SLURM_JOB_ID" ]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    JOB_NAME="koalai_totalseg"
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
# The nnsyn fork modifies the predictor for synthesis output → TotalSeg labels become invalid.
python -c "import nnunetv2; \
  assert 'baselines/koalAI' not in nnunetv2.__file__, \
  'nnsyn fork is active. Run: pip install nnunetv2==2.5.2 --force-reinstall  (then re-submit)'"

FORCE_FLAG=""
if [ "${FORCE:-0}" = "1" ]; then FORCE_FLAG="--force"; fi

python baselines/koalAI/mri2ct_scripts/totalseg_for_koalai.py \
    --shard "${SLURM_ARRAY_TASK_ID}/${SLURM_ARRAY_TASK_COUNT}" \
    $FORCE_FLAG
