#!/bin/bash
# Light CPU work — convert NIfTI → MHA into nnsyn origin layout.
# Usage: REGION=thorax bash sbatch/koalai_convert.sh
#        REGION=all bash sbatch/koalai_convert.sh  (all 5 regions sequentially)

set -eo pipefail
cd /home/minsukc/MRI2CT
source sbatch/koalai_env.sh

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate koalai

REGION=${REGION:-all}

KS=baselines/koalAI/mri2ct_scripts

# Generate per-region splits first if needed.
python "$KS/make_region_splits.py"

python "$KS/nnsyn_convert_to_mha.py" --region "$REGION"
