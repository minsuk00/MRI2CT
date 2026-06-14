#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=08:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# nnsyn plan_and_preprocess for one region (synth + seg datasets), then
# overwrite splits_final.json with our fold-0 split.
# Usage: REGION=thorax bash sbatch/koalai_preprocess.sh

set -eo pipefail
REGION=${REGION:?REGION env var is required (abdomen|thorax|head_neck|brain|pelvis|all)}

declare -A SYN_IDS=( [abdomen]=960 [thorax]=962 [head_neck]=964 [brain]=966 [pelvis]=968 )
declare -A SEG_IDS=( [abdomen]=961 [thorax]=963 [head_neck]=965 [brain]=967 [pelvis]=969 )

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="koalai_pp_${REGION}" --export=ALL,REGION="$REGION" "$0"
    exit
fi

cd /home/minsukc/MRI2CT
source sbatch/koalai_env.sh

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate koalai

run_one() {
    local r=$1
    local sid=${SYN_IDS[$r]}
    local gid=${SEG_IDS[$r]}
    export nnsyn_origin_dataset="$NNSYN_ORIGIN_ROOT/synthrad2025_task1_mri2ct_${r}"

    # Clean any stale Dataset96X folders from prior failed runs. nnsyn's synth
    # preprocess creates a temp Dataset{sid+1}_* that it rmtrees on success;
    # if it failed mid-flight, both Dataset{sid+1}_* (temp) and the persistent
    # Dataset{gid}_SEG_* can co-exist, tripping the duplicate-id check.
    for base in "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results"; do
        rm -rf "$base/Dataset${sid}_synthrad2025_task1_mri2ct_${r}" \
               "$base/Dataset${gid}_synthrad2025_task1_mri2ct_${r}" \
               "$base/Dataset${gid}_SEG_synthrad2025_task1_mri2ct_${r}"
    done

    echo "=== [$r] synth plan_and_preprocess (d=$sid) ==="
    nnsyn_plan_and_preprocess -d "$sid" -c 3d_fullres \
        -pl nnUNetPlannerResEncL -p nnUNetResEncUNetLPlans \
        --preprocessing_input MR --preprocessing_target CT --use_mask

    echo "=== [$r] seg plan_and_preprocess (dseg=$gid) ==="
    nnsyn_plan_and_preprocess_seg -d "$sid" -dseg "$gid" \
        -c 3d_fullres -p nnUNetResEncUNetLPlans

    echo "=== [$r] override splits_final.json ==="
    python baselines/koalAI/mri2ct_scripts/nnsyn_write_splits.py --region "$r"
}

if [ "$REGION" = "all" ]; then
    for r in abdomen thorax head_neck brain pelvis; do run_one "$r"; done
else
    run_one "$REGION"
fi
