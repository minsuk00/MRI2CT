#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Predict full-volume sCTs using a trained synthesis model for one fold + subset.
# Reads subject IDs from the fold's subjects.json under the chosen subset key
# (val for honest validation metrics, test for held-out evaluation) and links
# the matching INPUT_IMAGES / MASKS into per-(region,fold,subset) staging dirs
# before invoking nnsyn_predict's sliding-window inference.
# Usage:
#   REGION=thorax                       sbatch sbatch/koalai_predict.sh   # fold 0 center-wise, test (default)
#   REGION=thorax FOLD=1                sbatch sbatch/koalai_predict.sh   # fold 1 random, test
#   REGION=thorax SUBSET=val            sbatch sbatch/koalai_predict.sh   # fold 0 center-wise, val
#   REGION=thorax FOLD=1 SUBSET=val     sbatch sbatch/koalai_predict.sh   # fold 1 random, val

set -eo pipefail
REGION=${REGION:?REGION env var is required (abdomen|thorax|head_neck|brain|pelvis)}
FOLD=${FOLD:-0}
SUBSET=${SUBSET:-test}
if [ "$SUBSET" != "val" ] && [ "$SUBSET" != "test" ]; then
    echo "SUBSET must be 'val' or 'test' (got: $SUBSET)" >&2
    exit 1
fi

declare -A SYN_IDS=( [abdomen]=960 [thorax]=962 [head_neck]=964 [brain]=966 [pelvis]=968 )

if [ -z "$SLURM_JOB_ID" ]; then
    mkdir -p /home/minsukc/MRI2CT/slurm_logs/
    sbatch --job-name="koalai_pred_${REGION}_f${FOLD}_${SUBSET}" \
           --export=ALL,REGION="$REGION",FOLD="$FOLD",SUBSET="$SUBSET" "$0"
    exit
fi

cd /home/minsukc/MRI2CT
source sbatch/koalai_env.sh

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate koalai

SYN_ID=${SYN_IDS[$REGION]}

# Fold → which split's test subjects to predict on, and a label for output dirs.
if [ "$FOLD" = "0" ]; then
    SPLIT_TAG=centerwise
    SUBJECTS_JSON="splits/koalai/${REGION}_subjects.json"
else
    SPLIT_TAG=random
    SUBJECTS_JSON="splits/koalai/${REGION}_random_subjects.json"
fi

ORIGIN_DIR="$NNSYN_ORIGIN_ROOT/synthrad2025_task1_mri2ct_${REGION}"
INPUT_DIR="$NNSYN_WORKSPACE/predict_inputs/${REGION}/fold${FOLD}_${SPLIT_TAG}/${SUBSET}"
MASK_DIR="$NNSYN_WORKSPACE/predict_masks/${REGION}/fold${FOLD}_${SPLIT_TAG}/${SUBSET}"
OUT_DIR="$NNSYN_WORKSPACE/predictions/${REGION}/fold${FOLD}_${SPLIT_TAG}/${SUBSET}"
mkdir -p "$INPUT_DIR" "$MASK_DIR" "$OUT_DIR"

# Build symlink farms of this fold's chosen-subset subjects' MR + mask files
python - <<PYEOF
import json, os
with open("$SUBJECTS_JSON") as f:
    subjects = json.load(f)["$SUBSET"]
for subj in subjects:
    for src_rel, dst_dir in (
        (f"INPUT_IMAGES/{subj}_0000.mha", "$INPUT_DIR"),
        (f"MASKS/{subj}.mha",             "$MASK_DIR"),
    ):
        src = os.path.join("$ORIGIN_DIR", src_rel)
        dst = os.path.join(dst_dir, os.path.basename(src))
        if not os.path.exists(dst):
            os.symlink(src, dst)
print(f"[fold $FOLD / $SPLIT_TAG / $SUBSET] Linked {len(subjects)} subjects from $SUBJECTS_JSON")
PYEOF

# --revert_norm un-z-scores the prediction back to HU (pred*std + mean, bg -> -1000).
# WITHOUT it the output stays in normalized space. HU volumes are written to
# "${OUT_DIR}_revert_norm/" (raw normalized ones moved to .../backup_normalised/).
# => EVALUATION (step 7) MUST read "${OUT_DIR}_revert_norm/", not "${OUT_DIR}".
nnsyn_predict -d "$SYN_ID" -i "$INPUT_DIR" -o "$OUT_DIR" -m "$MASK_DIR" \
    -c 3d_fullres -p nnUNetResEncUNetLPlans \
    -tr nnUNetTrainer_nnsyn_loss_map -f "$FOLD" --revert_norm

echo "[koalai_predict] HU sCTs written to: ${OUT_DIR}_revert_norm/  (use THIS dir for evaluation)"
