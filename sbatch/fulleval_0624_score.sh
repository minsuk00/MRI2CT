#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64g
#SBATCH --time=06:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# full_eval_20260624: scoring pipeline. Only THREE models are scored fresh —
#   maisi (NEW ep5600), amix_bw4, unet_bw4 (NEW dice_bone_w 0.4 runs).
# The other FIVE (amix, unet, mcddpm, cwdm, koalAI) have byte-identical predictions
# to full_eval_20260617 (symlinked raw/volumes) and the Dice/teacher math is unchanged,
# so their per-subject scores are spliced in from 0617 rather than recomputed (~62% less GPU).
# Submit with: sbatch --dependency=afterok:<maisi_jobid> sbatch/fulleval_0624_score.sh
EVAL=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260624
REUSE_CSV=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/full_eval_20260617/metrics/per_subject.csv

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct
export PYTORCH_ALLOC_CONF=expandable_segments:True
cd /home/minsukc/MRI2CT
set -x

# 1) Assemble unified per-model volume tree (symlinks; koalAI volumes pre-linked from 0617).
python src/evaluate/assemble_unified_volumes.py --eval_root "$EVAL"

# 2) Score ONLY the three new/changed models (both tracks + Bone/all-class Dice).
python src/evaluate/score_all_models.py --eval_root "$EVAL" \
    --models maisi amix_bw4 unet_bw4

# 3) Splice in the five unchanged models' scores from 0617 and re-aggregate to 8 models.
python src/evaluate/merge_per_subject.py --eval_root "$EVAL" \
    --reuse_csv "$REUSE_CSV" \
    --reuse_models amix unet mcddpm cwdm koalAI

# 4a) Error-bar metric figures (Track A/B + per-region).
python src/evaluate/plot_metric_bars.py --eval_root "$EVAL"

# 4b) Qualitative figures (1 representative subject per region x 8 models).
python src/evaluate/visualize_full_eval.py --eval_root "$EVAL"

# 5) Self-contained HTML report (+ repo copy).
python src/evaluate/build_eval_report.py --eval_root "$EVAL" \
    --repo_copy _html/full_eval_20260624.html

echo "[fulleval_0624_score] DONE -> $EVAL/report.html and _html/full_eval_20260624.html"
