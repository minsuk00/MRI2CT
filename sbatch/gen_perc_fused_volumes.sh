#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --gpu_cmode=shared
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=48g
#SBATCH --time=12:00:00
#SBATCH --job-name=gen_perc_fused_volumes
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Generate sCT volumes (sample.nii.gz, HU) for the 4-way fused-LNCC perceptual ablation,
# all at the parity epoch 300 (equal samples-seen: bs8x500 == bs4x1000 == 4000 patches/ep).
#   nbn71048  unet, no-perc  (L1+SSIM+Dice)                  -> src/unet_baseline/validate.py
#   827la6dp  amix v1.4, no-perc (anatomix backbone, perc=0) -> src/amix/validate.py
#   mwrwxvvu  unet, perceptual LNCC (Python separable)       -> src/unet_baseline/validate.py
#   fxudaqcp  unet, perceptual LNCC (fused CUDA kernel)      -> src/unet_baseline/validate.py
# Teacher is OFF here; score_perc_ablation.py re-runs the teacher for Hard Bone Dice.
# Output tree matches the existing perc_ablation layout: <RAW>/<tag>/<subj>/sample.nii.gz.

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash --root-prefix "$MAMBA_ROOT_PREFIX")"
micromamba activate mrct

# amix's frozen extractor uses a compiled backward path that needs contiguous cuDNN
# workspace; harmless for the plain U-Net runs.
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd /home/minsukc/MRI2CT

RUNS=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/wandb_logs/runs
RAW=/gpfs/accounts/jjparkcv_root/jjparkcv98/minsukc/MRI2CT/evaluation_results/perc_ablation_fused_20260624/raw
SPLIT=splits/center_wise_split.txt
mkdir -p "$RAW"

run_val () {  # $1=script  $2=checkpoint  $3=tag
    echo "================================================================"
    echo "[gen] $3  <-  $2"
    echo "================================================================"
    python "$1" \
        --checkpoint "$2" \
        --split_file "$SPLIT" \
        --split_name val \
        --out_dir "$RAW/$3" \
        --teacher_weights_path none
}

run_val src/unet_baseline/validate.py "$RUNS/20260611_1957_nbn71048/unet_baseline_epoch00300.pt"      nbn71048_ep300
run_val src/amix/validate.py          "$RUNS/20260611_1957_827la6dp/anatomix_translator_epoch00300.pt" 827la6dp_ep300
run_val src/unet_baseline/validate.py "$RUNS/20260618_0156_mwrwxvvu/unet_baseline_epoch00300.pt"      mwrwxvvu_ep300
run_val src/unet_baseline/validate.py "$RUNS/20260618_2104_fxudaqcp/unet_baseline_epoch00300.pt"      fxudaqcp_ep300

echo "[gen] ✅ all 4 variants done -> $RAW"
