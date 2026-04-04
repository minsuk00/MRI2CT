#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%A_%a_encode.log
#SBATCH --job-name=maisi_encode

NUM_NODES=5
OUTPUT_DIR="/home/minsukc/MRI2CT/dataset/1.5mm_registered_maisi_encoding"

cd /home/minsukc/MRI2CT
micromamba run -n mrct python src/maisi_baseline/encode_all_volumes.py \
    --output_dir "$OUTPUT_DIR" \
    --node_id "$SLURM_ARRAY_TASK_ID" \
    --num_nodes "$NUM_NODES" \
    --force
