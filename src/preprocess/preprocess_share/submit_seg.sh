#!/bin/bash
#SBATCH --job-name=totalseg
#SBATCH --account=jjparkcv98
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=07:00:00
#SBATCH --array=0-9
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/%u/logs/%x-%A_%a.log

SCRIPT_PATH="src/preprocess_share/totalsegmentator_batch_sharded.py"
python $SCRIPT_PATH --part $SLURM_ARRAY_TASK_ID --total_parts 10
