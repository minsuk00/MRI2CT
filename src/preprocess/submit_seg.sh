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

# NOTE: The %A_%a in the output path helps track individual array task logs.
# Ensure the directory /home/minsukc/logs/ exists or change the path.

python src/preprocess/totalsegmentator_batch_sharded.py --part $SLURM_ARRAY_TASK_ID --total_parts 10

# ls -lt /home/minsukc/logs/ | head -n 10
# tail -f /home/minsukc/logs/totalseg-123456_0.log
# grep -i "error\|fail" /home/minsukc/logs/totalseg-123456_*.log