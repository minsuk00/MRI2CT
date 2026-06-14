#!/bin/bash
#SBATCH --account=jjparkcv98
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=08:00:00
#SBATCH --mail-user=minsukc@umich.edu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --output=/home/minsukc/MRI2CT/slurm_logs/%j_%x.log

# Three-way deformable-registration comparison (anatomix vs elastix MR->CT vs elastix CT->MR)
# for 2 subjects x 5 regions. Elastix is CPU-only; runs the SynthRAD configs verbatim.
# Volumes -> GPFS (reg_compare_vols/), PNGs -> notebooks/reg_compare/out/.
# Usage: sbatch sbatch/reg_compare_sweep.sh

set -u
cd /home/minsukc/MRI2CT
mkdir -p /home/minsukc/MRI2CT/slurm_logs/

export MAMBA_EXE='/home/minsukc/.local/bin/micromamba'
export MAMBA_ROOT_PREFIX='/home/minsukc/micromamba'
eval "$("$MAMBA_EXE" shell hook --shell bash)"
micromamba activate elastix

declare -a JOBS=(
  "1THA001 thorax" "1THA002 thorax"
  "1ABA005 abdomen" "1ABA009 abdomen"
  "1HNA001 head_neck" "1HNA004 head_neck"
  "1BA001 brain" "1BA005 brain"
  "1PA001 pelvis" "1PA004 pelvis"
)
for j in "${JOBS[@]}"; do
  set -- $j
  echo "=== $(date +%H:%M:%S) START $1 ($2) ==="
  python notebooks/reg_compare/run_compare.py --id "$1" --region "$2"
  echo "=== $(date +%H:%M:%S) DONE  $1 (exit $?) ==="
done
echo "ALL DONE $(date +%H:%M:%S)"
