#!/bin/bash
set -u
cd /home/minsukc/MRI2CT
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
  micromamba run -n elastix python notebooks/reg_compare/run_compare.py --id "$1" --region "$2"
  echo "=== $(date +%H:%M:%S) DONE  $1 ==="
done
echo "ALL DONE $(date +%H:%M:%S)"
