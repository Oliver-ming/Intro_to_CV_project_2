#!/usr/bin/env bash
# Part 1 difference on all 12 VisA categories.
# One OpenCV thread so this does not take the CPUs the AnomalyDINO jobs are using.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
DATA="${DATA:-data/VisA_pytorch/1cls}"
LOG="${LOG:-outputs/part1/visa_difference.log}"
mkdir -p outputs/part1 outputs/splits
exec > >(tee -a "$LOG") 2>&1
echo "PART1_VISA_DIFFERENCE_START $(date -Is)"
"$PY" -m src.inspection.build_splits --dataset visa --data-root "$DATA" --out-dir outputs/splits/visa
export CV2_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
"$PY" -m src.part1.run_difference \
  --data-root "$DATA" \
  --split-dir outputs/splits/visa \
  --out-dir outputs/part1/visa_difference
echo "PART1_VISA_DIFFERENCE_DONE $(date -Is)"
