#!/usr/bin/env bash
# Part 1 on all 15 MVTec categories. Prints PART1_DONE when both methods finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
DATA="${DATA:-data/mvtec_anomaly_detection}"
LOG="${LOG:-outputs/part1/run.log}"
mkdir -p outputs/part1
exec > >(tee -a "$LOG") 2>&1
echo "PART1_START $(date -Is)"
"$PY" -m src.inspection.build_splits --data-root "$DATA"
"$PY" -m src.part1.run_difference --data-root "$DATA"
"$PY" -m src.part1.train_ae --data-root "$DATA"
echo "PART1_DONE $(date -Is)"
