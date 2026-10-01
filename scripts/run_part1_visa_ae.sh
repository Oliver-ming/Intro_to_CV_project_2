#!/usr/bin/env bash
# Part 1 autoencoder on all 12 VisA categories. Uses one free GPU.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
DATA="${DATA:-data/VisA_pytorch/1cls}"
LOG="${LOG:-outputs/part1/visa_autoencoder.log}"
mkdir -p outputs/part1 outputs/splits
exec > >(tee -a "$LOG") 2>&1
echo "PART1_VISA_AE_START $(date -Is)"
"$PY" -m src.inspection.build_splits --dataset visa --data-root "$DATA" --out-dir outputs/splits/visa
"$PY" -m src.part1.train_ae \
  --data-root "$DATA" \
  --split-dir outputs/splits/visa \
  --out-dir outputs/part1/visa_autoencoder \
  --device "${DEVICE:-cuda:0}"
echo "PART1_VISA_AE_DONE $(date -Is)"
