#!/usr/bin/env bash
# Train AnomalyCLIP prompts the way the official script does:
# first on VisA (used when testing MVTec), then on MVTec (used when testing VisA).
# Prints ANOMALYCLIP_TRAIN_DONE when both checkpoints exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_train.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/third_party/AnomalyCLIP"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYCLIP_TRAIN_START $(date -Is) gpu=${CUDA_VISIBLE_DEVICES}"
"$PY" train.py \
  --dataset visa \
  --train_data_path "$ROOT/data/visa_raw" \
  --save_path "$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale" \
  --features_list 24 \
  --image_size 518 \
  --batch_size 8 \
  --epoch 15 \
  --save_freq 1 \
  --print_freq 1 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4
"$PY" train.py \
  --dataset mvtec \
  --train_data_path "$ROOT/data/mvtec_anomaly_detection" \
  --save_path "$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale_visa" \
  --features_list 24 \
  --image_size 518 \
  --batch_size 8 \
  --epoch 15 \
  --save_freq 1 \
  --print_freq 1 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4
echo "ANOMALYCLIP_TRAIN_DONE $(date -Is)"
