#!/usr/bin/env bash
# Official AnomalyDINO on all 15 MVTec categories.
# Prints ANOMALYDINO_MVTEC_DONE when every shot and seed has been evaluated.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalydino_mvtec.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/third_party/AnomalyDINO"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYDINO_MVTEC_START $(date -Is)"
"$PY" run_anomalydino.py \
  --dataset MVTec \
  --data_root "$ROOT/data/mvtec_anomaly_detection" \
  --model_name dinov2_vits14 \
  --resolution 448 \
  --preprocess agnostic \
  --shots 1 2 4 8 \
  --num_seeds 3 \
  --faiss_on_cpu \
  --eval_clf \
  --eval_segm \
  --device cuda:1 \
  --warmup_iters 1
echo "ANOMALYDINO_MVTEC_DONE $(date -Is)"
