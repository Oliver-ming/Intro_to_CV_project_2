#!/usr/bin/env bash
# Official AnomalyDINO on VisA. Uses GPU 3 so it does not share a device
# with the MVTec run. Prints ANOMALYDINO_VISA_DONE when evaluation finishes.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalydino_visa.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/third_party/AnomalyDINO"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYDINO_VISA_START $(date -Is)"
"$PY" run_anomalydino.py \
  --dataset VisA \
  --data_root "$ROOT/data/VisA_pytorch/1cls" \
  --model_name dinov2_vits14 \
  --resolution 448 \
  --preprocess agnostic \
  --shots 1 2 4 8 \
  --num_seeds 3 \
  --faiss_on_cpu \
  --eval_clf \
  --eval_segm \
  --device cuda:3 \
  --warmup_iters 1
echo "ANOMALYDINO_VISA_DONE $(date -Is)"
