#!/usr/bin/env bash
# Test AnomalyCLIP one direction at a time.
# VisA-trained prompts are evaluated on MVTec as soon as that checkpoint exists.
# MVTec-trained prompts are evaluated on VisA when the second checkpoint exists.
# Prints ANOMALYCLIP_TEST_DONE only after both evaluations have metrics.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_test.log}"
VISA_CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale/epoch_15.pth"
MVTEC_CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale_visa/epoch_15.pth"
MVTEC_OUT="$ROOT/outputs/part2/anomalyclip/mvtec_zero_shot"
VISA_OUT="$ROOT/outputs/part2/anomalyclip/visa_zero_shot"
mkdir -p "$(dirname "$LOG")"
if [[ ! -f "$VISA_CKPT" && ! -f "$MVTEC_CKPT" ]]; then
  echo "waiting for $VISA_CKPT" >&2
  exit 2
fi
cd "$ROOT/third_party/AnomalyCLIP"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT")
from pathlib import Path
from src.part2.ensure_anomalyclip_metrics import ensure_anomalyclip_image_scores, ensure_anomalyclip_metrics
print("anomalyclip metrics hook", "inserted" if ensure_anomalyclip_metrics(Path("test.py")) else "already present")
print("anomalyclip image scores", "inserted" if ensure_anomalyclip_image_scores(Path("test.py")) else "already present")
PY
exec > >(tee -a "$LOG") 2>&1

run_one() {
  local dataset="$1" data_path="$2" save_path="$3" checkpoint="$4"
  if [[ ! -f "$checkpoint" ]]; then
    echo "waiting for $checkpoint"
    return 2
  fi
  if [[ -f "$save_path/metrics.json" ]]; then
    echo "skip $dataset"
    return 0
  fi
  echo "ANOMALYCLIP_TEST_START $dataset $(date -Is)"
  "$PY" test.py \
    --dataset "$dataset" \
    --data_path "$data_path" \
    --save_path "$save_path" \
    --checkpoint_path "$checkpoint" \
    --features_list 24 \
    --image_size 518 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4
}

run_one mvtec "$ROOT/data/mvtec_anomaly_detection" "$MVTEC_OUT" "$VISA_CKPT" || true
run_one visa "$ROOT/data/visa_raw" "$VISA_OUT" "$MVTEC_CKPT" || true
if [[ -f "$MVTEC_OUT/metrics.json" && -f "$VISA_OUT/metrics.json" ]]; then
  echo "ANOMALYCLIP_TEST_DONE $(date -Is)"
else
  exit 2
fi
