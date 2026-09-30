#!/usr/bin/env bash
# Test AnomalyCLIP after both prompt checkpoints exist.
# VisA-trained prompts are evaluated on MVTec, and MVTec-trained prompts on VisA.
# Prints ANOMALYCLIP_TEST_DONE when both evaluations finish.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_test.log}"
VISA_CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale/epoch_15.pth"
MVTEC_CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale_visa/epoch_15.pth"
mkdir -p "$(dirname "$LOG")"
if [[ ! -f "$VISA_CKPT" || ! -f "$MVTEC_CKPT" ]]; then
  echo "waiting for $VISA_CKPT and $MVTEC_CKPT" >&2
  exit 2
fi
cd "$ROOT/third_party/AnomalyCLIP"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"
"$PY" - <<PY
import sys
sys.path.insert(0, "$ROOT")
from pathlib import Path
from src.part2.ensure_anomalyclip_metrics import ensure_anomalyclip_metrics
changed = ensure_anomalyclip_metrics(Path("test.py"))
print("anomalyclip metrics hook", "inserted" if changed else "already present")
PY
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYCLIP_TEST_START $(date -Is)"
"$PY" test.py \
  --dataset mvtec \
  --data_path "$ROOT/data/mvtec_anomaly_detection" \
  --save_path "$ROOT/outputs/part2/anomalyclip/mvtec_zero_shot" \
  --checkpoint_path "$VISA_CKPT" \
  --features_list 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4
"$PY" test.py \
  --dataset visa \
  --data_path "$ROOT/data/visa_raw" \
  --save_path "$ROOT/outputs/part2/anomalyclip/visa_zero_shot" \
  --checkpoint_path "$MVTEC_CKPT" \
  --features_list 24 \
  --image_size 518 \
  --depth 9 \
  --n_ctx 12 \
  --t_n_ctx 4
echo "ANOMALYCLIP_TEST_DONE $(date -Is)"
