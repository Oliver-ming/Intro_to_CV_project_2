#!/usr/bin/env bash
# Same validation-tail scoring as the 448 run, at the 672 resolution used for multi-scale fusion.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalydino_heldout_672.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYDINO_HELDOUT_672_START $(date -Is)"
"$PY" scripts/score_anomalydino_heldout.py \
  --device "${DEVICE:-cuda:7}" \
  --resolution 672 \
  --out-dir "$ROOT/outputs/part2/anomalydino_heldout/672" \
  --shots 1 2 4 8 \
  --num-seeds 3
echo "ANOMALYDINO_HELDOUT_672_DONE $(date -Is)"
