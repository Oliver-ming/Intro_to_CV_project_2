#!/usr/bin/env bash
# Score the defect-free validation tail with AnomalyDINO's official references.
# Prints ANOMALYDINO_HELDOUT_DONE when every requested shot and seed is written.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalydino_heldout.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYDINO_HELDOUT_START $(date -Is)"
"$PY" scripts/score_anomalydino_heldout.py \
  --device "${DEVICE:-cuda:6}" \
  --resolution "${RESOLUTION:-448}" \
  --shots 1 2 4 8 \
  --num-seeds 3
echo "ANOMALYDINO_HELDOUT_DONE $(date -Is)"
