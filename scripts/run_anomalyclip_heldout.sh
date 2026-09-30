#!/usr/bin/env bash
# Prompt scores on the MVTec validation tail, after the VisA-trained checkpoint exists.
# Prints ANOMALYCLIP_HELDOUT_DONE when every category has been written.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale/epoch_15.pth"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_heldout.log}"
mkdir -p "$(dirname "$LOG")"
if [[ ! -f "$CKPT" ]]; then
  echo "waiting for $CKPT" >&2
  exit 2
fi
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYCLIP_HELDOUT_START $(date -Is)"
"$PY" scripts/score_anomalyclip_heldout.py \
  --checkpoint "$CKPT" \
  --device "${DEVICE:-cuda:5}"
echo "ANOMALYCLIP_HELDOUT_DONE $(date -Is)"
