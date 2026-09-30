#!/usr/bin/env bash
# Start AnomalyCLIP evaluation when a finished checkpoint appears.
# Checks every 15 minutes. The test script itself refuses to run on a
# partial epoch. After the MVTec metrics and the validation-tail scores
# exist, fuse them with the 1-shot visual distances.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_watch.log}"
VISA_CKPT="$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale/epoch_15.pth"
MVTEC_METRICS="$ROOT/outputs/part2/anomalyclip/mvtec_zero_shot/metrics.json"
HELD_LOG="$ROOT/outputs/part2/anomalyclip_heldout.log"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec 9>"$ROOT/outputs/part2/anomalyclip_watch.lock"
if ! flock -n 9; then
  echo "anomalyclip watch already running" >&2
  exit 0
fi
while true; do
  set +e
  bash "$ROOT/scripts/run_anomalyclip_test.sh" >>"$LOG" 2>&1
  if [[ -f "$VISA_CKPT" && -f "$MVTEC_METRICS" ]]; then
    if [[ ! -f "$HELD_LOG" ]] || ! grep -q ANOMALYCLIP_HELDOUT_DONE "$HELD_LOG"; then
      bash "$ROOT/scripts/run_anomalyclip_heldout.sh" >>"$LOG" 2>&1
    fi
    bash "$ROOT/scripts/run_prompt_fusion.sh" >>"$LOG" 2>&1
    fusion=$?
    if [[ "$fusion" -eq 0 ]]; then
      echo "ANOMALYCLIP_WATCH_DONE $(date -Is)" >>"$LOG"
      exit 0
    fi
  fi
  set -e
  sleep 900
done
