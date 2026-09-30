#!/usr/bin/env bash
# Score a category only after its patch grids are all on disk.
# Wait 15 minutes between checks. Stop when the three AnomalyDINO map jobs
# have printed their own completion markers.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/score_finished.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
while true; do
  set +e
  "$PY" -m src.part2.score_finished_maps --stop-when-maps-done >>"$LOG" 2>&1
  code=$?
  set -e
  if [[ "$code" -eq 0 ]]; then
    echo "MAP_SCORER_DONE $(date -Is)" >>"$LOG"
    break
  fi
  sleep 900
done
