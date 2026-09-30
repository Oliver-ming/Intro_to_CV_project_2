#!/usr/bin/env bash
# Fuse AnomalyDINO distances with AnomalyCLIP prompt scores on MVTec.
# Exits 2 until both the test scores and the validation-tail scores exist.
# Prints PROMPT_FUSION_DONE when all 15 categories have a fused file.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
PROMPT_SCORES="${PROMPT_SCORES:-$ROOT/outputs/part2/anomalyclip/mvtec_zero_shot/image_scores.json}"
PROMPT_VAL="${PROMPT_VAL:-$ROOT/outputs/part2/anomalyclip_heldout/mvtec}"
VISUAL_MEASUREMENTS="${VISUAL_MEASUREMENTS:-$ROOT/third_party/AnomalyDINO/results_MVTec/dinov2_vits14_448/1-shot_preprocess=agnostic/measurements_seed=0.csv}"
VISUAL_VAL="${VISUAL_VAL:-$ROOT/outputs/part2/anomalydino_heldout/448/1-shot/seed=0}"
OUT="${OUT:-$ROOT/outputs/part3/prompt_fusion/1shot_seed0}"
if [[ ! -f "$PROMPT_SCORES" ]]; then
  echo "waiting for $PROMPT_SCORES" >&2
  exit 2
fi
cd "$ROOT"
"$PY" - <<PY
from pathlib import Path
from src.inspection.categories import MVTEC_CATEGORIES
from src.part2.build_fusion import fuse_ready_categories
out = Path("$OUT")
written = fuse_ready_categories(
    list(MVTEC_CATEGORIES),
    Path("$VISUAL_MEASUREMENTS"),
    Path("$VISUAL_VAL"),
    Path("$PROMPT_SCORES"),
    Path("$PROMPT_VAL"),
    Path("$ROOT/data/mvtec_anomaly_detection"),
    out,
)
done = [category for category in MVTEC_CATEGORIES if (out / f"{category}.json").is_file()]
print({"written": written, "done": len(done)})
if len(done) != len(MVTEC_CATEGORIES):
    raise SystemExit(2)
print("PROMPT_FUSION_DONE")
PY
