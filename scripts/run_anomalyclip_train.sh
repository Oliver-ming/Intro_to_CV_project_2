#!/usr/bin/env bash
# Train AnomalyCLIP prompts the way the official script does:
# first on VisA (used when testing MVTec), then on MVTec (used when testing VisA).
# The official Dataset defaults to meta.json["test"]. That is required:
# the train split has no defect masks, and the prompt loss needs them.
# VisA test images are not MVTec images, so the MVTec evaluation stays zero-shot.
# Prints ANOMALYCLIP_TRAIN_DONE when both checkpoints exist.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part2/anomalyclip_train.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/third_party/AnomalyCLIP"
"$PY" - <<'PY'
from pathlib import Path
path = Path("AnomalyCLIP_lib/model_load.py")
text = path.read_text()
old = 'os.path.expanduser("/remote-home/iot_zhouqihang/root/.cache/clip")'
new = 'os.path.expanduser("~/.cache/clip")'
if old in text:
    path.write_text(text.replace(old, new))
PY
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
exec > >(tee -a "$LOG") 2>&1
echo "ANOMALYCLIP_TRAIN_START $(date -Is) gpu=${CUDA_VISIBLE_DEVICES}"

run_train() {
  local dataset="$1" data_path="$2" save_path="$3"
  if [[ -f "$save_path/epoch_15.pth" ]]; then
    echo "skip $dataset, checkpoint exists"
    return 0
  fi
  "$PY" train.py \
    --dataset "$dataset" \
    --train_data_path "$data_path" \
    --save_path "$save_path" \
    --features_list 24 \
    --image_size 518 \
    --batch_size 8 \
    --epoch 15 \
    --save_freq 1 \
    --print_freq 1 \
    --depth 9 \
    --n_ctx 12 \
    --t_n_ctx 4
}

run_train visa "$ROOT/data/visa_raw" "$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale"
run_train mvtec "$ROOT/data/mvtec_anomaly_detection" "$ROOT/outputs/part2/anomalyclip/9_12_4_multiscale_visa"
echo "ANOMALYCLIP_TRAIN_DONE $(date -Is)"
