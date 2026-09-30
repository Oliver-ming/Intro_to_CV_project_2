#!/usr/bin/env bash
# Train the Part 1 autoencoder on every MVTec category.
# Safe to overlap the difference job: this uses a GPU, difference uses the CPU.
# A later invocation skips categories whose scores.json already exists.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="${PY:-/home/zhangyuming/CV_project/env_cv/bin/python}"
LOG="${LOG:-$ROOT/outputs/part1/autoencoder.log}"
mkdir -p "$(dirname "$LOG")"
cd "$ROOT"
exec > >(tee -a "$LOG") 2>&1
echo "AUTOENCODER_START $(date -Is)"
"$PY" -m src.part1.train_ae --device "${DEVICE:-cuda:0}"
echo "AUTOENCODER_DONE $(date -Is)"
