"""Runtime numbers already written by AnomalyDINO, plus a GPU-memory helper."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def summarize_measurements(csv_path: Path) -> dict[str, float]:
    """Mean memory-bank build time and per-image scoring time, in seconds."""
    bank_times = []
    score_times = []
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            bank_times.append(float(row["MemoryBank_Time"]))
            score_times.append(float(row["Inference_Time"]))
    if not score_times:
        raise ValueError(f"no measurements in {csv_path}")
    return {
        "n_images": float(len(score_times)),
        "memory_bank_seconds_mean": sum(bank_times) / len(bank_times),
        "scoring_seconds_mean": sum(score_times) / len(score_times),
    }


def reset_gpu_peak(device: str = "cuda:0") -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_mb(device: str = "cuda:0") -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
