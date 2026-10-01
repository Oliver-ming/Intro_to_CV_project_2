"""Runtime numbers already written by AnomalyDINO, plus a GPU-memory helper."""

from __future__ import annotations

import csv
from pathlib import Path

import torch


def summarize_measurements(csv_path: Path) -> dict[str, float]:
    """Mean memory-bank build time and per-image scoring time, in seconds.

    The bank time is repeated on every image row, so this mean is weighted
    by how many test images a category has.
    """
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


def summarize_by_category(csv_path: Path) -> dict:
    """One memory-bank time per category, plus the mean scoring time per image."""
    bank_time: dict[str, float] = {}
    score_times: dict[str, list[float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row["Object"]
            bank_time[category] = float(row["MemoryBank_Time"])
            score_times.setdefault(category, []).append(float(row["Inference_Time"]))
    if not score_times:
        raise ValueError(f"no measurements in {csv_path}")
    per_category = {}
    for category, times in score_times.items():
        per_category[category] = {
            "n_images": len(times),
            "memory_bank_seconds": bank_time[category],
            "scoring_seconds_mean": sum(times) / len(times),
        }
    bank_values = [row["memory_bank_seconds"] for row in per_category.values()]
    all_scores = [value for times in score_times.values() for value in times]
    return {
        "n_images": len(all_scores),
        "n_categories": len(per_category),
        "scoring_seconds_mean": sum(all_scores) / len(all_scores),
        "memory_bank_seconds_mean": sum(bank_values) / len(bank_values),
        "memory_bank_seconds_min": min(bank_values),
        "memory_bank_seconds_max": max(bank_values),
        "per_category": per_category,
    }


def reset_gpu_peak(device: str = "cuda:0") -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


def peak_gpu_mb(device: str = "cuda:0") -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated(device) / (1024 * 1024)


def bank_slot_table(
    shots: tuple[int, ...] = (1, 2, 4, 8),
    resolutions: tuple[int, ...] = (448, 672),
) -> list[dict[str, int]]:
    """Upper bound on stored patch vectors before foreground masking."""
    return [
        {"shots": shot, "resolution": resolution, "slots": memory_bank_slots(shot, resolution)}
        for resolution in resolutions
        for shot in shots
    ]


def memory_bank_slots(shots: int, resolution: int, patch: int = 14, rotations: int = 8) -> int:
    """Upper bound on stored patch vectors before foreground masking.

    AnomalyDINO's agnostic setting rotates each reference through eight angles.
    """
    if shots < 1 or resolution < patch:
        raise ValueError("shots and resolution must cover at least one patch")
    grid = resolution // patch
    return shots * rotations * grid * grid
