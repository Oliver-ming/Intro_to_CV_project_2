"""Mean and standard deviation across AnomalyDINO's per-seed metric files."""

from __future__ import annotations

import numpy as np

SEED_SUMMARY_KEYS = (
    "mean_classification_au_roc",
    "mean_classification_ap",
    "mean_classification_f1",
    "mean_au_pro",
    "mean_segmentation_au_roc",
    "mean_segmentation_f1",
)


def mean_and_std(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("need at least one seed")
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {"mean": float(array.mean()), "std": std, "n": int(array.size)}


def aggregate_seed_metrics(seed_payloads: list[dict]) -> dict[str, dict[str, float]]:
    summary = {}
    for key in SEED_SUMMARY_KEYS:
        present = [payload[key] for payload in seed_payloads if key in payload]
        if present:
            summary[key] = mean_and_std(present)
    return summary
