"""Save detection scores and OK/NG decisions without reading test labels."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.inspection.threshold import decide, mean_top1p, operating_threshold_from_maps


def _pack(maps: dict[str, np.ndarray], threshold: float) -> dict[str, dict[str, float | str]]:
    packed: dict[str, dict[str, float | str]] = {}
    for path, anomaly in maps.items():
        hottest = float(np.max(anomaly))
        packed[path] = {
            "detection_score": mean_top1p(anomaly),
            "max_pixel": hottest,
            "decision": decide(hottest, threshold),
        }
    return packed


def write_scores(
    destination: Path,
    validation_maps: dict[str, np.ndarray],
    test_maps: dict[str, np.ndarray],
    extra: dict | None = None,
) -> dict:
    threshold = operating_threshold_from_maps(list(validation_maps.values()))
    payload = {
        "threshold": threshold,
        "threshold_rule": "mean_plus_3std_of_validation_pixels",
        "detection_score": "mean_of_highest_1_percent",
        "decision_rule": "max_pixel_above_threshold",
        "validation": _pack(validation_maps, threshold),
        "test": _pack(test_maps, threshold),
    }
    if extra:
        payload.update(extra)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
