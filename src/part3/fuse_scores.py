"""Calibrate two image-level detection scores on normal validation images, then average them."""

from __future__ import annotations

import numpy as np


def _mean_std(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("need validation scores to calibrate")
    std = float(array.std())
    return float(array.mean()), std if std > 1e-8 else 1.0


def fuse_detection_scores(
    left: list[float],
    right: list[float],
    left_validation: list[float],
    right_validation: list[float],
) -> np.ndarray:
    if len(left) != len(right):
        raise ValueError("the two score lists must describe the same images")
    left_mean, left_std = _mean_std(left_validation)
    right_mean, right_std = _mean_std(right_validation)
    left_z = (np.asarray(left, dtype=np.float64) - left_mean) / left_std
    right_z = (np.asarray(right, dtype=np.float64) - right_mean) / right_std
    return 0.5 * (left_z + right_z)
