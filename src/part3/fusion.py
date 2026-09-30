"""Combine two anomaly maps after calibrating each on normal validation pixels."""

from __future__ import annotations

import numpy as np


def pixel_mean_std(maps: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(anomaly, dtype=np.float64).ravel() for anomaly in maps])
    if values.size == 0:
        raise ValueError("need validation pixels to calibrate a map")
    mean = float(values.mean())
    variance = float(np.square(values - mean).mean())
    return mean, variance ** 0.5


def fuse_maps(
    left: np.ndarray,
    right: np.ndarray,
    left_mean: float,
    left_std: float,
    right_mean: float,
    right_std: float,
) -> np.ndarray:
    """Average of two z-scored maps. A zero standard deviation is treated as one."""
    if left.shape != right.shape:
        raise ValueError(f"map shapes differ: {left.shape} vs {right.shape}")
    left_scale = left_std if left_std > 1e-8 else 1.0
    right_scale = right_std if right_std > 1e-8 else 1.0
    left_z = (np.asarray(left, dtype=np.float64) - left_mean) / left_scale
    right_z = (np.asarray(right, dtype=np.float64) - right_mean) / right_scale
    return (0.5 * (left_z + right_z)).astype(np.float32)
