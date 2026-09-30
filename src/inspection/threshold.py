"""Operating point and the two scores that must not be mixed.

The detection score follows AnomalyDINO: the mean of the highest 1% of
values. The operating point follows MVTec AD 2: mean + 3 population
standard deviations of the raw pixels on defect-free validation maps.
Image OK/NG compares the maximum pixel with that same operating point.
"""

from __future__ import annotations

import math

import numpy as np


def mean_top1p(values: np.ndarray) -> float:
    """Match AnomalyDINO's ``mean_top1p``: mean of the highest 1% of values."""
    flat = np.asarray(values).ravel()
    if flat.size == 0:
        raise ValueError("need at least one value")
    count = int(flat.size * 0.01)
    if count == 0:
        return float(np.max(flat))
    top = np.partition(flat, flat.size - count)[-count:]
    return float(np.mean(top))


def operating_threshold(values: np.ndarray) -> float:
    """Mean + 3 population standard deviations.

    ``values`` are pixel anomaly values from defect-free validation maps,
    not image-level scores.
    """
    flat = np.asarray(values, dtype=np.float64).ravel()
    if flat.size == 0:
        raise ValueError("need at least one pixel to set the operating point")
    mean = float(flat.mean())
    if flat.size == 1:
        return mean
    variance = float(np.square(flat - mean).sum() / flat.size)
    return mean + 3.0 * math.sqrt(variance)


def operating_threshold_from_maps(maps: list[np.ndarray]) -> float:
    """Same cut as ``operating_threshold`` without stacking every map at once."""
    count = 0
    mean = 0.0
    second_moment = 0.0
    for anomaly in maps:
        flat = np.asarray(anomaly, dtype=np.float64).ravel()
        batch_count = int(flat.size)
        if batch_count == 0:
            continue
        batch_mean = float(flat.mean())
        batch_moment = float(np.square(flat - batch_mean).sum())
        if count == 0:
            count = batch_count
            mean = batch_mean
            second_moment = batch_moment
            continue
        new_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * batch_count / new_count
        second_moment = second_moment + batch_moment + delta * delta * count * batch_count / new_count
        count = new_count
    if count == 0:
        raise ValueError("need at least one pixel to set the operating point")
    if count == 1:
        return mean
    return mean + 3.0 * math.sqrt(second_moment / count)


def decide(max_pixel: float, threshold: float) -> str:
    """NG when the hottest pixel is strictly above the operating point."""
    return "NG" if max_pixel > threshold else "OK"
