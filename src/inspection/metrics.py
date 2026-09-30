"""Image-level and pixel-level metrics for the shared evaluation protocol.

AUPRO integrates the per-region overlap curve up to a false-positive rate of
0.3, which is the limit used by MVTec AD and by AnomalyDINO.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def safe_auroc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, scores))


def safe_ap(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(np.uint8).ravel()
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if labels.min() == labels.max():
        return float("nan")
    return float(average_precision_score(labels, scores))


def false_positive_rate(labels: list[str], decisions: list[str]) -> float:
    """Fraction of OK images that the operating point marks NG."""
    ok = [(label, decision) for label, decision in zip(labels, decisions) if label == "OK"]
    if not ok:
        return float("nan")
    return sum(decision == "NG" for _, decision in ok) / len(ok)


def _regions(mask: np.ndarray) -> list[np.ndarray]:
    labeled, count = ndimage.label(mask.astype(bool), structure=np.ones((3, 3), dtype=int))
    return [np.flatnonzero(labeled == index) for index in range(1, count + 1)]


def aupro(
    anomaly_maps: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
    integration_limit: float = 0.3,
) -> float:
    """Area under the PRO curve, normalized by ``integration_limit``."""
    if len(anomaly_maps) != len(ground_truth_masks) or not anomaly_maps:
        raise ValueError("anomaly maps and ground-truth masks must be paired and non-empty")
    maps = [np.asarray(anomaly, dtype=np.float64) for anomaly in anomaly_maps]
    masks = [np.asarray(mask).astype(bool) for mask in ground_truth_masks]
    for anomaly, mask in zip(maps, masks):
        if anomaly.shape != mask.shape:
            raise ValueError(f"shape mismatch {anomaly.shape} vs {mask.shape}")

    region_ids: list[tuple[int, np.ndarray]] = []
    normal_ids: list[tuple[int, np.ndarray]] = []
    for index, (anomaly, mask) in enumerate(zip(maps, masks)):
        region_ids.extend((index, pixels) for pixels in _regions(mask))
        normal_ids.append((index, np.flatnonzero(~mask)))
    if not region_ids:
        return float("nan")
    normal_count = sum(pixels.size for _, pixels in normal_ids)
    if normal_count == 0:
        return float("nan")

    pooled = np.concatenate([anomaly.ravel() for anomaly in maps])
    if pooled.size > 256:
        thresholds = np.unique(np.quantile(pooled, np.linspace(0.0, 1.0, 256)))[::-1]
    else:
        thresholds = np.unique(pooled)[::-1]
    fprs = [0.0]
    pros = [0.0]
    for threshold in thresholds:
        false_positives = 0
        overlaps = []
        for index, pixels in normal_ids:
            if pixels.size:
                false_positives += int(np.count_nonzero(maps[index].ravel()[pixels] >= threshold))
        for index, pixels in region_ids:
            overlaps.append(float(np.count_nonzero(maps[index].ravel()[pixels] >= threshold)) / pixels.size)
        fprs.append(false_positives / normal_count)
        pros.append(float(np.mean(overlaps)))

    order = np.argsort(fprs)
    fprs_sorted = np.asarray(fprs, dtype=np.float64)[order]
    pros_sorted = np.asarray(pros, dtype=np.float64)[order]
    cutoff = fprs_sorted <= integration_limit
    if cutoff.sum() < 2:
        return float("nan")
    xs = fprs_sorted[cutoff]
    ys = pros_sorted[cutoff]
    if xs[-1] < integration_limit:
        xs = np.append(xs, integration_limit)
        ys = np.append(ys, ys[-1])
    area = float(np.trapezoid(ys, xs))
    return area / integration_limit
