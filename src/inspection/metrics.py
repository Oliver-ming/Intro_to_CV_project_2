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


def _integrate_until(fprs: np.ndarray, pros: np.ndarray, limit: float) -> float:
    """Trapezoid up to ``limit``, interpolating the cutoff the way MVTec does."""
    xs = np.asarray(fprs, dtype=np.float64)
    ys = np.asarray(pros, dtype=np.float64)
    if xs.size < 2 or xs[-1] < limit:
        return float("nan")
    insert_at = int(np.searchsorted(xs, limit, side="left"))
    correction = 0.0
    if insert_at >= xs.size or xs[insert_at] != limit:
        if insert_at == 0 or insert_at >= xs.size:
            return float("nan")
        span = xs[insert_at] - xs[insert_at - 1]
        y_at_limit = ys[insert_at - 1] + (ys[insert_at] - ys[insert_at - 1]) * (limit - xs[insert_at - 1]) / span
        correction = 0.5 * (y_at_limit + ys[insert_at - 1]) * (limit - xs[insert_at - 1])
        xs = xs[:insert_at]
        ys = ys[:insert_at]
    else:
        xs = xs[: insert_at + 1]
        ys = ys[: insert_at + 1]
    area = 0.0
    if xs.size >= 2:
        area = float(np.sum(0.5 * (ys[1:] + ys[:-1]) * (xs[1:] - xs[:-1])))
    return (area + correction) / limit


def aupro(
    anomaly_maps: list[np.ndarray],
    ground_truth_masks: list[np.ndarray],
    integration_limit: float = 0.3,
) -> float:
    """Area under the PRO curve, normalized by ``integration_limit``.

    The curve is the one used by MVTec and AnomalyDINO: sort every pixel once,
    then accumulate false positives on normal pixels and per-region overlap.
    Images may differ in height and width.
    """
    if len(anomaly_maps) != len(ground_truth_masks) or not anomaly_maps:
        raise ValueError("anomaly maps and ground-truth masks must be paired and non-empty")
    maps = [np.asarray(anomaly, dtype=np.float64) for anomaly in anomaly_maps]
    masks = [np.asarray(mask).astype(bool) for mask in ground_truth_masks]
    for anomaly, mask in zip(maps, masks):
        if anomaly.shape != mask.shape:
            raise ValueError(f"shape mismatch {anomaly.shape} vs {mask.shape}")

    score_parts: list[np.ndarray] = []
    false_positive_parts: list[np.ndarray] = []
    overlap_parts: list[np.ndarray] = []
    region_count = 0
    normal_count = 0
    structure = np.ones((3, 3), dtype=int)
    for anomaly, mask in zip(maps, masks):
        labeled, n_components = ndimage.label(mask, structure=structure)
        region_count += int(n_components)
        normal = labeled == 0
        normal_count += int(normal.sum())
        overlap = np.zeros(anomaly.shape, dtype=np.float64)
        for component in range(1, n_components + 1):
            region = labeled == component
            overlap[region] = 1.0 / float(region.sum())
        score_parts.append(anomaly.ravel())
        false_positive_parts.append(normal.ravel().astype(np.float64))
        overlap_parts.append(overlap.ravel())
    if region_count == 0 or normal_count == 0:
        return float("nan")

    scores = np.concatenate(score_parts)
    false_positives = np.concatenate(false_positive_parts)
    overlaps = np.concatenate(overlap_parts)
    order = np.argsort(scores, kind="mergesort")[::-1]
    scores = scores[order]
    fprs = np.cumsum(false_positives[order]) / normal_count
    pros = np.cumsum(overlaps[order]) / region_count
    keep = np.empty(scores.size, dtype=bool)
    keep[:-1] = scores[:-1] != scores[1:]
    keep[-1] = True
    fprs = np.clip(fprs[keep], 0.0, 1.0)
    pros = np.clip(pros[keep], 0.0, 1.0)
    fprs = np.concatenate((np.array([0.0]), fprs, np.array([1.0])))
    pros = np.concatenate((np.array([0.0]), pros, np.array([1.0])))
    return _integrate_until(fprs, pros, integration_limit)
