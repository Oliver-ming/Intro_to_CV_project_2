"""Operating point: mean + 3 standard deviations of normal image scores."""

from __future__ import annotations

import math


def operating_threshold(normal_scores: list[float]) -> float:
    """Return the OK/NG cut computed only from defect-free validation scores.

    A single score has no deviation, so the cut falls back to that score.
    """
    if not normal_scores:
        raise ValueError("need at least one normal score to set the operating point")
    mean = sum(normal_scores) / len(normal_scores)
    if len(normal_scores) == 1:
        return mean
    variance = sum((score - mean) ** 2 for score in normal_scores) / len(normal_scores)
    return mean + 3.0 * math.sqrt(variance)


def decide(score: float, threshold: float) -> str:
    """NG when the score is strictly above the operating point."""
    return "NG" if score > threshold else "OK"
