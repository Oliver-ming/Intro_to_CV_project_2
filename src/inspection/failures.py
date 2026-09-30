"""Group wrong OK/NG decisions by the cause the inspection report needs."""

from __future__ import annotations

import numpy as np


def categorize_failure(
    label: str,
    decision: str,
    anomaly: np.ndarray,
    mask: np.ndarray | None,
    small_defect_pixels: int = 64,
) -> str | None:
    """Return a cause name for a wrong decision, or None when the decision is right."""
    if label == decision:
        return None
    heat = np.asarray(anomaly, dtype=np.float64)
    height, width = heat.shape[:2]
    border = max(1, min(height, width) // 10)
    border_mask = np.zeros_like(heat, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_mean = float(heat[border_mask].mean()) if border_mask.any() else 0.0
    center_mean = float(heat[~border_mask].mean()) if (~border_mask).any() else 0.0
    if label == "OK" and decision == "NG":
        if border_mean > center_mean:
            return "background_or_border"
        return "normal_variation"
    defect_pixels = 0 if mask is None else int(np.count_nonzero(mask))
    if defect_pixels > 0 and defect_pixels < small_defect_pixels:
        return "small_defect"
    return "missed_defect"
