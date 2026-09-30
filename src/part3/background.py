"""Foreground mask used where AnomalyDINO's PCA mask fails.

GrabCut is initialized from a centered rectangle, so thin or dark objects
that the first principal component marks as background can still be kept.
"""

from __future__ import annotations

import cv2
import numpy as np


def grabcut_foreground(image_rgb: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Return a boolean mask that is true on the estimated foreground."""
    if image_rgb.dtype != np.uint8:
        image = np.clip(image_rgb * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image_rgb
    height, width = image.shape[:2]
    margin_x = max(1, width // 8)
    margin_y = max(1, height // 8)
    rectangle = (margin_x, margin_y, width - 2 * margin_x, height - 2 * margin_y)
    mask = np.zeros((height, width), np.uint8)
    background = np.zeros((1, 65), np.float64)
    foreground = np.zeros((1, 65), np.float64)
    cv2.grabCut(
        cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
        mask,
        rectangle,
        background,
        foreground,
        iterations,
        cv2.GC_INIT_WITH_RECT,
    )
    return (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)


def suppress_background(anomaly: np.ndarray, foreground: np.ndarray) -> np.ndarray:
    """Zero anomaly scores outside the foreground. The map shape is unchanged."""
    if anomaly.shape[:2] != foreground.shape[:2]:
        foreground = cv2.resize(
            foreground.astype(np.uint8),
            (anomaly.shape[1], anomaly.shape[0]),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    suppressed = np.array(anomaly, copy=True)
    suppressed[~foreground.astype(bool)] = 0
    return suppressed
