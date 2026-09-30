"""Fuse a fine anomaly map with a coarser one after z-score calibration."""

from __future__ import annotations

import cv2
import numpy as np

from src.part3.fusion import fuse_maps


def upsample_to(coarse: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if coarse.shape == shape:
        return np.asarray(coarse, dtype=np.float32)
    return cv2.resize(np.asarray(coarse, dtype=np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)


def fuse_resolutions(
    fine: np.ndarray,
    coarse: np.ndarray,
    fine_mean: float,
    fine_std: float,
    coarse_mean: float,
    coarse_std: float,
) -> np.ndarray:
    """Calibrate both resolutions on normal pixels, then average them on the fine grid."""
    coarse_on_fine = upsample_to(coarse, fine.shape[:2])
    return fuse_maps(fine, coarse_on_fine, fine_mean, fine_std, coarse_mean, coarse_std)
