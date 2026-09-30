"""Turn AnomalyDINO patch distances into the smoothed map used for segmentation."""

from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter


def patch_distances_to_map(distances: np.ndarray, image_shape: tuple[int, int], sigma: float = 4.0) -> np.ndarray:
    """Resize a patch grid to ``image_shape`` and smooth it, matching ``dists2map``."""
    grid = np.asarray(distances, dtype=np.float64)
    if grid.ndim != 2:
        raise ValueError("patch distances must be a 2D grid")
    height, width = image_shape
    resized = cv2.resize(grid, (width, height), interpolation=cv2.INTER_LINEAR)
    return gaussian_filter(resized, sigma=sigma)
