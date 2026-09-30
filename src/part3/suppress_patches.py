"""Clear AnomalyDINO patch distances that fall outside a GrabCut foreground.

The official agnostic run leaves cable and transistor unmasked. This edits
those patch grids only. The detection score stays the mean of the highest 1%
of patch distances, including the zeros written into the background.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.part3.background import grabcut_foreground


def suppress_patch_grid(grid: np.ndarray, image_rgb: np.ndarray) -> np.ndarray:
    if grid.ndim != 2:
        raise ValueError("patch distances must be a 2D grid")
    foreground = grabcut_foreground(image_rgb)
    mask = cv2.resize(
        foreground.astype(np.uint8),
        (grid.shape[1], grid.shape[0]),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    suppressed = np.array(grid, copy=True)
    suppressed[~mask] = 0
    return suppressed
