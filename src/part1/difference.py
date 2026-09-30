"""Align a test image to the nearest normal reference and score the residual."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from src.inspection.images import read_rgb, resize_rgb, to_float_rgb, to_gray_float


def _coarse_distance(test_gray: np.ndarray, reference_gray: np.ndarray) -> float:
    test_small = cv2.resize(test_gray, (64, 64), interpolation=cv2.INTER_AREA)
    reference_small = cv2.resize(reference_gray, (64, 64), interpolation=cv2.INTER_AREA)
    return float(np.mean((test_small - reference_small) ** 2))


def _ecc_warp(template: np.ndarray, moving: np.ndarray) -> tuple[np.ndarray | None, float]:
    """Return the affine warp that aligns ``moving`` onto ``template``, plus ECC."""
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
    try:
        correlation, warp = cv2.findTransformECC(
            template,
            moving,
            warp,
            cv2.MOTION_AFFINE,
            criteria,
            None,
            5,
        )
    except cv2.error:
        return None, -1.0
    return warp, float(correlation)


def _residual_map(test_rgb: np.ndarray, reference_rgb: np.ndarray, warp: np.ndarray | None) -> np.ndarray:
    """Pixel residual in the test image's coordinate frame."""
    height, width = test_rgb.shape[:2]
    if warp is None:
        aligned = test_rgb
        reference = reference_rgb
    else:
        aligned = cv2.warpAffine(
            test_rgb,
            warp,
            (width, height),
            flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
            borderMode=cv2.BORDER_REPLICATE,
        )
        reference = reference_rgb

    abs_diff = np.mean(np.abs(aligned - reference), axis=2)
    test_gray = cv2.cvtColor(aligned, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY)
    _, ssim_map = structural_similarity(
        reference_gray,
        test_gray,
        data_range=1.0,
        win_size=7,
        full=True,
    )
    residual = 0.5 * abs_diff + 0.5 * (1.0 - ssim_map)
    residual = cv2.GaussianBlur(residual.astype(np.float32), (0, 0), sigmaX=2.0)

    if warp is None:
        return residual

    inverse = cv2.invertAffineTransform(warp)
    restored = cv2.warpAffine(
        residual,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP,
    )
    return restored


def difference_map(
    test_path: Path,
    reference_paths: list[Path],
    image_size: int,
    shortlist: int = 3,
    min_correlation: float = 0.5,
) -> tuple[np.ndarray, str]:
    """Score ``test_path`` against the closest references.

    Returns the anomaly map at ``image_size`` and the reference path that was used.
    The map is later resized to the original image only for pixel metrics.
    """
    if not reference_paths:
        raise ValueError("reference pool is empty")

    test_rgb = to_float_rgb(resize_rgb(read_rgb(test_path), image_size))
    test_gray = to_gray_float((test_rgb * 255).astype(np.uint8))

    ranked: list[tuple[float, Path, np.ndarray]] = []
    for reference_path in reference_paths:
        reference_rgb = to_float_rgb(resize_rgb(read_rgb(reference_path), image_size))
        reference_gray = to_gray_float((reference_rgb * 255).astype(np.uint8))
        ranked.append((_coarse_distance(test_gray, reference_gray), reference_path, reference_rgb))
    ranked.sort(key=lambda item: item[0])

    best_map = None
    best_correlation = -1.0
    best_path = ranked[0][1]
    for _, reference_path, reference_rgb in ranked[:shortlist]:
        reference_gray = to_gray_float((reference_rgb * 255).astype(np.uint8))
        warp, correlation = _ecc_warp(reference_gray, test_gray)
        if warp is None or correlation < min_correlation:
            candidate = _residual_map(test_rgb, reference_rgb, None)
            correlation = -1.0 if correlation < min_correlation else correlation
        else:
            candidate = _residual_map(test_rgb, reference_rgb, warp)
        if correlation > best_correlation or best_map is None:
            best_correlation = correlation
            best_map = candidate
            best_path = reference_path

    assert best_map is not None
    return best_map.astype(np.float32), str(best_path)


def image_score(anomaly_map: np.ndarray) -> float:
    return float(np.max(anomaly_map))
