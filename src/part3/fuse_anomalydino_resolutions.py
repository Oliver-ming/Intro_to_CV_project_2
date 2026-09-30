"""Fuse AnomalyDINO's 448 and 672 patch grids.

The finer grid is 672. The coarser grid is resized onto it. Both are z-scored
with the defect-free validation maps from the same shot and seed, then averaged.
The detection score remains the mean of the highest 1% of values. Pixel metrics
use that same 672 patch grid, with ground-truth masks resized onto it.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.inspection.images import read_mask
from src.inspection.metrics import aupro, safe_ap, safe_auroc
from src.inspection.splits import load_split
from src.inspection.threshold import mean_top1p
from src.part3.fusion import pixel_mean_std
from src.part3.multiscale import fuse_resolutions, upsample_to


def _validation_stats(map_dir: Path) -> tuple[float, float]:
    maps = [np.load(path) for path in sorted(map_dir.glob("*.npy"))]
    if not maps:
        raise FileNotFoundError(map_dir)
    return pixel_mean_std(maps)


def _grid(root: Path, relative: str) -> np.ndarray:
    path = root / Path(relative).with_suffix(".npy")
    if not path.is_file():
        raise FileNotFoundError(path)
    return np.load(path)


def _summary(labels: np.ndarray, scores: list[float], maps: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    values = np.asarray(scores, dtype=np.float64)
    flat_labels = np.concatenate([(mask > 0).astype(np.uint8).ravel() for mask in masks])
    flat_scores = np.concatenate([np.asarray(anomaly, dtype=np.float64).ravel() for anomaly in maps])
    return {
        "image_auroc": safe_auroc(labels, values),
        "image_ap": safe_ap(labels, values),
        "pixel_auroc": safe_auroc(flat_labels, flat_scores),
        "pixel_ap": safe_ap(flat_labels, flat_scores),
        "aupro": aupro(maps, masks, integration_limit=0.3),
    }


def fuse_category(
    fine_root: Path,
    coarse_root: Path,
    fine_validation: Path,
    coarse_validation: Path,
    split_path: Path,
    data_root: Path,
) -> dict:
    split = load_split(split_path)
    fine_mean, fine_std = _validation_stats(fine_validation)
    coarse_mean, coarse_std = _validation_stats(coarse_validation)
    labels = []
    fine_scores = []
    coarse_scores = []
    fused_scores = []
    fine_maps = []
    coarse_maps = []
    fused_maps = []
    masks = []
    category = split["category"]
    for record in split["test"]:
        relative = Path(record["path"])
        if relative.parts and relative.parts[0] == category:
            relative = Path(*relative.parts[1:])
        fine = _grid(fine_root, relative.as_posix())
        coarse = _grid(coarse_root, relative.as_posix())
        fused = fuse_resolutions(fine, coarse, fine_mean, fine_std, coarse_mean, coarse_std)
        if record["mask"] is None:
            mask = np.zeros(fine.shape[:2], dtype=np.uint8)
        else:
            full_mask = read_mask(data_root / record["mask"])
            mask = cv2.resize(full_mask, (fine.shape[1], fine.shape[0]), interpolation=cv2.INTER_NEAREST)
        labels.append(1 if record["label"] == "NG" else 0)
        fine_scores.append(mean_top1p(fine))
        coarse_scores.append(mean_top1p(coarse))
        fused_scores.append(mean_top1p(fused))
        fine_maps.append(fine)
        coarse_maps.append(upsample_to(coarse, fine.shape[:2]))
        fused_maps.append(fused)
        masks.append(mask)
    labels_array = np.asarray(labels)
    return {
        "category": category,
        "n_test": int(labels_array.size),
        "pixel_resolution": "672_patch_grid",
        "fine_672": _summary(labels_array, fine_scores, fine_maps, masks),
        "coarse_448": _summary(labels_array, coarse_scores, coarse_maps, masks),
        "fused": _summary(labels_array, fused_scores, fused_maps, masks),
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-root", type=Path, required=True)
    parser.add_argument("--coarse-root", type=Path, required=True)
    parser.add_argument("--fine-heldout", type=Path, required=True)
    parser.add_argument("--coarse-heldout", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    categories = args.categories
    if not categories:
        categories = sorted(path.name for path in args.fine_root.iterdir() if path.is_dir())
    rows = []
    for category in categories:
        fine_val = args.fine_heldout / category / "maps"
        coarse_val = args.coarse_heldout / category / "maps"
        if not fine_val.is_dir() or not coarse_val.is_dir():
            print(f"skip {category}: validation maps missing", flush=True)
            continue
        print(f"fuse {category}", flush=True)
        rows.append(
            fuse_category(
                args.fine_root / category,
                args.coarse_root / category,
                fine_val,
                coarse_val,
                args.split_dir / f"{category}.json",
                args.data_root,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
