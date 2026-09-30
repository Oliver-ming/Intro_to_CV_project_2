"""Compare AnomalyDINO patch scores before and after GrabCut.

Only cable and transistor are expected. The score is still the mean of the
highest 1% patch distances. Pixel numbers here are on the 32 by 32 patch grid,
not the full-resolution maps used in the official table.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.inspection.images import read_mask, read_rgb
from src.inspection.metrics import aupro, safe_ap, safe_auroc
from src.inspection.splits import load_split
from src.inspection.threshold import mean_top1p
from src.part3.suppress_patches import suppress_patch_grid
from src.part3.targets import needs_grabcut


def _paired_scores(
    map_root: Path,
    data_root: Path,
    split: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    labels = []
    before_scores = []
    after_scores = []
    before_maps = []
    after_maps = []
    masks = []
    for record in split["test"]:
        relative = Path(record["path"])
        grid_path = map_root / relative.with_suffix(".npy")
        if not grid_path.is_file():
            continue
        grid = np.load(grid_path)
        image = read_rgb(data_root / record["path"])
        suppressed = suppress_patch_grid(grid, image)
        labels.append(1 if record["label"] == "NG" else 0)
        before_scores.append(mean_top1p(grid))
        after_scores.append(mean_top1p(suppressed))
        if record["mask"] is None:
            mask = np.zeros(grid.shape, dtype=np.uint8)
        else:
            mask = read_mask(data_root / record["mask"])
            mask = cv2.resize(mask, (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_NEAREST)
        before_maps.append(grid)
        after_maps.append(suppressed)
        masks.append(mask)
    if not labels:
        raise FileNotFoundError(f"no patch grids under {map_root}")
    return (
        np.asarray(labels),
        np.asarray(before_scores),
        np.asarray(after_scores),
        before_maps,
        after_maps,
        masks,
    )


def _pixel_summary(maps: list[np.ndarray], masks: list[np.ndarray]) -> dict[str, float]:
    flat_labels = np.concatenate([(mask > 0).astype(np.uint8).ravel() for mask in masks])
    flat_scores = np.concatenate([anomaly.ravel() for anomaly in maps])
    return {
        "pixel_auroc": safe_auroc(flat_labels, flat_scores),
        "pixel_ap": safe_ap(flat_labels, flat_scores),
        "aupro": aupro(maps, masks, integration_limit=0.3),
    }


def compare_category(map_root: Path, data_root: Path, split_path: Path) -> dict:
    split = load_split(split_path)
    category = split["category"]
    if not needs_grabcut(category):
        raise ValueError(f"{category} is not a PCA-mask failure")
    labels, before, after, before_maps, after_maps, masks = _paired_scores(map_root, data_root, split)
    return {
        "category": category,
        "n_test": int(labels.size),
        "before": {
            "image_auroc": safe_auroc(labels, before),
            "image_ap": safe_ap(labels, before),
            **_pixel_summary(before_maps, masks),
        },
        "after": {
            "image_auroc": safe_auroc(labels, after),
            "image_ap": safe_ap(labels, after),
            **_pixel_summary(after_maps, masks),
        },
        "pixel_resolution": "patch_grid",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--categories", nargs="+", default=["cable", "transistor"])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for category in args.categories:
        print(f"suppress {category}", flush=True)
        rows.append(
            compare_category(args.map_root, args.data_root, args.split_dir / f"{category}.json")
        )
        print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
