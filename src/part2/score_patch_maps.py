"""Image metrics from AnomalyDINO patch-distance grids.

The official metrics file is written only after every category in a seed
finishes. These scores use the same rule, the mean of the highest 1% patch
distances, so a finished category can be read before that file exists.
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


def category_image_metrics(map_dir: Path, split_path: Path) -> dict:
    split = load_split(split_path)
    category = split["category"]
    labels = []
    scores = []
    missing = []
    for record in split["test"]:
        relative = Path(record["path"])
        if relative.parts and relative.parts[0] == category:
            relative = Path(*relative.parts[1:])
        grid_path = map_dir / relative.with_suffix(".npy")
        if not grid_path.is_file():
            missing.append(record["path"])
            continue
        labels.append(1 if record["label"] == "NG" else 0)
        scores.append(mean_top1p(np.load(grid_path)))
    if missing:
        raise FileNotFoundError(f"{category} is missing {len(missing)} patch grids, first {missing[0]}")
    labels_array = np.asarray(labels)
    values = np.asarray(scores)
    return {
        "category": category,
        "n_test": int(labels_array.size),
        "image_auroc": safe_auroc(labels_array, values),
        "image_ap": safe_ap(labels_array, values),
    }


def category_pixel_metrics(map_dir: Path, split_path: Path, data_root: Path) -> dict:
    """Image scores plus pixel metrics on the patch grid."""
    summary = category_image_metrics(map_dir, split_path)
    split = load_split(split_path)
    category = summary["category"]
    maps = []
    masks = []
    for record in split["test"]:
        relative = Path(record["path"])
        if relative.parts and relative.parts[0] == category:
            relative = Path(*relative.parts[1:])
        grid = np.load(map_dir / relative.with_suffix(".npy"))
        if record["mask"] is None:
            mask = np.zeros(grid.shape[:2], dtype=np.uint8)
        else:
            full_mask = read_mask(data_root / record["mask"])
            mask = cv2.resize(full_mask, (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_NEAREST)
        maps.append(grid)
        masks.append(mask)
    flat_labels = np.concatenate([(mask > 0).astype(np.uint8).ravel() for mask in masks])
    flat_scores = np.concatenate([np.asarray(grid, dtype=np.float64).ravel() for grid in maps])
    summary.update(
        {
            "pixel_auroc": safe_auroc(flat_labels, flat_scores),
            "pixel_ap": safe_ap(flat_labels, flat_scores),
            "aupro": aupro(maps, masks, integration_limit=0.3),
            "pixel_resolution": "patch_grid",
        }
    )
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()
    rows = []
    for category in args.categories:
        split_path = args.split_dir / f"{category}.json"
        map_dir = args.map_root / category
        if args.data_root is None:
            row = category_image_metrics(map_dir, split_path)
        else:
            row = category_pixel_metrics(map_dir, split_path, args.data_root)
        rows.append(row)
        print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
