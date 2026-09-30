"""Score a finished AnomalyDINO VisA seed from its patch grids.

VisA labels come from the folder name: ``test/good`` is normal and ``test/bad``
is anomalous. Pixel metrics resize the official mask onto the patch grid.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.inspection.images import read_mask
from src.inspection.metrics import aupro, safe_ap, safe_auroc
from src.inspection.threshold import mean_top1p


def visa_image_metrics(map_dir: Path) -> dict:
    """Image AUROC and AP only. Pixel metrics wait for the official file."""
    category = map_dir.name
    labels = []
    scores = []
    for grid_path in sorted(map_dir.rglob("*.npy")):
        relative = grid_path.relative_to(map_dir)
        if len(relative.parts) < 2 or relative.parts[1] not in {"good", "bad"}:
            continue
        labels.append(0 if relative.parts[1] == "good" else 1)
        scores.append(mean_top1p(np.load(grid_path)))
    if not labels:
        raise FileNotFoundError(f"no patch grids in {map_dir}")
    labels_array = np.asarray(labels)
    values = np.asarray(scores)
    return {
        "category": category,
        "n_test": int(labels_array.size),
        "image_auroc": safe_auroc(labels_array, values),
        "image_ap": safe_ap(labels_array, values),
    }


def score_visa_category(map_dir: Path, data_root: Path) -> dict:
    category = map_dir.name
    labels = []
    scores = []
    grids = []
    masks = []
    for grid_path in sorted(map_dir.rglob("*.npy")):
        relative = grid_path.relative_to(map_dir)
        if len(relative.parts) < 2 or relative.parts[1] not in {"good", "bad"}:
            continue
        kind = relative.parts[1]
        grid = np.load(grid_path)
        labels.append(0 if kind == "good" else 1)
        scores.append(mean_top1p(grid))
        if kind == "good":
            mask = np.zeros(grid.shape[:2], dtype=np.uint8)
        else:
            mask_path = data_root / category / "ground_truth" / "bad" / f"{grid_path.stem}.png"
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            mask = cv2.resize(read_mask(mask_path), (grid.shape[1], grid.shape[0]), interpolation=cv2.INTER_NEAREST)
        grids.append(grid)
        masks.append(mask)
    if not labels:
        raise FileNotFoundError(f"no patch grids in {map_dir}")
    labels_array = np.asarray(labels)
    values = np.asarray(scores)
    flat_labels = np.concatenate([(mask > 0).astype(np.uint8).ravel() for mask in masks])
    flat_scores = np.concatenate([np.asarray(grid, dtype=np.float64).ravel() for grid in grids])
    return {
        "category": category,
        "n_test": int(labels_array.size),
        "image_auroc": safe_auroc(labels_array, values),
        "image_ap": safe_ap(labels_array, values),
        "pixel_auroc": safe_auroc(flat_labels, flat_scores),
        "pixel_ap": safe_ap(flat_labels, flat_scores),
        "aupro": aupro(grids, masks, integration_limit=0.3),
        "pixel_resolution": "patch_grid",
    }


def score_visa_seed(map_root: Path, data_root: Path) -> list[dict]:
    rows = [score_visa_category(path, data_root) for path in sorted(map_root.iterdir()) if path.is_dir()]
    if not rows:
        raise FileNotFoundError(f"no categories in {map_root}")
    return rows


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/VisA_pytorch/1cls"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = score_visa_seed(args.map_root, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    means = {
        key: sum(row[key] for row in rows) / len(rows)
        for key in ("image_auroc", "image_ap", "pixel_auroc", "aupro")
    }
    print(json.dumps({"n": len(rows), **means}))


if __name__ == "__main__":
    main()
