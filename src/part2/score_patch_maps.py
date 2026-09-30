"""Image metrics from AnomalyDINO patch-distance grids.

The official metrics file is written only after every category in a seed
finishes. These scores use the same rule, the mean of the highest 1% patch
distances, so a finished category can be read before that file exists.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.inspection.metrics import safe_ap, safe_auroc
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--map-root", type=Path, required=True)
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--categories", nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for category in args.categories:
        rows.append(category_image_metrics(args.map_root / category, args.split_dir / f"{category}.json"))
        print(json.dumps(rows[-1]), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
