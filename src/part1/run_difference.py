"""Run the registration-and-difference baseline on one dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.inspection.reporting import write_scores
from src.inspection.splits import load_split
from src.part1.difference import difference_map, image_score


def _score_paths(
    data_root: Path,
    relative_paths: list[str],
    references: list[Path],
    image_size: int,
    map_dir: Path,
) -> dict[str, float]:
    scores = {}
    for relative in tqdm(relative_paths, leave=False):
        anomaly, _reference = difference_map(
            data_root / relative, references, image_size=image_size
        )
        stem = Path(relative).stem
        parent = Path(relative).parent.name
        np.save(map_dir / f"{parent}_{stem}.npy", anomaly)
        scores[relative] = image_score(anomaly)
    return scores


def run_category(split_path: Path, data_root: Path, out_dir: Path, image_size: int) -> None:
    split = load_split(split_path)
    category = split["category"]
    category_dir = out_dir / category
    map_dir = category_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    references = [data_root / relative for relative in split["reference_pool"]]
    validation_scores = _score_paths(
        data_root, split["threshold_validation"], references, image_size, map_dir
    )
    test_paths = [record["path"] for record in split["test"]]
    test_scores = _score_paths(data_root, test_paths, references, image_size, map_dir)
    write_scores(
        category_dir / "scores.json",
        validation_scores,
        test_scores,
        extra={
            "method": "part1_difference",
            "category": category,
            "image_size": image_size,
            "labels_used_for_threshold": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/part1/difference"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--categories", nargs="*", default=None)
    args = parser.parse_args()
    split_paths = sorted(args.split_dir.glob("*.json"))
    if args.categories:
        wanted = set(args.categories)
        split_paths = [path for path in split_paths if path.stem in wanted]
    if not split_paths:
        raise SystemExit(f"no splits in {args.split_dir}")
    for split_path in split_paths:
        print(f"difference: {split_path.stem}")
        run_category(split_path, args.data_root, args.out_dir, args.image_size)
    print(json.dumps({"out_dir": str(args.out_dir), "categories": [path.stem for path in split_paths]}))


if __name__ == "__main__":
    main()
