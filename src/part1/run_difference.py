"""Run the registration-and-difference baseline on one dataset split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.inspection.evaluate_run import image_metrics, pixel_metrics
from src.inspection.reporting import write_scores
from src.inspection.splits import load_split
from src.part1.difference import difference_map, prepare_reference


def _collect_maps(
    data_root: Path,
    relative_paths: list[str],
    prepared: list[tuple[Path, np.ndarray, np.ndarray]],
    image_size: int,
    map_dir: Path,
) -> dict[str, np.ndarray]:
    maps = {}
    for relative in tqdm(relative_paths, leave=False):
        anomaly, _reference = difference_map(
            data_root / relative,
            [],
            image_size=image_size,
            prepared=prepared,
        )
        stem = Path(relative).stem
        parent = Path(relative).parent.name
        np.save(map_dir / f"{parent}_{stem}.npy", anomaly)
        maps[relative] = anomaly
    return maps


def write_category_metrics(split: dict, scores_path: Path, data_root: Path) -> dict:
    """Image metrics plus pixel metrics on the saved 256 map."""
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    summary = image_metrics(split, scores)
    summary.update(pixel_metrics(split, data_root, scores_path.parent / "maps", align_to="map"))
    summary["pixel_resolution"] = "anomaly_map"
    (scores_path.parent / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def run_category(split_path: Path, data_root: Path, out_dir: Path, image_size: int) -> None:
    split = load_split(split_path)
    category = split["category"]
    category_dir = out_dir / category
    if (category_dir / "scores.json").is_file():
        print(f"skip difference: {category}")
        return
    map_dir = category_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    references = [data_root / relative for relative in split["reference_pool"]]
    prepared = [prepare_reference(path, image_size) for path in references]
    validation_maps = _collect_maps(
        data_root, split["threshold_validation"], prepared, image_size, map_dir
    )
    test_paths = [record["path"] for record in split["test"]]
    test_maps = _collect_maps(data_root, test_paths, prepared, image_size, map_dir)
    write_scores(
        category_dir / "scores.json",
        validation_maps,
        test_maps,
        extra={
            "method": "part1_difference",
            "category": category,
            "image_size": image_size,
            "labels_used_for_threshold": False,
        },
    )
    write_category_metrics(split, category_dir / "scores.json", data_root)


def main() -> None:
    thread_limit = os.environ.get("CV2_NUM_THREADS")
    if thread_limit:
        cv2.setNumThreads(int(thread_limit))
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
