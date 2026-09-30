"""Recompute detection scores after a map has been edited.

Background suppression and multi-scale fusion both write new ``.npy`` maps.
The OK/NG threshold has to be fit again on the validation maps, still without
reading test labels.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.inspection.reporting import write_scores
from src.inspection.splits import load_split
from src.part3.apply_background import map_name


def load_named_maps(relative_paths: list[str], map_dir: Path) -> dict[str, np.ndarray]:
    maps = {}
    for relative in relative_paths:
        path = map_dir / map_name(relative)
        if not path.is_file():
            raise FileNotFoundError(path)
        maps[relative] = np.load(path)
    return maps


def rescore_directory(split_path: Path, map_dir: Path, out_path: Path, extra: dict | None = None) -> dict:
    split = load_split(split_path)
    validation = load_named_maps(list(split["threshold_validation"]), map_dir)
    test_paths = [record["path"] for record in split["test"]]
    test = load_named_maps(test_paths, map_dir)
    payload_extra = {"category": split["category"], "labels_used_for_threshold": False}
    if extra:
        payload_extra.update(extra)
    return write_scores(out_path, validation, test, extra=payload_extra)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--method", default="rescored")
    args = parser.parse_args()
    payload = rescore_directory(args.split, args.map_dir, args.out, extra={"method": args.method})
    print(json.dumps({"out": str(args.out), "threshold": payload["threshold"], "n_test": len(payload["test"])}))


if __name__ == "__main__":
    main()
