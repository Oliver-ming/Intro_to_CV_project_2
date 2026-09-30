"""Clear background scores on saved anomaly maps using GrabCut."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.inspection.images import read_rgb, resize_rgb
from src.inspection.splits import load_split
from src.part3.background import grabcut_foreground, suppress_background


def map_name(relative: str) -> str:
    path = Path(relative)
    return f"{path.parent.name}_{path.stem}.npy"


def suppress_directory(
    split_path: Path,
    data_root: Path,
    map_dir: Path,
    out_dir: Path,
    image_size: int,
) -> int:
    split = load_split(split_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    relatives = list(split["threshold_validation"]) + [record["path"] for record in split["test"]]
    for relative in relatives:
        source = map_dir / map_name(relative)
        if not source.is_file():
            continue
        anomaly = np.load(source)
        image = resize_rgb(read_rgb(data_root / relative), image_size)
        foreground = grabcut_foreground(image)
        np.save(out_dir / source.name, suppress_background(anomaly, foreground))
        written += 1
    if written == 0:
        raise FileNotFoundError(f"no maps from {map_dir} matched {split_path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--map-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-size", type=int, default=256)
    args = parser.parse_args()
    count = suppress_directory(args.split, args.data_root, args.map_dir, args.out_dir, args.image_size)
    print(f"wrote {count} maps to {args.out_dir}")


if __name__ == "__main__":
    main()
