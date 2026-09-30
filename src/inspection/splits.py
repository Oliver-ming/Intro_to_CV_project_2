"""Shared normal-image splits.

Threshold images are held out from defect-free training images. Few-shot
references and the Part 1 normality model both use only the remaining pool.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.inspection.categories import IMAGE_SUFFIXES, MVTEC_CATEGORIES


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def hold_out_validation(
    paths: list[Path], val_count: int = 10, seed: int = 0
) -> tuple[list[Path], list[Path]]:
    """Return ``(reference_pool, threshold_validation)``.

    The draw is deterministic for a fixed sorted path list and seed.
    """
    if not paths:
        raise ValueError("no defect-free images to split")
    ordered = sorted(paths)
    if len(ordered) == 1:
        return ordered, ordered
    count = min(val_count, len(ordered) - 1)
    rng = random.Random(seed)
    chosen = set(rng.sample(range(len(ordered)), count))
    validation = [ordered[index] for index in range(len(ordered)) if index in chosen]
    pool = [ordered[index] for index in range(len(ordered)) if index not in chosen]
    return pool, validation


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_mvtec_category(
    data_root: Path, category: str, val_count: int = 10, seed: int = 0
) -> dict:
    category_root = data_root / category
    train_good = list_images(category_root / "train" / "good")
    if not train_good:
        raise FileNotFoundError(f"missing train/good for {category} under {data_root}")
    pool, validation = hold_out_validation(train_good, val_count=val_count, seed=seed)

    test_records = []
    test_root = category_root / "test"
    for defect_dir in sorted(path for path in test_root.iterdir() if path.is_dir()):
        defect_type = defect_dir.name
        for image_path in list_images(defect_dir):
            record = {
                "path": _relative(image_path, data_root),
                "label": "OK" if defect_type == "good" else "NG",
                "defect_type": defect_type,
                "mask": None,
            }
            if defect_type != "good":
                mask_path = (
                    category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask{image_path.suffix}"
                )
                if not mask_path.is_file():
                    png_mask = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                    mask_path = png_mask if png_mask.is_file() else mask_path
                if not mask_path.is_file():
                    raise FileNotFoundError(f"missing mask for {image_path}")
                record["mask"] = _relative(mask_path, data_root)
            test_records.append(record)

    return {
        "dataset": "mvtec",
        "category": category,
        "seed": seed,
        "reference_pool": [_relative(path, data_root) for path in pool],
        "threshold_validation": [_relative(path, data_root) for path in validation],
        "test": test_records,
    }


def write_mvtec_splits(
    data_root: Path, out_dir: Path, val_count: int = 10, seed: int = 0
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for category in MVTEC_CATEGORIES:
        payload = build_mvtec_category(data_root, category, val_count=val_count, seed=seed)
        destination = out_dir / f"{category}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(destination)
    return written


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
