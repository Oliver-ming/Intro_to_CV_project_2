"""Shared normal-image splits.

MVTec few-shot slices stay on the sorted filename prefix, matching
AnomalyDINO. Threshold images are taken from the tail so those slices do
not move. The wild-set helper still draws a fixed random holdout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.inspection.categories import IMAGE_SUFFIXES, MVTEC_CATEGORIES, VISA_CATEGORIES

OFFICIAL_SHOTS = (1, 2, 4, 8, 16)
OFFICIAL_SEEDS = 3


def list_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def split_mvtec_normals(
    paths: list[Path],
    val_count: int = 10,
    shots: tuple[int, ...] = OFFICIAL_SHOTS,
    num_seeds: int = OFFICIAL_SEEDS,
) -> tuple[list[Path], list[Path], int]:
    """Return ``(reference_pool, threshold_validation, protected_prefix_length)``.

    The pool always contains the official prefix
    ``sorted(paths)[:protected_prefix_length]`` for the largest shot that
    still leaves a tail. Validation images are the last ``val_count`` names
    in that tail. Nothing is removed from disk.
    """
    ordered = sorted(paths)
    if not ordered:
        raise ValueError("no defect-free images to split")
    if len(ordered) == 1:
        return ordered, [], 1
    protected = 0
    for shot in sorted(shots, reverse=True):
        need = shot * num_seeds
        if len(ordered) - need >= 1:
            protected = need
            break
    tail = ordered[protected:]
    count = min(val_count, len(tail))
    validation = tail[-count:] if count else []
    held = set(validation)
    pool = [path for path in ordered if path not in held]
    return pool, validation, protected


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


def defect_mask_path(category_root: Path, defect_type: str, image_path: Path) -> Path:
    """Mask next to a defect image.

    MVTec names the mask ``<stem>_mask.png``. VisA names it ``<stem>.png``.
    """
    ground_truth = category_root / "ground_truth" / defect_type
    candidates = [
        ground_truth / f"{image_path.stem}_mask{image_path.suffix}",
        ground_truth / f"{image_path.stem}_mask.png",
        ground_truth / f"{image_path.stem}.png",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"missing mask for {image_path}")


def build_mvtec_category(
    data_root: Path, category: str, val_count: int = 10, dataset: str = "mvtec"
) -> dict:
    category_root = data_root / category
    train_good = list_images(category_root / "train" / "good")
    if not train_good:
        raise FileNotFoundError(f"missing train/good for {category} under {data_root}")
    pool, validation, protected = split_mvtec_normals(train_good, val_count=val_count)
    ordered = sorted(train_good)
    official_prefix = [_relative(path, data_root) for path in ordered[:protected]]

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
                mask_path = defect_mask_path(category_root, defect_type, image_path)
                record["mask"] = _relative(mask_path, data_root)
            test_records.append(record)

    return {
        "dataset": dataset,
        "category": category,
        "reference_pool": [_relative(path, data_root) for path in pool],
        "threshold_validation": [_relative(path, data_root) for path in validation],
        "official_shots": list(OFFICIAL_SHOTS),
        "official_seeds": list(range(OFFICIAL_SEEDS)),
        "official_sampling": "sorted train/good filenames, slice [seed * k:(seed + 1) * k]",
        "protected_prefix_length": protected,
        "official_reference_prefix": official_prefix,
        "test": test_records,
    }


def write_category_splits(
    data_root: Path,
    out_dir: Path,
    categories: list[str],
    dataset: str,
    val_count: int = 10,
) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for category in categories:
        payload = build_mvtec_category(data_root, category, val_count=val_count, dataset=dataset)
        destination = out_dir / f"{category}.json"
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(destination)
    return written


def write_mvtec_splits(data_root: Path, out_dir: Path, val_count: int = 10) -> list[Path]:
    return write_category_splits(data_root, out_dir, list(MVTEC_CATEGORIES), "mvtec", val_count=val_count)


def write_visa_splits(data_root: Path, out_dir: Path, val_count: int = 10) -> list[Path]:
    return write_category_splits(data_root, out_dir, list(VISA_CATEGORIES), "visa", val_count=val_count)


def load_split(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
