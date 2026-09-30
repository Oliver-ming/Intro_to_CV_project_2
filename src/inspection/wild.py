"""Split for the self-collected mouse images.

Controlled defect-free images are divided with the fixed seed agreed for the
wild set. Lighting and background shifts stay in the test set.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.inspection.categories import IMAGE_SUFFIXES
from src.inspection.splits import hold_out_validation, list_images

DEFECT_TYPES = ("scratch", "ink", "tape_or_missing")
SHIFT_CONDITIONS = ("shift_light", "shift_background")


def _relative(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def build_wild_split(root: Path, val_count: int = 10, seed: int = 0) -> dict:
    goods = list_images(root / "controlled" / "good")
    pool, validation = hold_out_validation(goods, val_count=val_count, seed=seed)
    tests = []
    for condition in SHIFT_CONDITIONS:
        for image in list_images(root / condition / "good"):
            tests.append(
                {
                    "path": _relative(image, root),
                    "label": "OK",
                    "defect_type": condition,
                    "mask": None,
                }
            )
    for defect in DEFECT_TYPES:
        for image in list_images(root / "controlled" / defect):
            mask = root / "masks" / defect / f"{image.stem}.png"
            if not mask.is_file():
                raise FileNotFoundError(mask)
            if mask.suffix.lower() not in IMAGE_SUFFIXES:
                raise FileNotFoundError(mask)
            tests.append(
                {
                    "path": _relative(image, root),
                    "label": "NG",
                    "defect_type": defect,
                    "mask": _relative(mask, root),
                }
            )
    return {
        "dataset": "wild_mouse",
        "category": "mouse",
        "seed": seed,
        "reference_pool": [_relative(path, root) for path in pool],
        "threshold_validation": [_relative(path, root) for path in validation],
        "test": tests,
    }


def write_wild_split(root: Path, destination: Path, val_count: int = 10, seed: int = 0) -> Path:
    payload = build_wild_split(root, val_count=val_count, seed=seed)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return destination
