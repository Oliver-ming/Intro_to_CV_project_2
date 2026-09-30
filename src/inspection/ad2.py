"""MVTec AD 2 public test images.

The public split is the only one with pixel ground truth. Private images are
not read here, and nothing in this module writes a benchmark submission.
"""

from __future__ import annotations

from pathlib import Path

AD2_OBJECTS = (
    "can",
    "fabric",
    "fruit_jelly",
    "rice",
    "sheet_metal",
    "vial",
    "wallplugs",
    "walnuts",
)


def public_test_records(root: Path, category: str) -> list[dict]:
    """Defect-free and anomalous images from ``test_public``, with masks on the latter."""
    if category not in AD2_OBJECTS:
        raise ValueError(f"unknown MVTec AD 2 category {category}")
    base = root / category / "test_public"
    records = []
    for label, folder in (("OK", "good"), ("NG", "bad")):
        image_dir = base / folder
        if not image_dir.is_dir():
            raise FileNotFoundError(image_dir)
        for image in sorted(image_dir.glob("*.png")):
            mask = None
            if label == "NG":
                mask = base / "ground_truth" / "bad" / f"{image.stem}_mask.png"
                if not mask.is_file():
                    raise FileNotFoundError(mask)
            records.append(
                {
                    "path": image.relative_to(root).as_posix(),
                    "label": label,
                    "mask": None if mask is None else mask.relative_to(root).as_posix(),
                }
            )
    if not records:
        raise FileNotFoundError(f"no public test images for {category} under {root}")
    return records
