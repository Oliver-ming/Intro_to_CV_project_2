"""Link the official VisA 1-class split into AnomalyDINO's folder layout.

The archive stays in ``data/visa_raw``. ``data/VisA_pytorch/1cls`` only
contains symlinks, with anomaly images under ``test/bad`` and masks named
``<stem>.png`` inside ``ground_truth/bad``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_symlink() or destination.exists():
        return
    destination.symlink_to(source.resolve())


def prepare(raw_root: Path, out_root: Path) -> dict[str, int]:
    table = raw_root / "split_csv" / "1cls.csv"
    if not table.is_file():
        raise FileNotFoundError(f"missing {table}; extract VisA_20220922.tar into {raw_root}")
    counts = {"train_good": 0, "test_good": 0, "test_bad": 0, "masks": 0}
    with table.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            image = raw_root / row["image"]
            if not image.is_file():
                raise FileNotFoundError(image)
            name = image.name
            category = out_root / row["object"]
            if row["split"] == "train":
                link(image, category / "train" / "good" / name)
                counts["train_good"] += 1
            elif row["label"] == "normal":
                link(image, category / "test" / "good" / name)
                counts["test_good"] += 1
            else:
                link(image, category / "test" / "bad" / name)
                counts["test_bad"] += 1
                mask = raw_root / row["mask"]
                if not mask.is_file():
                    raise FileNotFoundError(mask)
                link(mask, category / "ground_truth" / "bad" / f"{image.stem}.png")
                counts["masks"] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, default=Path("data/visa_raw"))
    parser.add_argument("--out-root", type=Path, default=Path("data/VisA_pytorch/1cls"))
    args = parser.parse_args()
    counts = prepare(args.raw_root, args.out_root)
    print(counts)


if __name__ == "__main__":
    main()
