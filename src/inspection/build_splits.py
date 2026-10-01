"""Write the shared MVTec normal-image split used by every method."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.inspection.splits import write_mvtec_splits, write_visa_splits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--val-count", type=int, default=10)
    parser.add_argument("--dataset", choices=("mvtec", "visa"), default="mvtec")
    args = parser.parse_args()
    if args.dataset == "visa":
        written = write_visa_splits(args.data_root, args.out_dir, val_count=args.val_count)
    else:
        written = write_mvtec_splits(args.data_root, args.out_dir, val_count=args.val_count)
    print(f"wrote {len(written)} splits to {args.out_dir}")


if __name__ == "__main__":
    main()
