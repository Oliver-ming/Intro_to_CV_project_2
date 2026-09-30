"""Fuse two directories of anomaly maps that share filenames."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.part3.fusion import pixel_mean_std
from src.part3.multiscale import fuse_resolutions, upsample_to


def fuse_directory(
    fine_dir: Path,
    coarse_dir: Path,
    validation_names: list[str],
    out_dir: Path,
) -> Path:
    if not validation_names:
        raise ValueError("need validation maps to calibrate the fusion")
    fine_validation = []
    coarse_validation = []
    for name in validation_names:
        fine = np.load(fine_dir / name)
        coarse = upsample_to(np.load(coarse_dir / name), fine.shape[:2])
        fine_validation.append(fine)
        coarse_validation.append(coarse)
    fine_mean, fine_std = pixel_mean_std(fine_validation)
    coarse_mean, coarse_std = pixel_mean_std(coarse_validation)
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for fine_path in sorted(fine_dir.glob("*.npy")):
        coarse_path = coarse_dir / fine_path.name
        if not coarse_path.is_file():
            continue
        fine = np.load(fine_path)
        fused = fuse_resolutions(
            fine,
            np.load(coarse_path),
            fine_mean,
            fine_std,
            coarse_mean,
            coarse_std,
        )
        np.save(out_dir / fine_path.name, fused)
        written += 1
    if written == 0:
        raise FileNotFoundError(f"no shared maps in {fine_dir} and {coarse_dir}")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fine-dir", type=Path, required=True)
    parser.add_argument("--coarse-dir", type=Path, required=True)
    parser.add_argument("--validation", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    names = [path.name for path in args.validation]
    fuse_directory(args.fine_dir, args.coarse_dir, names, args.out_dir)
    print(f"wrote {args.out_dir}")


if __name__ == "__main__":
    main()
