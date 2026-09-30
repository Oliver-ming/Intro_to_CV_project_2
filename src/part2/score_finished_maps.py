"""Score AnomalyDINO categories only after every test grid is on disk.

Official ``metrics_seed`` files appear after a whole seed, including the
slow full-resolution pass. This reads patch grids one finished category
at a time and writes image AUROC and AP. Categories that already have an
official metrics entry, or an output file, are left alone.
"""

from __future__ import annotations

import json
from pathlib import Path

from src.inspection.categories import IMAGE_SUFFIXES
from src.inspection.completion import completion_status
from src.inspection.splits import load_split
from src.part2.score_patch_maps import category_image_metrics
from src.part2.score_visa_maps import visa_image_metrics

MAP_JOBS = ("anomalydino_mvtec", "anomalydino_mvtec_672", "anomalydino_visa")


def _npy_count(directory: Path) -> int:
    return sum(1 for path in directory.rglob("*.npy") if path.is_file())


def _image_count(directory: Path) -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


def visa_expected_count(data_root: Path, category: str) -> int:
    root = data_root / category / "test"
    return _image_count(root / "good") + _image_count(root / "bad")


def _official_categories(metrics_path: Path) -> set[str]:
    if not metrics_path.is_file():
        return set()
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    return {key for key, value in payload.items() if isinstance(value, dict) and "classification_AUROC" in value}


def score_ready_maps(
    anomalydino_root: Path,
    out_dir: Path,
    mvtec_splits: Path,
    visa_root: Path,
) -> list[dict]:
    """Write one JSON per newly finished category. Return the new rows."""
    written = []
    if not anomalydino_root.is_dir():
        return written
    for seed_dir in sorted(anomalydino_root.glob("results_*/**/anomaly_maps/seed=*")):
        if not seed_dir.is_dir():
            continue
        shot_dir = seed_dir.parent.parent
        dataset = shot_dir.parent.parent.name.removeprefix("results_")
        model = shot_dir.parent.name
        finished = _official_categories(shot_dir / f"metrics_{seed_dir.name}.json")
        for category_dir in sorted(path for path in seed_dir.iterdir() if path.is_dir()):
            category = category_dir.name
            if category in finished:
                continue
            destination = out_dir / dataset / model / shot_dir.name / seed_dir.name / f"{category}.json"
            if destination.is_file():
                continue
            summary = _score_one(dataset, category, category_dir, mvtec_splits, visa_root)
            if summary is None:
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "dataset": dataset,
                "model_resolution": model,
                "shot_dir": shot_dir.name,
                "seed": seed_dir.name,
                **summary,
            }
            destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            written.append(payload)
    return written


def _score_one(
    dataset: str,
    category: str,
    category_dir: Path,
    mvtec_splits: Path,
    visa_root: Path,
) -> dict | None:
    try:
        if dataset == "MVTec":
            split_path = mvtec_splits / f"{category}.json"
            if not split_path.is_file():
                return None
            expected = len(load_split(split_path)["test"])
            if expected == 0 or _npy_count(category_dir) != expected:
                return None
            return category_image_metrics(category_dir, split_path)
        if dataset == "VisA":
            expected = visa_expected_count(visa_root, category)
            if expected == 0 or _npy_count(category_dir) != expected:
                return None
            summary = visa_image_metrics(category_dir)
            if summary["n_test"] != expected:
                return None
            return summary
    except (OSError, ValueError, FileNotFoundError, json.JSONDecodeError):
        return None
    return None


def main() -> None:
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--stop-when-maps-done",
        action="store_true",
        help="Exit 0 when the three AnomalyDINO map jobs have printed their markers, else exit 2.",
    )
    args = parser.parse_args()
    root = args.root
    written = score_ready_maps(
        root / "third_party/AnomalyDINO",
        root / "outputs/part2/anomalydino_partial/image_scores",
        root / "outputs/splits/mvtec",
        root / "data/VisA_pytorch/1cls",
    )
    print(json.dumps({"wrote": len(written), "categories": [row["category"] for row in written]}), flush=True)
    if not args.stop_when_maps_done:
        return
    status = completion_status(root)
    if all(status[name] for name in MAP_JOBS):
        print("MAP_SCORER_DONE", flush=True)
        return
    sys.exit(2)


if __name__ == "__main__":
    main()
