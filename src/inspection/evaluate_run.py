"""Turn a saved score file into image-level metrics. Pixel metrics are optional."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.inspection.images import read_mask
from src.inspection.metrics import aupro, false_positive_rate, safe_ap, safe_auroc
from src.inspection.splits import load_split


def map_filename(relative: str) -> str:
    path = Path(relative)
    return f"{path.parent.name}_{path.stem}.npy"


def image_metrics(split: dict, scores: dict) -> dict:
    labels = []
    label_names = []
    detection = []
    decisions = []
    for record in split["test"]:
        entry = scores["test"][record["path"]]
        labels.append(1 if record["label"] == "NG" else 0)
        label_names.append(record["label"])
        detection.append(entry["detection_score"])
        decisions.append(entry["decision"])
    return {
        "image_auroc": safe_auroc(np.asarray(labels), np.asarray(detection)),
        "image_ap": safe_ap(np.asarray(labels), np.asarray(detection)),
        "false_positive_rate": false_positive_rate(label_names, decisions),
        "n_test": len(labels),
    }


def pixel_metrics(
    split: dict,
    data_root: Path,
    map_dir: Path,
    integration_limit: float = 0.3,
    align_to: str = "mask",
) -> dict:
    if align_to not in {"mask", "map"}:
        raise ValueError("align_to must be 'mask' or 'map'")
    maps = []
    masks = []
    for record in split["test"]:
        anomaly = np.load(map_dir / map_filename(record["path"]))
        if record["mask"] is None:
            if align_to == "map":
                mask = np.zeros(anomaly.shape[:2], dtype=np.uint8)
            else:
                image_path = data_root / record["path"]
                sample = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
                if sample is None:
                    raise FileNotFoundError(image_path)
                mask = np.zeros(sample.shape, dtype=np.uint8)
        else:
            mask = read_mask(data_root / record["mask"])
        if align_to == "map":
            mask = cv2.resize(mask, (anomaly.shape[1], anomaly.shape[0]), interpolation=cv2.INTER_NEAREST)
            resized = anomaly
        else:
            resized = cv2.resize(anomaly, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
        maps.append(resized)
        masks.append(mask)
    flat_labels = np.concatenate([(mask > 0).astype(np.uint8).ravel() for mask in masks])
    flat_scores = np.concatenate([anomaly.ravel() for anomaly in maps])
    return {
        "pixel_auroc": safe_auroc(flat_labels, flat_scores),
        "pixel_ap": safe_ap(flat_labels, flat_scores),
        "aupro": aupro(maps, masks, integration_limit=integration_limit),
        "aupro_integration_limit": integration_limit,
    }


def evaluate_directory(split_path: Path, scores_path: Path, data_root: Path) -> dict:
    split = load_split(split_path)
    scores = json.loads(scores_path.read_text(encoding="utf-8"))
    summary = image_metrics(split, scores)
    map_dir = scores_path.parent / "maps"
    if map_dir.is_dir():
        summary.update(pixel_metrics(split, data_root, map_dir))
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    summary = evaluate_directory(args.split, args.scores, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
