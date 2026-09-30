"""Score one test image with the Part 1 difference baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.inspection.splits import load_split
from src.inspection.threshold import decide, mean_top1p
from src.part1.difference import difference_map, image_score


def _overlay(image_path: Path, anomaly: np.ndarray, destination: Path, decision: str | None, score: float) -> None:
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(image_path)
    heat = cv2.resize(anomaly, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    heat = heat - heat.min()
    denom = float(heat.max()) or 1.0
    heat_u8 = np.clip(heat / denom * 255.0, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(bgr, 0.55, colored, 0.45, 0)
    label = f"score {score:.4f}"
    if decision is not None:
        label = f"{decision}  {label}"
    cv2.putText(blended, label, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), blended)


def inspect_image(
    image_path: Path,
    split_path: Path,
    data_root: Path,
    scores_path: Path | None,
    image_size: int,
    out_path: Path,
) -> dict:
    split = load_split(split_path)
    references = [data_root / relative for relative in split["reference_pool"]]
    anomaly, used = difference_map(image_path, references, image_size=image_size)
    payload = {
        "image": str(image_path),
        "reference": used,
        "detection_score": mean_top1p(anomaly),
        "max_pixel": image_score(anomaly),
        "decision": None,
        "threshold": None,
    }
    if scores_path is not None and scores_path.is_file():
        stored = json.loads(scores_path.read_text(encoding="utf-8"))
        payload["threshold"] = stored["threshold"]
        payload["decision"] = decide(payload["max_pixel"], stored["threshold"])
    _overlay(image_path, anomaly, out_path, payload["decision"], payload["detection_score"])
    payload["heatmap"] = str(out_path)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--scores", type=Path, default=None)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--out", type=Path, default=Path("outputs/demo/overlay.png"))
    args = parser.parse_args()
    result = inspect_image(args.image, args.split, args.data_root, args.scores, args.image_size, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
