"""Build the score files that prompt-visual fusion reads.

AnomalyDINO's measurement table has one detection score per test image.
AnomalyCLIP's test hook writes the same kind of list. Validation scores come
from the held-out tail. Fusion matches the two files on those paths.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path


def scores_from_measurements(csv_path: Path, category: str) -> dict[str, float]:
    scores: dict[str, float] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["Object"] != category:
                continue
            scores[f"{category}/test/{row['Sample']}"] = float(row["Anomaly_Score"])
    if not scores:
        raise ValueError(f"no measurements for {category} in {csv_path}")
    return scores


def validation_detection_scores(scores_path: Path) -> dict[str, float]:
    payload = json.loads(scores_path.read_text(encoding="utf-8"))
    return {key: float(row["detection_score"]) for key, row in payload["validation"].items()}


def relative_to_root(img_path: str, data_root: Path) -> str:
    path = Path(img_path)
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError:
        return path.as_posix()


def group_anomalyclip_scores(rows: list[dict], data_root: Path) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, float]] = {}
    for row in rows:
        relative = relative_to_root(row["img_path"], data_root)
        category = relative.split("/", 1)[0]
        grouped.setdefault(category, {})[relative] = float(row["detection_score"])
    return grouped


def detection_score_file(
    validation: dict[str, float],
    test: dict[str, float],
    extra: dict | None = None,
) -> dict:
    if not validation or not test:
        raise ValueError("fusion needs both validation and test scores")

    def pack(scores: dict[str, float]) -> dict[str, dict[str, float]]:
        return {key: {"detection_score": float(value)} for key, value in scores.items()}

    payload = {"validation": pack(validation), "test": pack(test)}
    if extra:
        payload.update(extra)
    return payload


def write_detection_score_file(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
