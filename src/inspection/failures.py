"""Group wrong OK/NG decisions by the cause the inspection report needs."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.inspection.evaluate_run import map_filename
from src.inspection.images import read_mask
from src.inspection.splits import load_split


def categorize_failure(
    label: str,
    decision: str,
    anomaly: np.ndarray,
    mask: np.ndarray | None,
    small_defect_pixels: int = 64,
) -> str | None:
    """Return a cause name for a wrong decision, or None when the decision is right."""
    if label == decision:
        return None
    heat = np.asarray(anomaly, dtype=np.float64)
    height, width = heat.shape[:2]
    border = max(1, min(height, width) // 10)
    border_mask = np.zeros_like(heat, dtype=bool)
    border_mask[:border, :] = True
    border_mask[-border:, :] = True
    border_mask[:, :border] = True
    border_mask[:, -border:] = True
    border_mean = float(heat[border_mask].mean()) if border_mask.any() else 0.0
    center_mean = float(heat[~border_mask].mean()) if (~border_mask).any() else 0.0
    if label == "OK" and decision == "NG":
        if border_mean > center_mean:
            return "background_or_border"
        return "normal_variation"
    defect_pixels = 0 if mask is None else int(np.count_nonzero(mask))
    if defect_pixels > 0 and defect_pixels < small_defect_pixels:
        return "small_defect"
    return "missed_defect"


def collect_failures(split: dict, scores: dict, map_dir: Path, data_root: Path) -> list[dict]:
    """Wrong decisions for one category, using the saved anomaly maps."""
    rows = []
    for record in split["test"]:
        entry = scores["test"][record["path"]]
        anomaly = np.load(map_dir / map_filename(record["path"]))
        mask = None if record["mask"] is None else read_mask(data_root / record["mask"])
        cause = categorize_failure(record["label"], entry["decision"], anomaly, mask)
        if cause is None:
            continue
        rows.append(
            {
                "path": record["path"],
                "label": record["label"],
                "decision": entry["decision"],
                "cause": cause,
                "detection_score": entry["detection_score"],
            }
        )
    return rows


def summarize_causes(rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row["cause"]] = counts.get(row["cause"], 0) + 1
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=Path, required=True)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    split = load_split(args.split)
    scores = json.loads(args.scores.read_text(encoding="utf-8"))
    rows = collect_failures(split, scores, args.scores.parent / "maps", args.data_root)
    payload = {"n_failures": len(rows), "causes": summarize_causes(rows), "failures": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"n_failures": payload["n_failures"], "causes": payload["causes"]}))


if __name__ == "__main__":
    main()
