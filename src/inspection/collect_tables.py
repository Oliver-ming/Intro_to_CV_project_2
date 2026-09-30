"""Gather finished metric files into one comparison table.

Part 1 rows come from ``metrics.json`` written by ``evaluate_run``.
AnomalyDINO rows come from each ``metrics_seed=*.json``. Seeds are averaged
with the sample standard deviation. Missing files are skipped, so this can
run before every job has finished.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.inspection.aggregate_seeds import SEED_SUMMARY_KEYS, aggregate_seed_metrics, mean_and_std
from src.inspection.categories import MVTEC_CATEGORIES

PART1_KEYS = (
    "image_auroc",
    "image_ap",
    "pixel_auroc",
    "pixel_ap",
    "aupro",
    "false_positive_rate",
)

CATEGORY_METRICS = (
    "classification_AUROC",
    "classification_AP",
    "seg_AUROC",
    "seg_AUPRO",
)

_SHOT_DIR = re.compile(r"^(?P<shot>\d+)-shot_preprocess=(?P<preprocess>.+)$")


def _mean(values: list[float]) -> float | None:
    finite = [value for value in values if value == value]
    if not finite:
        return None
    return float(sum(finite) / len(finite))


def collect_part1(method_dir: Path) -> dict:
    per_category = {}
    for category in MVTEC_CATEGORIES:
        path = method_dir / category / "metrics.json"
        if not path.is_file():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        per_category[category] = {key: payload[key] for key in PART1_KEYS if key in payload}
    summary = {}
    for key in PART1_KEYS:
        present = [row[key] for row in per_category.values() if key in row]
        if present:
            summary[key] = _mean(present)
    return {"per_category": per_category, "mean": summary, "n_categories": len(per_category)}


def per_category_from_seeds(seed_payloads: list[dict]) -> dict[str, dict[str, dict[str, float]]]:
    names: set[str] = set()
    for payload in seed_payloads:
        for key, value in payload.items():
            if isinstance(value, dict):
                names.add(key)
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for category in sorted(names):
        metrics = {}
        for metric in CATEGORY_METRICS:
            values = []
            for payload in seed_payloads:
                row = payload.get(category)
                if isinstance(row, dict) and metric in row:
                    values.append(float(row[metric]))
            if values:
                metrics[metric] = mean_and_std(values)
        if metrics:
            summary[category] = metrics
    return summary


def collect_anomalydino(results_root: Path) -> list[dict]:
    """One record per dataset, model resolution, shot, and preprocessing."""
    found: list[tuple[tuple, dict]] = []
    if not results_root.is_dir():
        return []
    for metrics_path in sorted(results_root.glob("results_*/**/metrics_seed=*.json")):
        shot_dir = metrics_path.parent
        match = _SHOT_DIR.match(shot_dir.name)
        if match is None:
            continue
        key = (
            shot_dir.parent.parent.name.removeprefix("results_"),
            shot_dir.parent.name,
            int(match.group("shot")),
            match.group("preprocess"),
        )
        found.append((key, json.loads(metrics_path.read_text(encoding="utf-8"))))
    grouped: dict[tuple, list[dict]] = {}
    for key, payload in found:
        grouped.setdefault(key, []).append(payload)
    table = []
    for key, payloads in sorted(grouped.items()):
        dataset, model_resolution, shot, preprocess = key
        table.append(
            {
                "dataset": dataset,
                "model_resolution": model_resolution,
                "shot": shot,
                "preprocess": preprocess,
                "metrics": aggregate_seed_metrics(payloads),
                "per_category": per_category_from_seeds(payloads),
            }
        )
    return table


def write_comparison(out_path: Path, part1_dirs: dict[str, Path], anomalydino_root: Path | None) -> dict:
    table = {
        "part1": {name: collect_part1(path) for name, path in part1_dirs.items()},
        "anomalydino": collect_anomalydino(anomalydino_root) if anomalydino_root else [],
        "anomalydino_keys": list(SEED_SUMMARY_KEYS),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")
    return table


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("outputs/tables/comparison.json"))
    args = parser.parse_args()
    root = args.root
    table = write_comparison(
        args.out,
        {
            "difference": root / "outputs/part1/difference",
            "autoencoder": root / "outputs/part1/autoencoder",
        },
        root / "third_party/AnomalyDINO",
    )
    part1_counts = {name: row["n_categories"] for name, row in table["part1"].items()}
    print(json.dumps({"part1_categories": part1_counts, "anomalydino_rows": len(table["anomalydino"])}))


if __name__ == "__main__":
    main()
