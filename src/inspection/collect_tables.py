"""Gather finished metric files into one comparison table.

Part 1 rows come from ``metrics.json`` written by ``evaluate_run``.
AnomalyDINO rows come from each ``metrics_seed=*.json``. Seeds are averaged
with the sample standard deviation. The multi-scale file contributes the
448, 672, and fused means. Missing files are skipped, so this can run
before every job has finished.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.inspection.aggregate_seeds import SEED_SUMMARY_KEYS, aggregate_seed_metrics, mean_and_std
from src.inspection.categories import MVTEC_CATEGORIES
from src.inspection.efficiency import summarize_by_category

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


FUSION_BRANCHES = ("coarse_448", "fine_672", "fused")
FUSION_KEYS = ("image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "aupro")


def collect_multiscale(path: Path) -> dict:
    """Mean the 448, 672, and fused scores stored for one finished shot."""
    if not path.is_file():
        return {"n_categories": 0, "per_category": {}, "mean": {}}
    rows = json.loads(path.read_text(encoding="utf-8"))
    per_category = {}
    for row in rows:
        per_category[row["category"]] = {
            branch: {key: float(row[branch][key]) for key in FUSION_KEYS} for branch in FUSION_BRANCHES
        }
    mean = {}
    for branch in FUSION_BRANCHES:
        mean[branch] = {}
        for key in FUSION_KEYS:
            values = [per_category[category][branch][key] for category in per_category]
            mean[branch][key] = _mean(values)
    return {"n_categories": len(per_category), "per_category": per_category, "mean": mean}


def collect_efficiency(results_root: Path) -> list[dict]:
    """Timing rows for seeds whose official metrics file already exists.

    A measurements CSV grows while a seed is still running. The matching
    ``metrics_seed`` file is written only after that seed finishes, so
    partial tables stay out of the comparison.
    """
    rows = []
    if not results_root.is_dir():
        return rows
    for csv_path in sorted(results_root.glob("results_*/**/measurements_seed=*.csv")):
        metrics_name = csv_path.name.replace("measurements_", "metrics_").replace(".csv", ".json")
        if not (csv_path.parent / metrics_name).is_file():
            continue
        shot_dir = csv_path.parent
        match = _SHOT_DIR.match(shot_dir.name)
        if match is None:
            continue
        seed_text = csv_path.stem.split("=", 1)[1]
        summary = summarize_by_category(csv_path)
        rows.append(
            {
                "dataset": shot_dir.parent.parent.name.removeprefix("results_"),
                "model_resolution": shot_dir.parent.name,
                "shot": int(match.group("shot")),
                "preprocess": match.group("preprocess"),
                "seed": int(seed_text),
                "n_images": summary["n_images"],
                "n_categories": summary["n_categories"],
                "scoring_seconds_mean": summary["scoring_seconds_mean"],
                "memory_bank_seconds_mean": summary["memory_bank_seconds_mean"],
                "memory_bank_seconds_min": summary["memory_bank_seconds_min"],
                "memory_bank_seconds_max": summary["memory_bank_seconds_max"],
                "per_category": summary["per_category"],
            }
        )
    return rows


def collect_partial_images(results_root: Path, partial_root: Path) -> list[dict]:
    """Image AUROC and AP, including seeds whose official file is not written yet.

    Official ``metrics_seed`` wins when both exist. Seed means use only the
    categories present in every seed, so a half-finished seed does not change
    the denominator.
    """
    grouped: dict[tuple, dict[int, dict[str, dict[str, float]]]] = {}

    def bucket(dataset: str, model: str, shot_dir: str) -> dict[int, dict[str, dict[str, float]]]:
        return grouped.setdefault((dataset, model, shot_dir), {})

    if results_root.is_dir():
        for metrics_path in sorted(results_root.glob("results_*/**/metrics_seed=*.json")):
            shot_dir = metrics_path.parent
            if _SHOT_DIR.match(shot_dir.name) is None:
                continue
            payload = json.loads(metrics_path.read_text(encoding="utf-8"))
            rows = {}
            for category, value in payload.items():
                if isinstance(value, dict) and "classification_AUROC" in value and "classification_AP" in value:
                    rows[category] = {
                        "image_auroc": float(value["classification_AUROC"]),
                        "image_ap": float(value["classification_AP"]),
                    }
            if rows:
                seed = int(metrics_path.stem.split("=", 1)[1])
                dataset = shot_dir.parent.parent.name.removeprefix("results_")
                bucket(dataset, shot_dir.parent.name, shot_dir.name)[seed] = rows

    if partial_root.is_dir():
        for path in sorted(partial_root.glob("*/*/*/*/*.json")):
            shot_dir = path.parent.parent.name
            if _SHOT_DIR.match(shot_dir) is None or not path.parent.name.startswith("seed="):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            category = payload.get("category")
            if not isinstance(category, str) or "image_auroc" not in payload or "image_ap" not in payload:
                continue
            seed = int(path.parent.name.split("=", 1)[1])
            dataset = path.parent.parent.parent.parent.name
            model = path.parent.parent.parent.name
            seeds = bucket(dataset, model, shot_dir)
            if category in seeds.get(seed, {}):
                continue
            seeds.setdefault(seed, {})[category] = {
                "image_auroc": float(payload["image_auroc"]),
                "image_ap": float(payload["image_ap"]),
            }

    table = []
    for (dataset, model, shot_name), seeds in sorted(grouped.items()):
        match = _SHOT_DIR.match(shot_name)
        if match is None or not seeds:
            continue
        categories: set[str] = set()
        for rows in seeds.values():
            categories.update(rows)
        per_category = {}
        for category in sorted(categories):
            metrics = {}
            for key in ("image_auroc", "image_ap"):
                values = [rows[category][key] for rows in seeds.values() if category in rows]
                if values:
                    metrics[key] = mean_and_std(values)
            per_category[category] = metrics
        shared = set.intersection(*(set(rows) for rows in seeds.values()))
        seed_means = {}
        if shared:
            for key in ("image_auroc", "image_ap"):
                per_seed = [
                    sum(rows[category][key] for category in shared) / len(shared) for rows in seeds.values()
                ]
                seed_means[key] = mean_and_std(per_seed)
        table.append(
            {
                "dataset": dataset,
                "model_resolution": model,
                "shot": int(match.group("shot")),
                "preprocess": match.group("preprocess"),
                "n_seeds": len(seeds),
                "n_shared_categories": len(shared),
                "seed_means": seed_means,
                "per_category": per_category,
            }
        )
    return table


def collect_anomalyclip(directory: Path) -> list[dict]:
    """Read metrics.json files written after the AnomalyCLIP test hook."""
    rows = []
    if not directory.is_dir():
        return rows
    for path in sorted(directory.glob("**/metrics.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        categories = payload.get("per_category")
        if not isinstance(categories, list) or not categories:
            continue
        if not isinstance(categories[0], dict) or "category" not in categories[0]:
            continue
        rows.append({"path": str(path), **payload})
    return rows


def write_comparison(
    out_path: Path,
    part1_dirs: dict[str, Path],
    anomalydino_root: Path | None,
    anomalyclip_dir: Path | None = None,
    multiscale_path: Path | None = None,
    partial_image_root: Path | None = None,
) -> dict:
    table = {
        "part1": {name: collect_part1(path) for name, path in part1_dirs.items()},
        "anomalydino": collect_anomalydino(anomalydino_root) if anomalydino_root else [],
        "anomalyclip": collect_anomalyclip(anomalyclip_dir) if anomalyclip_dir else [],
        "multiscale": collect_multiscale(multiscale_path) if multiscale_path else {"n_categories": 0, "per_category": {}, "mean": {}},
        "efficiency": collect_efficiency(anomalydino_root) if anomalydino_root else [],
        "anomalydino_image": collect_partial_images(anomalydino_root, partial_image_root)
        if anomalydino_root is not None and partial_image_root is not None
        else [],
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
            "grabcut": root / "outputs/part3/grabcut",
        },
        root / "third_party/AnomalyDINO",
        root / "outputs/part2/anomalyclip",
        root / "outputs/part3/multiscale/1shot_seed0_all15.json",
        root / "outputs/part2/anomalydino_partial/image_scores",
    )
    part1_counts = {name: row["n_categories"] for name, row in table["part1"].items()}
    print(json.dumps({
        "part1_categories": part1_counts,
        "anomalydino_rows": len(table["anomalydino"]),
        "anomalyclip_rows": len(table["anomalyclip"]),
        "multiscale_categories": table["multiscale"]["n_categories"],
        "efficiency_rows": len(table["efficiency"]),
        "image_rows": len(table["anomalydino_image"]),
    }))


if __name__ == "__main__":
    main()
