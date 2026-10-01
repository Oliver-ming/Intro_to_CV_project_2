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
from src.inspection.categories import MVTEC_CATEGORIES, MVTEC_OBJECTS, MVTEC_TEXTURES
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


def _group_mean(per_category: dict, names: tuple[str, ...]) -> dict:
    present = [category for category in names if category in per_category]
    summary = {}
    for key in PART1_KEYS:
        values = [per_category[category][key] for category in present if key in per_category[category]]
        if values:
            summary[key] = _mean(values)
    return {"n_categories": len(present), "mean": summary}


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
    return {
        "per_category": per_category,
        "mean": summary,
        "n_categories": len(per_category),
        "groups": {
            "texture": _group_mean(per_category, MVTEC_TEXTURES),
            "object": _group_mean(per_category, MVTEC_OBJECTS),
        },
    }


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


def group_seed_means(seed_payloads: list[dict], names: tuple[str, ...]) -> dict:
    """Mean each finished seed over a fixed category group, then across seeds.

    A seed is included only when every category in the group is present.
    """
    per_seed = []
    for payload in seed_payloads:
        if not all(isinstance(payload.get(name), dict) and "classification_AUROC" in payload[name] for name in names):
            continue
        per_seed.append(
            {
                metric: sum(float(payload[name][metric]) for name in names) / len(names)
                for metric in CATEGORY_METRICS
            }
        )
    metrics = {}
    if per_seed:
        for metric in CATEGORY_METRICS:
            metrics[metric] = mean_and_std([row[metric] for row in per_seed])
    return {"n_seeds": len(per_seed), "n_categories": len(names), "metrics": metrics}


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
                "groups": {
                    "object": group_seed_means(payloads, MVTEC_OBJECTS),
                    "texture": group_seed_means(payloads, MVTEC_TEXTURES),
                }
                if dataset == "MVTec"
                else {},
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


def collect_multiscale_seeds(paths: list[Path]) -> dict:
    """Mean and sample standard deviation across finished multi-scale seeds.

    Seed-level means use only categories present in every file, so a partial
    seed does not change the denominator.
    """
    per_seed = []
    for path in paths:
        if not path.is_file():
            continue
        item = collect_multiscale(path)
        if item["n_categories"]:
            per_seed.append(item)
    if not per_seed:
        return {"n_seeds": 0, "n_shared_categories": 0, "seed_means": {}, "per_category": {}}
    shared = set.intersection(*(set(item["per_category"]) for item in per_seed))
    if not shared:
        return {"n_seeds": len(per_seed), "n_shared_categories": 0, "seed_means": {}, "per_category": {}}
    per_category = {}
    for category in sorted(shared):
        per_category[category] = {}
        for branch in FUSION_BRANCHES:
            per_category[category][branch] = {}
            for key in FUSION_KEYS:
                values = [item["per_category"][category][branch][key] for item in per_seed]
                per_category[category][branch][key] = mean_and_std(values)
    seed_means = {}
    for branch in FUSION_BRANCHES:
        seed_means[branch] = {}
        for key in FUSION_KEYS:
            per_seed_mean = []
            for item in per_seed:
                values = [item["per_category"][category][branch][key] for category in shared]
                per_seed_mean.append(sum(values) / len(values))
            seed_means[branch][key] = mean_and_std(per_seed_mean)
    return {
        "n_seeds": len(per_seed),
        "n_shared_categories": len(shared),
        "seed_means": seed_means,
        "per_category": per_category,
    }


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


_CONTRAST_PAIRS = (
    ("image_auroc", "classification_AUROC", "image_auroc"),
    ("image_ap", "classification_AP", "image_ap"),
    ("pixel_auroc", "seg_AUROC", "pixel_auroc"),
)


def collect_prompt_fusion(directory: Path) -> dict:
    """Mean image metrics from fused prompt-visual files that already have them."""
    per_category = {}
    if directory.is_dir():
        for path in sorted(directory.glob("*/*.json")):
            if path.stem.endswith("_visual") or path.stem.endswith("_prompt"):
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            if "image_auroc" not in payload or "image_ap" not in payload:
                continue
            per_category[path.stem] = {
                "image_auroc": float(payload["image_auroc"]),
                "image_ap": float(payload["image_ap"]),
            }
    mean = {}
    for key in ("image_auroc", "image_ap"):
        values = [row[key] for row in per_category.values()]
        if values:
            mean[key] = _mean(values)
    return {"n_categories": len(per_category), "mean": mean, "per_category": per_category}


def contrast_zero_shot(anomalydino_rows: list[dict], anomalyclip_rows: list[dict]) -> list[dict]:
    """Place AnomalyCLIP's one zero-shot run next to AnomalyDINO 448 1-shot.

    AnomalyDINO contributes the mean and sample standard deviation across seeds.
    Pixel AUPRO is left out: AnomalyCLIP integrates 200 thresholds, while the
    AnomalyDINO number is the official sorted sweep.
    """
    prompts = {}
    for row in anomalyclip_rows:
        categories = {}
        for item in row.get("per_category") or []:
            if isinstance(item, dict) and "category" in item and "image_auroc" in item:
                categories[item["category"]] = item
        if categories:
            prompts[str(row.get("dataset", "")).casefold()] = categories
    contrasts = []
    for row in anomalydino_rows:
        if row.get("shot") != 1 or row.get("preprocess") != "agnostic":
            continue
        if "448" not in str(row.get("model_resolution", "")):
            continue
        clip = prompts.get(str(row.get("dataset", "")).casefold())
        if not clip:
            continue
        shared = sorted(set(row.get("per_category", {})) & set(clip))
        per_category = {}
        for category in shared:
            metrics = {}
            for name, dino_key, clip_key in _CONTRAST_PAIRS:
                dino_metric = row["per_category"][category].get(dino_key)
                if not isinstance(dino_metric, dict) or "mean" not in dino_metric or clip_key not in clip[category]:
                    continue
                clip_value = float(clip[category][clip_key])
                metrics[name] = {
                    "anomalydino_mean": float(dino_metric["mean"]),
                    "anomalydino_std": float(dino_metric["std"]),
                    "anomalydino_n": int(dino_metric["n"]),
                    "anomalyclip": clip_value,
                    "delta": clip_value - float(dino_metric["mean"]),
                }
            if metrics:
                per_category[category] = metrics
        if not per_category:
            continue
        means = {}
        for name, _, _ in _CONTRAST_PAIRS:
            deltas = [item[name]["delta"] for item in per_category.values() if name in item]
            dino_values = [item[name]["anomalydino_mean"] for item in per_category.values() if name in item]
            clip_values = [item[name]["anomalyclip"] for item in per_category.values() if name in item]
            if deltas:
                means[name] = {
                    "anomalydino": _mean(dino_values),
                    "anomalyclip": _mean(clip_values),
                    "delta": _mean(deltas),
                }
        contrasts.append(
            {
                "dataset": row["dataset"],
                "anomalydino": "448_1-shot",
                "n_categories": len(per_category),
                "mean": means,
                "per_category": per_category,
            }
        )
    return contrasts


def write_comparison(
    out_path: Path,
    part1_dirs: dict[str, Path],
    anomalydino_root: Path | None,
    anomalyclip_dir: Path | None = None,
    multiscale_path: Path | None = None,
    partial_image_root: Path | None = None,
    multiscale_seed_paths: list[Path] | None = None,
    prompt_fusion_dir: Path | None = None,
) -> dict:
    anomalydino = collect_anomalydino(anomalydino_root) if anomalydino_root else []
    anomalyclip = collect_anomalyclip(anomalyclip_dir) if anomalyclip_dir else []
    table = {
        "part1": {name: collect_part1(path) for name, path in part1_dirs.items()},
        "anomalydino": anomalydino,
        "anomalyclip": anomalyclip,
        "zero_shot_contrast": contrast_zero_shot(anomalydino, anomalyclip),
        "prompt_fusion": collect_prompt_fusion(prompt_fusion_dir) if prompt_fusion_dir else {"n_categories": 0, "mean": {}, "per_category": {}},
        "multiscale": collect_multiscale(multiscale_path) if multiscale_path else {"n_categories": 0, "per_category": {}, "mean": {}},
        "multiscale_seeds": collect_multiscale_seeds(multiscale_seed_paths or []),
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
        sorted((root / "outputs/part3/multiscale").glob("1shot_seed*_all15.json")),
        root / "outputs/part3/prompt_fusion",
    )
    part1_counts = {name: row["n_categories"] for name, row in table["part1"].items()}
    print(json.dumps({
        "part1_categories": part1_counts,
        "anomalydino_rows": len(table["anomalydino"]),
        "anomalyclip_rows": len(table["anomalyclip"]),
        "multiscale_categories": table["multiscale"]["n_categories"],
        "efficiency_rows": len(table["efficiency"]),
        "image_rows": len(table["anomalydino_image"]),
        "multiscale_seeds": table["multiscale_seeds"]["n_seeds"],
        "zero_shot_contrast": len(table["zero_shot_contrast"]),
        "prompt_fusion": table["prompt_fusion"]["n_categories"],
    }))


if __name__ == "__main__":
    main()
