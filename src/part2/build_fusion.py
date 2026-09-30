"""Join AnomalyDINO and AnomalyCLIP scores, then fuse them.

The visual side is a measurements table plus the held-out validation file.
The prompt side is ``image_scores.json`` plus the same kind of validation file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.part2.score_files import (
    detection_score_file,
    group_anomalyclip_scores,
    scores_from_measurements,
    validation_detection_scores,
    write_detection_score_file,
)
from src.part3.apply_score_fusion import fuse_score_files


def build_fusion(
    visual_measurements: Path,
    visual_validation: Path,
    prompt_image_scores: Path,
    prompt_validation: Path,
    data_root: Path,
    category: str,
    out_path: Path,
) -> dict:
    prompt_rows = json.loads(prompt_image_scores.read_text(encoding="utf-8"))
    grouped = group_anomalyclip_scores(prompt_rows, data_root)
    if category not in grouped:
        raise ValueError(f"no AnomalyCLIP scores for {category}")
    visual = detection_score_file(
        validation_detection_scores(visual_validation),
        scores_from_measurements(visual_measurements, category),
        extra={"method": "anomalydino", "category": category},
    )
    prompt = detection_score_file(
        validation_detection_scores(prompt_validation),
        grouped[category],
        extra={"method": "anomalyclip", "category": category},
    )
    work = out_path.parent
    visual_path = work / f"{category}_visual.json"
    prompt_path = work / f"{category}_prompt.json"
    write_detection_score_file(visual_path, visual)
    write_detection_score_file(prompt_path, prompt)
    return fuse_score_files(visual_path, prompt_path, out_path)


def fuse_ready_categories(
    categories: list[str],
    visual_measurements: Path,
    visual_validation_dir: Path,
    prompt_image_scores: Path,
    prompt_validation_dir: Path,
    data_root: Path,
    out_dir: Path,
) -> list[str]:
    """Fuse categories whose visual and prompt files both exist. Skip finished outputs."""
    if not prompt_image_scores.is_file() or not visual_measurements.is_file():
        return []
    written = []
    for category in categories:
        out_path = out_dir / f"{category}.json"
        visual_validation = visual_validation_dir / category / "scores.json"
        prompt_validation = prompt_validation_dir / category / "scores.json"
        if out_path.is_file() or not visual_validation.is_file() or not prompt_validation.is_file():
            continue
        build_fusion(
            visual_measurements,
            visual_validation,
            prompt_image_scores,
            prompt_validation,
            data_root,
            category,
            out_path,
        )
        written.append(category)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-measurements", type=Path, required=True)
    parser.add_argument("--visual-validation", type=Path, required=True)
    parser.add_argument("--prompt-image-scores", type=Path, required=True)
    parser.add_argument("--prompt-validation", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fused = build_fusion(
        args.visual_measurements,
        args.visual_validation,
        args.prompt_image_scores,
        args.prompt_validation,
        args.data_root,
        args.category,
        args.out,
    )
    print(json.dumps({"out": str(args.out), "n_test": len(fused["test"]), "threshold": fused["threshold"]}))


if __name__ == "__main__":
    main()
