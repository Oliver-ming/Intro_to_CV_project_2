"""Fuse two saved score files that use the shared detection_score field."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.inspection.threshold import decide, operating_threshold
from src.part3.fuse_scores import fuse_detection_scores


def _scores(payload: dict, section: str, keys: list[str]) -> list[float]:
    return [float(payload[section][key]["detection_score"]) for key in keys]


def fuse_score_files(left_path: Path, right_path: Path, out_path: Path) -> dict:
    left = json.loads(left_path.read_text(encoding="utf-8"))
    right = json.loads(right_path.read_text(encoding="utf-8"))
    validation_keys = sorted(set(left["validation"]) & set(right["validation"]))
    test_keys = sorted(set(left["test"]) & set(right["test"]))
    if not validation_keys or not test_keys:
        raise ValueError("the two score files do not share validation and test images")
    fused_validation = fuse_detection_scores(
        _scores(left, "validation", validation_keys),
        _scores(right, "validation", validation_keys),
        _scores(left, "validation", validation_keys),
        _scores(right, "validation", validation_keys),
    )
    fused_test = fuse_detection_scores(
        _scores(left, "test", test_keys),
        _scores(right, "test", test_keys),
        _scores(left, "validation", validation_keys),
        _scores(right, "validation", validation_keys),
    )
    threshold = operating_threshold(fused_validation)
    payload = {
        "threshold": threshold,
        "threshold_rule": "mean_plus_3std_of_fused_validation_detection_scores",
        "detection_score": "mean_of_zscored_prompt_and_visual_scores",
        "validation": {
            key: {
                "detection_score": float(score),
                "decision": decide(float(score), threshold),
            }
            for key, score in zip(validation_keys, fused_validation)
        },
        "test": {
            key: {
                "detection_score": float(score),
                "decision": decide(float(score), threshold),
            }
            for key, score in zip(test_keys, fused_test)
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    fuse_score_files(args.left, args.right, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
