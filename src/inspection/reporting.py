"""Save image scores and the operating point without reading test labels."""

from __future__ import annotations

import json
from pathlib import Path

from src.inspection.threshold import decide, operating_threshold


def write_scores(
    destination: Path,
    validation_scores: dict[str, float],
    test_scores: dict[str, float],
    extra: dict | None = None,
) -> dict:
    threshold = operating_threshold(list(validation_scores.values()))
    payload = {
        "threshold": threshold,
        "validation_scores": validation_scores,
        "test_scores": test_scores,
        "validation_decisions": {
            path: decide(score, threshold) for path, score in validation_scores.items()
        },
        "test_decisions": {
            path: decide(score, threshold) for path, score in test_scores.items()
        },
    }
    if extra:
        payload.update(extra)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload
