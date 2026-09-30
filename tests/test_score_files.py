"""Score files used to fuse a prompt score with a visual distance."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.part2.score_files import (
    detection_score_file,
    group_anomalyclip_scores,
    scores_from_measurements,
    validation_detection_scores,
)


class ScoreFileTest(unittest.TestCase):
    def test_measurements_use_the_test_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "measurements.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Object": "bottle",
                        "Sample": "broken_large/000.png",
                        "Anomaly_Score": "0.42",
                        "MemoryBank_Time": "1",
                        "Inference_Time": "0.1",
                    }
                )
                writer.writerow(
                    {
                        "Object": "cable",
                        "Sample": "good/000.png",
                        "Anomaly_Score": "0.10",
                        "MemoryBank_Time": "1",
                        "Inference_Time": "0.1",
                    }
                )
            scores = scores_from_measurements(path, "bottle")
        self.assertEqual(scores, {"bottle/test/broken_large/000.png": 0.42})

    def test_heldout_validation_and_prompt_scores_share_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            heldout = root / "scores.json"
            heldout.write_text(
                json.dumps({"validation": {"bottle/train/good/199.png": {"detection_score": 0.2}}}),
                encoding="utf-8",
            )
            grouped = group_anomalyclip_scores(
                [{"img_path": str(root / "data" / "bottle/test/good/000.png"), "detection_score": 0.8}],
                root / "data",
            )
            payload = detection_score_file(
                validation_detection_scores(heldout),
                grouped["bottle"],
            )
        self.assertIn("bottle/train/good/199.png", payload["validation"])
        self.assertEqual(payload["test"]["bottle/test/good/000.png"]["detection_score"], 0.8)


if __name__ == "__main__":
    unittest.main()
