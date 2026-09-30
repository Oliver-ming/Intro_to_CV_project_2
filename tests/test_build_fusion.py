"""Fusing a visual distance with a prompt score on the same images."""

from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.part2.build_fusion import build_fusion, fuse_ready_categories


class BuildFusionTest(unittest.TestCase):
    def test_equal_and_opposite_z_scores_cancel(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            measurements = root / "measurements.csv"
            with measurements.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Object": "bottle",
                        "Sample": "broken_large/000.png",
                        "Anomaly_Score": "2",
                        "MemoryBank_Time": "1",
                        "Inference_Time": "0.1",
                    }
                )
            visual_validation = root / "visual.json"
            visual_validation.write_text(
                json.dumps(
                    {
                        "validation": {
                            "bottle/train/good/000.png": {"detection_score": 0},
                            "bottle/train/good/001.png": {"detection_score": 2},
                        }
                    }
                ),
                encoding="utf-8",
            )
            prompt_validation = root / "prompt_validation.json"
            prompt_validation.write_text(visual_validation.read_text(encoding="utf-8"), encoding="utf-8")
            image_scores = root / "image_scores.json"
            image_scores.write_text(
                json.dumps(
                    [
                        {
                            "img_path": str(data / "bottle/test/broken_large/000.png"),
                            "detection_score": 0,
                        }
                    ]
                ),
                encoding="utf-8",
            )
            fused = build_fusion(
                measurements,
                visual_validation,
                image_scores,
                prompt_validation,
                data,
                "bottle",
                root / "fused.json",
            )
        self.assertAlmostEqual(fused["test"]["bottle/test/broken_large/000.png"]["detection_score"], 0.0)
        self.assertEqual(fused["test"]["bottle/test/broken_large/000.png"]["decision"], "OK")

    def test_ready_categories_skip_missing_prompt_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            measurements = root / "measurements.csv"
            with measurements.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=["Object", "Sample", "Anomaly_Score", "MemoryBank_Time", "Inference_Time"],
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "Object": "bottle",
                        "Sample": "broken_large/000.png",
                        "Anomaly_Score": "2",
                        "MemoryBank_Time": "1",
                        "Inference_Time": "0.1",
                    }
                )
            visual = root / "visual" / "bottle"
            visual.mkdir(parents=True)
            (visual / "scores.json").write_text(
                json.dumps({"validation": {"bottle/train/good/000.png": {"detection_score": 0}, "bottle/train/good/001.png": {"detection_score": 2}}}),
                encoding="utf-8",
            )
            prompt = root / "prompt" / "bottle"
            prompt.mkdir(parents=True)
            (prompt / "scores.json").write_text((visual / "scores.json").read_text(encoding="utf-8"), encoding="utf-8")
            image_scores = root / "image_scores.json"
            image_scores.write_text(
                json.dumps([{"img_path": str(data / "bottle/test/broken_large/000.png"), "detection_score": 0}]),
                encoding="utf-8",
            )
            written = fuse_ready_categories(
                ["bottle", "cable"],
                measurements,
                root / "visual",
                image_scores,
                root / "prompt",
                data,
                root / "out",
            )
            self.assertEqual(written, ["bottle"])
            self.assertTrue((root / "out" / "bottle.json").is_file())
            again = fuse_ready_categories(
                ["bottle"],
                measurements,
                root / "visual",
                image_scores,
                root / "prompt",
                data,
                root / "out",
            )
            self.assertEqual(again, [])


if __name__ == "__main__":
    unittest.main()
