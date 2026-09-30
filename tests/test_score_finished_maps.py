"""A category is scored only when its patch-grid count matches the split."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.part2.score_finished_maps import score_ready_maps


class ScoreFinishedMapsTest(unittest.TestCase):
    def test_partial_category_is_skipped_until_the_count_matches(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seed = root / "results_MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic" / "anomaly_maps" / "seed=2" / "bottle"
            (seed / "test" / "good").mkdir(parents=True)
            (seed / "test" / "broken").mkdir(parents=True)
            np.save(seed / "test" / "good" / "000.npy", np.array([[0.1, 0.2], [0.0, 0.3]], dtype=np.float32))
            split_dir = root / "splits"
            split_dir.mkdir()
            split = {
                "category": "bottle",
                "test": [
                    {"path": "bottle/test/good/000.png", "label": "OK", "mask": None},
                    {"path": "bottle/test/broken/000.png", "label": "NG", "mask": None},
                ],
            }
            (split_dir / "bottle.json").write_text(json.dumps(split), encoding="utf-8")
            out = root / "out"
            self.assertEqual(score_ready_maps(root, out, split_dir, root / "visa"), [])
            np.save(seed / "test" / "broken" / "000.npy", np.array([[0.9, 0.8], [0.7, 1.0]], dtype=np.float32))
            written = score_ready_maps(root, out, split_dir, root / "visa")
            self.assertEqual([row["category"] for row in written], ["bottle"])
            self.assertEqual(score_ready_maps(root, out, split_dir, root / "visa"), [])

    def test_official_metrics_skip_that_category(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "results_MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic"
            seed = shot / "anomaly_maps" / "seed=0" / "bottle" / "test" / "good"
            seed.mkdir(parents=True)
            np.save(seed / "000.npy", np.zeros((2, 2), dtype=np.float32))
            split_dir = root / "splits"
            split_dir.mkdir()
            split = {"category": "bottle", "test": [{"path": "bottle/test/good/000.png", "label": "OK", "mask": None}]}
            (split_dir / "bottle.json").write_text(json.dumps(split), encoding="utf-8")
            (shot / "metrics_seed=0.json").write_text(
                json.dumps({"bottle": {"classification_AUROC": 0.99}}),
                encoding="utf-8",
            )
            self.assertEqual(score_ready_maps(root, root / "out", split_dir, root / "visa"), [])


if __name__ == "__main__":
    unittest.main()
