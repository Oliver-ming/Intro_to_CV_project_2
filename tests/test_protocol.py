"""Checks that do not need images or a GPU."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.inspection.splits import build_mvtec_category, hold_out_validation, split_mvtec_normals
from src.inspection.threshold import decide, mean_top1p, operating_threshold


class SplitTest(unittest.TestCase):
    def test_wild_holdout_keeps_twenty_references(self) -> None:
        paths = [Path(f"{index:03d}.png") for index in range(30)]
        pool_a, val_a = hold_out_validation(paths, val_count=10, seed=0)
        pool_b, val_b = hold_out_validation(paths, val_count=10, seed=0)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 10)
        self.assertEqual(len(pool_a), 20)
        self.assertEqual(set(pool_a).intersection(val_a), set())

    def test_mvtec_tail_stays_out_of_the_official_prefix(self) -> None:
        paths = [Path(f"{index:03d}.png") for index in range(60)]
        pool, validation, protected = split_mvtec_normals(paths, val_count=10)
        prefix = paths[:protected]
        self.assertEqual(protected, 48)
        self.assertEqual(validation, paths[-10:])
        self.assertTrue(set(prefix).isdisjoint(validation))
        self.assertTrue(set(prefix).issubset(pool))

    def test_visa_defect_mask_is_the_stem_png(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            category = root / "candle"
            (category / "train" / "good").mkdir(parents=True)
            (category / "test" / "bad").mkdir(parents=True)
            (category / "test" / "good").mkdir(parents=True)
            (category / "ground_truth" / "bad").mkdir(parents=True)
            for index in range(60):
                (category / "train" / "good" / f"{index:04d}.JPG").write_bytes(b"x")
            (category / "test" / "good" / "000.JPG").write_bytes(b"x")
            (category / "test" / "bad" / "000.JPG").write_bytes(b"x")
            (category / "ground_truth" / "bad" / "000.png").write_bytes(b"x")
            payload = build_mvtec_category(root, "candle", dataset="visa")
        self.assertEqual(payload["dataset"], "visa")
        self.assertEqual(payload["protected_prefix_length"], 48)
        self.assertEqual(payload["threshold_validation"], [f"candle/train/good/{index:04d}.JPG" for index in range(50, 60)])
        bad = next(record for record in payload["test"] if record["label"] == "NG")
        self.assertEqual(bad["mask"], "candle/ground_truth/bad/000.png")
        good = next(record for record in payload["test"] if record["label"] == "OK")
        self.assertIsNone(good["mask"])


class ScoreTest(unittest.TestCase):
    def test_threshold_is_mean_plus_three_population_std_of_pixels(self) -> None:
        pixels = np.array([1.0, 2.0, 3.0])
        mean = 2.0
        std = (2.0 / 3.0) ** 0.5
        threshold = operating_threshold(pixels)
        self.assertAlmostEqual(threshold, mean + 3.0 * std)
        self.assertEqual(decide(threshold, threshold), "OK")
        self.assertEqual(decide(threshold + 1e-6, threshold), "NG")

    def test_detection_score_is_mean_of_highest_one_percent(self) -> None:
        values = np.arange(100, dtype=np.float64)
        self.assertEqual(mean_top1p(values), 99.0)
        short = np.arange(50, dtype=np.float64)
        self.assertEqual(mean_top1p(short), 49.0)


if __name__ == "__main__":
    unittest.main()
