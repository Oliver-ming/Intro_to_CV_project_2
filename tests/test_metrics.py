"""Synthetic maps for the metric definitions."""

from __future__ import annotations

import unittest

import numpy as np

from src.inspection.metrics import aupro, false_positive_rate, safe_ap, safe_auroc


class MetricTest(unittest.TestCase):
    def test_perfect_separation_scores_one(self) -> None:
        labels = np.array([0, 0, 1, 1])
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        self.assertAlmostEqual(safe_auroc(labels, scores), 1.0)
        self.assertGreater(safe_ap(labels, scores), 0.99)

    def test_false_positive_rate_counts_only_ok_images(self) -> None:
        labels = ["OK", "OK", "NG"]
        decisions = ["NG", "OK", "NG"]
        self.assertAlmostEqual(false_positive_rate(labels, decisions), 0.5)

    def test_aupro_is_high_when_the_defect_is_the_hottest_region(self) -> None:
        anomaly = np.zeros((8, 8), dtype=np.float64)
        mask = np.zeros((8, 8), dtype=np.uint8)
        anomaly[2:4, 2:4] = 1.0
        mask[2:4, 2:4] = 1
        score = aupro([anomaly], [mask], integration_limit=0.3)
        self.assertGreater(score, 0.9)


if __name__ == "__main__":
    unittest.main()
