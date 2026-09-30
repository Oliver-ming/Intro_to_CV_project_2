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

    def test_aupro_is_one_when_the_defect_is_strictly_hotter(self) -> None:
        anomaly = np.zeros((8, 8), dtype=np.float64)
        mask = np.zeros((8, 8), dtype=np.uint8)
        anomaly[2:4, 2:4] = 1.0
        mask[2:4, 2:4] = 1
        score = aupro([anomaly], [mask], integration_limit=0.3)
        self.assertAlmostEqual(score, 1.0)

    def test_aupro_matches_a_direct_threshold_sweep(self) -> None:
        anomaly = np.array(
            [
                [0.2, 0.9, 0.1],
                [0.4, 0.8, 0.0],
                [0.3, 0.7, 0.5],
            ],
            dtype=np.float64,
        )
        mask = np.zeros((3, 3), dtype=np.uint8)
        mask[0, 1] = 1
        mask[1, 1] = 1
        score = aupro([anomaly], [mask], integration_limit=0.3)
        normal = np.flatnonzero(mask.ravel() == 0)
        region = np.flatnonzero(mask.ravel() == 1)
        flat = anomaly.ravel()
        fprs = [0.0]
        pros = [0.0]
        for threshold in np.unique(flat)[::-1]:
            fprs.append(float(np.count_nonzero(flat[normal] >= threshold)) / normal.size)
            pros.append(float(np.count_nonzero(flat[region] >= threshold)) / region.size)
        fprs.append(1.0)
        pros.append(1.0)
        order = np.argsort(fprs)
        xs = np.asarray(fprs, dtype=np.float64)[order]
        ys = np.asarray(pros, dtype=np.float64)[order]
        cutoff = xs <= 0.3
        area = float(np.trapezoid(ys[cutoff], xs[cutoff]))
        if xs[cutoff][-1] < 0.3:
            span = 0.3 - xs[cutoff][-1]
            next_index = int(np.searchsorted(xs, 0.3, side="left"))
            y_at = ys[next_index - 1] + (ys[next_index] - ys[next_index - 1]) * span / (xs[next_index] - xs[next_index - 1])
            area += 0.5 * (ys[cutoff][-1] + y_at) * span
        self.assertAlmostEqual(score, area / 0.3, places=6)

    def test_aupro_accepts_images_of_different_sizes(self) -> None:
        first = np.zeros((4, 4), dtype=np.float64)
        first_mask = np.zeros((4, 4), dtype=np.uint8)
        first[0, 0] = 1.0
        first_mask[0, 0] = 1
        second = np.zeros((5, 3), dtype=np.float64)
        second_mask = np.zeros((5, 3), dtype=np.uint8)
        second[1, 1] = 1.0
        second_mask[1, 1] = 1
        score = aupro([first, second], [first_mask, second_mask])
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
