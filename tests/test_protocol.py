"""Checks that do not need images or a GPU."""

from __future__ import annotations

import unittest
from pathlib import Path

from src.inspection.splits import hold_out_validation
from src.inspection.threshold import decide, operating_threshold


class SplitTest(unittest.TestCase):
    def test_holdout_is_deterministic_and_keeps_twenty_references(self) -> None:
        paths = [Path(f"{index:03d}.png") for index in range(30)]
        pool_a, val_a = hold_out_validation(paths, val_count=10, seed=0)
        pool_b, val_b = hold_out_validation(paths, val_count=10, seed=0)
        self.assertEqual(pool_a, pool_b)
        self.assertEqual(val_a, val_b)
        self.assertEqual(len(val_a), 10)
        self.assertEqual(len(pool_a), 20)
        self.assertEqual(set(pool_a).intersection(val_a), set())

    def test_threshold_is_mean_plus_three_std(self) -> None:
        scores = [1.0, 2.0, 3.0]
        mean = 2.0
        std = (2.0 / 3.0) ** 0.5
        self.assertAlmostEqual(operating_threshold(scores), mean + 3.0 * std)
        self.assertEqual(decide(mean + 3.0 * std, mean + 3.0 * std), "OK")
        self.assertEqual(decide(mean + 3.0 * std + 1e-6, mean + 3.0 * std), "NG")


if __name__ == "__main__":
    unittest.main()
