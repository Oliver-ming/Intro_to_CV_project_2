"""Seed aggregation for the few-shot stability table."""

from __future__ import annotations

import unittest

from src.inspection.aggregate_seeds import aggregate_seed_metrics


class AggregateTest(unittest.TestCase):
    def test_three_seeds_report_mean_and_sample_std(self) -> None:
        seeds = [
            {"mean_classification_au_roc": 0.90},
            {"mean_classification_au_roc": 0.96},
            {"mean_classification_au_roc": 0.93},
        ]
        summary = aggregate_seed_metrics(seeds)["mean_classification_au_roc"]
        self.assertAlmostEqual(summary["mean"], 0.93)
        self.assertEqual(summary["n"], 3)
        self.assertGreater(summary["std"], 0)


if __name__ == "__main__":
    unittest.main()
