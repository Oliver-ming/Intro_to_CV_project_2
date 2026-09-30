"""Comparison table built from metric files that already exist."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.inspection.collect_tables import collect_anomalydino, collect_part1, write_comparison


class CollectTablesTest(unittest.TestCase):
    def test_part1_mean_skips_categories_without_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bottle = root / "bottle"
            bottle.mkdir()
            (bottle / "metrics.json").write_text(
                json.dumps({"image_auroc": 0.8, "image_ap": 0.7, "aupro": 0.6}),
                encoding="utf-8",
            )
            summary = collect_part1(root)
            self.assertEqual(summary["n_categories"], 1)
            self.assertAlmostEqual(summary["mean"]["image_auroc"], 0.8)
            self.assertNotIn("cable", summary["per_category"])

    def test_anomalydino_seeds_are_averaged(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "results_MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic"
            shot.mkdir(parents=True)
            for seed, score in ((0, 0.90), (1, 0.96)):
                payload = {
                    "mean_classification_au_roc": score,
                    "bottle": {"classification_AUROC": score},
                }
                (shot / f"metrics_seed={seed}.json").write_text(json.dumps(payload), encoding="utf-8")
            rows = collect_anomalydino(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["dataset"], "MVTec")
            self.assertEqual(rows[0]["shot"], 1)
            self.assertAlmostEqual(rows[0]["metrics"]["mean_classification_au_roc"]["mean"], 0.93)
            self.assertAlmostEqual(rows[0]["per_category"]["bottle"]["classification_AUROC"]["mean"], 0.93)

    def test_write_comparison_creates_the_table_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            out = root / "comparison.json"
            table = write_comparison(out, {"difference": root / "missing"}, None)
            self.assertEqual(table["part1"]["difference"]["n_categories"], 0)
            self.assertTrue(out.is_file())


if __name__ == "__main__":
    unittest.main()
