"""Comparison table built from metric files that already exist."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.inspection.collect_tables import (
    collect_anomalyclip,
    collect_anomalydino,
    collect_efficiency,
    collect_multiscale,
    collect_partial_images,
    collect_part1,
    write_comparison,
)


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

    def test_anomalyclip_metrics_keep_the_category_name(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "mvtec_zero_shot"
            run.mkdir()
            (run / "metrics.json").write_text(
                json.dumps({"dataset": "mvtec", "per_category": [{"category": "bottle", "image_auroc": 0.91}]}),
                encoding="utf-8",
            )
            rows = collect_anomalyclip(root)
            self.assertEqual(rows[0]["per_category"][0]["category"], "bottle")

    def test_multiscale_averages_each_resolution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "fusion.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "category": "bottle",
                            "coarse_448": {"image_auroc": 0.4, "image_ap": 0.4, "pixel_auroc": 0.4, "pixel_ap": 0.4, "aupro": 0.4},
                            "fine_672": {"image_auroc": 0.6, "image_ap": 0.6, "pixel_auroc": 0.6, "pixel_ap": 0.6, "aupro": 0.6},
                            "fused": {"image_auroc": 0.8, "image_ap": 0.8, "pixel_auroc": 0.8, "pixel_ap": 0.8, "aupro": 0.8},
                        },
                        {
                            "category": "cable",
                            "coarse_448": {"image_auroc": 0.2, "image_ap": 0.2, "pixel_auroc": 0.2, "pixel_ap": 0.2, "aupro": 0.2},
                            "fine_672": {"image_auroc": 0.2, "image_ap": 0.2, "pixel_auroc": 0.2, "pixel_ap": 0.2, "aupro": 0.2},
                            "fused": {"image_auroc": 0.4, "image_ap": 0.4, "pixel_auroc": 0.4, "pixel_ap": 0.4, "aupro": 0.4},
                        },
                    ]
                ),
                encoding="utf-8",
            )
            summary = collect_multiscale(path)
            self.assertEqual(summary["n_categories"], 2)
            self.assertAlmostEqual(summary["mean"]["fused"]["image_auroc"], 0.6)
            self.assertEqual(collect_multiscale(Path(tmp) / "missing.json")["n_categories"], 0)

    def test_efficiency_skips_a_seed_without_official_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "results_MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic"
            shot.mkdir(parents=True)
            header = "Object,Sample,Anomaly_Score,MemoryBank_Time,Inference_Time\n"
            (shot / "measurements_seed=0.csv").write_text(
                header + "bottle,good/000,0.1,1.5,0.2\n",
                encoding="utf-8",
            )
            (shot / "measurements_seed=1.csv").write_text(
                header + "bottle,good/000,0.1,4.0,0.8\n",
                encoding="utf-8",
            )
            (shot / "metrics_seed=0.json").write_text("{}", encoding="utf-8")
            rows = collect_efficiency(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seed"], 0)
            self.assertAlmostEqual(rows[0]["scoring_seconds_mean"], 0.2)
            self.assertEqual(rows[0]["dataset"], "MVTec")

    def test_partial_image_scores_share_categories_across_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            shot = root / "results" / "results_MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic"
            shot.mkdir(parents=True)
            (shot / "metrics_seed=0.json").write_text(
                json.dumps(
                    {
                        "bottle": {"classification_AUROC": 0.90, "classification_AP": 0.80},
                        "cable": {"classification_AUROC": 0.50, "classification_AP": 0.40},
                    }
                ),
                encoding="utf-8",
            )
            partial = root / "partial" / "MVTec" / "dinov2_vits14_448" / "1-shot_preprocess=agnostic" / "seed=2"
            partial.mkdir(parents=True)
            (partial / "bottle.json").write_text(
                json.dumps({"category": "bottle", "image_auroc": 1.0, "image_ap": 1.0}),
                encoding="utf-8",
            )
            (partial / "tile.json").write_text(
                json.dumps({"category": "tile", "image_auroc": 0.2, "image_ap": 0.2}),
                encoding="utf-8",
            )
            rows = collect_partial_images(root / "results", root / "partial")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["n_shared_categories"], 1)
            self.assertAlmostEqual(rows[0]["seed_means"]["image_auroc"]["mean"], 0.95)
            self.assertAlmostEqual(rows[0]["per_category"]["bottle"]["image_ap"]["mean"], 0.9)
            self.assertEqual(rows[0]["per_category"]["tile"]["image_auroc"]["n"], 1)


if __name__ == "__main__":
    unittest.main()
