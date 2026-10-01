"""AnomalyDINO's measurement CSV is the source for scoring time."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inspection.efficiency import bank_slot_table, memory_bank_slots, summarize_by_category, summarize_measurements


class EfficiencyTest(unittest.TestCase):
    def test_mean_scoring_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            path.write_text(
                "Object,Sample,Anomaly_Score,MemoryBank_Time,Inference_Time\n"
                "bottle,good/000,0.1,1.0,0.2\n"
                "bottle,good/001,0.2,1.0,0.4\n",
                encoding="utf-8",
            )
            summary = summarize_measurements(path)
        self.assertEqual(summary["n_images"], 2)
        self.assertAlmostEqual(summary["memory_bank_seconds_mean"], 1.0)
        self.assertAlmostEqual(summary["scoring_seconds_mean"], 0.3)

    def test_one_shot_448_bank_has_eight_rotations(self) -> None:
        self.assertEqual(memory_bank_slots(1, 448), 8 * 32 * 32)

    def test_eight_shot_672_is_eighteen_times_one_shot_448(self) -> None:
        rows = {(row["shots"], row["resolution"]): row["slots"] for row in bank_slot_table()}
        self.assertEqual(rows[(8, 672)], 18 * rows[(1, 448)])

    def test_bank_time_is_once_per_category(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "measurements.csv"
            path.write_text(
                "Object,Sample,Anomaly_Score,MemoryBank_Time,Inference_Time\n"
                "bottle,good/000,0.1,1.0,0.2\n"
                "bottle,good/001,0.2,1.0,0.4\n"
                "cable,good/000,0.3,3.0,0.9\n",
                encoding="utf-8",
            )
            summary = summarize_by_category(path)
        self.assertEqual(summary["n_categories"], 2)
        self.assertEqual(summary["n_images"], 3)
        self.assertAlmostEqual(summary["memory_bank_seconds_mean"], 2.0)
        self.assertAlmostEqual(summary["scoring_seconds_mean"], 0.5)
        self.assertAlmostEqual(summary["per_category"]["cable"]["memory_bank_seconds"], 3.0)


if __name__ == "__main__":
    unittest.main()
