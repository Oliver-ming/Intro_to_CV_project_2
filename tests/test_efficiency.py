"""AnomalyDINO's measurement CSV is the source for scoring time."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inspection.efficiency import summarize_measurements


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


if __name__ == "__main__":
    unittest.main()
