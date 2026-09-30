"""Patch-grid smoothing and the AnomalyCLIP metrics hook."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.part2.ensure_anomalyclip_metrics import MARKER, ensure_anomalyclip_metrics
from src.part2.references import official_reference_names
from src.part3.patch_maps import patch_distances_to_map


class Part2HelperTest(unittest.TestCase):
    def test_official_slice_matches_sorted_filenames(self) -> None:
        names = ["010.png", "000.png", "002.png", "001.png"]
        self.assertEqual(official_reference_names(names, shots=2, seed=1), ["002.png", "010.png"])

    def test_patch_map_has_the_image_shape(self) -> None:
        grid = np.zeros((4, 4), dtype=np.float64)
        grid[1, 1] = 1.0
        smoothed = patch_distances_to_map(grid, (16, 20))
        self.assertEqual(smoothed.shape, (16, 20))
        self.assertGreater(smoothed.max(), 0)

    def test_anomalyclip_hook_is_idempotent(self) -> None:
        source = Path("third_party/AnomalyCLIP/test.py").read_text(encoding="utf-8")
        if MARKER in source:
            source = source.split("    # PROJECT2_METRICS_JSON")[0]
            source = source.replace("    saved_rows = []\n", "", 1)
            start = source.find('        saved = {"category": obj}')
            end = source.find("        table_ls.append(table)\n")
            source = source[:start] + source[end:]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test.py"
            path.write_text(source, encoding="utf-8")
            self.assertTrue(ensure_anomalyclip_metrics(path))
            self.assertFalse(ensure_anomalyclip_metrics(path))
            patched = path.read_text(encoding="utf-8")
        self.assertIn(MARKER, patched)
        self.assertIn("metrics.json", patched)


if __name__ == "__main__":
    unittest.main()
