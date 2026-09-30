"""VisA patch grids are scored from good and bad folder names."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.part2.score_visa_maps import score_visa_category


class VisaMapScoreTest(unittest.TestCase):
    def test_bad_folder_is_the_positive_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            maps = root / "maps" / "candle" / "test"
            (maps / "good").mkdir(parents=True)
            (maps / "bad").mkdir(parents=True)
            np.save(maps / "good" / "000.npy", np.zeros((4, 4)))
            defect = np.zeros((4, 4))
            defect[1, 1] = 1
            np.save(maps / "bad" / "000.npy", defect)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1, 1] = 255
            mask_dir = root / "data" / "candle" / "ground_truth" / "bad"
            mask_dir.mkdir(parents=True)
            cv2.imwrite(str(mask_dir / "000.png"), mask)
            summary = score_visa_category(root / "maps" / "candle", root / "data")
        self.assertAlmostEqual(summary["image_auroc"], 1.0)
        self.assertGreater(summary["aupro"], 0.9)


if __name__ == "__main__":
    unittest.main()
