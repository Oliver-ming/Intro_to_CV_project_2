"""Image metrics can be read from saved patch grids."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.part2.score_patch_maps import category_image_metrics, category_pixel_metrics


class ScorePatchMapsTest(unittest.TestCase):
    def test_perfect_patch_scores_have_auroc_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            maps = root / "maps"
            (maps / "test" / "good").mkdir(parents=True)
            (maps / "test" / "bad").mkdir(parents=True)
            np.save(maps / "test" / "good" / "000.npy", np.zeros((4, 4)))
            bad = np.zeros((4, 4), dtype=np.float64)
            bad[1, 1] = 1
            np.save(maps / "test" / "bad" / "000.npy", bad)
            mask = np.zeros((4, 4), dtype=np.uint8)
            mask[1, 1] = 255
            mask_path = root / "mask.png"
            import cv2
            cv2.imwrite(str(mask_path), mask)
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "category": "bottle",
                        "test": [
                            {"path": "bottle/test/good/000.png", "label": "OK", "mask": None},
                            {"path": "bottle/test/bad/000.png", "label": "NG", "mask": "mask.png"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = category_image_metrics(maps, split)
            self.assertAlmostEqual(summary["image_auroc"], 1.0)
            self.assertEqual(summary["n_test"], 2)
            pixels = category_pixel_metrics(maps, split, root)
            self.assertGreater(pixels["pixel_auroc"], 0.9)
            self.assertGreater(pixels["aupro"], 0.9)


if __name__ == "__main__":
    unittest.main()
