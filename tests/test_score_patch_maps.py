"""Image metrics can be read from saved patch grids."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.part2.score_patch_maps import category_image_metrics


class ScorePatchMapsTest(unittest.TestCase):
    def test_perfect_patch_scores_have_auroc_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            maps = root / "maps"
            (maps / "test" / "good").mkdir(parents=True)
            (maps / "test" / "bad").mkdir(parents=True)
            np.save(maps / "test" / "good" / "000.npy", np.zeros((4, 4)))
            np.save(maps / "test" / "bad" / "000.npy", np.ones((4, 4)))
            split = root / "split.json"
            split.write_text(
                json.dumps(
                    {
                        "category": "bottle",
                        "test": [
                            {"path": "bottle/test/good/000.png", "label": "OK"},
                            {"path": "bottle/test/bad/000.png", "label": "NG"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            summary = category_image_metrics(maps, split)
        self.assertAlmostEqual(summary["image_auroc"], 1.0)
        self.assertEqual(summary["n_test"], 2)


if __name__ == "__main__":
    unittest.main()
