"""Wrong decisions are grouped into the report's failure causes."""

from __future__ import annotations

import unittest

import numpy as np

from src.inspection.failures import categorize_failure


class FailureTest(unittest.TestCase):
    def test_correct_decision_has_no_cause(self) -> None:
        heat = np.zeros((20, 20), dtype=np.float32)
        self.assertIsNone(categorize_failure("OK", "OK", heat, None))

    def test_border_hotspot_on_a_normal_image_is_background(self) -> None:
        heat = np.zeros((20, 20), dtype=np.float32)
        heat[0, :] = 5
        self.assertEqual(categorize_failure("OK", "NG", heat, None), "background_or_border")

    def test_tiny_ground_truth_is_a_small_defect(self) -> None:
        heat = np.ones((20, 20), dtype=np.float32)
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[10, 10] = 255
        self.assertEqual(categorize_failure("NG", "OK", heat, mask), "small_defect")

    def test_collect_failures_keeps_only_wrong_decisions(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from src.inspection.failures import collect_failures, summarize_causes

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            maps = root / "maps"
            maps.mkdir()
            heat = np.zeros((8, 8), dtype=np.float32)
            heat[0, :] = 1
            np.save(maps / "good_000.npy", heat)
            np.save(maps / "broken_000.npy", np.zeros((8, 8), dtype=np.float32))
            mask_path = root / "mask.png"
            import cv2
            cv2.imwrite(str(mask_path), np.full((8, 8), 255, dtype=np.uint8))
            split = {
                "test": [
                    {"path": "bottle/test/good/000.png", "label": "OK", "mask": None},
                    {"path": "bottle/test/broken/000.png", "label": "NG", "mask": "mask.png"},
                ]
            }
            scores = {
                "test": {
                    "bottle/test/good/000.png": {"decision": "NG", "detection_score": 0.9},
                    "bottle/test/broken/000.png": {"decision": "OK", "detection_score": 0.1},
                }
            }
            rows = collect_failures(split, scores, maps, root)
        self.assertEqual(summarize_causes(rows)["background_or_border"], 1)
        self.assertEqual(summarize_causes(rows)["missed_defect"], 1)
        self.assertFalse(json.dumps(rows) == "")


if __name__ == "__main__":
    unittest.main()
