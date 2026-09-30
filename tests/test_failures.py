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


if __name__ == "__main__":
    unittest.main()
