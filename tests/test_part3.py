"""Part 3 pieces that do not need a finished baseline run."""

from __future__ import annotations

import unittest

import numpy as np

from src.part3.background import grabcut_foreground, suppress_background
from src.part3.fusion import fuse_maps


class BackgroundTest(unittest.TestCase):
    def test_centered_object_is_kept_and_border_is_cleared(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[20:44, 20:44] = 220
        foreground = grabcut_foreground(image, iterations=2)
        anomaly = np.ones((64, 64), dtype=np.float32)
        suppressed = suppress_background(anomaly, foreground)
        self.assertGreater(foreground[32, 32], 0)
        self.assertEqual(suppressed[2, 2], 0)
        self.assertEqual(suppressed[32, 32], 1)


class FusionTest(unittest.TestCase):
    def test_calibrated_average_keeps_the_shared_peak(self) -> None:
        left = np.zeros((4, 4), dtype=np.float32)
        right = np.zeros((4, 4), dtype=np.float32)
        left[1, 1] = 10
        right[1, 1] = 2
        fused = fuse_maps(left, right, left_mean=0, left_std=1, right_mean=0, right_std=1)
        self.assertEqual(int(np.argmax(fused)), int(np.ravel_multi_index((1, 1), fused.shape)))


if __name__ == "__main__":
    unittest.main()
