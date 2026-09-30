"""Part 3 pieces that do not need a finished baseline run."""

from __future__ import annotations

import unittest

import numpy as np

from src.part3.background import grabcut_foreground, suppress_background
from src.part3.fusion import fuse_maps
from src.part3.multiscale import fuse_resolutions


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


class MultiScaleTest(unittest.TestCase):
    def test_fine_peak_survives_a_flat_coarse_map(self) -> None:
        fine = np.zeros((8, 8), dtype=np.float32)
        fine[3, 5] = 8
        coarse = np.zeros((2, 2), dtype=np.float32)
        fused = fuse_resolutions(fine, coarse, 0, 1, 0, 1)
        self.assertEqual(fused.shape, fine.shape)
        self.assertEqual(int(np.argmax(fused)), int(np.ravel_multi_index((3, 5), fused.shape)))


class ApplyMultiScaleTest(unittest.TestCase):
    def test_directory_fusion_writes_a_map(self) -> None:
        import tempfile
        from pathlib import Path

        from src.part3.apply_multiscale import fuse_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fine = root / "fine"
            coarse = root / "coarse"
            fine.mkdir()
            coarse.mkdir()
            np.save(fine / "a.npy", np.zeros((4, 4), dtype=np.float32))
            np.save(coarse / "a.npy", np.zeros((2, 2), dtype=np.float32))
            out = fuse_directory(fine, coarse, ["a.npy"], root / "fused")
            self.assertTrue((out / "a.npy").is_file())
            self.assertEqual(np.load(out / "a.npy").shape, (4, 4))


if __name__ == "__main__":
    unittest.main()
