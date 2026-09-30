"""The demo overlay is a real image with the score written on it."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.demo.inspect import _overlay


class OverlayTest(unittest.TestCase):
    def test_writes_a_nonempty_png(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            canvas = np.zeros((64, 64, 3), dtype=np.uint8)
            canvas[16:48, 16:48] = (40, 40, 200)
            cv2.imwrite(str(image), canvas)
            anomaly = np.zeros((64, 64), dtype=np.float32)
            anomaly[20:30, 20:30] = 1
            destination = root / "overlay.png"
            _overlay(image, anomaly, destination, "NG", 0.42)
            saved = cv2.imread(str(destination), cv2.IMREAD_COLOR)
        self.assertIsNotNone(saved)
        self.assertEqual(saved.shape, (64, 64, 3))
        self.assertGreater(int(saved.max()), 0)


if __name__ == "__main__":
    unittest.main()
