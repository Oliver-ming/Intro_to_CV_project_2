"""A synthetic scratch should peak in the test image, not be shifted away."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.part1.difference import difference_map


class DifferenceSmokeTest(unittest.TestCase):
    def test_blob_stays_on_the_defect(self) -> None:
        reference = np.full((128, 128, 3), 180, np.uint8)
        test = reference.copy()
        test[40:55, 70:85] = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ref_path = root / "ref.png"
            test_path = root / "test.png"
            cv2.imwrite(str(ref_path), cv2.cvtColor(reference, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(test_path), cv2.cvtColor(test, cv2.COLOR_RGB2BGR))
            heat, _used = difference_map(test_path, [ref_path], image_size=128, shortlist=1)
        peak = np.unravel_index(int(np.argmax(heat)), heat.shape)
        self.assertGreater(heat[47, 77], heat[10, 10] * 2)
        self.assertTrue(40 <= peak[0] <= 55)
        self.assertTrue(70 <= peak[1] <= 85)


if __name__ == "__main__":
    unittest.main()
