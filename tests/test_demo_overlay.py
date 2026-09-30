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

    def test_saved_map_uses_the_stored_decision(self) -> None:
        import json

        from src.demo.inspect import overlay_saved_map

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "input.png"
            cv2.imwrite(str(image), np.zeros((32, 32, 3), dtype=np.uint8))
            anomaly = root / "map.npy"
            np.save(anomaly, np.ones((8, 8), dtype=np.float32))
            scores = root / "scores.json"
            scores.write_text(
                json.dumps({"test": {"cable/test/bent_wire/000.png": {"detection_score": 0.5, "decision": "NG"}}}),
                encoding="utf-8",
            )
            payload = overlay_saved_map(image, anomaly, scores, "cable/test/bent_wire/000.png", root / "out.png")
            self.assertEqual(payload["decision"], "NG")
            self.assertGreater(Path(payload["heatmap"]).stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
