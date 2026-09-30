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


class ApplyBackgroundTest(unittest.TestCase):
    def test_border_scores_are_cleared(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        import cv2

        from src.part3.apply_background import suppress_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_dir = root / "controlled" / "scratch"
            image_dir.mkdir(parents=True)
            image = np.zeros((64, 64, 3), dtype=np.uint8)
            image[20:44, 20:44] = 220
            cv2.imwrite(str(image_dir / "000.png"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            split = {
                "threshold_validation": [],
                "test": [{"path": "controlled/scratch/000.png", "label": "NG", "mask": None, "defect_type": "scratch"}],
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            map_dir = root / "maps"
            map_dir.mkdir()
            np.save(map_dir / "scratch_000.npy", np.ones((64, 64), dtype=np.float32))
            count = suppress_directory(split_path, root, map_dir, root / "out", image_size=64)
            saved = np.load(root / "out" / "scratch_000.npy")
        self.assertEqual(count, 1)
        self.assertEqual(saved[1, 1], 0)
        self.assertGreater(saved[32, 32], 0)


class FuseScoresTest(unittest.TestCase):
    def test_shared_high_score_stays_highest(self) -> None:
        from src.part3.fuse_scores import fuse_detection_scores

        fused = fuse_detection_scores(
            [0.1, 5.0],
            [0.2, 4.0],
            left_validation=[0.1, 0.2, 0.15],
            right_validation=[0.2, 0.25, 0.22],
        )
        self.assertGreater(fused[1], fused[0])


class ApplyScoreFusionTest(unittest.TestCase):
    def test_writes_decisions_for_shared_images(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from src.part3.apply_score_fusion import fuse_score_files

        def pack(values: dict[str, float]) -> dict:
            return {key: {"detection_score": value, "max_pixel": value, "decision": "OK"} for key, value in values.items()}

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left = {"validation": pack({"a": 0.1, "b": 0.2}), "test": pack({"c": 0.1, "d": 3.0})}
            right = {"validation": pack({"a": 0.2, "b": 0.3}), "test": pack({"c": 0.2, "d": 4.0})}
            (root / "left.json").write_text(json.dumps(left), encoding="utf-8")
            (root / "right.json").write_text(json.dumps(right), encoding="utf-8")
            payload = fuse_score_files(root / "left.json", root / "right.json", root / "fused.json")
        self.assertIn("c", payload["test"])
        self.assertIn("d", payload["test"])
        self.assertGreater(payload["test"]["d"]["detection_score"], payload["test"]["c"]["detection_score"])


class TargetTest(unittest.TestCase):
    def test_grabcut_is_only_for_the_pca_failures_and_the_mouse(self) -> None:
        from src.part3.targets import needs_grabcut

        self.assertTrue(needs_grabcut("cable"))
        self.assertTrue(needs_grabcut("transistor"))
        self.assertTrue(needs_grabcut("mouse"))
        self.assertFalse(needs_grabcut("bottle"))
        self.assertFalse(needs_grabcut("carpet"))


class RescoreTest(unittest.TestCase):
    def test_suppressed_border_lowers_the_detection_score(self) -> None:
        import json
        import tempfile
        from pathlib import Path

        from src.part3.rescore import rescore_directory

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            maps = root / "maps"
            maps.mkdir()
            heat = np.zeros((16, 16), dtype=np.float32)
            heat[0, :] = 1
            np.save(maps / "good_000.npy", heat)
            np.save(maps / "good_001.npy", np.zeros((16, 16), dtype=np.float32))
            split = {
                "category": "cable",
                "threshold_validation": ["cable/train/good/000.png"],
                "test": [{"path": "cable/test/good/001.png"}],
            }
            split_path = root / "split.json"
            split_path.write_text(json.dumps(split), encoding="utf-8")
            payload = rescore_directory(split_path, maps, root / "scores.json", extra={"method": "grabcut"})
        self.assertEqual(payload["method"], "grabcut")
        self.assertLess(payload["test"]["cable/test/good/001.png"]["detection_score"], 0.5)
        self.assertFalse(payload["labels_used_for_threshold"])


if __name__ == "__main__":
    unittest.main()
