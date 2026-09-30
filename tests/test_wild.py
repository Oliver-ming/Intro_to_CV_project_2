"""The wild-set split keeps shifts out of the reference pool."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inspection.wild import build_wild_split


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


class WildSplitTest(unittest.TestCase):
    def test_reference_pool_is_only_controlled_normals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(30):
                _touch(root / "controlled" / "good" / f"{index:03d}.png")
            for index in range(2):
                _touch(root / "shift_light" / "good" / f"{index:03d}.png")
                _touch(root / "controlled" / "scratch" / f"{index:03d}.png")
                _touch(root / "masks" / "scratch" / f"{index:03d}.png")
            payload = build_wild_split(root, val_count=10, seed=0)
        self.assertEqual(len(payload["reference_pool"]), 20)
        self.assertEqual(len(payload["threshold_validation"]), 10)
        self.assertTrue(all(path.startswith("controlled/good/") for path in payload["reference_pool"]))
        labels = {record["path"]: record["label"] for record in payload["test"]}
        self.assertEqual(labels["shift_light/good/000.png"], "OK")
        self.assertEqual(labels["controlled/scratch/000.png"], "NG")


if __name__ == "__main__":
    unittest.main()
