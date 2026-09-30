"""The MVTec AD 2 reader stays on the public test split."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inspection.ad2 import public_test_records


class AD2Test(unittest.TestCase):
    def test_public_bad_image_requires_a_mask_and_private_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            public = root / "vial" / "test_public"
            (public / "good").mkdir(parents=True)
            (public / "bad").mkdir()
            (public / "ground_truth" / "bad").mkdir(parents=True)
            private = root / "vial" / "test_private"
            private.mkdir()
            (public / "good" / "000.png").write_bytes(b"good")
            (public / "bad" / "001.png").write_bytes(b"bad")
            (public / "ground_truth" / "bad" / "001_mask.png").write_bytes(b"mask")
            (private / "000.png").write_bytes(b"private")
            records = public_test_records(root, "vial")
        self.assertEqual([row["label"] for row in records], ["OK", "NG"])
        self.assertIsNone(records[0]["mask"])
        self.assertEqual(records[1]["mask"], "vial/test_public/ground_truth/bad/001_mask.png")
        self.assertTrue(all("test_private" not in row["path"] for row in records))


if __name__ == "__main__":
    unittest.main()
