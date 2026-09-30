"""Completion markers are read from logs, not from process lists."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.inspection.completion import completion_status


class CompletionTest(unittest.TestCase):
    def test_marker_means_done(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log = root / "outputs" / "part1"
            log.mkdir(parents=True)
            (log / "run.log").write_text("PART1_DONE 2026-10-01\n", encoding="utf-8")
            status = completion_status(root)
        self.assertTrue(status["part1"])
        self.assertFalse(status["anomalydino_mvtec"])


if __name__ == "__main__":
    unittest.main()
