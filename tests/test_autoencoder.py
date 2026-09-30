"""The normal-only autoencoder trains and can be resumed."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from src.part1.autoencoder import ConvAutoencoder, reconstruction_loss
from src.part1.train_ae import scores_ready


class AutoencoderTest(unittest.TestCase):
    def test_loss_backpropagates_on_a_divisible_input(self) -> None:
        model = ConvAutoencoder()
        images = torch.rand(2, 3, 32, 32)
        loss = reconstruction_loss(model(images), images)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))

    def test_finished_category_is_detected_from_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(scores_ready(root, "bottle"))
            category = root / "bottle"
            category.mkdir()
            (category / "scores.json").write_text("{}", encoding="utf-8")
            self.assertTrue(scores_ready(root, "bottle"))


if __name__ == "__main__":
    unittest.main()
