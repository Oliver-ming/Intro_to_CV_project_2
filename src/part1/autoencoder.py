"""Compact convolutional autoencoder trained on defect-free images only."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class ConvAutoencoder(nn.Module):
    """Encoder-decoder for a square input whose side is divisible by 16."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(images))


def _gaussian_window(channel: int, window_size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coords = torch.arange(window_size, dtype=torch.float32, device=device) - window_size // 2
    kernel = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    window = kernel[:, None] @ kernel[None, :]
    return window.expand(channel, 1, window_size, window_size).contiguous()


def ssim_map(prediction: torch.Tensor, target: torch.Tensor, window_size: int = 7) -> torch.Tensor:
    """Per-image SSIM averaged over channels. Both tensors are in ``[0, 1]``."""
    channel = prediction.shape[1]
    window = _gaussian_window(channel, window_size, 1.5, prediction.device)
    padding = window_size // 2
    mu_x = F.conv2d(prediction, window, padding=padding, groups=channel)
    mu_y = F.conv2d(target, window, padding=padding, groups=channel)
    sigma_x = F.conv2d(prediction * prediction, window, padding=padding, groups=channel) - mu_x ** 2
    sigma_y = F.conv2d(target * target, window, padding=padding, groups=channel) - mu_y ** 2
    sigma_xy = F.conv2d(prediction * target, window, padding=padding, groups=channel) - mu_x * mu_y
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (numerator / denominator).mean(dim=1)


def reconstruction_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pixel = torch.mean(torch.abs(prediction - target))
    structure = torch.mean(1.0 - ssim_map(prediction, target))
    return pixel + structure
