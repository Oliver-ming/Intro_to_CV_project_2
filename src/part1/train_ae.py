"""Train one autoencoder per category on the reference pool, then score the test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.inspection.images import read_rgb, resize_rgb, to_float_rgb
from src.inspection.reporting import write_scores
from src.inspection.splits import load_split
from src.part1.autoencoder import ConvAutoencoder, reconstruction_loss
from src.part1.difference import image_score


class ReferenceDataset(Dataset):
    def __init__(self, paths: list[Path], image_size: int) -> None:
        self.paths = paths
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = to_float_rgb(resize_rgb(read_rgb(self.paths[index]), self.image_size))
        tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        return tensor


def anomaly_map(model: ConvAutoencoder, image: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().to(device)
    with torch.no_grad():
        restored = model(tensor)
    error = torch.mean(torch.abs(restored - tensor), dim=1)[0]
    blurred = cv2.GaussianBlur(error.cpu().numpy().astype(np.float32), (0, 0), sigmaX=2.0)
    return blurred


def score_paths(
    model: ConvAutoencoder,
    data_root: Path,
    relative_paths: list[str],
    image_size: int,
    map_dir: Path,
    device: torch.device,
) -> dict[str, float]:
    scores = {}
    for relative in tqdm(relative_paths, leave=False):
        image = to_float_rgb(resize_rgb(read_rgb(data_root / relative), image_size))
        heat = anomaly_map(model, image, device)
        stem = Path(relative).stem
        parent = Path(relative).parent.name
        np.save(map_dir / f"{parent}_{stem}.npy", heat)
        scores[relative] = image_score(heat)
    return scores


def train_one(
    references: list[Path],
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> ConvAutoencoder:
    dataset = ReferenceDataset(references, image_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    model = ConvAutoencoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _epoch in range(epochs):
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = reconstruction_loss(model(batch), batch)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def run_category(
    split_path: Path,
    data_root: Path,
    out_dir: Path,
    image_size: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> None:
    split = load_split(split_path)
    category = split["category"]
    category_dir = out_dir / category
    map_dir = category_dir / "maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    references = [data_root / relative for relative in split["reference_pool"]]
    model = train_one(references, image_size, epochs, batch_size, lr, device)
    torch.save(model.state_dict(), category_dir / "autoencoder.pt")
    validation_scores = score_paths(
        model, data_root, split["threshold_validation"], image_size, map_dir, device
    )
    test_paths = [record["path"] for record in split["test"]]
    test_scores = score_paths(model, data_root, test_paths, image_size, map_dir, device)
    write_scores(
        category_dir / "scores.json",
        validation_scores,
        test_scores,
        extra={
            "method": "part1_autoencoder",
            "category": category,
            "image_size": image_size,
            "epochs": epochs,
            "labels_used_for_threshold": False,
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("data/mvtec_anomaly_detection"))
    parser.add_argument("--split-dir", type=Path, default=Path("outputs/splits/mvtec"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/part1/autoencoder"))
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.image_size % 16 != 0:
        raise SystemExit("image-size must be divisible by 16")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    split_paths = sorted(args.split_dir.glob("*.json"))
    if args.categories:
        wanted = set(args.categories)
        split_paths = [path for path in split_paths if path.stem in wanted]
    if not split_paths:
        raise SystemExit(f"no splits in {args.split_dir}")
    for split_path in split_paths:
        print(f"autoencoder: {split_path.stem} on {device}")
        run_category(
            split_path,
            args.data_root,
            args.out_dir,
            args.image_size,
            args.epochs,
            args.batch_size,
            args.lr,
            device,
        )
    print(json.dumps({"out_dir": str(args.out_dir), "device": str(device)}))


if __name__ == "__main__":
    main()
