"""Score the defect-free validation tail with AnomalyDINO's official few-shot references.

The official runner only scores ``test/``. The operating point needs the
validation tail, which is never used as a reference. This script builds the
same memory bank and writes detection scores plus smoothed maps.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DINO_ROOT = ROOT / "third_party" / "AnomalyDINO"


def _import_anomalydino():
    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path.insert(0, str(DINO_ROOT))
    from src.backbones import get_model
    from src.post_eval import mean_top1p
    from src.utils import augment_image, dists2map, get_dataset_info

    for key in list(sys.modules):
        if key == "src" or key.startswith("src."):
            del sys.modules[key]
    sys.path.remove(str(DINO_ROOT))
    return get_model, mean_top1p, augment_image, dists2map, get_dataset_info


def score_category(
    model,
    mean_top1p,
    augment_image,
    dists2map,
    data_root: Path,
    category: str,
    references: list[str],
    images: list[str],
    masking: bool,
    rotation: bool,
    map_dir: Path,
    faiss_on_cpu: bool,
) -> dict[str, dict[str, float]]:
    import cv2
    import faiss
    import numpy as np
    import torch

    features_ref = []
    with torch.inference_mode():
        for name in references:
            image_ref = cv2.cvtColor(cv2.imread(str(data_root / category / "train" / "good" / name), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            augmented = augment_image(image_ref) if rotation else [image_ref]
            for image in augmented:
                tensor, grid_size = model.prepare_image(image)
                features = model.extract_features(tensor)
                mask = model.compute_background_mask(features, grid_size, threshold=10, masking_type=False)
                features_ref.append(features[mask])
        bank = np.concatenate(features_ref, axis=0).astype("float32")
        index = faiss.IndexFlatL2(bank.shape[1]) if faiss_on_cpu else faiss.GpuIndexFlatL2(faiss.StandardGpuResources(), bank.shape[1])
        faiss.normalize_L2(bank)
        index.add(bank)
        map_dir.mkdir(parents=True, exist_ok=True)
        scored = {}
        for relative in images:
            image_bgr = cv2.imread(str(data_root / relative), cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            tensor, grid_size = model.prepare_image(image)
            features = model.extract_features(tensor)
            mask = model.compute_background_mask(features, grid_size, threshold=10, masking_type=masking)
            kept = np.ascontiguousarray(features[mask], dtype=np.float32)
            faiss.normalize_L2(kept)
            distances, _matches = index.search(kept, k=1)
            distances = distances / 2
            output = np.zeros(mask.shape, dtype=np.float64)
            output[mask] = distances.squeeze()
            grid = output.reshape(grid_size)
            anomaly = dists2map(grid, image.shape)
            stem = Path(relative).stem
            np.save(map_dir / f"{stem}.npy", anomaly)
            scored[relative] = {
                "detection_score": float(mean_top1p(output)),
                "max_pixel": float(np.max(anomaly)),
            }
    return scored


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mvtec_anomaly_detection")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/splits/mvtec")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/part2/anomalydino_heldout/448")
    parser.add_argument("--resolution", type=int, default=448)
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--num-seeds", type=int, default=3)
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--device", default="cuda:6")
    parser.add_argument("--dataset", default="MVTec")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.inspection.splits import load_split
    from src.inspection.threshold import decide, operating_threshold_from_maps
    from src.part2.references import official_reference_names

    get_model, mean_top1p, augment_image, dists2map, get_dataset_info = _import_anomalydino()
    import numpy as np

    _objects, _anomalies, masking_default, rotation_default = get_dataset_info(args.dataset, "agnostic", data_path=str(args.data_root))
    model = get_model("dinov2_vits14", "cuda", smaller_edge_size=args.resolution)
    split_paths = sorted(args.split_dir.glob("*.json"))
    if args.categories:
        wanted = set(args.categories)
        split_paths = [path for path in split_paths if path.stem in wanted]
    for shot in args.shots:
        for seed in range(args.num_seeds):
            for split_path in split_paths:
                split = load_split(split_path)
                category = split["category"]
                destination = args.out_dir / f"{shot}-shot" / f"seed={seed}" / category / "scores.json"
                if destination.is_file():
                    print(f"skip {destination}")
                    continue
                folder = args.data_root / category / "train" / "good"
                names = official_reference_names(
                    [path.name for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}],
                    shot,
                    seed,
                )
                images = list(split["threshold_validation"])
                if args.max_images:
                    images = images[: args.max_images]
                print(f"heldout {category} shot={shot} seed={seed} refs={len(names)} images={len(images)}")
                map_dir = destination.parent / "maps"
                scored = score_category(
                    model,
                    mean_top1p,
                    augment_image,
                    dists2map,
                    args.data_root,
                    category,
                    names,
                    images,
                    masking_default[category],
                    rotation_default[category],
                    map_dir,
                    faiss_on_cpu=True,
                )
                maps = [np.load(map_dir / f"{Path(relative).stem}.npy") for relative in images]
                threshold = operating_threshold_from_maps(maps)
                payload = {
                    "method": "anomalydino",
                    "category": category,
                    "resolution": args.resolution,
                    "shot": shot,
                    "seed": seed,
                    "references": names,
                    "threshold": threshold,
                    "threshold_rule": "mean_plus_3std_of_validation_pixels",
                    "detection_score": "mean_of_highest_1_percent_patches",
                    "validation": {
                        relative: {
                            **row,
                            "decision": decide(row["max_pixel"], threshold),
                        }
                        for relative, row in scored.items()
                    },
                }
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
                print(f"wrote {destination}")


if __name__ == "__main__":
    main()
