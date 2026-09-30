"""Score the defect-free validation tail with a trained AnomalyCLIP prompt.

The official test loop only reads ``meta.json`` test images. Fusion needs a
prompt score on the same validation tail used for the visual distances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANOMALYCLIP = ROOT / "third_party" / "AnomalyCLIP"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/mvtec_anomaly_detection")
    parser.add_argument("--split-dir", type=Path, default=ROOT / "outputs/splits/mvtec")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "outputs/part2/anomalyclip_heldout/mvtec")
    parser.add_argument("--dataset", default="mvtec")
    parser.add_argument("--categories", nargs="*", default=None)
    parser.add_argument("--device", default="cuda:5")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--features-list", nargs="+", type=int, default=[24])
    parser.add_argument("--depth", type=int, default=9)
    parser.add_argument("--n-ctx", type=int, default=12)
    parser.add_argument("--t-n-ctx", type=int, default=4)
    args = parser.parse_args()
    if not args.checkpoint.is_file():
        raise SystemExit(f"missing checkpoint {args.checkpoint}")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device[-1])
    sys.path.insert(0, str(ANOMALYCLIP))
    os.chdir(ANOMALYCLIP)

    import torch
    from PIL import Image

    import AnomalyCLIP_lib
    from prompt_ensemble import AnomalyCLIP_PromptLearner
    from utils import get_transform

    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from src.inspection.splits import load_split

    class _Args:
        image_size = args.image_size

    preprocess, _target = get_transform(_Args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    parameters = {
        "Prompt_length": args.n_ctx,
        "learnabel_text_embedding_depth": args.depth,
        "learnabel_text_embedding_length": args.t_n_ctx,
    }
    model, _ = AnomalyCLIP_lib.load("ViT-L/14@336px", device=device, design_details=parameters)
    model.eval()
    prompt_learner = AnomalyCLIP_PromptLearner(model.to("cpu"), parameters)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    prompt_learner.load_state_dict(checkpoint["prompt_learner"])
    prompt_learner.to(device)
    model.to(device)
    model.visual.DAPM_replace(DPAM_layer=20)
    prompts, tokenized_prompts, compound_prompts_text = prompt_learner(cls_id=None)
    text_features = model.encode_text_learn(prompts, tokenized_prompts, compound_prompts_text).float()
    text_features = torch.stack(torch.chunk(text_features, dim=0, chunks=2), dim=1)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    split_paths = sorted(args.split_dir.glob("*.json"))
    if args.categories:
        wanted = set(args.categories)
        split_paths = [path for path in split_paths if path.stem in wanted]
    for split_path in split_paths:
        split = load_split(split_path)
        category = split["category"]
        destination = args.out_dir / category / "scores.json"
        if destination.is_file():
            print(f"skip {destination}")
            continue
        validation = {}
        for relative in split["threshold_validation"]:
            image = preprocess(Image.open(args.data_root / relative).convert("RGB")).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features, _patch_features = model.encode_image(image, args.features_list, DPAM_layer=20)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                text_probs = image_features @ text_features.permute(0, 2, 1)
                text_probs = (text_probs / 0.07).softmax(-1)[:, 0, 1]
            validation[relative] = {"detection_score": float(text_probs.detach().cpu().reshape(-1)[0])}
            print(f"{relative} {validation[relative]['detection_score']:.4f}")
        payload = {
            "method": "anomalyclip",
            "category": category,
            "dataset": args.dataset,
            "checkpoint": str(args.checkpoint),
            "detection_score": "prompt_image_probability",
            "validation": validation,
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {destination}")


if __name__ == "__main__":
    main()
