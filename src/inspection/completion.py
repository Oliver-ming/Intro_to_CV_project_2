"""Report which long-run completion markers are present. Does not start jobs."""

from __future__ import annotations

import argparse
from pathlib import Path

MARKERS = {
    "part1": "PART1_DONE",
    "anomalydino_mvtec": "ANOMALYDINO_MVTEC_DONE",
    "anomalydino_mvtec_672": "ANOMALYDINO_MVTEC_672_DONE",
    "anomalydino_visa": "ANOMALYDINO_VISA_DONE",
    "anomalyclip_train": "ANOMALYCLIP_TRAIN_DONE",
    "anomalyclip_test": "ANOMALYCLIP_TEST_DONE",
    "anomalyclip_heldout": "ANOMALYCLIP_HELDOUT_DONE",
    "anomalydino_heldout": "ANOMALYDINO_HELDOUT_DONE",
    "anomalydino_heldout_672": "ANOMALYDINO_HELDOUT_672_DONE",
    "autoencoder": "AUTOENCODER_DONE",
}

LOGS = {
    "part1": "outputs/part1/run.log",
    "anomalydino_mvtec": "outputs/part2/anomalydino_mvtec.log",
    "anomalydino_mvtec_672": "outputs/part2/anomalydino_mvtec_672.log",
    "anomalydino_visa": "outputs/part2/anomalydino_visa.log",
    "anomalyclip_train": "outputs/part2/anomalyclip_train.log",
    "anomalyclip_test": "outputs/part2/anomalyclip_test.log",
    "anomalyclip_heldout": "outputs/part2/anomalyclip_heldout.log",
    "anomalydino_heldout": "outputs/part2/anomalydino_heldout.log",
    "anomalydino_heldout_672": "outputs/part2/anomalydino_heldout_672.log",
    "autoencoder": "outputs/part1/autoencoder.log",
}


def heldout_done_marker(resolution: int) -> str | None:
    """Marker printed after a full validation-tail run at one resolution."""
    if resolution == 448:
        return MARKERS["anomalydino_heldout"]
    if resolution == 672:
        return MARKERS["anomalydino_heldout_672"]
    return None


def completion_status(root: Path) -> dict[str, bool]:
    status = {}
    for name, marker in MARKERS.items():
        log_path = root / LOGS[name]
        text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.is_file() else ""
        status[name] = marker in text
    return status


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    for name, done in completion_status(args.root).items():
        print(f"{name}: {'done' if done else 'running_or_missing'}")


if __name__ == "__main__":
    main()
