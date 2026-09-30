"""Official few-shot reference names: a deterministic slice of sorted filenames."""

from __future__ import annotations


def official_reference_names(names: list[str], shots: int, seed: int) -> list[str]:
    ordered = sorted(names)
    if shots < 0:
        return ordered
    if shots < 1 or seed < 0:
        raise ValueError("shots and seed must be non-negative, or shots must be -1")
    return ordered[seed * shots : (seed + 1) * shots]
