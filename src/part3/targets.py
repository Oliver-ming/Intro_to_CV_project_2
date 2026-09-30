"""Categories where AnomalyDINO's PCA foreground mask is not used.

The paper's masking test fails on cable and transistor, so the official
agnostic setting leaves them unmasked. GrabCut is applied only there, and
later on the self-collected mouse. Other MVTec categories keep the official mask.
"""

from __future__ import annotations

PCA_MASK_FAILURES = ("cable", "transistor")


def needs_grabcut(category: str) -> bool:
    return category in PCA_MASK_FAILURES or category == "mouse"
