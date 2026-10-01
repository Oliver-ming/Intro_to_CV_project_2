"""Product categories used by the shared evaluation protocol."""

MVTEC_CATEGORIES = [
    "carpet",
    "grid",
    "leather",
    "tile",
    "wood",
    "bottle",
    "cable",
    "capsule",
    "hazelnut",
    "metal_nut",
    "pill",
    "screw",
    "toothbrush",
    "transistor",
    "zipper",
]

MVTEC_TEXTURES = ("carpet", "grid", "leather", "tile", "wood")
MVTEC_OBJECTS = tuple(category for category in MVTEC_CATEGORIES if category not in MVTEC_TEXTURES)

VISA_CATEGORIES = [
    "candle",
    "capsules",
    "cashew",
    "chewinggum",
    "fryum",
    "macaroni1",
    "macaroni2",
    "pcb1",
    "pcb2",
    "pcb3",
    "pcb4",
    "pipe_fryum",
]
VISA_PASSING = ("candle", "capsules", "cashew", "chewinggum", "fryum", "pipe_fryum")

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
