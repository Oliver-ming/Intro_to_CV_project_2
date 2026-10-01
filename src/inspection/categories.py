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

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
