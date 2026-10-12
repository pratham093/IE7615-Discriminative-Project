import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", PROJECT_DIR.parent / "IE7615 Deep Learning for AI"))
OUTPUT_DIR = PROJECT_DIR / "outputs"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", PROJECT_DIR.parent / "milestone1_classification" / "outputs" / "cache"))

SEED = 42
IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BG_REGEX = r"_bg[_\d]"  # case-insensitive; OBJ013 uses OBJ013_BG001.jpg
