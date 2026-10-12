import os
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent

DATASET_ROOT = Path(os.environ.get("DATASET_ROOT", PROJECT_DIR.parent / "IE7615 Deep Learning for AI"))
OBJECT_IDS_CSV = Path(os.environ.get("OBJECT_IDS", PROJECT_DIR.parent / "object_ids.csv"))

OUTPUT_DIR = PROJECT_DIR / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
CACHE_DIR = Path(os.environ.get("CACHE_DIR", OUTPUT_DIR / "cache"))  # Colab: read from a local copy
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
METRIC_DIR = OUTPUT_DIR / "metrics"
for d in (LOG_DIR, CACHE_DIR, CHECKPOINT_DIR, FIGURE_DIR, METRIC_DIR):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42
IMG_SIZE = 224
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
BG_REGEX = r"_bg[_\d]"  # case-insensitive; OBJ013 uses OBJ013_BG001.jpg
BG_MODE = "exclude"  # or "bg_class"
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

MODEL_NAMES = ("custom_cnn", "resnet50", "efficientnet_b0")
BATCH_SIZE = 64
EPOCHS = {"custom_cnn": 40, "resnet50": 25, "efficientnet_b0": 25}
LR = {"custom_cnn": 1e-3, "resnet50": 1e-4, "efficientnet_b0": 1e-4}  # fine-tune lr for pretrained models
HEAD_EPOCHS = 5
HEAD_LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
PATIENCE = 6
SMOKE_CLASSES = 10
SMOKE_EPOCHS = 2
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

YOLO_WEIGHTS = "yolo11s-cls.pt"
YOLO_RUN = "yolo11s_cls"
YOLO_DATA_DIR = OUTPUT_DIR / "yolo_data"
YOLO_EPOCHS = 30
YOLO_BATCH = 64
YOLO_PATIENCE = 8
