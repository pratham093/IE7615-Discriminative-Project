"""PyTorch datasets that read the Stage 1 cache (outputs/cache/{train,val,test}.npz)."""

import csv

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import v2

import config


def train_transform():
    return v2.Compose([
        v2.RandomResizedCrop(config.IMG_SIZE, scale=(0.7, 1.0), antialias=True),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15, interpolation=v2.InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def eval_transform():
    return v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
    ])


def load_classes():
    with open(config.CACHE_DIR / "classes.csv", newline="") as f:
        return [(r["object_id"], r["object_name"]) for r in csv.DictReader(f)]


class CachedImages(Dataset):
    def __init__(self, split, transform, num_classes=None):
        data = np.load(config.CACHE_DIR / f"{split}.npz")
        x, y, paths = data["x"], data["y"], data["paths"]
        if num_classes is not None:  # smoke test: keep labels 0..num_classes-1
            keep = y < num_classes
            x, y, paths = x[keep], y[keep], paths[keep]
        self.x = torch.from_numpy(x).permute(0, 3, 1, 2)  # NHWC uint8 -> NCHW view
        self.y = torch.from_numpy(y)
        self.paths = paths
        self.transform = transform

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.transform(self.x[i]), self.y[i]
