"""Writes the cached images as PNGs in the folder layout Ultralytics expects (outputs/yolo_data)."""

import shutil
from pathlib import PurePosixPath

import numpy as np
from PIL import Image

import config
from datasets import load_classes


def main():
    out = config.YOLO_DATA_DIR
    if out.exists():
        shutil.rmtree(out)
    classes = [object_id for object_id, _ in load_classes()]

    for split in ("train", "val", "test"):
        data = np.load(config.CACHE_DIR / f"{split}.npz")
        for x, y, path in zip(data["x"], data["y"], data["paths"]):
            folder = out / split / classes[y]
            folder.mkdir(parents=True, exist_ok=True)
            dst = folder / (PurePosixPath(str(path)).stem + ".png")
            if dst.exists():
                raise SystemExit(f"two images would share the name {dst}")
            Image.fromarray(x).save(dst, compress_level=1)
        written = sum(1 for _ in (out / split).rglob("*.png"))
        n_classes = sum(1 for d in (out / split).iterdir() if d.is_dir())
        print(f"{split:<5} {written} images in {n_classes} class folders")
        if written != len(data["y"]):
            raise SystemExit(f"{split}: wrote {written} files but the cache has {len(data['y'])}")
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
