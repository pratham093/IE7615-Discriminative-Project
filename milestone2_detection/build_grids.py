"""Adds more 2x2 grids and the 3x3 grids to data/. Each split only uses its own photos.

python build_grids.py [--n2 1000] [--n3 2000] [--seed 7]
"""

import argparse
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CELL = 224
# split that supplies the photos -> split the grids are written to
SPLITS = {"train": "train", "val": "val", "test_single": "test_multi"}


def load_photos(split):
    photos = {}  # class id -> [(image path, [(cls, cx, cy, w, h), ...])]
    for img in sorted((DATA / split / "images").glob("OBJ*.jpg")):
        boxes = [tuple(map(float, line.split())) for line in (DATA / split / "labels" / f"{img.stem}.txt").read_text().split("\n") if line.strip()]
        photos.setdefault(int(boxes[0][0]), []).append((img, boxes))
    return photos


def make_grid(photos, k, rng):
    canvas = Image.new("RGB", (k * CELL, k * CELL))
    labels = []
    for i, cls in enumerate(rng.sample(sorted(photos), k * k)):
        path, boxes = rng.choice(photos[cls])
        im = Image.open(path).convert("RGB").resize((CELL, CELL))
        flip = rng.random() < 0.5
        if flip:
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im = ImageEnhance.Brightness(im).enhance(rng.uniform(0.75, 1.25))
        ox, oy = (i % k) * CELL, (i // k) * CELL
        canvas.paste(im, (ox, oy))
        for c, cx, cy, w, h in boxes:
            cx = 1 - cx if flip else cx
            labels.append(f"{int(c)} {(ox + cx * CELL) / (k * CELL):.6f} {(oy + cy * CELL) / (k * CELL):.6f} {w / k:.6f} {h / k:.6f}")
    return canvas, labels


def draw(img, labels, names):
    img = img.copy()
    d = ImageDraw.Draw(img)
    W, H = img.size
    for line in labels:
        c, cx, cy, w, h = map(float, line.split())
        x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
        d.rectangle([x1, y1, x1 + w * W, y1 + h * H], outline=(0, 255, 0), width=2)
        d.rectangle([x1, y1, x1 + 48, y1 + 12], fill=(0, 255, 0))
        d.text((x1 + 2, y1), names[int(c)], fill=(0, 0, 0))
    return img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n2", type=int, default=1000, help="extra 2x2 training grids")
    parser.add_argument("--n3", type=int, default=2000, help="3x3 training grids")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    names = json.loads((DATA / "classes.json").read_text())
    (ROOT / "samples").mkdir(exist_ok=True)

    for src, dst in SPLITS.items():
        photos = load_photos(src)
        for old in (DATA / dst / "images").glob(f"multi_{dst}_g[23]_*.jpg"):
            old.unlink()
            (DATA / dst / "labels" / f"{old.stem}.txt").unlink(missing_ok=True)
        (DATA / dst / "labels.cache").unlink(missing_ok=True)
        scale = 1 if dst == "train" else 5
        for k, n in ((2, args.n2 // scale), (3, args.n3 // scale)):
            for j in range(n):
                img, labels = make_grid(photos, k, rng)
                stem = f"multi_{dst}_g{k}_{j:05d}"
                img.save(DATA / dst / "images" / f"{stem}.jpg", quality=92)
                (DATA / dst / "labels" / f"{stem}.txt").write_text("\n".join(labels) + "\n")
                if dst == "test_multi" and j == 0:
                    img.save(ROOT / "samples" / f"sample_{k}x{k}.jpg", quality=95)
                    draw(img, labels, names).save(ROOT / "samples" / f"sample_{k}x{k}_boxes.jpg", quality=95)
            print(f"{dst}: {n} {k}x{k} grids from {sum(map(len, photos.values()))} {src} photos", flush=True)


if __name__ == "__main__":
    main()
