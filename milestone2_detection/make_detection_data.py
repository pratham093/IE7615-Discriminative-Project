"""Builds the YOLO detection data from the boxes: single photos, 2x2 grids and pasted images for each split.

python make_detection_data.py [--multi 4000] [--out data]
"""

import argparse
import hashlib
import json
import re
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageOps

import config
from annotate import images, load_classes, load_state, photos

SIZE = 448
TILE = config.IMG_SIZE
MIN_VISIBLE = 0.6
# background photos that show class objects or clutter (checked by eye); OBJ065's are copies of watch photos
BG_SKIP = {
    "OBJ001_bg_004", "OBJ003_bg_003", "OBJ005_bg_003", "OBJ006_bg_002", "OBJ008_bg_003", "OBJ009_bg_003",
    "OBJ012_bg_003", "OBJ014_bg_001", "OBJ014_bg_005", "OBJ015_bg_003", "OBJ015_bg_004", "OBJ016_bg_001",
    "OBJ016_bg_003", "OBJ020_bg_001", "OBJ020_bg_003", "OBJ021_bg_001", "OBJ024_bg_003", "OBJ025_bg_003",
    "OBJ026_bg_003", "OBJ026_bg_004", "OBJ027_bg_001", "OBJ027_bg_002", "OBJ027_bg_003", "OBJ028_bg_001",
    "OBJ030_bg_003", "OBJ032_bg_003", "OBJ032_bg_004", "OBJ035_bg_002", "OBJ036_bg_001", "OBJ037_bg_005",
    "OBJ038_bg_005", "OBJ039_bg_004", "OBJ043_bg_001", "OBJ044_bg_001", "OBJ044_bg_002", "OBJ047_bg_004",
    "OBJ048_bg_001", "OBJ048_bg_002", "OBJ048_bg_003", "OBJ048_bg_005", "OBJ049_bg_001", "OBJ050_bg_001",
    "OBJ052_bg_001", "OBJ052_bg_004", "OBJ052_bg_005", "OBJ054_bg_002", "OBJ055_bg_001", "OBJ055_bg_003",
    "OBJ056_bg_003", "OBJ057_bg_004", "OBJ058_bg_003", "OBJ059_bg_002", "OBJ059_bg_005", "OBJ061_bg_001",
    "OBJ061_bg_002", "OBJ061_bg_003", "OBJ061_bg_004", "OBJ063_bg_004", "OBJ064_bg_003", "OBJ064_bg_004",
    "OBJ065_bg_001", "OBJ065_bg_002", "OBJ065_bg_003", "OBJ065_bg_004", "OBJ065_bg_005", "OBJ068_bg_001",
    "OBJ068_bg_005", "OBJ070_bg_005",
}


def backgrounds():
    out = {"train": [], "val": [], "test": []}
    for p in sorted(config.DATASET_ROOT.rglob("*")):
        if p.suffix.lower() not in config.IMAGE_EXTS or not re.search(config.BG_REGEX, p.name, re.I) or p.stem in BG_SKIP:
            continue
        h = int(hashlib.md5(p.stem.encode()).hexdigest(), 16) % 100
        out["train" if h < 70 else "val" if h < 85 else "test"].append(p)
    return out


def load_bg(path):
    with Image.open(path) as im:
        return ImageOps.fit(ImageOps.exif_transpose(im).convert("RGB"), (TILE, TILE), Image.LANCZOS)


def yolo_lines(boxes, size, label_of):
    return [f"{label_of[o]} {(x1 + x2) / 2 / size:.6f} {(y1 + y2) / 2 / size:.6f} {(x2 - x1) / size:.6f} {(y2 - y1) / size:.6f}"
            for o, x1, y1, x2, y2 in boxes]


def save(img, boxes, folder, name, size, label_of):
    img.save(folder / "images" / f"{name}.jpg", quality=95)
    (folder / "labels" / f"{name}.txt").write_text("\n".join(yolo_lines(boxes, size, label_of)) + "\n")


def flip_boxes(boxes, w):
    return [[o, w - x2, y1, w - x1, y2] for o, x1, y1, x2, y2 in boxes]


def grid(rng, by_class, data):
    classes = rng.choice(sorted(by_class), 4, replace=False)
    canvas = Image.new("RGB", (SIZE, SIZE))
    boxes = []
    for k, c in enumerate(classes):
        p, e = by_class[c][rng.integers(len(by_class[c]))]
        img = Image.fromarray(np.asarray(data[p["split"]][p["index"]]))
        b = [list(x) for x in e["boxes"]]
        if rng.random() < 0.5:
            img, b = ImageOps.mirror(img), flip_boxes(b, TILE)
        dx, dy = (k % 2) * TILE, (k // 2) * TILE
        canvas.paste(img, (dx, dy))
        boxes += [[o, x1 + dx, y1 + dy, x2 + dx, y2 + dy] for o, x1, y1, x2, y2 in b]
    return canvas, boxes


def paste(rng, by_class, data, bgs):
    bg = load_bg(bgs[rng.integers(len(bgs))])
    if rng.random() < 0.5:
        bg = ImageOps.mirror(bg)
    canvas = bg.resize((SIZE, SIZE), Image.LANCZOS)
    owner = np.full((SIZE, SIZE), -1, dtype=np.int32)
    placed = []  # [object_id, x1, y1, x2, y2, pixels it owned when placed]
    for c in rng.choice(sorted(by_class), rng.integers(3, 7), replace=False):
        p, e = by_class[c][rng.integers(len(by_class[c]))]
        own = [b for b in e["boxes"] if b[0] == c]
        _, bx1, by1, bx2, by2 = own[rng.integers(len(own))]
        cx1, cy1 = max(0, int(bx1) - 3), max(0, int(by1) - 3)
        cx2, cy2 = min(TILE, int(np.ceil(bx2)) + 3), min(TILE, int(np.ceil(by2)) + 3)
        crop = Image.fromarray(np.asarray(data[p["split"]][p["index"]])).crop((cx1, cy1, cx2, cy2))
        inner = []
        for o, x1, y1, x2, y2 in e["boxes"]:
            ix1, iy1, ix2, iy2 = max(x1, cx1), max(y1, cy1), min(x2, cx2), min(y2, cy2)
            if ix2 > ix1 and iy2 > iy1 and (ix2 - ix1) * (iy2 - iy1) >= 0.3 * (x2 - x1) * (y2 - y1):
                inner.append([o, ix1 - cx1, iy1 - cy1, ix2 - cx1, iy2 - cy1])
        w, h = crop.size
        scale = min(rng.uniform(80, 260) / max(w, h), 2.0, (SIZE - 8) / max(w, h))
        nw, nh = max(8, round(w * scale)), max(8, round(h * scale))
        crop = crop.resize((nw, nh), Image.LANCZOS)
        if rng.random() < 0.5:
            crop, inner = ImageOps.mirror(crop), flip_boxes(inner, w)
        crop = ImageEnhance.Brightness(crop).enhance(rng.uniform(0.85, 1.15))
        for _ in range(40):
            x, y = int(rng.integers(0, SIZE - nw + 1)), int(rng.integers(0, SIZE - nh + 1))
            after = owner.copy()
            after[y:y + nh, x:x + nw] = -1
            if all((after[q[2]:q[4], q[1]:q[3]] == j).sum() >= MIN_VISIBLE * q[5] for j, q in enumerate(placed)):
                break
        else:
            continue
        canvas.paste(crop, (x, y))
        owner = after
        for o, x1, y1, x2, y2 in inner:
            q = [o, round(x + x1 * scale), round(y + y1 * scale), round(x + x2 * scale), round(y + y2 * scale)]
            q[3], q[4] = max(q[3], q[1] + 1), max(q[4], q[2] + 1)
            region = owner[q[2]:q[4], q[1]:q[3]]
            region[region == -1] = len(placed)
            placed.append(q + [int((region == len(placed)).sum())])
    return canvas, [q[:5] for q in placed]


def preview(folder, names, out_path, size):
    tiles = []
    for n in names:
        img = Image.open(folder / "images" / f"{n}.jpg").convert("RGB")
        g = ImageDraw.Draw(img)
        for row in (folder / "labels" / f"{n}.txt").read_text().strip().splitlines():
            c, xc, yc, bw, bh = row.split()
            xc, yc, bw, bh = (float(v) * size for v in (xc, yc, bw, bh))
            g.rectangle([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], outline="#00e000", width=2)
            g.rectangle([xc - bw / 2, yc - bh / 2, xc - bw / 2 + 46, yc - bh / 2 + 11], fill="#00e000")
            g.text((xc - bw / 2 + 2, yc - bh / 2), load_classes()[int(c)][0], fill="black")
        tiles.append(img.resize((300, 300)))
    sheet = Image.new("RGB", (4 * 306 + 6, -(-len(tiles) // 4) * 306 + 6), "white")
    for k, t in enumerate(tiles):
        sheet.paste(t, (6 + (k % 4) * 306, 6 + (k // 4) * 306))
    sheet.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--multi", type=int, default=4000)
    parser.add_argument("--out", default="data")
    args = parser.parse_args()

    out = config.PROJECT_DIR / args.out if not args.out.startswith("/") else type(config.PROJECT_DIR)(args.out)
    if out.exists():
        shutil.rmtree(out)
    classes = load_classes()
    label_of = {c: i for i, (c, _) in enumerate(classes)}
    state, data, bgs = load_state(), images(), backgrounds()
    rows = [(p, state[p["path"]]) for p in photos() if state.get(p["path"], {}).get("status") == "done"]
    counts = {}
    for split, n_multi in (("train", args.multi), ("val", args.multi // 5), ("test", args.multi // 5)):
        rng = np.random.default_rng(config.SEED + ("train", "val", "test").index(split))
        mine = [(p, e) for p, e in rows if p["split"] == split]
        by_class = {}
        for p, e in mine:
            by_class.setdefault(p["object_id"], []).append((p, e))
        single = out / ("test_single" if split == "test" else split)
        multi = out / ("test_multi" if split == "test" else split)
        for f in {single, multi}:
            (f / "images").mkdir(parents=True, exist_ok=True)
            (f / "labels").mkdir(parents=True, exist_ok=True)
        for p, e in mine:
            img = Image.fromarray(np.asarray(data[split][p["index"]]))
            save(img, e["boxes"], single, p["path"].split("/")[-1].rsplit(".", 1)[0], TILE, label_of)
        kinds = {"grid": 0, "paste": 0}
        n_boxes = 0
        for k in range(n_multi):
            kind = "grid" if k % 2 == 0 else "paste"
            img, boxes = grid(rng, by_class, data) if kind == "grid" else paste(rng, by_class, data, bgs[split])
            save(img, boxes, multi, f"multi_{split}_{kind}_{k:05d}", SIZE, label_of)
            kinds[kind] += 1
            n_boxes += len(boxes)
        counts[split] = {"single_photos": len(mine), "multi_images": n_multi, **kinds,
                         "multi_boxes": n_boxes, "backgrounds": len(bgs[split])}
        preview(multi, [f"multi_{split}_{'grid' if k % 2 == 0 else 'paste'}_{k:05d}" for k in range(12)],
                out / f"preview_{split}.jpg", SIZE)
        print(split, counts[split], flush=True)
    (out / "classes.json").write_text(json.dumps([c for c, _ in classes]))
    (out / "summary.json").write_text(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
