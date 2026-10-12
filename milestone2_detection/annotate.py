"""Command line tool used to place the bounding boxes by hand on the 224x224 photos.

python annotate.py sheets --cls OBJ001 --round 1
python annotate.py add --sheet OBJ001_r1_01 < boxes.txt     lines like "3 OBJ001 60 84 195 219" or "3 exclude <reason>"
python annotate.py zoom --sheet OBJ001_r1_01 --tiles 6 7
python annotate.py verify --cls OBJ001
python annotate.py fix --sheet OBJ001_v01 < fixes.txt
python annotate.py ref --cls OBJ064 OBJ020
python annotate.py status
python annotate.py export
"""

import argparse
import csv
import json
import sys
from datetime import datetime, timezone

import numpy as np
from PIL import Image, ImageDraw

import config

OUT = config.OUTPUT_DIR / "boxes_manual"
STATE = OUT / "annotations.json"
PILOT = config.PROJECT_DIR.parent / "Milestone2_Detection" / "source_boxes.csv"
SPLITS = ("train", "val", "test")
ROUND1 = {"train": 20, "val": 5, "test": 5}
GRID_TILE, GRID_COLS, PER_GRID = 300, 4, 16
VERIFY_TILE, VERIFY_COLS, PER_VERIFY = 320, 5, 20  # big enough to see a box that is off by a few pixels


def load_classes():
    with open(config.CACHE_DIR / "classes.csv", newline="") as f:
        return [(r["object_id"], r["object_name"]) for r in csv.DictReader(f)]


def photos():
    out = []
    for s in SPLITS:
        with np.load(config.CACHE_DIR / f"{s}.npz") as f:
            y, paths = f["y"], f["paths"]
        ids = [c for c, _ in load_classes()]
        out += [{"path": str(p), "split": s, "index": i, "object_id": ids[int(c)]} for i, (c, p) in enumerate(zip(y, paths))]
    return out


def images():
    out = {}
    for s in SPLITS:
        x_path = config.OUTPUT_DIR / "boxes" / f"{s}_x.npy"
        if not x_path.is_file():
            x_path.parent.mkdir(parents=True, exist_ok=True)
            with np.load(config.CACHE_DIR / f"{s}.npz") as f:
                np.save(x_path, f["x"])
        out[s] = np.load(x_path, mmap_mode="r")
    return out


def load_state():
    return json.loads(STATE.read_text()) if STATE.is_file() else {}


def save_state(state):
    OUT.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state, indent=0))


def log(event, **kw):
    with open(OUT / "decisions.jsonl", "a") as f:
        f.write(json.dumps({"time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": event, **kw}) + "\n")


def round_of(p, cls_photos):
    same = sorted((q for q in cls_photos if q["split"] == p["split"]), key=lambda q: q["path"])
    k = ROUND1[p["split"]]
    picks = {same[round(j * (len(same) - 1) / max(k - 1, 1))]["path"] for j in range(min(k, len(same)))}
    return 1 if p["path"] in picks else 2


def parse_line(line, valid_ids):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    tile, rest = line.split(None, 1)
    if rest.startswith("exclude"):
        return int(tile), {"status": "excluded", "reason": rest[len("exclude"):].strip() or "not visible", "boxes": []}
    boxes = []
    for part in rest.split(";"):
        oid, *xyxy = part.split()
        x1, y1, x2, y2 = (float(v) for v in xyxy)
        if oid not in valid_ids or not (0 <= x1 < x2 <= 224 and 0 <= y1 < y2 <= 224):
            raise SystemExit(f"bad box on tile {tile}: {part.strip()}")
        boxes.append([oid, x1, y1, x2, y2])
    return int(tile), {"status": "done", "reason": "", "boxes": boxes}


def import_pilot():
    state = load_state()
    by_path = {(p["path"], p["split"]): p for p in photos()}
    n = 0
    with open(PILOT, newline="") as f:
        for r in csv.DictReader(f):
            p = by_path[(r["source_path"], r["split"])]
            entry = state.setdefault(p["path"], {"split": p["split"], "index": p["index"], "photo_object_id": p["object_id"],
                                                 "status": "done", "reason": "", "boxes": [], "source": "codex pilot"})
            x2, y2 = min(float(r["xmax"]), 224), min(float(r["ymax"]), 224)
            entry["boxes"].append([r["object_id"], float(r["xmin"]), float(r["ymin"]), x2, y2])
            n += 1
    save_state(state)
    log("import_pilot", boxes=n)
    print(f"imported {n} pilot boxes")


def draw_grid(img, scale, step=16):
    g = ImageDraw.Draw(img)
    for v in range(0, 225, step):
        color = "#ffeb3b" if v % 32 == 0 else "#9e9e9e"
        g.line([(v * scale, 0), (v * scale, 224 * scale)], fill=color, width=1)
        g.line([(0, v * scale), (224 * scale, v * scale)], fill=color, width=1)
    for v in range(0, 224, 32):
        for x, y in ((v * scale + 2, 1), (1, v * scale + 2)):
            g.rectangle([x, y, x + 17, y + 10], fill="black")
            g.text((x + 1, y - 1), str(v), fill="#ffeb3b")


def draw_boxes(img, boxes, scale, own):
    g = ImageDraw.Draw(img)
    for oid, x1, y1, x2, y2 in boxes:
        color = "#00e000" if oid == own else "#ff00ff"
        g.rectangle([x1 * scale, y1 * scale, x2 * scale, y2 * scale], outline=color, width=2)
        g.rectangle([x1 * scale, y1 * scale, x1 * scale + 46, y1 * scale + 11], fill=color)
        g.text((x1 * scale + 2, y1 * scale), oid, fill="black")


def render(name, items, data, tile, cols, grid, kind):
    rows = -(-len(items) // cols)
    sheet = Image.new("RGB", (cols * (tile + 10) + 10, rows * (tile + 30) + 10), "white")
    d = ImageDraw.Draw(sheet)
    scale = tile / 224
    for k, (p, entry) in enumerate(items):
        img = Image.fromarray(np.asarray(data[p["split"]][p["index"]])).resize((tile, tile), Image.LANCZOS)
        if grid:
            draw_grid(img, scale)
        if entry:
            draw_boxes(img, entry["boxes"], scale, p["object_id"])
        x, y = 10 + (k % cols) * (tile + 10), 10 + (k // cols) * (tile + 30)
        sheet.paste(img, (x, y + 20))
        tag = "" if not entry or entry["status"] == "done" else " EXCLUDED"
        d.text((x, y + 4), f"{k + 1}. {p['object_id']} {p['split']} {p['path'].split('/')[-1]}{tag}", fill="black")
    folder = OUT / "sheets" / kind
    folder.mkdir(parents=True, exist_ok=True)
    sheet.save(folder / f"{name}.png")
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    manifest[name] = [p["path"] for p, _ in items]
    manifest_path.write_text(json.dumps(manifest, indent=0))
    return folder / f"{name}.png"


def sheets(cls, round_no):
    state = load_state()
    allp = [p for p in photos() if p["object_id"] == cls]
    todo = [p for p in allp if p["path"] not in state and (round_no == 2 or round_of(p, allp) == 1)]
    todo.sort(key=lambda p: (SPLITS.index(p["split"]), p["path"]))
    data = images()
    made = []
    for n, start in enumerate(range(0, len(todo), PER_GRID), 1):
        made.append(render(f"{cls}_r{round_no}_{n:02d}", [(p, None) for p in todo[start:start + PER_GRID]], data,
                           GRID_TILE, GRID_COLS, True, "grid"))
    print("\n".join(str(m) for m in made) if made else f"{cls}: nothing left for round {round_no}")


def add(sheet_name, kind="grid"):
    manifest = json.loads((OUT / "sheets" / kind / "manifest.json").read_text())
    paths = manifest[sheet_name]
    by_path = {p["path"]: p for p in photos()}
    valid = {c for c, _ in load_classes()}
    state = load_state()
    seen = set()
    for line in sys.stdin:
        parsed = parse_line(line, valid)
        if parsed is None:
            continue
        tile, entry = parsed
        p = by_path[paths[tile - 1]]
        if entry["status"] == "done" and p["object_id"] not in [b[0] for b in entry["boxes"]]:
            raise SystemExit(f"tile {tile}: no box for the photo's own object {p['object_id']}")
        old = state.get(p["path"])
        state[p["path"]] = {"split": p["split"], "index": p["index"], "photo_object_id": p["object_id"],
                            "source": "fixed" if kind == "verify" else "visual", **entry}
        log("fix" if kind == "verify" else "add", sheet=sheet_name, tile=tile, path=p["path"], new=entry,
            old=old if kind == "verify" else None)
        seen.add(tile)
    if kind == "grid" and seen != set(range(1, len(paths) + 1)):
        raise SystemExit(f"{sheet_name}: tiles without a decision: {sorted(set(range(1, len(paths) + 1)) - seen)}")
    save_state(state)
    print(f"{sheet_name}: {len(seen)} photos recorded")


def zoom(sheet_name, tiles):
    paths = json.loads((OUT / "sheets" / "grid" / "manifest.json").read_text())[sheet_name]
    by_path = {p["path"]: p for p in photos()}
    data = images()
    sheet = Image.new("RGB", (2 * 458 + 10, -(-len(tiles) // 2) * 478 + 10), "white")
    d = ImageDraw.Draw(sheet)
    for k, t in enumerate(tiles):
        p = by_path[paths[t - 1]]
        img = Image.fromarray(np.asarray(data[p["split"]][p["index"]])).resize((448, 448), Image.LANCZOS)
        draw_grid(img, 2.0, step=8)
        x, y = 10 + (k % 2) * 458, 10 + (k // 2) * 478
        sheet.paste(img, (x, y + 20))
        d.text((x, y + 4), f"tile {t}. {p['object_id']} {p['path'].split('/')[-1]}", fill="black")
    folder = OUT / "sheets" / "zoom"
    folder.mkdir(parents=True, exist_ok=True)
    sheet.save(folder / f"{sheet_name}_z.png")
    print(folder / f"{sheet_name}_z.png")


def ref(classes):
    allp = photos()
    data = images()
    names = dict(load_classes())
    sheet = Image.new("RGB", (6 * 160 + 10, len(classes) * 180 + 10), "white")
    d = ImageDraw.Draw(sheet)
    for r, c in enumerate(classes):
        ps = sorted((p for p in allp if p["object_id"] == c and p["split"] == "train"), key=lambda p: p["path"])
        for k, p in enumerate(ps[::max(1, len(ps) // 6)][:6]):
            sheet.paste(Image.fromarray(np.asarray(data[p["split"]][p["index"]])).resize((150, 150)), (10 + k * 160, r * 180 + 24))
        d.text((10, r * 180 + 6), f"{c} {names[c]}", fill="black")
    folder = OUT / "sheets" / "ref"
    folder.mkdir(parents=True, exist_ok=True)
    out = folder / ("_".join(classes) + ".png")
    sheet.save(out)
    print(out)


def verify(cls, tiles=None):
    state = load_state()
    items = [(p, state[p["path"]]) for p in photos() if p["object_id"] == cls and p["path"] in state]
    items.sort(key=lambda t: (SPLITS.index(t[0]["split"]), t[0]["path"]))
    data = images()
    if tiles:  # re-check of fixed tiles (numbered as on the first verify sheet), drawn larger
        made = [render(f"{cls}_recheck", [items[t - 1] for t in tiles], data, 400, 4, False, "recheck")]
        print(made[0])
        return
    made = [render(f"{cls}_v{n:02d}", items[s:s + PER_VERIFY], data, VERIFY_TILE, VERIFY_COLS, False, "verify")
            for n, s in enumerate(range(0, len(items), PER_VERIFY), 1)]
    print("\n".join(str(m) for m in made))


def status():
    state = load_state()
    rows = {}
    for p in photos():
        r = rows.setdefault(p["object_id"], {"photos": 0, "done": 0, "excluded": 0, "boxes": 0, "other": 0})
        r["photos"] += 1
        e = state.get(p["path"])
        if e:
            r[e["status"]] += 1
            r["boxes"] += len(e["boxes"])
            r["other"] += sum(b[0] != p["object_id"] for b in e["boxes"])
    total = {k: sum(r[k] for r in rows.values()) for k in ("photos", "done", "excluded", "boxes", "other")}
    print(f"total: {total}")
    progress = {"updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "total": total, "per_class": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "progress.json").write_text(json.dumps(progress, indent=1))


def export():
    state = load_state()
    label_of = {c: i for i, (c, _) in enumerate(load_classes())}
    with open(OUT / "annotations.csv", "w", newline="") as fa, open(OUT / "photos.csv", "w", newline="") as fp:
        wa, wp = csv.writer(fa), csv.writer(fp)
        wa.writerow(["path", "split", "index", "photo_object_id", "box_object_id", "is_primary", "x1", "y1", "x2", "y2", "source"])
        wp.writerow(["path", "split", "index", "photo_object_id", "status", "n_boxes", "reason", "source"])
        for path, e in sorted(state.items()):
            wp.writerow([path, e["split"], e["index"], e["photo_object_id"], e["status"], len(e["boxes"]), e["reason"], e["source"]])
            lines = []
            for oid, x1, y1, x2, y2 in e["boxes"]:
                wa.writerow([path, e["split"], e["index"], e["photo_object_id"], oid, int(oid == e["photo_object_id"]),
                             x1, y1, x2, y2, e["source"]])
                lines.append(f"{label_of[oid]} {(x1 + x2) / 448:.6f} {(y1 + y2) / 448:.6f} {(x2 - x1) / 224:.6f} {(y2 - y1) / 224:.6f}")
            if e["status"] == "done":
                folder = OUT / "labels" / e["split"]
                folder.mkdir(parents=True, exist_ok=True)
                (folder / (path.split("/")[-1].rsplit(".", 1)[0] + ".txt")).write_text("\n".join(lines) + "\n")
    print(f"exported {len(state)} photos")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["import-pilot", "sheets", "add", "verify", "fix", "verified", "zoom", "ref", "status", "export"])
    parser.add_argument("--cls", nargs="*")
    parser.add_argument("--round", type=int, default=1, choices=[1, 2])
    parser.add_argument("--sheet")
    parser.add_argument("--tiles", nargs="*", type=int)
    args = parser.parse_args()
    if args.command == "import-pilot":
        import_pilot()
    elif args.command == "sheets":
        sheets(args.cls[0], args.round)
    elif args.command == "add":
        add(args.sheet)
    elif args.command == "verify":
        verify(args.cls[0], args.tiles)
    elif args.command == "fix":
        add(args.sheet, kind="verify")
    elif args.command == "verified":
        paths = json.loads((OUT / "sheets" / "verify" / "manifest.json").read_text())[args.sheet]
        log("verified", sheet=args.sheet, photos=len(paths))
        print(f"{args.sheet}: {len(paths)} photos marked verified")
    elif args.command == "ref":
        ref(args.cls)
    elif args.command == "zoom":
        zoom(args.sheet, args.tiles)
    elif args.command == "status":
        status()
    else:
        export()


if __name__ == "__main__":
    main()
