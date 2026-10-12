"""Stage 0: read-only check of the class dataset.

python inspect_data.py [--data_root PATH] [--object_ids PATH] [--workers 16]
"""

import argparse
import csv
import hashlib
import io
import re
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import config

Image.MAX_IMAGE_PIXELS = None

EXPECTED_PER_CLASS = 100
LARGE_SIDE = 1024
OBJ_ID_RE = re.compile(r"OBJ[_\- ]?(\d+)", re.IGNORECASE)
COURSE_ID_RE = re.compile(r"^OBJ\d{3}$")
MAYBE_BG_RE = re.compile(r"bg|background", re.IGNORECASE)
SYSTEM_FILES = {"thumbs.db", "desktop.ini", "icon\r"}
FORMAT_FOR_EXT = {".jpg": {"JPEG", "MPO"}, ".jpeg": {"JPEG", "MPO"}, ".png": {"PNG"}}  # MPO = multi-frame phone JPEG


def norm_key(text):
    return re.sub(r"[^a-z0-9]", "", text.lower())


def name_key(text):
    # "images_Water Bottle" -> "waterbottle"
    return norm_key(re.sub(r"^images?[_\- ]", "", text.strip(), flags=re.IGNORECASE))


def id_number(text):
    # "OBJ007" -> 7, "images_OBJ_0070" -> 70
    m = OBJ_ID_RE.search(text)
    return int(m.group(1)) if m else None


def is_hidden(rel):
    return any(part.startswith(".") for part in rel.parts) or rel.name.lower() in SYSTEM_FILES


def is_bg(path):
    return re.search(config.BG_REGEX, path.name, re.IGNORECASE) is not None


def load_object_ids(path):
    # Sheets exports can have title rows above the real header, so look for the row with "Object ID"
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))
    header_row = next((i for i, row in enumerate(rows) if "objectid" in map(norm_key, row)), None)
    if header_row is None:
        raise SystemExit(f"{path} has no 'Object ID' column. First rows: {rows[:3]}")
    columns = rows[header_row]
    col = {}
    for i, c in enumerate(columns):
        col.setdefault(norm_key(c), i)
    entries = []
    for row in rows[header_row + 1:]:
        get = lambda key: row[col[key]].strip() if key in col and col[key] < len(row) else ""
        entry = {"object_id": get("objectid"), "object_name": get("objectname"), "folder": get("objectfolder")}
        if any(entry.values()):
            entries.append(entry)
    return entries, columns, header_row


def match_folder(folder, top_file_id, by_folder, by_number, by_name):
    candidates = [
        ("csv folder column", by_folder.get(folder.strip().lower(), [])),
        ("ID in folder name", by_number.get(id_number(folder), [])),
        ("ID in filenames", by_number.get(top_file_id, [])),
        ("object name", by_name.get(name_key(folder), [])),
    ]
    for method, hits in candidates:
        if len(hits) == 1:
            return hits[0], method, ""
        if len(hits) > 1:
            return None, "AMBIGUOUS", f"{method} fits {', '.join(h['object_id'] or h['object_name'] for h in hits)}"
    return None, "UNMATCHED", ""


def scan_file(path):
    info = {"path": path, "md5": None, "format": None, "mode": None, "size": None,
            "orientation": 1, "error": None}
    try:
        data = path.read_bytes()
        info["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            info.update(format=im.format, mode=im.mode, size=im.size,
                        orientation=im.getexif().get(0x0112, 1))
            im.draft("RGB", (config.IMG_SIZE, config.IMG_SIZE))
            im.load()  # verify() misses truncated JPEGs, a full decode doesn't
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
    return info


def describe(values):
    if not values:
        return "n/a"
    return (f"min {min(values)}, max {max(values)}, mean {statistics.mean(values):.1f}, "
            f"median {statistics.median(values):g}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=config.DATASET_ROOT)
    parser.add_argument("--object_ids", type=Path, default=config.OBJECT_IDS_CSV)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = args.data_root
    if not root.is_dir():
        parser.error(f"dataset root not found: {root} (set DATASET_ROOT or pass --data_root)")

    class_dirs = sorted((d for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")),
                        key=lambda d: d.name.lower())
    root_files = [f for f in root.iterdir() if f.is_file() and not is_hidden(Path(f.name))]

    folders = {}
    for d in class_dirs:
        info = {"images": [], "other": [], "hidden": 0, "subdirs": 0}
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(d)
            if is_hidden(rel):
                info["hidden"] += p.is_file()
            elif p.is_dir():
                info["subdirs"] += 1
            elif p.suffix.lower() in config.IMAGE_EXTS:
                info["images"].append(p)
            else:
                info["other"].append(p)
        folders[d.name] = info

    all_images = [p for info in folders.values() for p in info["images"]]
    folder_of = {p: name for name, info in folders.items() for p in info["images"]}
    print(f"Found {len(class_dirs)} folders, {len(all_images)} images. Reading with {args.workers} threads...",
          flush=True)
    scans = {}
    with ThreadPoolExecutor(args.workers) as pool:
        for i, s in enumerate(pool.map(scan_file, all_images), 1):
            scans[s["path"]] = s
            if i % 500 == 0 or i == len(all_images):
                print(f"  {i}/{len(all_images)}", flush=True)
    readable = [s for s in scans.values() if s["error"] is None]
    unreadable = [s for s in scans.values() if s["error"] is not None]

    by_hash = defaultdict(list)
    for s in scans.values():
        if s["md5"]:
            by_hash[s["md5"]].append(s["path"])
    dup_groups = [sorted(g) for g in by_hash.values() if len(g) > 1]
    within_dups = [g for g in dup_groups if len({folder_of[p] for p in g}) == 1]
    cross_dups = [g for g in dup_groups if len({folder_of[p] for p in g}) > 1]
    dup_extra = Counter()
    for g in dup_groups:
        for name, n in Counter(folder_of[p] for p in g).items():
            dup_extra[name] += n - 1

    entries, csv_columns, header_row = [], [], 0
    if args.object_ids.is_file():
        entries, csv_columns, header_row = load_object_ids(args.object_ids)
    by_folder, by_number, by_name = defaultdict(list), defaultdict(list), defaultdict(list)
    for e in entries:
        if e["folder"]:
            by_folder[e["folder"].lower()].append(e)
        if id_number(e["object_id"]) is not None:
            by_number[id_number(e["object_id"])].append(e)
        for key in {name_key(e["object_name"]), name_key(e["object_id"])} - {""}:
            by_name[key].append(e)

    for name, info in folders.items():
        file_ids = Counter(n for n in (id_number(p.stem) for p in info["images"]) if n is not None)
        top_file_id = file_ids.most_common(1)[0][0] if file_ids else None
        entry, method, note = match_folder(name, top_file_id, by_folder, by_number, by_name)
        notes = [note] if note else []
        if entry:
            target = id_number(entry["object_id"])
            for where, n in (("folder name", id_number(name)), ("filenames", top_file_id)):
                if n is not None and n != target:
                    notes.append(f"{where} suggest OBJ number {n}, matched {entry['object_id']}")
        ok = [scans[p] for p in info["images"] if scans[p]["error"] is None]
        info.update(
            entry=entry, method=method, notes=notes,
            n_images=len(info["images"]),
            n_bg=sum(is_bg(p) for p in info["images"]),
            n_maybe_bg=sum(bool(MAYBE_BG_RE.search(p.stem)) and not is_bg(p) for p in info["images"]),
            n_unreadable=len(info["images"]) - len(ok),
            n_off_size=sum(s["size"] != (config.IMG_SIZE, config.IMG_SIZE) for s in ok),
            n_non_rgb=sum(s["mode"] != "RGB" for s in ok),
        )
    id_folders = defaultdict(list)
    for name, info in folders.items():
        if info["entry"]:
            id_folders[info["entry"]["object_id"]].append(name)

    def flags(info):
        f = []
        if info["entry"] is None:
            f.append(info["method"])
        elif len(id_folders[info["entry"]["object_id"]]) > 1:
            f.append("ID shared with another folder")
        elif not COURSE_ID_RE.match(info["entry"]["object_id"]):
            f.append("ID not OBJ###")
        if info["entry"] and info["notes"]:
            f.append("ID mismatch")
        if info["n_images"] == 0:
            f.append("EMPTY")
        elif info["n_images"] != EXPECTED_PER_CLASS:
            f.append(f"{'>' if info['n_images'] > EXPECTED_PER_CLASS else '<'}{EXPECTED_PER_CLASS} images")
        if info["n_bg"] == 0:
            f.append("no bg")
        if info["n_maybe_bg"]:
            f.append("bg naming differs")
        if info["n_unreadable"]:
            f.append("unreadable")
        if info["subdirs"]:
            f.append("subfolders")
        if info["other"]:
            f.append("non-image files")
        return f

    out = []
    say = out.append
    rel = lambda p: str(p.relative_to(root))

    say("STAGE 0 - DATASET INSPECTION REPORT")
    say(f"Dataset root : {root}")
    say(f"Object IDs   : {args.object_ids}" + ("" if entries else "  ** NOT FOUND or empty, no mapping possible **"))

    say("\n== 1. Overview")
    say(f"Class folders                 : {len(class_dirs)}")
    say(f"Image files (.jpg/.jpeg/.png) : {len(all_images)} ({len(readable)} readable, {len(unreadable)} unreadable)")
    say(f"Other files in class folders  : {sum(len(i['other']) for i in folders.values())}")
    say(f"Hidden/system files skipped   : {sum(i['hidden'] for i in folders.values())} (.DS_Store, ._* etc.)")
    say(f"Folders with subfolders       : {sum(1 for i in folders.values() if i['subdirs'])}"
        " (their images are counted for the parent folder)")
    if root_files:
        say(f"Loose files at dataset root   : {len(root_files)} -> {', '.join(f.name for f in root_files[:10])}")

    say("\n== 2. object_ids.csv")
    if entries:
        ids = [e["object_id"] for e in entries]
        missing = [e["object_name"] or "?" for e in entries if not e["object_id"]]
        bad_format = sorted({i for i in ids if i and not COURSE_ID_RE.match(i)})
        repeated = sorted(i for i, n in Counter(i for i in ids if i).items() if n > 1)
        unused = sorted(i for i in set(ids) - set(id_folders) if i)
        say(f"Header  : row {header_row + 1}" + (f" ({header_row} row(s) above it skipped)" if header_row else ""))
        say(f"Columns : {[c for c in csv_columns if c.strip()]}"
            + (f" (+{sum(1 for c in csv_columns if not c.strip())} unnamed)" if any(not c.strip() for c in csv_columns) else ""))
        say(f"Rows    : {len(entries)} ({sum(1 for i in ids if i)} with an Object ID)")
        say(f"Rows without an Object ID  : {len(missing)}" + (f" -> {missing}" if missing else ""))
        say(f"IDs not in OBJ### format   : {len(bad_format)}" + (f" -> {bad_format}" if bad_format else ""))
        say(f"IDs listed more than once  : {len(repeated)}" + (f" -> {repeated}" if repeated else ""))
        say(f"IDs with no matching folder: {len(unused)}" + (f" -> {unused}" if unused else ""))
    else:
        say("Not available.")

    say("\n== 3. Folder -> Object ID mapping")
    if not entries:
        say("Skipped, no object_ids.csv.")
    say("Match order: csv folder column, OBJ number in folder name, OBJ number in filenames, object name.")
    for method, n in Counter(i["method"] for i in folders.values()).most_common():
        say(f"  {method:<18}: {n} folders")
    problems = [(n, i) for n, i in folders.items() if entries and
                (i["entry"] is None or i["notes"] or len(id_folders[i["entry"]["object_id"]]) > 1)]
    say(f"Folders needing attention: {len(problems)}")
    for name, info in problems:
        target = info["entry"]["object_id"] if info["entry"] else "-"
        say(f"  {name!r} -> {target} [{info['method']}] {'; '.join(info['notes'])}")
    for oid, names in sorted(id_folders.items()):
        if len(names) > 1:
            say(f"  {oid} is matched by several folders: {names}")

    say("\n== 4. Images per folder")
    counts = [i["n_images"] for i in folders.values()]
    over = {n: i["n_images"] for n, i in folders.items() if i["n_images"] > EXPECTED_PER_CLASS}
    under = {n: i["n_images"] for n, i in folders.items() if i["n_images"] < EXPECTED_PER_CLASS}
    say(f"All images      : {describe(counts)}")
    say(f"Object (non-bg) : {describe([i['n_images'] - i['n_bg'] for i in folders.values()])}")
    say(f"Exactly {EXPECTED_PER_CLASS}     : {sum(c == EXPECTED_PER_CLASS for c in counts)} folders")
    say(f"More than {EXPECTED_PER_CLASS}  : {len(over)}" + (f" -> {over}" if over else ""))
    say(f"Fewer than {EXPECTED_PER_CLASS} : {len(under)}" + (f" -> {under}" if under else ""))

    say(f"\n== 5. Background images (filename matches {config.BG_REGEX!r}, case-insensitive)")
    say(f"Total: {sum(i['n_bg'] for i in folders.values())}")
    say("bg images per folder -> number of folders: "
        + ", ".join(f"{k}: {v}" for k, v in sorted(Counter(i['n_bg'] for i in folders.values()).items())))
    no_bg = [n for n, i in folders.items() if i["n_bg"] == 0]
    say(f"Folders with no bg images: {len(no_bg)}" + (f" -> {no_bg}" if no_bg else ""))
    maybe = {n: i["n_maybe_bg"] for n, i in folders.items() if i["n_maybe_bg"]}
    say(f"Files mentioning bg/background that don't match (NOT treated as background): "
        f"{sum(maybe.values())}" + (f" -> {maybe}" if maybe else ""))
    for name in list(maybe)[:5]:
        examples = [p.name for p in folders[name]["images"] if MAYBE_BG_RE.search(p.stem) and not is_bg(p)][:3]
        say(f"    e.g. {name}: {examples}")

    say("\n== 6. File extensions")
    ext_counts = Counter(p.suffix.lower() or "(none)" for i in folders.values() for p in i["images"] + i["other"])
    for ext, n in ext_counts.most_common():
        say(f"  {ext:<8} {n:>6}" + ("" if ext in config.IMAGE_EXTS else "   <- not accepted"))
    others = [p for i in folders.values() for p in i["other"]]
    for p in others[:20]:
        say(f"    {rel(p)}")
    if len(others) > 20:
        say(f"    ... and {len(others) - 20} more")
    mismatch = [s for s in readable if s["format"] not in FORMAT_FOR_EXT[s["path"].suffix.lower()]]
    say(f"Extension doesn't match actual format: {len(mismatch)}")
    for s in mismatch[:10]:
        say(f"    {rel(s['path'])} is {s['format']}")

    say("\n== 7. Image sizes (readable images, before EXIF rotation)")
    target = (config.IMG_SIZE, config.IMG_SIZE)
    sizes = Counter(s["size"] for s in readable)
    say(f"Exactly {target[0]}x{target[1]}  : {sizes[target]} of {len(readable)}")
    say(f"Distinct sizes   : {len(sizes)}")
    say(f"Non-square       : {sum(n for (w, h), n in sizes.items() if w != h)}")
    say(f"Longer side > {LARGE_SIDE}: {sum(n for (w, h), n in sizes.items() if max(w, h) > LARGE_SIDE)}")
    if readable:
        biggest = max(readable, key=lambda s: s["size"][0] * s["size"][1])
        say(f"Largest          : {biggest['size'][0]}x{biggest['size'][1]} ({rel(biggest['path'])})")
    say("Most common      : " + ", ".join(f"{w}x{h} ({n})" for (w, h), n in sizes.most_common(10)))
    off = {n: i["n_off_size"] for n, i in folders.items() if i["n_off_size"]}
    say(f"Folders with non-{target[0]}x{target[1]} images: {len(off)}" + (f" -> {off}" if off else ""))
    rotated = [s for s in readable if s["orientation"] not in (1, None)]
    say(f"Images with an EXIF rotation tag: {len(rotated)}"
        + (f" in {sorted({folder_of[s['path']] for s in rotated})}" if rotated else ""))

    say("\n== 8. Colour modes")
    for mode, n in Counter(s["mode"] for s in readable).most_common():
        where = sorted({folder_of[s["path"]] for s in readable if s["mode"] == mode})
        say(f"  {mode:<6} {n:>6}" + ("" if mode == "RGB" else f"   in {where}"))
    say("Formats: " + ", ".join(f"{f} ({n})" for f, n in Counter(s["format"] for s in readable).most_common()))

    say("\n== 9. Unreadable files")
    say(f"Count: {len(unreadable)}")
    for s in sorted(unreadable, key=lambda s: s["path"]):
        say(f"  {rel(s['path'])}: {s['error']}")

    say("\n== 10. Exact duplicates (identical bytes)")
    say(f"Groups: {len(dup_groups)}, extra copies: {sum(len(g) - 1 for g in dup_groups)}")
    say(f"Within one folder: {len(within_dups)} groups")
    for g in within_dups[:30]:
        mixed = len({is_bg(p) for p in g}) > 1
        say(f"  {folder_of[g[0]]}: {[p.name for p in g]}" + ("  <- same image named as bg AND object" if mixed else ""))
    if len(within_dups) > 30:
        say(f"  ... and {len(within_dups) - 30} more")
    say(f"Across folders: {len(cross_dups)} groups")
    for g in cross_dups:
        say(f"  {[rel(p) for p in g]}")

    say("\n== 11. Per-folder table")
    header = (f"{'Folder':<34} {'Object ID':<11} {'Matched by':<18} {'Imgs':>5} {'BG':>3} "
              f"{'Off-sz':>6} {'NonRGB':>6} {'Bad':>4} {'Dup':>4}  Flags")
    say(header)
    say("-" * len(header))
    for name, info in folders.items():
        oid = info["entry"]["object_id"] if info["entry"] else "-"
        say(f"{name[:34]:<34} {oid[:11]:<11} {info['method']:<18} {info['n_images']:>5} {info['n_bg']:>3} "
            f"{info['n_off_size']:>6} {info['n_non_rgb']:>6} {info['n_unreadable']:>4} {dup_extra[name]:>4}  "
            + ", ".join(flags(info)))

    report = "\n".join(out) + "\n"
    report_path = config.LOG_DIR / "inspection_report.txt"
    report_path.write_text(report)

    mapping_path = config.LOG_DIR / "folder_mapping.csv"
    with open(mapping_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["folder", "object_id", "object_name", "matched_by", "note", "n_images", "n_bg"])
        for name, info in folders.items():
            e = info["entry"] or {}
            w.writerow([name, e.get("object_id", ""), e.get("object_name", ""), info["method"],
                        "; ".join(info["notes"]), info["n_images"], info["n_bg"]])

    print(report)
    print(f"Saved {report_path}\nSaved {mapping_path}")


if __name__ == "__main__":
    main()
