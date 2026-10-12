"""Stage 1: dedup, resize, split 70/15/15 and cache the dataset. Run inspect_data.py first.

python preprocess.py [--workers 16]
"""

import argparse
import csv
import hashlib
import io
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor

import imagehash
import numpy as np
from PIL import Image, ImageOps

import config
from inspect_data import is_bg, is_hidden

Image.MAX_IMAGE_PIXELS = None

BACKGROUND = "BACKGROUND"
NEAR_DUP_MAX_DIST = 4
COPY_SUFFIX = re.compile(r"\s?\(\d+\)$")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
SPLITS = ("train", "val", "test")


def load_mapping(folders):
    path = config.LOG_DIR / "folder_mapping.csv"
    if not path.is_file():
        raise SystemExit(f"{path} not found, run inspect_data.py first")
    with open(path, newline="") as f:
        rows = {r["folder"]: r for r in csv.DictReader(f)}
    problems = [f"{n} has no Object ID" for n in folders if not rows.get(n, {}).get("object_id")]
    ids = Counter(rows[n]["object_id"] for n in folders if rows.get(n, {}).get("object_id"))
    problems += [f"{oid} is used by {k} folders" for oid, k in ids.items() if k > 1]
    if problems:
        raise SystemExit("folder_mapping.csv is not usable:\n  " + "\n  ".join(problems)
                         + "\nAdd object_ids.csv and rerun inspect_data.py, or fix the mapping by hand.")
    return {n: (rows[n]["object_id"], rows[n]["object_name"]) for n in folders}


def load_image(path):
    rec = {"path": path, "md5": None, "error": None, "orientation": 1, "orig_size": None, "orig_mode": None,
           "pixels": None, "phash": None}
    try:
        data = path.read_bytes()
        rec["md5"] = hashlib.md5(data).hexdigest()
        with Image.open(io.BytesIO(data)) as im:
            im.verify()
        with Image.open(io.BytesIO(data)) as im:
            rec["orientation"] = im.getexif().get(0x0112, 1)
            rec["orig_size"], rec["orig_mode"] = im.size, im.mode
            im = ImageOps.exif_transpose(im).convert("RGB")
            im = ImageOps.fit(im, (config.IMG_SIZE, config.IMG_SIZE), Image.LANCZOS)  # center crop + resize
        rec["pixels"] = np.asarray(im, dtype=np.uint8)
        rec["phash"] = np.packbits(imagehash.phash(im).hash.flatten())
    except Exception as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    return rec


def near_duplicate_pairs(hashes, max_dist, chunk=512):
    h = np.stack(hashes)
    pairs = []
    for start in range(0, len(h), chunk):
        dist = POPCOUNT[h[start:start + chunk, None, :] ^ h[None, :, :]].sum(-1)
        i, j = np.nonzero(dist <= max_dist)
        keep = i + start < j
        pairs += zip(i[keep] + start, j[keep], dist[i[keep], j[keep]])
    return pairs


def near_duplicate_groups(n, pairs):
    # connected components (union-find); returns lists of indices, sorted for determinism
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i, j, _ in pairs:
        parent[find(i)] = find(j)
    groups = defaultdict(list)
    for k in range(n):
        groups[find(k)].append(k)
    return sorted(groups.values())


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    if config.BG_MODE not in ("exclude", "bg_class"):
        raise SystemExit(f"BG_MODE must be 'exclude' or 'bg_class', got {config.BG_MODE!r}")

    root = config.DATASET_ROOT
    folders = sorted((d.name for d in root.iterdir() if d.is_dir() and not d.name.startswith(".")), key=str.lower)
    mapping = load_mapping(folders)
    rel = lambda p: p.relative_to(root).as_posix()
    folder_of = lambda p: p.relative_to(root).parts[0]

    excluded = []  # (path, object_id, step, reason)

    candidates = []
    for name in folders:
        for p in sorted((root / name).rglob("*")):
            if p.is_dir():
                continue
            if is_hidden(p.relative_to(root / name)):
                excluded.append((rel(p), mapping[name][0], "scan", "hidden/system file"))
            elif p.suffix.lower() not in config.IMAGE_EXTS:
                excluded.append((rel(p), mapping[name][0], "scan", f"unsupported file type {p.suffix}"))
            else:
                candidates.append(p)

    print(f"Processing {len(candidates)} images with {args.workers} threads...", flush=True)
    recs = []
    with ThreadPoolExecutor(args.workers) as pool:
        for i, r in enumerate(pool.map(load_image, candidates), 1):
            recs.append(r)
            if i % 1000 == 0 or i == len(candidates):
                print(f"  {i}/{len(candidates)}", flush=True)

    valid = []
    for r in recs:
        if r["error"]:
            excluded.append((rel(r["path"]), mapping[folder_of(r["path"])][0], "validate", f"unreadable: {r['error']}"))
        else:
            valid.append(r)

    by_md5 = defaultdict(list)
    for r in valid:
        by_md5[r["md5"]].append(r)
    unique, cross_class = [], []
    dup_kinds = Counter()
    for group in by_md5.values():
        # keep an object-named copy over a bg-named one, and "x.jpg" over a download copy "x(1).jpg"
        group.sort(key=lambda r: (is_bg(r["path"]), COPY_SUFFIX.search(r["path"].stem) is not None, rel(r["path"])))
        keep = group[0]
        unique.append(keep)
        for r in group[1:]:
            oid = mapping[folder_of(r["path"])][0]
            if folder_of(r["path"]) != folder_of(keep["path"]):
                kind, reason = "across classes", f"exact duplicate of {rel(keep['path'])} (different class)"
                cross_class.append((rel(keep["path"]), rel(r["path"]), mapping[folder_of(keep["path"])][0], oid))
            elif is_bg(r["path"]) and not is_bg(keep["path"]):
                kind, reason = "bg-named copy of an object image", f"bg-named copy of object image {rel(keep['path'])}"
            else:
                kind, reason = "within class", f"exact duplicate of {rel(keep['path'])}"
            dup_kinds[kind] += 1
            excluded.append((rel(r["path"]), oid, "deduplicate", reason))
    unique.sort(key=lambda r: rel(r["path"]))

    final = []
    for r in unique:
        name = folder_of(r["path"])
        if is_bg(r["path"]):
            if config.BG_MODE == "exclude":
                excluded.append((rel(r["path"]), mapping[name][0], "background", "background image (BG_MODE=exclude)"))
                continue
            r["object_id"] = BACKGROUND
        else:
            r["object_id"] = mapping[name][0]
        final.append(r)

    classes = sorted({r["object_id"] for r in final} - {BACKGROUND})
    if config.BG_MODE == "bg_class":
        classes.append(BACKGROUND)
    label_of = {c: i for i, c in enumerate(classes)}
    name_of = {oid: oname for oid, oname in mapping.values()}
    name_of[BACKGROUND] = "background (no object)"
    pairs = near_duplicate_pairs([r["phash"] for r in final], NEAR_DUP_MAX_DIST)
    groups = near_duplicate_groups(len(final), pairs)
    for g, members in enumerate(groups):
        for k in members:
            final[k]["group"] = g if len(members) > 1 else ""

    # Group-aware stratified split: per class, shuffle the near-duplicate groups (singletons are
    # groups of one) and fill test, then val, up to their target sizes; the rest go to train.
    ratios = config.SPLIT_RATIOS
    rng = np.random.default_rng(config.SEED)
    groups_by_class = defaultdict(list)
    for members in groups:
        groups_by_class[final[members[0]]["object_id"]].append(members)
    for c in classes:
        class_groups = groups_by_class[c]
        rng.shuffle(class_groups)
        n = sum(len(g) for g in class_groups)
        target = {"test": round(n * ratios["test"]), "val": round(n * ratios["val"])}
        filled = Counter()
        for members in class_groups:
            split = next((s for s in ("test", "val") if filled[s] + len(members) <= target[s]), "train")
            filled[split] += len(members)
            for k in members:
                final[k]["split"] = split

    near_rows = []
    for i, j, d in sorted(pairs, key=lambda p: (p[2], p[0], p[1])):
        a, b = final[i], final[j]
        near_rows.append((rel(a["path"]), rel(b["path"]), a["object_id"], b["object_id"], int(d),
                          a["split"], b["split"], a["object_id"] == b["object_id"], a["split"] != b["split"]))

    for split in SPLITS:
        members = [r for r in final if r["split"] == split]
        np.savez(config.CACHE_DIR / f"{split}.npz",
                 x=np.stack([r["pixels"] for r in members]),
                 y=np.array([label_of[r["object_id"]] for r in members], dtype=np.int64),
                 paths=np.array([rel(r["path"]) for r in members]))

    write_csv(config.CACHE_DIR / "splits.csv", ["path", "object_id", "label_idx", "split", "near_dup_group"],
              [(rel(r["path"]), r["object_id"], label_of[r["object_id"]], r["split"], r["group"]) for r in final])
    write_csv(config.CACHE_DIR / "classes.csv", ["label_idx", "object_id", "object_name"],
              [(i, c, name_of.get(c, "")) for i, c in enumerate(classes)])
    write_csv(config.LOG_DIR / "preprocessing_log.csv", ["path", "object_id", "step", "reason"], excluded)
    write_csv(config.LOG_DIR / "exif_rotated.csv", ["path", "object_id", "exif_orientation", "in_final_dataset"],
              [(rel(r["path"]), mapping[folder_of(r["path"])][0], r["orientation"], "split" in r)
               for r in valid if r["orientation"] not in (1, None)])
    write_csv(config.LOG_DIR / "cross_class_duplicates.csv", ["kept", "removed", "kept_object_id", "removed_object_id"],
              cross_class)
    write_csv(config.LOG_DIR / "near_duplicates.csv",
              ["path_a", "path_b", "object_id_a", "object_id_b", "hamming", "split_a", "split_b",
               "same_class", "crosses_split"], near_rows)

    # summary
    raw_per_class = Counter(mapping[folder_of(r["path"])][0] for r in recs)
    removed_per_class = Counter(oid for _, oid, step, _ in excluded if step in ("validate", "deduplicate"))
    bg_per_class = Counter(oid for _, oid, step, _ in excluded if step == "background")
    split_counts = defaultdict(Counter)
    for r in final:
        split_counts[r["object_id"]][r["split"]] += 1
    rotated = [r for r in final if r["orientation"] not in (1, None)]
    resized = [r for r in final if r["orig_size"] != (config.IMG_SIZE, config.IMG_SIZE)]
    crossing = [row for row in near_rows if row[-1]]

    out = []
    say = out.append
    say("STAGE 1 - PREPROCESSING SUMMARY")
    say(f"Dataset root: {root}")
    say(f"Seed {config.SEED}, image size {config.IMG_SIZE}x{config.IMG_SIZE}, BG_MODE={config.BG_MODE}, "
        f"background rule: filename matches {config.BG_REGEX!r} (case-insensitive)")
    say(f"Split ratios: {ratios} (stratified by class, near-duplicate groups kept together)")
    say("")
    say("Step counts")
    say(f"  Non-image / system files skipped : {sum(1 for e in excluded if e[2] == 'scan')}")
    say(f"  Image files found                : {len(recs)}")
    say(f"  - unreadable                     : {len(recs) - len(valid):>4}  -> {len(valid)}")
    say(f"  - exact duplicates               : {sum(dup_kinds.values()):>4}  -> {len(unique)}"
        + (f"  ({', '.join(f'{k}: {v}' for k, v in dup_kinds.items())})" if dup_kinds else ""))
    if config.BG_MODE == "exclude":
        say(f"  - background images (excluded)   : {len(unique) - len(final):>4}  -> {len(final)}")
    else:
        say(f"  Background images kept as {BACKGROUND}: {sum(r['object_id'] == BACKGROUND for r in final)}")
    say(f"  Final dataset                    : {len(final)} images, {len(classes)} classes")
    say(f"  Split                            : " + ", ".join(f"{s} {sum(r['split'] == s for r in final)}" for s in SPLITS))
    say("")
    say("Image handling (final dataset)")
    say(f"  EXIF-rotated before cropping : {len(rotated)} (listed in exif_rotated.csv)")
    say(f"  Not originally {config.IMG_SIZE}x{config.IMG_SIZE} (center-cropped + resized): {len(resized)}")
    for r in resized:
        say(f"    {rel(r['path'])} was {r['orig_size'][0]}x{r['orig_size'][1]}")
    say(f"  Source colour modes          : {dict(Counter(r['orig_mode'] for r in final))}, all converted to RGB")
    say("")
    say(f"Near-duplicates (pHash Hamming distance <= {NEAR_DUP_MAX_DIST}, logged only, not removed)")
    say(f"  Pairs               : {len(near_rows)}")
    multi = [g for g in groups if len(g) > 1]
    say(f"  Groups (connected)  : {len(multi)} groups covering {sum(map(len, multi))} images, largest {max(map(len, multi), default=0)}; "
        "each group is kept in one split")
    say(f"  Same class          : {sum(1 for row in near_rows if row[7])}")
    say(f"  Different classes   : {sum(1 for row in near_rows if not row[7])}")
    say(f"  Crossing splits     : {len(crossing)}  "
        + ", ".join(f"{a}-{b}: {n}" for (a, b), n in sorted(Counter(tuple(sorted((row[5], row[6]))) for row in crossing).items())))
    say(f"  Crossing pairs involving test: {sum(1 for row in crossing if 'test' in (row[5], row[6]))}")
    say(f"  Exact duplicates across classes: {len(cross_class)}")
    say("")
    say("Per-class counts")
    header = (f"{'Object ID':<11} {'Object name':<28} {'Raw':>4} {'Dup/bad':>7} {'BG excl':>7} {'Final':>5} "
              f"{'Train':>5} {'Val':>4} {'Test':>4}")
    say(header)
    say("-" * len(header))
    for c in classes:
        sc = split_counts[c]
        say(f"{c:<11} {name_of.get(c, '')[:28]:<28} {raw_per_class.get(c, 0):>4} {removed_per_class.get(c, 0):>7} "
            f"{bg_per_class.get(c, 0):>7} {sum(sc.values()):>5} {sc['train']:>5} {sc['val']:>4} {sc['test']:>4}")
    totals = Counter()
    for c in classes:
        totals.update(split_counts[c])
    say("-" * len(header))
    say(f"{'Total':<40} {len(recs):>4} {sum(removed_per_class.values()):>7} {sum(bg_per_class.values()):>7} "
        f"{len(final):>5} {totals['train']:>5} {totals['val']:>4} {totals['test']:>4}")

    summary = "\n".join(out) + "\n"
    (config.LOG_DIR / "preprocessing_summary.txt").write_text(summary)
    print(summary)
    print(f"Cache: {', '.join(str(config.CACHE_DIR / f'{s}.npz') for s in SPLITS)}")
    print(f"Logs : {config.LOG_DIR}")


if __name__ == "__main__":
    main()
