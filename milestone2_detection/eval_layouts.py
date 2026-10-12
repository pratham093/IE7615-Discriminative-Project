"""Scores the final model separately on the 2x2, 3x3, pasted and single test images.

python eval_layouts.py [--weights results/yolov8s_best.pt] [--batch 8]
"""

import argparse
import json
from pathlib import Path

from ultralytics import YOLO

import config
from train_detect import IMG_SIZE, device, scores

DATA = config.PROJECT_DIR / "data"
OUT = config.OUTPUT_DIR / "m2_results"
GROUPS = {"2x2": ("multi_test_grid_", "multi_test_multi_g2_"), "3x3": ("multi_test_multi_g3_",), "paste": ("multi_test_paste_",)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(config.PROJECT_DIR / "results" / "yolov8s_best.pt"))
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    names = json.loads((DATA / "classes.json").read_text())
    lists = OUT / "layout_lists"
    lists.mkdir(parents=True, exist_ok=True)
    tests = {"single": str((DATA / "test_single" / "images").resolve())}
    for g, prefixes in GROUPS.items():
        files = sorted(p.resolve() for p in (DATA / "test_multi" / "images").glob("*.jpg") if p.name.startswith(prefixes))
        (lists / f"{g}.txt").write_text("\n".join(map(str, files)) + "\n")
        tests[g] = str(lists / f"{g}.txt")

    dev, gpu = device()
    model = YOLO(args.weights)
    out = {"weights": args.weights, "device": gpu, "imgsz": IMG_SIZE}
    for g, path in tests.items():
        yaml = lists / f"data_{g}.yaml"
        yaml.write_text("\n".join([f"path: {DATA.resolve().as_posix()}", "train: train/images", "val: val/images",
                                   f"test: {path}", f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]) + "\n")
        m = model.val(data=str(yaml), split="test", imgsz=IMG_SIZE, batch=args.batch, device=dev, workers=2,
                      project=str(OUT / "runs"), name=f"layout_{g}", exist_ok=True, verbose=False)
        n = sum(1 for _ in open(path)) if path.endswith(".txt") else len(list(Path(path).glob("*.jpg")))
        out[g] = {"images": n, **scores(m)}
        print(f"{g}: {n} images  P {out[g]['precision']:.4f}  R {out[g]['recall']:.4f}  "
              f"mAP50 {out[g]['map50']:.4f}  mAP50-95 {out[g]['map50_95']:.4f}", flush=True)
    (OUT / "layout_metrics.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
