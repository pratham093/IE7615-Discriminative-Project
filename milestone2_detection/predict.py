"""Runs the detector on images and prints each object's ID, confidence and box.

python predict.py IMAGE_OR_FOLDER [...] [--weights results/yolov8s_best.pt] [--conf 0.25] [--out predictions]
"""

import argparse
import json
from pathlib import Path

import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect(paths):
    images = []
    for p in map(Path, paths):
        images += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    return images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+", help="image files or folders of images")
    parser.add_argument("--weights", default=str(ROOT / "results" / "yolov8s_best.pt"))
    parser.add_argument("--conf", type=float, default=0.25, help="minimum confidence to report a detection")
    parser.add_argument("--out", default=str(ROOT / "predictions"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    dev = 0 if torch.cuda.is_available() else "cpu"
    imgsz = model.ckpt.get("train_args", {}).get("imgsz", 640)  # predict at the size the model was trained at

    report = {}
    for path in collect(args.images):
        r = model.predict(str(path), imgsz=imgsz, conf=args.conf, device=dev, verbose=False)[0]
        dets = []
        for (x1, y1, x2, y2), c, k in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist(), r.boxes.cls.tolist()):
            dets.append({"object_id": r.names[int(k)], "confidence": round(c, 4),
                         "box_xyxy": [round(x1), round(y1), round(x2), round(y2)],
                         "center": [round((x1 + x2) / 2), round((y1 + y2) / 2)]})
        dets.sort(key=lambda d: (d["box_xyxy"][1], d["box_xyxy"][0]))  # top-to-bottom, left-to-right
        report[path.name] = dets

        print(f"\n{path.name}: {len(dets)} object(s)")
        for d in dets:
            print(f"  {d['object_id']}  conf {d['confidence']:.2f}  box (x1, y1, x2, y2) = {tuple(d['box_xyxy'])}"
                  f"  center {tuple(d['center'])}")
        r.save(filename=str(out / f"{path.stem}.jpg"))

    (out / "predictions.json").write_text(json.dumps(report, indent=2))
    print(f"\nsaved annotated images and predictions.json to {out}")


if __name__ == "__main__":
    main()
