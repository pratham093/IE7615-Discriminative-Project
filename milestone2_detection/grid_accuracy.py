"""Demo-style check on the test grids: objects found with the right ID, extra boxes, grids fully right.

python grid_accuracy.py [--weights results/yolov8s_best.pt] [--conf 0.25]
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from ultralytics import YOLO

import config
from train_detect import IMG_SIZE, device

OUT = config.OUTPUT_DIR / "m2_results"


def iou(a, b):
    ix = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    iy = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = ix * iy
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def ground_truth(image):
    label = image.parent.parent / "labels" / f"{image.stem}.txt"
    rows = np.array([list(map(float, r.split())) for r in label.read_text().split("\n") if r.strip()])
    return rows[:, 0].astype(int), rows[:, 1:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(config.PROJECT_DIR / "results" / "yolov8s_best.pt"))
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()

    dev, _ = device()
    model = YOLO(args.weights)
    names = model.names
    report = {"conf": args.conf, "iou": 0.5}
    for layout in ("2x2", "3x3"):
        images = [Path(p) for p in (OUT / "layout_lists" / f"{layout}.txt").read_text().splitlines() if p.strip()]
        n_obj = found = wrong = exact = 0
        missed, confused, miss_kind, extra_kind = Counter(), Counter(), Counter(), Counter()
        for k in range(0, len(images), 16):
            for img, r in zip(images[k:k + 16], model.predict([str(p) for p in images[k:k + 16]], imgsz=IMG_SIZE,
                                                             conf=args.conf, device=dev, verbose=False)):
                gt_cls, gt = ground_truth(img)
                w, h = r.orig_shape[1], r.orig_shape[0]
                gt_xyxy = np.stack([(gt[:, 0] - gt[:, 2] / 2) * w, (gt[:, 1] - gt[:, 3] / 2) * h,
                                    (gt[:, 0] + gt[:, 2] / 2) * w, (gt[:, 1] + gt[:, 3] / 2) * h], 1)
                pr = r.boxes.xyxy.cpu().numpy()
                pr_cls = r.boxes.cls.cpu().numpy().astype(int)
                ov = iou(pr, gt_xyxy) if len(pr) else np.zeros((0, len(gt)))
                used_p, used_g = set(), set()
                for i, j in sorted(((i, j) for i in range(len(pr)) for j in range(len(gt))
                                    if ov[i, j] >= 0.5 and pr_cls[i] == gt_cls[j]), key=lambda t: -ov[t]):
                    if i not in used_p and j not in used_g:
                        used_p.add(i)
                        used_g.add(j)
                for j in set(range(len(gt))) - used_g:
                    missed[names[gt_cls[j]]] += 1
                    near = [i for i in range(len(pr)) if i not in used_p and ov[i, j] >= 0.5]
                    if near:
                        confused[f"{names[gt_cls[j]]} -> {names[pr_cls[near[0]]]}"] += 1
                        miss_kind["wrong ID"] += 1
                    elif any(i not in used_p and pr_cls[i] == gt_cls[j] and ov[i, j] >= 0.1 for i in range(len(pr))):
                        miss_kind["right ID, box off (IoU < 0.5)"] += 1
                    else:
                        miss_kind["not detected"] += 1
                for i in set(range(len(pr))) - used_p:
                    if any(j in used_g and gt_cls[j] == pr_cls[i] and ov[i, j] >= 0.3 for j in range(len(gt))):
                        extra_kind["second box on a found object"] += 1
                    elif len(gt) and ov[i].max() >= 0.1:
                        extra_kind["on a labelled object (wrong ID or loose box)"] += 1
                    else:
                        extra_kind["on no labelled object"] += 1
                n_obj += len(gt)
                found += len(used_g)
                wrong += len(pr) - len(used_p)
                exact += len(used_g) == len(gt) and len(used_p) == len(pr)
        report[layout] = {"grids": len(images), "objects": n_obj, "found_with_right_id": found,
                          "found_pct": round(100 * found / n_obj, 2), "wrong_detections": wrong,
                          "grids_exactly_right": exact, "grids_exactly_right_pct": round(100 * exact / len(images), 2),
                          "misses": dict(miss_kind), "wrong_detection_kinds": dict(extra_kind),
                          "most_missed": missed.most_common(10), "most_confused": confused.most_common(10)}
        print(layout, {k: v for k, v in report[layout].items() if not k.startswith(("most", "misses", "wrong_detection_kinds"))}, flush=True)
    (OUT / "grid_accuracy.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
