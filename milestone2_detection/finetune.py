"""10 more epochs (mosaic off, low lr) on the run that stopped at epoch 29, then keeps whichever model is better on val.

python finetune.py [--epochs 10] [--batch 16] [--workers 2]
"""

import argparse
import json
import shutil
import time

import torch
import ultralytics
from ultralytics import YOLO

from train_detect import DATA, IMG_SIZE, RESULTS, ROOT, SEED, device, scores, write_yaml


def evaluate(weights, yamls, args, dev, workers, tag):
    model = YOLO(str(weights))
    out = {}
    for split, yaml in (("val", yamls["multi"]), ("test_single", yamls["single"]), ("test_multi", yamls["multi"])):
        m = model.val(data=str(yaml), split="val" if split == "val" else "test", imgsz=IMG_SIZE, batch=args.batch,
                      device=dev, workers=workers, project=str(ROOT / "runs"), name=f"{tag}_{split}", exist_ok=True)
        out[split] = scores(m)
        print(f"[{tag}] {split}: P {out[split]['precision']:.4f}  R {out[split]['recall']:.4f}  "
              f"mAP50 {out[split]['map50']:.4f}  mAP50-95 {out[split]['map50_95']:.4f}", flush=True)
    return out


def fitness(s):
    return 0.1 * s["map50"] + 0.9 * s["map50_95"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    dev, gpu = device()
    yamls = {"single": write_yaml("test_single"), "multi": write_yaml("test_multi")}
    start_weights = RESULTS / "yolov8s_e29_best.pt"
    print(f"device: {gpu}, ultralytics {ultralytics.__version__}, torch {torch.__version__}", flush=True)

    start = time.time()
    YOLO(str(start_weights)).train(
        data=str(yamls["multi"]), imgsz=IMG_SIZE, epochs=args.epochs, batch=args.batch, seed=SEED, device=dev,
        workers=args.workers, optimizer="SGD", lr0=0.002, lrf=0.1, warmup_epochs=0, close_mosaic=args.epochs,
        patience=args.epochs, project=str(ROOT / "runs"), name="yolov8s_ft", exist_ok=True)
    train_time = time.time() - start
    ft_weights = ROOT / "runs" / "yolov8s_ft" / "weights" / "best.pt"

    results = {"e29": evaluate(start_weights, yamls, args, dev, workers=args.workers, tag="yolov8s_e29"),
               "finetuned": evaluate(ft_weights, yamls, args, dev, workers=args.workers, tag="yolov8s_ft")}
    kept = max(results, key=lambda k: fitness(results[k]["val"]))
    final = ft_weights if kept == "finetuned" else start_weights
    shutil.copy(final, RESULTS / "yolov8s_best.pt")

    model = YOLO(str(RESULTS / "yolov8s_best.pt"))
    out = {"model": "yolov8s", "kept": kept, "device": gpu, "ultralytics": ultralytics.__version__,
           "torch": torch.__version__, "imgsz": IMG_SIZE, "batch": args.batch,
           "schedule": "29 epochs of an 80-epoch run (optimizer=auto picked MuSGD lr0 0.01, mosaic on), then "
                       f"{args.epochs} epochs SGD lr0 0.002 -> 0.0002, mosaic off",
           "finetune_time_s": round(train_time, 1), "params": sum(p.numel() for p in model.model.parameters()),
           **{k: v for k, v in results[kept].items()},
           "other_model": {"name": "e29" if kept == "finetuned" else "finetuned",
                           **results["e29" if kept == "finetuned" else "finetuned"]}}
    (RESULTS / "yolov8s.json").write_text(json.dumps(out, indent=2))

    samples = sorted((DATA / "test_multi" / "images").glob("*.jpg"))[::80][:16]
    model.predict([str(p) for p in samples], imgsz=IMG_SIZE, conf=0.25, device=dev, save=True,
                  project=str(RESULTS), name="yolov8s_samples", exist_ok=True, verbose=False)
    print(f"[yolov8s] kept {kept}, fine-tune took {train_time / 60:.1f} min -> {RESULTS / 'yolov8s.json'}", flush=True)


if __name__ == "__main__":
    main()
