"""Fine-tunes YOLOv8s on data/ and scores it on val, test_single and test_multi.

python train_detect.py [--only yolov8s] [--epochs 80] [--batch 16] [--workers N] [--smoke_test]
"""

import argparse
import json
import os
import platform
import shutil
import time
from pathlib import Path

import torch
import ultralytics
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = {"yolov8s": "yolov8s.pt", "yolo26s": "yolo26s.pt"}
IMG_SIZE = 672  # 3x3 grids are 672 px, so each 224 px cell keeps full resolution
SEED = 42


def write_yaml(test_split):
    names = json.loads((DATA / "classes.json").read_text())
    lines = [f"path: {DATA.as_posix()}", "train: train/images", "val: val/images", f"test: {test_split}/images",
             f"nc: {len(names)}", "names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    path = ROOT / f"data_{test_split}.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def device():
    if torch.cuda.is_available():
        return 0, torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "mps", "Apple MPS"
    return "cpu", platform.processor() or "CPU"


def scores(metrics):
    box = metrics.box
    per_class = {r["Class"]: {"images": int(r["Images"]), "instances": int(r["Instances"]), "precision": round(r["Box-P"], 4),
                              "recall": round(r["Box-R"], 4), "map50": round(r["mAP50"], 4), "map50_95": round(r["mAP50-95"], 4)}
                 for r in metrics.summary()}
    return {"precision": round(box.mp, 4), "recall": round(box.mr, 4), "map50": round(box.map50, 4),
            "map50_95": round(box.map, 4), "speed_ms_per_image": metrics.speed, "per_class": per_class}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(MODELS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, help="dataloader workers (default min(cpu_count, 8) on CUDA, else 2)")
    parser.add_argument("--smoke_test", action="store_true", help="1 epoch on 5%% of the training images")
    args = parser.parse_args()

    dev, gpu = device()
    workers = args.workers if args.workers is not None else (min(os.cpu_count() or 1, 8) if dev == 0 else 2)
    yaml_single, yaml_multi = write_yaml("test_single"), write_yaml("test_multi")
    RESULTS.mkdir(exist_ok=True)
    print(f"device: {gpu}, ultralytics {ultralytics.__version__}, torch {torch.__version__}", flush=True)

    for name, weights in MODELS.items():
        if args.only and name != args.only:
            continue
        run = name + ("_smoke" if args.smoke_test else "")
        epochs = 1 if args.smoke_test else args.epochs
        model = YOLO(weights)  # downloaded on first use
        start = time.time()
        model.train(data=str(yaml_multi), imgsz=IMG_SIZE, epochs=epochs, batch=args.batch, patience=20, seed=SEED,
                    device=dev, workers=workers, fraction=0.05 if args.smoke_test else 1.0,
                    project=str(ROOT / "runs"), name=run, exist_ok=True)
        train_time = time.time() - start

        best = ROOT / "runs" / run / "weights" / "best.pt"
        shutil.copy(best, RESULTS / f"{run}_best.pt")
        model = YOLO(str(best))
        out = {"model": name, "run": run, "pretrained_weights": weights, "device": gpu, "ultralytics": ultralytics.__version__,
               "torch": torch.__version__, "imgsz": IMG_SIZE, "batch": args.batch, "epochs_planned": epochs,
               "train_time_s": round(train_time, 1), "params": sum(p.numel() for p in model.model.parameters())}
        for split, yaml in (("val", yaml_multi), ("test_single", yaml_single), ("test_multi", yaml_multi)):
            m = model.val(data=str(yaml), split="val" if split == "val" else "test", imgsz=IMG_SIZE, batch=args.batch,
                          device=dev, workers=workers, project=str(ROOT / "runs"), name=f"{run}_{split}", exist_ok=True)
            out[split] = scores(m)
            print(f"[{run}] {split}: mAP50 {out[split]['map50']:.4f}  mAP50-95 {out[split]['map50_95']:.4f}", flush=True)
        (RESULTS / f"{run}.json").write_text(json.dumps(out, indent=2))

        samples = sorted((DATA / "test_multi" / "images").glob("*.jpg"))[::50][:16]
        model.predict([str(p) for p in samples], imgsz=IMG_SIZE, conf=0.25, device=dev, save=True,
                      project=str(RESULTS), name=f"{run}_samples", exist_ok=True, verbose=False)
        print(f"[{run}] done in {train_time / 60:.1f} min -> {RESULTS / f'{run}.json'}", flush=True)


if __name__ == "__main__":
    main()
