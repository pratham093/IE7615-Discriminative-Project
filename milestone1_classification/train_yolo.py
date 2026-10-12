"""Stage 3: fine-tune yolo11s-cls on the same split.

python train_yolo.py [--epochs 30] [--batch 64] [--workers N] [--smoke_test]
"""

import argparse
import csv
import json
import os
import shutil
import time

import ultralytics
from ultralytics import YOLO

import config
from train import hardware_info, pick_device


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=config.YOLO_EPOCHS)
    parser.add_argument("--batch", type=int, default=config.YOLO_BATCH)
    parser.add_argument("--workers", type=int, help="dataloader workers (default min(cpu_count, 8) on CUDA, else 2)")
    parser.add_argument("--smoke_test", action="store_true", help="2 epochs on 10%% of the training images")
    args = parser.parse_args()

    if not (config.YOLO_DATA_DIR / "train").is_dir():
        raise SystemExit(f"{config.YOLO_DATA_DIR} not found, run make_yolo_dirs.py first")
    device = pick_device()
    hardware = hardware_info(device)
    run = config.YOLO_RUN + ("_smoke" if args.smoke_test else "")
    epochs = config.SMOKE_EPOCHS if args.smoke_test else args.epochs
    workers = args.workers if args.workers is not None else (min(os.cpu_count() or 1, 8) if device.type == "cuda" else 2)
    count = lambda split: sum(1 for _ in (config.YOLO_DATA_DIR / split).rglob("*.png"))
    print(f"[{run}] hardware: {hardware}", flush=True)

    model = YOLO(str(config.CHECKPOINT_DIR / config.YOLO_WEIGHTS))  # downloaded on first use
    start = time.time()
    model.train(data=str(config.YOLO_DATA_DIR), imgsz=config.IMG_SIZE, epochs=epochs, batch=args.batch,
                patience=config.YOLO_PATIENCE, seed=config.SEED, deterministic=True,
                device=0 if device.type == "cuda" else device.type, workers=workers,
                fraction=0.1 if args.smoke_test else 1.0,
                project=str(config.OUTPUT_DIR / "yolo_runs"), name=run, exist_ok=True)
    train_time = time.time() - start

    save_dir = model.trainer.save_dir
    ckpt_path = config.CHECKPOINT_DIR / f"{run}_best.pt"
    shutil.copy(save_dir / "weights" / "best.pt", ckpt_path)
    best_model = YOLO(str(ckpt_path))
    n_params = sum(p.numel() for p in best_model.model.parameters())

    with open(save_dir / "results.csv", newline="") as f:
        results = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(f)]
    log = [{"epoch": int(float(r["epoch"])), "train_loss": round(float(r["train/loss"]), 5),
            "val_loss": round(float(r["val/loss"]), 5), "val_acc": round(float(r["metrics/accuracy_top1"]), 5),
            "val_top5": round(float(r["metrics/accuracy_top5"]), 5), "lr": float(r.get("lr/pg0", "nan")),
            "hardware": hardware["gpu"]} for r in results]
    best = max(log, key=lambda r: (r["val_acc"] + r["val_top5"]) / 2)

    metric_dir = config.METRIC_DIR / run
    metric_dir.mkdir(parents=True, exist_ok=True)
    with open(metric_dir / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=log[0].keys())
        w.writeheader()
        w.writerows(log)
    summary = {"model": config.YOLO_RUN, "run": run, "smoke_test": args.smoke_test, "hardware": hardware,
               "dataloader_workers": workers, "device": str(device), "num_classes": len(best_model.names),
               "train_images": count("train"), "val_images": count("val"), "params_total": n_params,
               "pretrained_weights": config.YOLO_WEIGHTS, "ultralytics": ultralytics.__version__,
               "batch_size": args.batch, "epochs_planned": epochs, "epochs_run": len(log),
               "early_stopped": len(log) < epochs, "best_epoch": best["epoch"], "best_val_acc": best["val_acc"],
               "best_val_top5": best["val_top5"], "best_selected_by": "(top1 + top5) / 2 on val",
               "augmentation": "Ultralytics default classification augmentation",
               "train_time_s": round(train_time, 1), "checkpoint": str(ckpt_path.relative_to(config.PROJECT_DIR))}
    (metric_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run}] done in {train_time / 60:.1f} min, best val acc {best['val_acc']:.4f} at epoch {best['epoch']}")
    print(f"[{run}] checkpoint {ckpt_path}\n[{run}] log {metric_dir / 'train_log.csv'}")


if __name__ == "__main__":
    main()
