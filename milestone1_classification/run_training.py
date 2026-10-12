"""Trains all four Milestone 1 models one after another.

python run_training.py [--smoke_test] [--workers N] [--only MODEL ...]
"""

import argparse
import json
import subprocess
import sys

import torch

import config

ORDER = ["efficientnet_b0", "resnet50", "custom_cnn", config.YOLO_RUN]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke_test", action="store_true", help="quick 2-epoch check of every model")
    parser.add_argument("--workers", type=int, help="DataLoader workers (train.py default: min(cpu_count, 8))")
    parser.add_argument("--only", nargs="+", choices=ORDER, help="train just these models")
    parser.add_argument("--allow_cpu", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available() and not args.allow_cpu:
        sys.exit("PyTorch can't see a CUDA GPU. Install the CUDA build of torch (see README.md), "
                 "check `nvidia-smi`, or pass --allow_cpu.")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__} | CUDA {torch.version.cuda}")

    models = args.only or ORDER
    status = {}
    for model in models:
        if model == config.YOLO_RUN:
            if not (config.YOLO_DATA_DIR / "train").is_dir():
                subprocess.run([sys.executable, "make_yolo_dirs.py"], cwd=config.PROJECT_DIR, check=True)
            cmd = [sys.executable, "train_yolo.py", "--epochs", str(config.YOLO_EPOCHS), "--batch", str(config.YOLO_BATCH)]
        else:
            cmd = [sys.executable, "train.py", "--model", model, "--epochs", str(config.EPOCHS[model]),
                   "--batch_size", str(config.BATCH_SIZE), "--lr", str(config.LR[model])]
        if args.workers is not None:
            cmd += ["--workers", str(args.workers)]
        if args.smoke_test:
            cmd.append("--smoke_test")
        print("\n" + " ".join(cmd), flush=True)
        status[model] = subprocess.run(cmd, cwd=config.PROJECT_DIR).returncode

    print("\nSummary")
    for model in models:
        run = model + ("_smoke" if args.smoke_test else "")
        path = config.METRIC_DIR / run / "train_summary.json"
        if status[model] != 0 or not path.is_file():
            print(f"  {model:<16} FAILED (exit code {status[model]})")
            continue
        s = json.loads(path.read_text())
        print(f"  {model:<16} best val acc {s['best_val_acc']:.4f} (epoch {s['best_epoch']}/{s['epochs_run']}) "
              f"{s['train_time_s'] / 60:.1f} min on {s['hardware']['gpu']}")
    sys.exit(1 if any(status.values()) else 0)


if __name__ == "__main__":
    main()
