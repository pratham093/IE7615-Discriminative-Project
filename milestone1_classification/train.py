"""Stage 2: train one model on the cached dataset.

python train.py --model {custom_cnn,resnet50,efficientnet_b0} [--epochs N] [--batch_size 64] [--lr LR] [--smoke_test]
"""

import argparse
import csv
import json
import os
import platform
import random
import subprocess
import time

import numpy as np
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from datasets import CachedImages, eval_transform, load_classes, train_transform
from models import build_model, finetune_modules, head_modules


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def hardware_info(device):
    info = {"device": device.type, "cpu_count": os.cpu_count(),
            "ram_gb": round(psutil.virtual_memory().total / 1e9, 1), "os": platform.platform(),
            "torch": torch.__version__}
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(0)
        info.update(gpu=props.name, gpu_memory_gb=round(props.total_memory / 1e9, 1), cuda=torch.version.cuda)
    elif device.type == "mps" and platform.system() == "Darwin":
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
        info["gpu"] = f"{chip.strip()} (MPS)"
    else:
        info["gpu"] = "none (CPU)"
    return info


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def set_trainable(model, modules):
    for p in model.parameters():
        p.requires_grad = False
    for m in modules:
        for p in m.parameters():
            p.requires_grad = True


def train_mode(model):
    model.train()
    for m in model.modules():  # frozen BatchNorm layers keep their ImageNet statistics
        if isinstance(m, nn.modules.batchnorm._BatchNorm) and not m.weight.requires_grad:
            m.eval()


def run_epoch(model, loader, criterion, device, use_amp, optimizer=None, scaler=None):
    training = optimizer is not None
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if training:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            seen += len(y)
    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=config.MODEL_NAMES)
    parser.add_argument("--epochs", type=int, help="total epochs (default from config.EPOCHS)")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, help="custom_cnn: training lr; pretrained: fine-tune lr (config.LR)")
    parser.add_argument("--workers", type=int, help="DataLoader workers (default min(cpu_count, 8) on CUDA, else 0)")
    parser.add_argument("--smoke_test", action="store_true")
    args = parser.parse_args()

    seed_everything(config.SEED)
    device = pick_device()
    use_amp = device.type == "cuda"
    epochs = config.SMOKE_EPOCHS if args.smoke_test else (args.epochs or config.EPOCHS[args.model])
    lr = args.lr or config.LR[args.model]
    workers = args.workers if args.workers is not None else (min(os.cpu_count() or 1, 8) if device.type == "cuda" else 0)
    hardware = hardware_info(device)
    n_classes = config.SMOKE_CLASSES if args.smoke_test else None
    run = args.model + ("_smoke" if args.smoke_test else "")

    classes = load_classes()[:n_classes]
    train_set = CachedImages("train", train_transform(), n_classes)
    val_set = CachedImages("val", eval_transform(), n_classes)
    loader_args = dict(batch_size=args.batch_size, num_workers=workers, pin_memory=device.type == "cuda",
                       worker_init_fn=seed_worker, persistent_workers=workers > 0)
    train_loader = DataLoader(train_set, shuffle=True, generator=torch.Generator().manual_seed(config.SEED),
                              **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)

    model = build_model(args.model, len(classes)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    criterion = nn.CrossEntropyLoss(label_smoothing=config.LABEL_SMOOTHING)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    if args.model == "custom_cnn":
        phases = [("all", [model], lr, epochs)]
    else:
        head_epochs = min(config.HEAD_EPOCHS, epochs - 1)
        phases = [("head", head_modules(model, args.model), config.HEAD_LR, head_epochs),
                  ("finetune", finetune_modules(model, args.model), lr, epochs - head_epochs)]

    print(f"[{run}] hardware: {hardware}", flush=True)
    print(f"[{run}] cache={config.CACHE_DIR} outputs={config.OUTPUT_DIR} workers={workers}", flush=True)
    print(f"[{run}] device={device} amp={use_amp} classes={len(classes)} train={len(train_set)} "
          f"val={len(val_set)} params={n_params:,} epochs={epochs}", flush=True)

    metric_dir = config.METRIC_DIR / run
    metric_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = config.CHECKPOINT_DIR / f"{run}_best.pt"
    log_path = metric_dir / "train_log.csv"
    log = []
    best_acc, best_epoch, best_loss = -1.0, 0, float("inf")
    epoch, stopped_early, trainable = 0, False, n_params
    start = time.time()

    for phase, modules, phase_lr, phase_epochs in phases:
        if phase_epochs <= 0:
            continue
        set_trainable(model, modules)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=phase_lr, weight_decay=config.WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=phase_epochs)
        bad_epochs = 0
        for _ in range(phase_epochs):
            epoch += 1
            t0 = time.time()
            current_lr = optimizer.param_groups[0]["lr"]
            train_mode(model)
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, use_amp, optimizer, scaler)
            model.eval()
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device, use_amp)
            scheduler.step()
            seconds = time.time() - t0

            improved = val_acc > best_acc
            if improved:
                best_acc, best_epoch = val_acc, epoch
                torch.save({"model": args.model, "state_dict": model.state_dict(),
                            "classes": [c for c, _ in classes], "class_names": [n for _, n in classes],
                            "epoch": epoch, "val_acc": val_acc, "img_size": config.IMG_SIZE,
                            "mean": config.IMAGENET_MEAN, "std": config.IMAGENET_STD}, ckpt_path)
            if val_loss < best_loss:
                best_loss, bad_epochs = val_loss, 0
            else:
                bad_epochs += 1

            log.append({"epoch": epoch, "phase": phase, "lr": current_lr, "train_loss": round(train_loss, 5),
                        "train_acc": round(train_acc, 5), "val_loss": round(val_loss, 5),
                        "val_acc": round(val_acc, 5), "seconds": round(seconds, 1), "hardware": hardware["gpu"]})
            with open(log_path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=log[0].keys())
                w.writeheader()
                w.writerows(log)
            print(f"[{run}] epoch {epoch}/{epochs} {phase:<8} lr {current_lr:.1e} | "
                  f"train {train_loss:.4f} / {train_acc:.4f} | val {val_loss:.4f} / {val_acc:.4f} | "
                  f"{seconds:.1f}s{'  * saved' if improved else ''}", flush=True)
            if bad_epochs >= config.PATIENCE:
                stopped_early = True
                print(f"[{run}] early stop: val loss hasn't improved for {config.PATIENCE} epochs", flush=True)
                break

    train_time = time.time() - start
    summary = {"model": args.model, "run": run, "smoke_test": args.smoke_test, "hardware": hardware,
               "dataloader_workers": workers, "device": str(device),
               "amp": use_amp, "num_classes": len(classes), "train_images": len(train_set),
               "val_images": len(val_set), "params_total": n_params, "params_trainable_last_phase": trainable,
               "batch_size": args.batch_size, "lr": lr, "head_lr": None if args.model == "custom_cnn" else config.HEAD_LR,
               "epochs_planned": epochs, "epochs_run": epoch, "early_stopped": stopped_early,
               "best_epoch": best_epoch, "best_val_acc": best_acc, "train_time_s": round(train_time, 1),
               "checkpoint": str(ckpt_path.relative_to(config.PROJECT_DIR))}
    (metric_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[{run}] done in {train_time / 60:.1f} min, best val acc {best_acc:.4f} at epoch {best_epoch}")
    print(f"[{run}] checkpoint {ckpt_path}\n[{run}] log {log_path}")


if __name__ == "__main__":
    main()
