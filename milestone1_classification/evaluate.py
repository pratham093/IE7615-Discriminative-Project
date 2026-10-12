"""Stage 4: test metrics, confusion matrices and training curves.

python evaluate.py [--model all|custom_cnn|resnet50|efficientnet_b0|yolo11s_cls]
"""

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import PurePosixPath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm
from matplotlib.ticker import PercentFormatter
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader

import config
from datasets import CachedImages, eval_transform, load_classes
from models import build_model
from train import pick_device

MODELS = ["efficientnet_b0", "resnet50", "custom_cnn", config.YOLO_RUN]
DISPLAY = {"efficientnet_b0": "EfficientNet-B0", "resnet50": "ResNet-50", "custom_cnn": "Custom CNN",
           config.YOLO_RUN: "YOLO11s-cls"}
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID = "#000000", "#52514e", "#e4e3df"
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281", "#0d366b"])
plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK, "axes.edgecolor": GRID,
                     "xtick.color": INK, "ytick.color": INK, "axes.grid": True, "grid.color": GRID,
                     "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False, "figure.facecolor": "white"})


def predict_torch(name, device):
    ckpt = torch.load(config.CHECKPOINT_DIR / f"{name}_best.pt", map_location="cpu")
    if ckpt["classes"] != [c for c, _ in load_classes()]:
        raise SystemExit(f"{name}: checkpoint classes don't match classes.csv")
    model = build_model(name, len(ckpt["classes"]), pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    test = CachedImages("test", eval_transform())
    probs = []
    with torch.no_grad():
        for x, _ in DataLoader(test, batch_size=128):
            probs.append(torch.softmax(model(x.to(device)), dim=1).cpu())
    return torch.cat(probs).numpy(), test.y.numpy(), [str(p) for p in test.paths]


def predict_yolo(device):
    from ultralytics import YOLO
    import make_yolo_dirs

    if not (config.YOLO_DATA_DIR / "test").is_dir():
        make_yolo_dirs.main()
    classes = [c for c, _ in load_classes()]
    data = np.load(config.CACHE_DIR / "test.npz")
    files = [str(config.YOLO_DATA_DIR / "test" / classes[y] / (PurePosixPath(str(p)).stem + ".png"))
             for y, p in zip(data["y"], data["paths"])]
    model = YOLO(str(config.CHECKPOINT_DIR / f"{config.YOLO_RUN}_best.pt"))
    column = [classes.index(model.names[i]) for i in range(len(model.names))]  # YOLO index -> our label
    probs = np.zeros((len(files), len(classes)), dtype=np.float32)
    for start in range(0, len(files), 64):
        results = model.predict(files[start:start + 64], imgsz=config.IMG_SIZE, verbose=False,
                                device=0 if device.type == "cuda" else device.type)
        for k, r in enumerate(results):
            probs[start + k, column] = r.probs.data.cpu().numpy()
    return probs, data["y"], [str(p) for p in data["paths"]]


def wilson_ci(correct, n, z=1.96):
    # 95% Wilson score interval for an accuracy measured on n test images
    p = correct / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return [round(float(centre - half), 4), round(float(centre + half), 4)]


def plot_confusion(cm, ids, title, path):
    norm_cm = cm / cm.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(18, 16))
    im = ax.imshow(norm_cm, cmap=BLUES, norm=PowerNorm(0.5, vmin=0, vmax=1))
    ax.grid(False)
    ax.set_xticks(range(len(ids)), ids, rotation=90, fontsize=5.5)
    ax.set_yticks(range(len(ids)), ids, fontsize=5.5)
    ax.set_xlabel("Predicted Object ID")
    ax.set_ylabel("True Object ID")
    ax.set_title(title, fontsize=14, loc="left")
    bar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.01)
    bar.set_label("Share of the true class (square-root colour scale, so single mistakes stay visible)")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_curves(name, path):
    with open(config.METRIC_DIR / name / "train_log.csv", newline="") as f:
        log = list(csv.DictReader(f))
    epochs = [int(r["epoch"]) for r in log]
    col = lambda key: [float(r[key]) for r in log]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.2))

    ax_loss.plot(epochs, col("train_loss"), color=BLUE, lw=2, label="Train")
    ax_loss.plot(epochs, col("val_loss"), color=ORANGE, lw=2, label="Validation")
    ax_loss.set_title("Loss", loc="left")
    if "train_acc" in log[0]:
        ax_acc.plot(epochs, col("train_acc"), color=BLUE, lw=2, label="Train")
        ax_acc.plot(epochs, col("val_acc"), color=ORANGE, lw=2, label="Validation")
    else:  # YOLO logs val top-1 and top-5 only
        ax_acc.plot(epochs, col("val_acc"), color=ORANGE, lw=2, label="Validation top-1")
        ax_acc.plot(epochs, col("val_top5"), color=AQUA, lw=2, label="Validation top-5")
    best = max(range(len(log)), key=lambda i: float(log[i]["val_acc"]))
    ax_acc.plot(epochs[best], float(log[best]["val_acc"]), "o", ms=8, color=ORANGE, mec="white", mew=2)
    ax_acc.annotate(f"best {float(log[best]['val_acc']):.1%} (epoch {epochs[best]})",
                    (epochs[best], float(log[best]["val_acc"])), xytext=(0, -18), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax_acc.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_acc.set_title("Accuracy", loc="left")

    if "phase" in log[0]:
        switch = next((e for e, r in zip(epochs, log) if r["phase"] == "finetune"), None)
        for ax in (ax_loss, ax_acc):
            if switch:
                ax.axvline(switch - 0.5, color=MUTED, lw=1, ls="--")
        if switch:
            ax_loss.text(switch - 0.3, ax_loss.get_ylim()[1], "fine-tuning starts", color=INK, fontsize=8, va="top")
    for ax in (ax_loss, ax_acc):
        ax.set_xlabel("Epoch")
        ax.legend(loc="best")
    fig.suptitle(f"{DISPLAY[name]} - training curves", x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def evaluate(name, device):
    classes = load_classes()
    ids = [c for c, _ in classes]
    name_of = dict(classes)
    t0 = time.time()
    probs, y, paths = predict_yolo(device) if name == config.YOLO_RUN else predict_torch(name, device)
    seconds = time.time() - t0
    pred = probs.argmax(1)
    top5 = np.argsort(-probs, axis=1)[:, :5]
    labels = np.arange(len(ids))
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)

    top1_correct = int((pred == y).sum())
    top5_correct = int(sum(t in row for t, row in zip(y, top5)))
    metrics = {"model": name, "test_images": int(len(y)), "top1_acc": top1_correct / len(y),
               "top1_ci95": wilson_ci(top1_correct, len(y)),
               "top5_acc": top5_correct / len(y), "top5_ci95": wilson_ci(top5_correct, len(y)),
               "macro_precision": float(precision.mean()), "macro_recall": float(recall.mean()),
               "macro_f1": float(f1.mean()), "misclassified": int((pred != y).sum()),
               "eval_device": str(device), "eval_seconds": round(seconds, 1),
               "checkpoint": f"outputs/checkpoints/{name}_best.pt"}
    metric_dir = config.METRIC_DIR / name
    figure_dir = config.FIGURE_DIR / name
    metric_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / "test_metrics.json").write_text(json.dumps(metrics, indent=2))

    mistakes = {i: Counter(pred[(y == i) & (pred != i)]) for i in labels}
    per_class = [(ids[i], name_of[ids[i]], int(support[i]), int(cm[i, i]), round(float(recall[i]), 4),
                  round(float(precision[i]), 4), round(float(f1[i]), 4)) for i in labels]
    write_csv(metric_dir / "per_class.csv",
              ["object_id", "object_name", "test_images", "correct", "accuracy", "precision", "f1"], per_class)
    write_csv(metric_dir / "test_predictions.csv", ["path", "true_id", "pred_id", "confidence", "correct"],
              [(p, ids[t], ids[q], round(float(probs[k, q]), 4), bool(t == q))
               for k, (p, t, q) in enumerate(zip(paths, y, pred))])

    off = [(cm[i, j], i, j) for i in labels for j in labels if i != j and cm[i, j] > 0]
    off.sort(key=lambda t: (-t[0], t[1], t[2]))
    write_csv(metric_dir / "confused_pairs.csv",
              ["rank", "true_id", "true_name", "predicted_id", "predicted_name", "count", "share_of_true_class"],
              [(r, ids[i], name_of[ids[i]], ids[j], name_of[ids[j]], int(n), round(n / support[i], 4))
               for r, (n, i, j) in enumerate(off[:10], 1)])

    worst = sorted(labels, key=lambda i: (recall[i], f1[i]))[:5]
    rows = []
    for r, i in enumerate(worst, 1):
        top_mistake = mistakes[i].most_common(1)
        mistake = f"{ids[top_mistake[0][0]]} ({name_of[ids[top_mistake[0][0]]]}) x{top_mistake[0][1]}" if top_mistake else ""
        rows.append((r, ids[i], name_of[ids[i]], int(support[i]), round(float(recall[i]), 4),
                     round(float(precision[i]), 4), round(float(f1[i]), 4), mistake))
    write_csv(metric_dir / "worst_classes.csv",
              ["rank", "object_id", "object_name", "test_images", "accuracy", "precision", "f1", "most_common_mistake"],
              rows)

    plot_confusion(cm, ids, f"{DISPLAY[name]} - normalized confusion matrix, test set "
                            f"(n = {len(y)}, top-1 {metrics['top1_acc']:.1%})",
                   figure_dir / "confusion_matrix.png")
    plot_curves(name, figure_dir / "training_curves.png")

    print(f"{DISPLAY[name]:<16} top-1 {metrics['top1_acc']:.4f} "
          f"(95% CI {metrics['top1_ci95'][0]:.4f}-{metrics['top1_ci95'][1]:.4f}) | top-5 {metrics['top5_acc']:.4f} | "
          f"macro F1 {metrics['macro_f1']:.4f} | {metrics['misclassified']} wrong of {len(y)} | {seconds:.0f}s")
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="all", choices=["all"] + MODELS)
    args = parser.parse_args()
    device = pick_device()
    for name in MODELS if args.model == "all" else [args.model]:
        if not (config.CHECKPOINT_DIR / f"{name}_best.pt").is_file():
            print(f"{name}: no checkpoint, skipped")
            continue
        evaluate(name, device)


if __name__ == "__main__":
    main()
