"""Stage 5: comparison table and charts for all models."""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import config
from datasets import load_classes
from evaluate import BLUE, DISPLAY, INK, MODELS, ORANGE


def plot_hard_images(path, models=("resnet50", "yolo11s_cls", "efficientnet_b0"), shown=10):
    names = dict(load_classes())
    preds = {}
    for m in models:
        with open(config.METRIC_DIR / m / "test_predictions.csv", newline="") as f:
            preds[m] = {r["path"]: r for r in csv.DictReader(f)}
    hard = [p for p in preds[models[0]] if all(preds[m][p]["correct"] == "False" for m in models)]
    data = np.load(config.CACHE_DIR / "test.npz")
    index = {str(p): i for i, p in enumerate(data["paths"])}
    fig, axes = plt.subplots(2, 5, figsize=(13, 6.4))
    for ax in axes.flat:
        ax.axis("off")
    for ax, p in zip(axes.flat, hard[:shown]):
        r = preds[models[0]][p]
        ax.imshow(data["x"][index[p]])
        ax.set_title(f"True: {names[r['true_id']]}\n{DISPLAY[models[0]]}: {names[r['pred_id']]}", fontsize=9, loc="left")
    fig.suptitle(f"The {len(hard)} test images that all three pretrained models misclassify"
                 + (f" (first {shown} shown)" if len(hard) > shown else ""), x=0.01, ha="left", fontsize=13)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return len(hard)


def main():
    rows = []
    for name in MODELS:
        train_path = config.METRIC_DIR / name / "train_summary.json"
        test_path = config.METRIC_DIR / name / "test_metrics.json"
        if not (train_path.is_file() and test_path.is_file()):
            print(f"{name}: missing train_summary.json or test_metrics.json, skipped")
            continue
        s, t = json.loads(train_path.read_text()), json.loads(test_path.read_text())
        rows.append({"model": DISPLAY[name], "approach": "from scratch" if name == "custom_cnn" else "transfer learning",
                     "params_m": round(s["params_total"] / 1e6, 2), "train_min": round(s["train_time_s"] / 60, 1),
                     "trained_on": s["hardware"]["gpu"], "best_val_acc": round(s["best_val_acc"], 4),
                     "test_top1": round(t["top1_acc"], 4), "test_top1_ci95_low": t["top1_ci95"][0],
                     "test_top1_ci95_high": t["top1_ci95"][1], "test_top5": round(t["top5_acc"], 4),
                     "test_macro_f1": round(t["macro_f1"], 4), "test_errors": t["misclassified"]})
    rows.sort(key=lambda r: (-r["test_macro_f1"], -r["test_top1"]))
    best = rows[0]

    with open(config.METRIC_DIR / "comparison.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    n_test = json.loads((config.METRIC_DIR / MODELS[0] / "test_metrics.json").read_text())["test_images"]
    lines = [f"Test split: {n_test} images, 73 classes. Sorted by test macro F1. "
             "95% CI = Wilson score interval for test top-1 accuracy.", "",
             "| Model | Approach | Params (M) | Train time (min) | Trained on | Best val acc | Test top-1 | Test top-1 95% CI | Test top-5 | Test macro F1 | Test errors |",
             "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        mark = "**" if r is best else ""
        lines.append(f"| {mark}{r['model']}{mark} | {r['approach']} | {r['params_m']} | {r['train_min']} | {r['trained_on']} | "
                     f"{r['best_val_acc']:.2%} | {mark}{r['test_top1']:.2%}{mark} | "
                     f"{r['test_top1_ci95_low']:.1%} to {r['test_top1_ci95_high']:.1%} | {r['test_top5']:.2%} | "
                     f"{mark}{r['test_macro_f1']:.4f}{mark} | {r['test_errors']} |")
    lines += ["", f"Best model by test macro F1: **{best['model']}** "
                  f"(macro F1 {best['test_macro_f1']:.4f}, top-1 {best['test_top1']:.2%})."]
    (config.METRIC_DIR / "comparison.md").write_text("\n".join(lines) + "\n")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(rows))
    width = 0.38
    for offset, key, label, color in ((-0.2, "test_top1", "Test top-1 accuracy", BLUE),
                                      (0.2, "test_macro_f1", "Test macro F1", ORANGE)):
        values = [r[key] for r in rows]
        bars = ax.bar(x + offset, values, width, color=color, label=label, edgecolor="white", linewidth=2)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}" if key == "test_top1" else f"{v:.3f}",
                    ha="center", va="bottom", fontsize=9, color=INK)
    ax.set_xticks(x, [r["model"] for r in rows])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score (0 to 1)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", ncols=2, bbox_to_anchor=(1, 1.12))
    ax.set_title(f"Test-set comparison ({n_test} images, 73 classes)", loc="left", pad=28)
    fig.tight_layout()
    fig.savefig(config.FIGURE_DIR / "comparison.png", dpi=200)
    plt.close(fig)

    n_hard = plot_hard_images(config.FIGURE_DIR / "hard_test_images.png")
    print("\n".join(lines))
    print(f"\n{n_hard} test images are misclassified by all three pretrained models")
    print(f"\nSaved {config.METRIC_DIR / 'comparison.csv'}, comparison.md and {config.FIGURE_DIR / 'comparison.png'}")


if __name__ == "__main__":
    main()
