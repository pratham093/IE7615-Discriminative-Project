"""Figures for the Milestone 2 report."""

import csv
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

import config

DATA = config.PROJECT_DIR / "data"
RES = config.OUTPUT_DIR / "m2_results"
FIG = config.OUTPUT_DIR / "m2_figures"
NAMES = json.loads((DATA / "classes.json").read_text())


def with_boxes(split, stem):
    img = Image.open(DATA / split / "images" / f"{stem}.jpg").convert("RGB")
    w, h = img.size
    g = ImageDraw.Draw(img)
    for row in (DATA / split / "labels" / f"{stem}.txt").read_text().strip().splitlines():
        c, xc, yc, bw, bh = map(float, row.split())
        x1, y1 = (xc - bw / 2) * w, (yc - bh / 2) * h
        g.rectangle([x1, y1, x1 + bw * w, y1 + bh * h], outline="#00c000", width=3)
        g.rectangle([x1, y1, x1 + 50, y1 + 13], fill="#00c000")
        g.text((x1 + 3, y1 + 1), NAMES[int(c)], fill="black")
    return img


def row_of(images, height=440, gap=12):
    images = [im.resize((round(im.width * height / im.height), height)) for im in images]
    out = Image.new("RGB", (sum(im.width for im in images) + gap * (len(images) - 1), height), "white")
    x = 0
    for im in images:
        out.paste(im, (x, 0))
        x += im.width + gap
    return out


def curves():
    read = lambda run: [{k.strip(): float(v) for k, v in r.items()} for r in csv.DictReader(open(RES / "runs" / run / "results.csv"))]
    main, ft = read("yolov8s"), read("yolov8s_ft")
    rows = main + ft
    ep = list(range(1, len(rows) + 1))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(ep, [r["metrics/mAP50(B)"] for r in rows], label="mAP50")
    ax[0].plot(ep, [r["metrics/mAP50-95(B)"] for r in rows], label="mAP50-95")
    for k in ("box", "cls", "dfl"):
        ax[1].plot(ep, [r[f"train/{k}_loss"] for r in rows], label=f"train {k}")
        ax[1].plot(ep, [r[f"val/{k}_loss"] for r in rows], "--", label=f"val {k}")
    for a, title in zip(ax, ("Validation mAP", "Loss")):
        a.axvline(len(main) + 0.5, color="gray", lw=1, ls=":")
        a.set_xlabel("epoch")
        a.set_title(title)
        a.grid(alpha=0.3)
        a.legend(fontsize=8, ncol=2 if title == "Loss" else 1)
    ax[0].text(len(main) + 1, 0.62, "fine-tune", color="gray", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig3_curves.png", dpi=200)


def per_class():
    m = json.loads((RES / "layout_metrics.json").read_text())
    rows = sorted(((c, v["map50_95"], v["map50"]) for c, v in m["3x3"]["per_class"].items()), key=lambda r: r[1])
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = range(len(rows))
    ax.bar(x, [r[2] for r in rows], color="#9ecae1", label="mAP50")
    ax.bar(x, [r[1] for r in rows], color="#3182bd", label="mAP50-95")
    ax.set_xticks(list(x), [r[0] for r in rows], rotation=90, fontsize=6)
    ax.set_ylim(0, 1)
    ax.set_ylabel("AP")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG / "fig5_per_class.png", dpi=200)
    return rows


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    verify = Image.open(config.OUTPUT_DIR / "boxes_manual" / "sheets" / "verify" / "OBJ070_v01.png")
    verify.crop((0, 0, verify.width, 710)).save(FIG / "fig1_annotation.png")
    row_of([with_boxes("train", "multi_train_grid_00000"), with_boxes("train", "multi_train_g3_00000"),
            with_boxes("train", "multi_train_paste_00001")]).save(FIG / "fig2_training_examples.png")
    curves()
    samples = RES / "results" / "yolov8s_samples"
    row_of([Image.open(samples / "multi_test_multi_g2_00000.jpg"), Image.open(samples / "multi_test_multi_g3_00040.jpg")],
           height=520).save(FIG / "fig4a_detections.png")
    row_of([Image.open(RES / "predictions" / "sample_3x3.jpg"), Image.open(samples / "multi_test_multi_g3_00200.jpg")],
           height=520).save(FIG / "fig4b_detections.png")
    rows = per_class()
    print("lowest 3x3 mAP50-95:", [(c, round(a, 3), round(b, 3)) for c, a, b in rows[:10]])
    print("figures in", FIG)


if __name__ == "__main__":
    main()
