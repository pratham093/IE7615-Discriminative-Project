"""Runs the trained models on the background-only photos to see how much they rely on the background."""

import csv

import numpy as np
import torch
from PIL import Image

import config
from datasets import eval_transform, load_classes
from evaluate import DISPLAY, MODELS
from models import build_model
from preprocess import load_image
from train import pick_device


def main():
    classes = [c for c, _ in load_classes()]
    with open(config.LOG_DIR / "preprocessing_log.csv", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["step"] == "background"]
    recs = [load_image(config.DATASET_ROOT / r["path"]) for r in rows]  # same EXIF / RGB / crop / resize as Stage 1
    bad = [r["path"] for r, rec in zip(rows, recs) if rec["error"]]
    if bad:
        raise SystemExit(f"could not read: {bad}")
    owner = np.array([classes.index(r["object_id"]) for r in rows])
    pixels = [rec["pixels"] for rec in recs]
    device = pick_device()
    print(f"{len(rows)} background photos from {len(set(owner))} classes, chance = {1 / len(classes):.1%}")

    results = []
    for name in MODELS:
        if name == config.YOLO_RUN:
            from ultralytics import YOLO
            model = YOLO(str(config.CHECKPOINT_DIR / f"{name}_best.pt"))
            column = [classes.index(model.names[i]) for i in range(len(model.names))]
            probs = np.zeros((len(pixels), len(classes)), dtype=np.float32)
            images = [Image.fromarray(p) for p in pixels]
            for start in range(0, len(images), 64):
                for k, r in enumerate(model.predict(images[start:start + 64], imgsz=config.IMG_SIZE, verbose=False,
                                                    device=0 if device.type == "cuda" else device.type)):
                    probs[start + k, column] = r.probs.data.cpu().numpy()
        else:
            ckpt = torch.load(config.CHECKPOINT_DIR / f"{name}_best.pt", map_location="cpu")
            model = build_model(name, len(classes), pretrained=False)
            model.load_state_dict(ckpt["state_dict"])
            model.to(device).eval()
            tf = eval_transform()
            x = torch.stack([tf(torch.from_numpy(p.copy()).permute(2, 0, 1)) for p in pixels])
            with torch.no_grad():
                probs = torch.cat([torch.softmax(model(x[i:i + 64].to(device)), dim=1).cpu()
                                   for i in range(0, len(x), 64)]).numpy()
        pred = probs.argmax(1)
        own = pred == owner
        top5 = np.mean([o in np.argsort(-p)[:5] for o, p in zip(owner, probs)])
        results.append({"model": DISPLAY[name], "background_photos": len(owner), "named_owner": int(own.sum()),
                        "named_owner_rate": round(float(own.mean()), 4), "owner_in_top5_rate": round(float(top5), 4),
                        "mean_confidence_when_owner": round(float(probs[own, pred[own]].mean()) if own.any() else 0.0, 4),
                        "chance_rate": round(1 / len(classes), 4)})
        r = results[-1]
        print(f"{r['model']:<16} names the photographer's object for {r['named_owner_rate']:.1%} "
              f"({r['named_owner']}/{len(owner)}) | in top-5 {r['owner_in_top5_rate']:.1%}")

    path = config.METRIC_DIR / "background_check.csv"
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=results[0].keys())
        w.writeheader()
        w.writerows(results)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
