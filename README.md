# IE 7615 Discriminative Project

Group: Aaditya Krishna, Pratham Shah, Sanjana Menon

The class dataset has 73 objects (OBJ001 to OBJ073), photographed by the students in the course.

- **Milestone 1:** classify a single-object photo with its Object ID. We compared a custom CNN, ResNet-50, EfficientNet-B0 and YOLO11s-cls.
- **Milestone 2:** find every class object in a multi-object image (2x2 and 3x3 grids of dataset photos) and give its Object ID and bounding box. We used YOLOv8s.

## Layout

```
object_ids.csv               Object ID, name and dataset folder for each class
requirements.txt
milestone1_classification/   preprocessing, the 4 classifiers, evaluation
  outputs/metrics, figures   our test results
milestone2_detection/        bounding boxes, detection data, YOLOv8s training and evaluation
  results/yolov8s_best.pt    final detection model
  outputs/boxes_manual/      all bounding boxes (annotations.json / .csv) and the labelling log
  outputs/m2_results/        test results
  outputs/m2_figures/        report figures
```

The dataset itself is not in the repo. Put the class folder `IE7615 Deep Learning for AI` (images_OBJ001 ... images_OBJ073) in the repo root, or set `DATASET_ROOT`.

## Setup

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126   # or the CPU / MPS build
pip install -r requirements.txt
```

## Milestone 1

```
cd milestone1_classification
python inspect_data.py
python preprocess.py           # dedup, 224x224, 70/15/15 split, writes outputs/cache
python run_training.py         # custom_cnn, resnet50, efficientnet_b0, yolo11s_cls
python evaluate.py
python compare.py
python check_backgrounds.py
```

| Model | Test top-1 | Macro F1 |
|---|---|---|
| ResNet-50 | 97.96% | 0.979 |
| YOLO11s-cls | 97.76% | 0.977 |
| EfficientNet-B0 | 96.89% | 0.968 |
| Custom CNN | 82.20% | 0.823 |

## Milestone 2

Run the detector on any image (a 2x2 or 3x3 grid, or a single photo):

```
cd milestone2_detection
python predict.py path/to/image.jpg
```

It prints one line per object with its Object ID, confidence and box, and saves the image with the boxes drawn in `predictions/`.

To rebuild the data and retrain (needs the Milestone 1 cache):

```
cd milestone2_detection
python make_detection_data.py          # single photos, 2x2 grids and pasted images -> data/
python build_grids.py --n2 1000 --n3 2000
python train_detect.py --only yolov8s --epochs 80
python finetune.py                     # 10 low-lr epochs, keeps the better model on val
python eval_layouts.py
python grid_accuracy.py
```

Every box was placed by eye with `annotate.py` (6,901 of 6,942 photos boxed, 7,157 boxes). The boxes are already in `outputs/boxes_manual/annotations.json`, so you don't need to annotate again to retrain.

Test results of the final model (test images are built only from test photos):

| Test images | mAP50 | mAP50-95 |
|---|---|---|
| 2x2 grids | 0.961 | 0.708 |
| 3x3 grids | 0.956 | 0.698 |
| Single photos | 0.954 | 0.686 |

At confidence 0.25 it finds 95.4% of the objects in the 2x2 test grids and 94.4% in the 3x3 grids with the right Object ID.
