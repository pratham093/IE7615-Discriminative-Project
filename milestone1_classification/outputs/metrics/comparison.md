Test split: 1028 images, 73 classes. Sorted by test macro F1. 95% CI = Wilson score interval for test top-1 accuracy.

| Model | Approach | Params (M) | Train time (min) | Trained on | Best val acc | Test top-1 | Test top-1 95% CI | Test top-5 | Test macro F1 | Test errors |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| **ResNet-50** | transfer learning | 23.66 | 4.8 | NVIDIA GeForce RTX 4070 Laptop GPU | 98.35% | **97.96%** | 96.9% to 98.7% | 99.61% | **0.9789** | 21 |
| YOLO11s-cls | transfer learning | 5.54 | 3.2 | NVIDIA GeForce RTX 4070 Laptop GPU | 97.47% | 97.76% | 96.7% to 98.5% | 99.51% | 0.9771 | 23 |
| EfficientNet-B0 | transfer learning | 4.1 | 3.7 | NVIDIA GeForce RTX 4070 Laptop GPU | 97.86% | 96.89% | 95.6% to 97.8% | 99.51% | 0.9680 | 32 |
| Custom CNN | from scratch | 4.75 | 10.9 | NVIDIA GeForce RTX 4070 Laptop GPU | 84.53% | 82.20% | 79.7% to 84.4% | 95.43% | 0.8226 | 183 |

Best model by test macro F1: **ResNet-50** (macro F1 0.9789, top-1 97.96%).
