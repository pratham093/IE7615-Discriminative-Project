"""custom_cnn (trained from scratch), resnet50 and efficientnet_b0 (ImageNet weights, new head)."""

import torch.nn as nn
from torchvision import models


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        channels = [3, 32, 64, 128, 256, 512]
        self.features = nn.Sequential(*[conv_block(a, b) for a, b in zip(channels, channels[1:])])
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(channels[-1], num_classes))

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes, pretrained=True):
    if name == "custom_cnn":
        return CustomCNN(num_classes)
    if name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
        return model
    raise ValueError(f"unknown model {name!r}")


def head_modules(model, name):
    return [model.fc] if name == "resnet50" else [model.classifier]


def finetune_modules(model, name):
    # phase 2 of the pretrained models: last stage(s) plus the head
    if name == "resnet50":
        return [model.layer4, model.fc]
    if name == "efficientnet_b0":
        return [model.features[6], model.features[7], model.features[8], model.classifier]
    return [model]
