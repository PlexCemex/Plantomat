from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNNEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.out_dim = 256

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return x.flatten(1)


def build_image_encoder(backbone: str = 'efficientnet_b0', pretrained: bool = True) -> Tuple[nn.Module, int]:
    backbone = backbone.lower()
    if backbone == 'simple_cnn':
        encoder = SimpleCNNEncoder()
        return encoder, encoder.out_dim

    try:
        if backbone == 'efficientnet_b0':
            from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            model = efficientnet_b0(weights=weights)
            out_dim = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, out_dim

        if backbone == 'resnet18':
            from torchvision.models import ResNet18_Weights, resnet18

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            model = resnet18(weights=weights)
            out_dim = model.fc.in_features
            model.fc = nn.Identity()
            return model, out_dim
    except Exception as exc:
        raise ImportError(
            'Не удалось создать torchvision-бэкбон. Установи совместимые torch/torchvision '
            'или переключись на --backbone simple_cnn.'
        ) from exc

    raise ValueError(f'Unsupported backbone: {backbone}')


class MultimodalClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        sensor_dim: int = 0,
        backbone: str = 'efficientnet_b0',
        pretrained: bool = True,
        dropout: float = 0.30,
    ) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone, image_dim = build_image_encoder(backbone=backbone, pretrained=pretrained)

        self.image_head = nn.Sequential(
            nn.Linear(image_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.sensor_dim = int(sensor_dim)
        if self.sensor_dim > 0:
            self.sensor_head = nn.Sequential(
                nn.Linear(self.sensor_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout / 2),
            )
            fusion_dim = 512 + 64
        else:
            self.sensor_head = None
            fusion_dim = 512

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone.parameters():
            param.requires_grad = trainable

    def forward(self, image: torch.Tensor, sensor: torch.Tensor | None = None) -> torch.Tensor:
        image_features = self.image_head(self.backbone(image))

        if self.sensor_head is not None and sensor is not None and sensor.numel() > 0:
            sensor_features = self.sensor_head(sensor)
            fused = torch.cat([image_features, sensor_features], dim=1)
        else:
            fused = image_features

        return self.classifier(fused)
