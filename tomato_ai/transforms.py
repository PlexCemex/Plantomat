from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence

import numpy as np
import torch
from PIL import Image, ImageEnhance


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Compose:
    def __init__(self, transforms: Sequence[Callable[[Image.Image], Image.Image | torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        out = image
        for transform in self.transforms:
            out = transform(out)
        return out


@dataclass
class Resize:
    size: int

    def __call__(self, image: Image.Image) -> Image.Image:
        return image.resize((self.size, self.size), Image.BILINEAR)


@dataclass
class CenterCrop:
    size: int

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        target = self.size
        left = max(0, int((width - target) / 2))
        upper = max(0, int((height - target) / 2))
        right = min(width, left + target)
        lower = min(height, upper + target)
        cropped = image.crop((left, upper, right, lower))
        if cropped.size != (target, target):
            cropped = cropped.resize((target, target), Image.BILINEAR)
        return cropped


@dataclass
class RandomHorizontalFlip:
    p: float = 0.5

    def __call__(self, image: Image.Image) -> Image.Image:
        if random.random() < self.p:
            return image.transpose(Image.FLIP_LEFT_RIGHT)
        return image


@dataclass
class RandomRotation:
    degrees: float = 10.0

    def __call__(self, image: Image.Image) -> Image.Image:
        angle = random.uniform(-self.degrees, self.degrees)
        return image.rotate(angle, resample=Image.BILINEAR)


@dataclass
class ColorJitter:
    brightness: float = 0.15
    contrast: float = 0.15
    saturation: float = 0.15

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            factor = random.uniform(max(0.0, 1.0 - self.brightness), 1.0 + self.brightness)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast > 0:
            factor = random.uniform(max(0.0, 1.0 - self.contrast), 1.0 + self.contrast)
            image = ImageEnhance.Contrast(image).enhance(factor)
        if self.saturation > 0:
            factor = random.uniform(max(0.0, 1.0 - self.saturation), 1.0 + self.saturation)
            image = ImageEnhance.Color(image).enhance(factor)
        return image


class ToTensorNormalize:
    def __init__(self, mean: Sequence[float] = IMAGENET_MEAN, std: Sequence[float] = IMAGENET_STD):
        self.mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.asarray(std, dtype=np.float32).reshape(1, 1, 3)

    def __call__(self, image: Image.Image) -> torch.Tensor:
        arr = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
        arr = (arr - self.mean) / self.std
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr.astype(np.float32))


def build_train_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(int(image_size * 1.10)),
            RandomHorizontalFlip(p=0.5),
            RandomRotation(degrees=10),
            ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10),
            CenterCrop(image_size),
            ToTensorNormalize(),
        ]
    )


def build_eval_transform(image_size: int) -> Compose:
    return Compose(
        [
            Resize(int(image_size * 1.05)),
            CenterCrop(image_size),
            ToTensorNormalize(),
        ]
    )
