from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .features import TabularFeatureBuilder


class MultimodalImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform,
        label_to_idx: Optional[Dict[str, int]] = None,
        feature_builder: Optional[TabularFeatureBuilder] = None,
        image_col: str = 'image_path',
        label_col: str = 'label',
    ) -> None:
        self.df = dataframe.reset_index(drop=True).copy()
        self.transform = transform
        self.label_to_idx = label_to_idx or {}
        self.feature_builder = feature_builder
        self.image_col = image_col
        self.label_col = label_col

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, path: str) -> Image.Image:
        return Image.open(path).convert('RGB')

    def _get_sensor_tensor(self, row: pd.Series) -> torch.Tensor:
        if self.feature_builder is None or self.feature_builder.feature_dim == 0:
            return torch.zeros(0, dtype=torch.float32)
        return torch.from_numpy(self.feature_builder.transform_row(row))

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        image_path = str(row[self.image_col])
        image = self._load_image(image_path)
        image_tensor = self.transform(image) if self.transform is not None else image
        sensor_tensor = self._get_sensor_tensor(row)

        item: Dict[str, Any] = {
            'image': image_tensor,
            'sensor': sensor_tensor,
            'path': image_path,
        }

        if self.label_col in self.df.columns and self.label_to_idx:
            label_name = str(row[self.label_col])
            item['label'] = torch.tensor(self.label_to_idx[label_name], dtype=torch.long)
            item['label_name'] = label_name

        return item
