from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from torch import nn


NUMERIC_SENSOR_COLUMNS = [
    'air_temp_c',
    'air_humidity_pct',
    'soil_moisture_pct',
    'solution_ph',
    'ec_ms_cm',
    'tds_ppm',
    'light_lux',
    'co2_ppm',
    'leaf_wetness',
]


class SensorPreprocessor:
    def __init__(self, numeric_cols: List[str] | None = None) -> None:
        self.numeric_cols = numeric_cols or NUMERIC_SENSOR_COLUMNS
        self.medians: Dict[str, float] = {}
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}
        self.stage_categories: List[str] = []

    def fit(self, df: pd.DataFrame) -> 'SensorPreprocessor':
        work = df.copy()
        if 'growth_stage' not in work.columns:
            work['growth_stage'] = 'unknown'
        work['growth_stage'] = work['growth_stage'].fillna('unknown').astype(str).str.lower()
        self.stage_categories = sorted(work['growth_stage'].unique().tolist())
        for col in self.numeric_cols:
            series = pd.to_numeric(work[col], errors='coerce') if col in work.columns else pd.Series(dtype=float)
            median = float(series.median()) if len(series.dropna()) else 0.0
            filled = series.fillna(median)
            mean = float(filled.mean()) if len(filled) else 0.0
            std = float(filled.std(ddof=0)) if len(filled) else 1.0
            if std == 0:
                std = 1.0
            self.medians[col] = median
            self.means[col] = mean
            self.stds[col] = std
        return self

    @property
    def feature_dim(self) -> int:
        return len(self.numeric_cols) + len(self.stage_categories)

    def _encode_stage(self, stage: str) -> np.ndarray:
        stage = (stage or 'unknown').lower()
        arr = np.zeros(len(self.stage_categories), dtype=np.float32)
        if stage in self.stage_categories:
            arr[self.stage_categories.index(stage)] = 1.0
        return arr

    def transform_df(self, df: pd.DataFrame) -> np.ndarray:
        work = df.copy()
        if 'growth_stage' not in work.columns:
            work['growth_stage'] = 'unknown'
        features = []
        for col in self.numeric_cols:
            series = pd.to_numeric(work[col], errors='coerce') if col in work.columns else pd.Series(np.nan, index=work.index)
            filled = series.fillna(self.medians[col]).astype(np.float32)
            scaled = (filled - self.means[col]) / self.stds[col]
            features.append(scaled.to_numpy().reshape(-1, 1))
        numeric = np.concatenate(features, axis=1) if features else np.zeros((len(work), 0), dtype=np.float32)
        stage_matrix = np.stack([self._encode_stage(s) for s in work['growth_stage'].astype(str).tolist()], axis=0)
        return np.concatenate([numeric.astype(np.float32), stage_matrix.astype(np.float32)], axis=1)

    def transform_row(self, row: Dict[str, Any]) -> np.ndarray:
        df = pd.DataFrame([row])
        return self.transform_df(df)[0]

    def state_dict(self) -> Dict[str, Any]:
        return {
            'numeric_cols': self.numeric_cols,
            'medians': self.medians,
            'means': self.means,
            'stds': self.stds,
            'stage_categories': self.stage_categories,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> 'SensorPreprocessor':
        obj = cls(state['numeric_cols'])
        obj.medians = {k: float(v) for k, v in state['medians'].items()}
        obj.means = {k: float(v) for k, v in state['means'].items()}
        obj.stds = {k: float(v) for k, v in state['stds'].items()}
        obj.stage_categories = list(state['stage_categories'])
        return obj


class SensorAutoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, bottleneck_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)
