from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd

from .utils import safe_float


@dataclass
class TabularFeatureBuilder:
    continuous_columns: List[str] = field(default_factory=list)
    categorical_columns: List[str] = field(default_factory=list)
    means: Dict[str, float] = field(default_factory=dict)
    stds: Dict[str, float] = field(default_factory=dict)
    categories: Dict[str, List[str]] = field(default_factory=dict)

    @classmethod
    def infer_and_fit(
        cls,
        df: pd.DataFrame,
        reserved_columns: Sequence[str],
        categorical_candidates: Sequence[str] | None = None,
    ) -> 'TabularFeatureBuilder':
        categorical_candidates = list(categorical_candidates or [])
        reserved = set(reserved_columns)

        continuous_columns: List[str] = []
        for col in df.columns:
            if col in reserved:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continuous_columns.append(col)

        categorical_columns = [col for col in categorical_candidates if col in df.columns and col not in reserved]

        builder = cls(continuous_columns=sorted(continuous_columns), categorical_columns=sorted(categorical_columns))
        builder.fit(df)
        return builder

    def fit(self, df: pd.DataFrame) -> None:
        self.means = {}
        self.stds = {}
        self.categories = {}

        for col in self.continuous_columns:
            series = pd.to_numeric(df[col], errors='coerce')
            mean = float(series.mean()) if series.notna().any() else 0.0
            std = float(series.std(ddof=0)) if series.notna().sum() > 1 else 1.0
            if std == 0.0 or np.isnan(std):
                std = 1.0
            self.means[col] = mean
            self.stds[col] = std

        for col in self.categorical_columns:
            values = (
                df[col]
                .fillna('unknown')
                .astype(str)
                .str.strip()
                .replace('', 'unknown')
                .unique()
                .tolist()
            )
            self.categories[col] = sorted(values)

    @property
    def feature_dim(self) -> int:
        dim = len(self.continuous_columns)
        for col in self.categorical_columns:
            dim += len(self.categories.get(col, []))
        return dim

    @property
    def feature_names(self) -> List[str]:
        names = list(self.continuous_columns)
        for col in self.categorical_columns:
            for cat in self.categories.get(col, []):
                names.append(f'{col}__{cat}')
        return names

    def _transform_continuous(self, row: Dict[str, Any] | pd.Series) -> List[float]:
        feats: List[float] = []
        for col in self.continuous_columns:
            value = safe_float(row.get(col, np.nan), np.nan)
            if np.isnan(value):
                value = self.means.get(col, 0.0)
            mean = self.means.get(col, 0.0)
            std = self.stds.get(col, 1.0)
            feats.append((value - mean) / std)
        return feats

    def _transform_categorical(self, row: Dict[str, Any] | pd.Series) -> List[float]:
        feats: List[float] = []
        for col in self.categorical_columns:
            value = row.get(col, 'unknown')
            value = 'unknown' if value is None else str(value).strip() or 'unknown'
            categories = self.categories.get(col, [])
            one_hot = [0.0] * len(categories)
            if value in categories:
                one_hot[categories.index(value)] = 1.0
            feats.extend(one_hot)
        return feats

    def transform_row(self, row: Dict[str, Any] | pd.Series) -> np.ndarray:
        feats = self._transform_continuous(row) + self._transform_categorical(row)
        return np.asarray(feats, dtype=np.float32)

    def transform_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        if self.feature_dim == 0:
            return np.zeros((len(df), 0), dtype=np.float32)
        return np.vstack([self.transform_row(row) for _, row in df.iterrows()])

    def state_dict(self) -> Dict[str, Any]:
        return {
            'continuous_columns': self.continuous_columns,
            'categorical_columns': self.categorical_columns,
            'means': self.means,
            'stds': self.stds,
            'categories': self.categories,
        }

    @classmethod
    def from_state_dict(cls, state: Dict[str, Any]) -> 'TabularFeatureBuilder':
        return cls(
            continuous_columns=list(state.get('continuous_columns', [])),
            categorical_columns=list(state.get('categorical_columns', [])),
            means=dict(state.get('means', {})),
            stds=dict(state.get('stds', {})),
            categories={k: list(v) for k, v in dict(state.get('categories', {})).items()},
        )
