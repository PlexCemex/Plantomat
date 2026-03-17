from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass
    os.environ['PYTHONHASHSEED'] = str(seed)


def slugify_label(label: str) -> str:
    label = label.replace('Tomato___', '')
    label = label.strip().lower()
    label = re.sub(r'[^a-z0-9]+', '_', label)
    label = re.sub(r'_+', '_', label).strip('_')
    return label


def read_json(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')


def detect_csv_separator(path: str | Path) -> str:
    sample = Path(path).read_text(encoding='utf-8', errors='ignore')[:4096]
    candidates = {',': sample.count(','), ';': sample.count(';'), '\t': sample.count('\t')}
    return max(candidates, key=candidates.get) if sample else ','


def stratified_split(
    df: pd.DataFrame,
    label_col: str,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    train_df, tmp_df = train_test_split(
        df,
        test_size=val_ratio + test_ratio,
        random_state=seed,
        stratify=df[label_col],
    )
    rel_test_ratio = test_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        tmp_df,
        test_size=rel_test_ratio,
        random_state=seed,
        stratify=tmp_df[label_col],
    )
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    return pd.concat([train_df, val_df, test_df], ignore_index=True)
