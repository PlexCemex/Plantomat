from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    except Exception:
        pass


def ensure_dir(path: os.PathLike[str] | str) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def load_yaml(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def save_json(data: Dict[str, Any], path: os.PathLike[str] | str) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: os.PathLike[str] | str) -> Dict[str, Any]:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def slugify_label(text: Any) -> str:
    value = str(text).strip().lower()
    value = value.replace('tomato___', '')
    value = value.replace('&', 'and')
    value = re.sub(r'[^a-z0-9а-яё]+', '_', value)
    value = re.sub(r'_+', '_', value)
    return value.strip('_')


def normalize_stage(stage: Optional[str], default: str = 'vegetative') -> str:
    if stage is None or str(stage).strip() == '':
        return default
    value = slugify_label(stage)
    aliases = {
        'germination': 'germination',
        'sprouting': 'germination',
        'seedling': 'seedling',
        'рассада': 'seedling',
        'vegetative': 'vegetative',
        'growth': 'vegetative',
        'vegetation': 'vegetative',
        'вегетация': 'vegetative',
        'flowering': 'flowering',
        'цветение': 'flowering',
        'fruiting': 'fruiting',
        'плодоношение': 'fruiting',
        'germination_and_early_growth': 'germination',
        'early_fruiting': 'flowering',
        'mature_fruiting': 'fruiting',
    }
    return aliases.get(value, value)


def resolve_image_paths(
    df: pd.DataFrame,
    csv_path: os.PathLike[str] | str,
    image_col: str = 'image_path',
) -> pd.DataFrame:
    csv_dir = Path(csv_path).resolve().parent
    out = df.copy()

    def _resolve(v: Any) -> str:
        path = Path(str(v))
        if path.is_absolute():
            return str(path)
        return str((csv_dir / path).resolve())

    out[image_col] = out[image_col].apply(_resolve)
    return out


def infer_device(preferred: Optional[str] = None) -> str:
    if preferred:
        return preferred
    try:
        import torch

        if torch.cuda.is_available():
            return 'cuda'
        return 'cpu'
    except Exception:
        return 'cpu'


def safe_float(value: Any, default: float = np.nan) -> float:
    try:
        if value is None:
            return default
        if isinstance(value, str) and value.strip() == '':
            return default
        return float(value)
    except Exception:
        return default


def stratified_split(
    df: pd.DataFrame,
    label_col: str = 'label',
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
    split_col: str = 'split',
) -> pd.DataFrame:
    if split_col in df.columns and df[split_col].notna().all():
        return df.copy()

    out = df.copy()
    out[split_col] = 'train'

    if label_col not in out.columns:
        return out

    labels = out[label_col].astype(str)
    class_counts = labels.value_counts()
    too_small = (class_counts < 3).any()

    indices = np.arange(len(out))
    stratify_labels = labels if not too_small else None

    temp_ratio = val_ratio + test_ratio
    if temp_ratio <= 0:
        return out

    train_idx, temp_idx = train_test_split(
        indices,
        test_size=temp_ratio,
        random_state=seed,
        shuffle=True,
        stratify=stratify_labels,
    )
    out.loc[temp_idx, split_col] = 'temp'

    if len(temp_idx) == 0:
        out[split_col] = 'train'
        return out

    if test_ratio <= 0:
        out.loc[temp_idx, split_col] = 'val'
        return out

    temp_df = out.iloc[temp_idx]
    if len(temp_df) < 2:
        out.loc[temp_df.index, split_col] = 'val'
        return out

    temp_labels = temp_df[label_col].astype(str)
    temp_counts = temp_labels.value_counts()
    temp_stratify = temp_labels if (temp_counts >= 2).all() else None
    relative_test = test_ratio / temp_ratio

    if len(temp_df) * relative_test < 1:
        out.loc[temp_df.index, split_col] = 'val'
        return out
    if len(temp_df) * (1 - relative_test) < 1:
        out.loc[temp_df.index, split_col] = 'test'
        return out

    val_sub_idx, test_sub_idx = train_test_split(
        np.arange(len(temp_df)),
        test_size=relative_test,
        random_state=seed,
        shuffle=True,
        stratify=temp_stratify,
    )

    real_temp_idx = temp_df.index.to_numpy()
    out.loc[real_temp_idx[val_sub_idx], split_col] = 'val'
    out.loc[real_temp_idx[test_sub_idx], split_col] = 'test'
    return out


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Iterable[str],
    out_path: os.PathLike[str] | str,
    title: str = 'Confusion matrix',
) -> None:
    import matplotlib.pyplot as plt

    class_names = list(class_names)
    fig, ax = plt.subplots(figsize=(max(8, len(class_names) * 0.8), max(6, len(class_names) * 0.6)))
    im = ax.imshow(cm, interpolation='nearest')
    fig.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], 'd'),
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    plt.close(fig)


def timestamp_now_str() -> str:
    from datetime import datetime

    return datetime.now().strftime('%Y%m%d_%H%M%S')
