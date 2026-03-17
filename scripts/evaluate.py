#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.datasets import MultimodalImageDataset  # noqa: E402
from tomato_ai.engine import evaluate  # noqa: E402
from tomato_ai.features import TabularFeatureBuilder  # noqa: E402
from tomato_ai.models import MultimodalClassifier  # noqa: E402
from tomato_ai.transforms import build_eval_transform  # noqa: E402
from tomato_ai.utils import ensure_dir, plot_confusion_matrix, resolve_image_paths, save_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Оценка обученной модели на val/test.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--num-workers', type=int, default=0)
    return parser.parse_args()


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj


def main() -> None:
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = ensure_dir(args.output_dir)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    cfg = checkpoint.get('config', {})
    image_col = cfg.get('image_col', 'image_path')
    label_col = cfg.get('label_col', 'label')
    split_col = cfg.get('split_col', 'split')
    image_size = int(checkpoint.get('image_size', cfg.get('image_size', 224)))

    df = pd.read_csv(args.csv)
    df = resolve_image_paths(df, args.csv, image_col=image_col)
    df = df[df[split_col] == args.split].reset_index(drop=True)
    if df.empty:
        raise RuntimeError(f'В split={args.split} нет строк.')

    label_to_idx = checkpoint['label_to_idx']
    idx_to_label = {int(k): v for k, v in checkpoint['idx_to_label'].items()}
    class_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    feature_builder = TabularFeatureBuilder.from_state_dict(checkpoint['feature_builder_state'])

    dataset = MultimodalImageDataset(
        df,
        transform=build_eval_transform(image_size),
        label_to_idx=label_to_idx,
        feature_builder=feature_builder,
        image_col=image_col,
        label_col=label_col,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=device.startswith('cuda'))

    model = MultimodalClassifier(
        num_classes=len(label_to_idx),
        sensor_dim=feature_builder.feature_dim,
        backbone=checkpoint.get('backbone', cfg.get('backbone', 'efficientnet_b0')),
        pretrained=False,
        dropout=float(cfg.get('dropout', 0.30)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    criterion = nn.CrossEntropyLoss()
    result = evaluate(model, loader, criterion, device=device, desc=f'Eval {args.split}')
    y_true = result['y_true']
    y_pred = result['y_pred']

    report = classification_report(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    plot_confusion_matrix(cm, class_names, out_dir / f'{args.split}_confusion_matrix.png', title=f'{args.split} confusion matrix')
    save_json(
        {
            'loss': float(result['loss']),
            'accuracy': float(result['accuracy']),
            'macro_f1': float(result['macro_f1']),
        },
        out_dir / f'{args.split}_metrics.json',
    )
    save_json(to_builtin(report), out_dir / f'{args.split}_classification_report.json')

    pred_rows = []
    for i, pred in enumerate(y_pred):
        prob = result['y_prob'][i]
        pred_rows.append(
            {
                'path': result['paths'][i],
                'true_label': class_names[y_true[i]],
                'pred_label': class_names[pred],
                'confidence': float(np.max(prob)),
            }
        )
    pd.DataFrame(pred_rows).to_csv(out_dir / f'{args.split}_predictions.csv', index=False)

    print(f'Готово. Метрики сохранены в {out_dir}')


if __name__ == '__main__':
    main()
