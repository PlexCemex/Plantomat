#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.datasets import MultimodalImageDataset  # noqa: E402
from tomato_ai.engine import evaluate, train_one_epoch  # noqa: E402
from tomato_ai.features import TabularFeatureBuilder  # noqa: E402
from tomato_ai.models import MultimodalClassifier  # noqa: E402
from tomato_ai.transforms import build_eval_transform, build_train_transform  # noqa: E402
from tomato_ai.utils import (  # noqa: E402
    ensure_dir,
    infer_device,
    load_yaml,
    plot_confusion_matrix,
    resolve_image_paths,
    save_json,
    seed_everything,
    stratified_split,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Обучение image-only или мультимодальной модели томатов.')
    parser.add_argument('--csv', type=str, required=True, help='CSV датасета.')
    parser.add_argument('--output-dir', type=str, required=True, help='Папка для чекпоинтов и метрик.')
    parser.add_argument('--config', type=str, default=None, help='YAML-конфиг.')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--resume-checkpoint', type=str, default=None)

    parser.add_argument('--backbone', type=str, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--image-size', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--dropout', type=float, default=None)
    parser.add_argument('--freeze-backbone-epochs', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)

    pretrained_group = parser.add_mutually_exclusive_group()
    pretrained_group.add_argument('--pretrained', dest='pretrained', action='store_true')
    pretrained_group.add_argument('--no-pretrained', dest='pretrained', action='store_false')
    parser.set_defaults(pretrained=None)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        'image_col': 'image_path',
        'label_col': 'label',
        'split_col': 'split',
        'categorical_feature_columns': ['growth_stage'],
        'reserved_columns': ['image_path', 'label', 'split', 'timestamp', 'plant_id', 'source_dataset'],
        'image_size': 224,
        'batch_size': 16,
        'num_workers': 0,
        'epochs': 20,
        'lr': 3e-4,
        'weight_decay': 1e-4,
        'dropout': 0.30,
        'backbone': 'efficientnet_b0',
        'pretrained': True,
        'freeze_backbone_epochs': 1,
        'early_stopping_patience': 5,
        'grad_clip': 1.0,
        'label_smoothing': 0.05,
        'mixed_precision': True,
        'seed': 42,
        'val_ratio': 0.15,
        'test_ratio': 0.15,
    }
    if args.config:
        config.update(load_yaml(args.config))

    for key, value in {
        'backbone': args.backbone,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'image_size': args.image_size,
        'lr': args.lr,
        'num_workers': args.num_workers,
        'dropout': args.dropout,
        'freeze_backbone_epochs': args.freeze_backbone_epochs,
        'seed': args.seed,
        'pretrained': args.pretrained,
    }.items():
        if value is not None:
            config[key] = value

    return config


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    return obj


def make_dataloader(dataset, batch_size: int, shuffle: bool, num_workers: int, device: str):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.startswith('cuda'),
    )


def save_split_predictions(
    result: Dict[str, Any],
    idx_to_label: Dict[int, str],
    out_csv: Path,
    y_true: List[int] | None = None,
) -> None:
    probs = result.get('y_prob')
    pred_idx = result.get('y_pred', [])
    paths = result.get('paths', [])
    rows = []
    for i, pred in enumerate(pred_idx):
        prob_vec = probs[i] if probs is not None and len(probs) > i else None
        conf = float(np.max(prob_vec)) if prob_vec is not None else math.nan
        row = {
            'path': paths[i] if i < len(paths) else '',
            'pred_idx': int(pred),
            'pred_label': idx_to_label[int(pred)],
            'confidence': conf,
        }
        if y_true is not None and len(y_true) > i:
            row['true_idx'] = int(y_true[i])
            row['true_label'] = idx_to_label[int(y_true[i])]
        rows.append(row)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    seed_everything(int(cfg['seed']))

    device = infer_device(args.device)
    output_dir = ensure_dir(args.output_dir)
    save_json(to_builtin(cfg), output_dir / 'resolved_config.json')

    df = pd.read_csv(args.csv)
    df = resolve_image_paths(df, args.csv, image_col=cfg['image_col'])
    df = stratified_split(
        df,
        label_col=cfg['label_col'],
        val_ratio=float(cfg['val_ratio']),
        test_ratio=float(cfg['test_ratio']),
        seed=int(cfg['seed']),
        split_col=cfg['split_col'],
    )

    train_df = df[df[cfg['split_col']] == 'train'].reset_index(drop=True)
    val_df = df[df[cfg['split_col']] == 'val'].reset_index(drop=True)
    test_df = df[df[cfg['split_col']] == 'test'].reset_index(drop=True)

    if train_df.empty:
        raise RuntimeError('Train split пустой.')
    if val_df.empty:
        raise RuntimeError('Val split пустой. Увеличь размер датасета или пересоздай split.')

    class_names = sorted(train_df[cfg['label_col']].astype(str).unique().tolist())
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    feature_builder = TabularFeatureBuilder.infer_and_fit(
        train_df,
        reserved_columns=cfg['reserved_columns'],
        categorical_candidates=cfg['categorical_feature_columns'],
    )

    train_ds = MultimodalImageDataset(
        train_df,
        transform=build_train_transform(int(cfg['image_size'])),
        label_to_idx=label_to_idx,
        feature_builder=feature_builder,
        image_col=cfg['image_col'],
        label_col=cfg['label_col'],
    )
    val_ds = MultimodalImageDataset(
        val_df,
        transform=build_eval_transform(int(cfg['image_size'])),
        label_to_idx=label_to_idx,
        feature_builder=feature_builder,
        image_col=cfg['image_col'],
        label_col=cfg['label_col'],
    )
    test_ds = MultimodalImageDataset(
        test_df,
        transform=build_eval_transform(int(cfg['image_size'])),
        label_to_idx=label_to_idx,
        feature_builder=feature_builder,
        image_col=cfg['image_col'],
        label_col=cfg['label_col'],
    )

    train_loader = make_dataloader(train_ds, int(cfg['batch_size']), True, int(cfg['num_workers']), device)
    val_loader = make_dataloader(val_ds, int(cfg['batch_size']), False, int(cfg['num_workers']), device)
    test_loader = make_dataloader(test_ds, int(cfg['batch_size']), False, int(cfg['num_workers']), device)

    model = MultimodalClassifier(
        num_classes=len(class_names),
        sensor_dim=feature_builder.feature_dim,
        backbone=str(cfg['backbone']),
        pretrained=bool(cfg['pretrained']),
        dropout=float(cfg['dropout']),
    ).to(device)

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print(f'Загружены веса из {args.resume_checkpoint}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg['lr']), weight_decay=float(cfg['weight_decay']))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

    train_targets = np.asarray([label_to_idx[str(v)] for v in train_df[cfg['label_col']].tolist()])
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.arange(len(class_names)),
        y=train_targets,
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32, device=device),
        label_smoothing=float(cfg['label_smoothing']),
    )

    scaler = torch.cuda.amp.GradScaler(enabled=bool(cfg['mixed_precision']) and device.startswith('cuda'))

    history: List[Dict[str, Any]] = []
    best_val_f1 = -1.0
    best_epoch = -1
    patience = 0
    best_path = output_dir / 'best_model.pt'

    for epoch in range(1, int(cfg['epochs']) + 1):
        freeze_epochs = int(cfg['freeze_backbone_epochs'])
        trainable = epoch > freeze_epochs or str(cfg['backbone']).lower() == 'simple_cnn' or not bool(cfg['pretrained'])
        model.set_backbone_trainable(trainable)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device=device,
            scaler=scaler if scaler.is_enabled() else None,
            grad_clip=float(cfg['grad_clip']) if cfg.get('grad_clip') is not None else None,
            desc=f'Train {epoch}/{cfg["epochs"]}',
        )
        val_metrics = evaluate(model, val_loader, criterion, device=device, desc=f'Val {epoch}/{cfg["epochs"]}')
        scheduler.step(float(val_metrics['loss']))

        record = {
            'epoch': epoch,
            'lr': float(optimizer.param_groups[0]['lr']),
            'train_loss': float(train_metrics['loss']),
            'train_acc': float(train_metrics['accuracy']),
            'train_macro_f1': float(train_metrics['macro_f1']),
            'val_loss': float(val_metrics['loss']),
            'val_acc': float(val_metrics['accuracy']),
            'val_macro_f1': float(val_metrics['macro_f1']),
        }
        history.append(record)
        print(json.dumps(record, ensure_ascii=False))

        if float(val_metrics['macro_f1']) > best_val_f1:
            best_val_f1 = float(val_metrics['macro_f1'])
            best_epoch = epoch
            patience = 0
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'backbone': cfg['backbone'],
                'num_classes': len(class_names),
                'label_to_idx': label_to_idx,
                'idx_to_label': idx_to_label,
                'feature_builder_state': feature_builder.state_dict(),
                'image_size': int(cfg['image_size']),
                'config': to_builtin(cfg),
            }
            torch.save(checkpoint, best_path)
        else:
            patience += 1

        if patience >= int(cfg['early_stopping_patience']):
            print('Early stopping.')
            break

    pd.DataFrame(history).to_csv(output_dir / 'history.csv', index=False)

    if not best_path.exists():
        raise RuntimeError('Не был сохранён ни один чекпоинт.')

    best_checkpoint = torch.load(best_path, map_location='cpu')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    model.to(device)

    save_json(
        {
            'best_epoch': int(best_epoch),
            'best_val_macro_f1': float(best_val_f1),
            'num_classes': len(class_names),
            'class_names': class_names,
            'feature_dim': int(feature_builder.feature_dim),
            'feature_names': feature_builder.feature_names,
            'train_size': int(len(train_df)),
            'val_size': int(len(val_df)),
            'test_size': int(len(test_df)),
            'device': device,
        },
        output_dir / 'training_summary.json',
    )

    def evaluate_and_save(split_name: str, loader, split_df: pd.DataFrame) -> None:
        if split_df.empty:
            return
        result = evaluate(model, loader, criterion, device=device, desc=f'{split_name.title()} final')
        y_true = result.get('y_true', [])
        y_pred = result.get('y_pred', [])
        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(class_names))),
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
        plot_confusion_matrix(cm, class_names, output_dir / f'{split_name}_confusion_matrix.png', title=f'{split_name} confusion matrix')
        save_json(to_builtin(report), output_dir / f'{split_name}_classification_report.json')
        save_json(
            {
                'loss': float(result['loss']),
                'accuracy': float(result['accuracy']),
                'macro_f1': float(result['macro_f1']),
            },
            output_dir / f'{split_name}_metrics.json',
        )
        save_split_predictions(result, idx_to_label, output_dir / f'{split_name}_predictions.csv', y_true=y_true)

    evaluate_and_save('val', val_loader, val_df)
    evaluate_and_save('test', test_loader, test_df)

    print('Обучение завершено.')
    print(f'Лучший чекпоинт: {best_path}')
    print(f'Папка артефактов: {output_dir}')


if __name__ == '__main__':
    main()
