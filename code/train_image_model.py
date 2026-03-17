#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import ImageCSVDataset, build_image_model  # noqa: E402
from plantomat.utils import ensure_dir, seed_everything, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Обучение модели по изображениям томата.')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='resnet18', choices=['resnet18', 'efficientnet_b0'])
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--pretrained', action='store_true')
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    losses, preds, targets = [], [], []
    model.train(train)
    iterator = tqdm(loader, leave=False)
    for images, labels in iterator:
        images = images.to(device)
        labels = labels.to(device)
        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        iterator.set_description(f"loss={sum(losses)/max(1, len(losses)):.4f}")
    acc = accuracy_score(targets, preds)
    return sum(losses) / max(1, len(losses)), acc


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    out_dir = ensure_dir(args.output_dir)

    df = pd.read_csv(args.csv)
    class_names = sorted(df['label'].unique().tolist())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()

    train_ds = ImageCSVDataset(train_df, class_to_idx, image_size=args.image_size, augment=True)
    val_ds = ImageCSVDataset(val_df, class_to_idx, image_size=args.image_size, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=device.type == 'cuda')
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')

    model = build_image_model(args.backbone, num_classes=len(class_names), pretrained=args.pretrained).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_val_acc = -1.0
    best_path = out_dir / 'best_image_model.pt'

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
        }
        history.append(row)
        print(row)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'backbone': args.backbone,
                'image_size': args.image_size,
                'class_names': class_names,
                'config': vars(args),
            }, best_path)

    pd.DataFrame(history).to_csv(out_dir / 'image_history.csv', index=False)

    if history:
        plt.figure(figsize=(7, 4))
        plt.plot([r['epoch'] for r in history], [r['train_acc'] for r in history], label='train_acc')
        plt.plot([r['epoch'] for r in history], [r['val_acc'] for r in history], label='val_acc')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'image_training_curve.png', dpi=150)
        plt.close()

    summary = {
        'csv': str(Path(args.csv).resolve()),
        'backbone': args.backbone,
        'image_size': args.image_size,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'best_val_acc': best_val_acc,
        'class_names': class_names,
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(test_df)),
        'best_checkpoint': str(best_path.resolve()),
    }
    write_json(summary, out_dir / 'image_training_summary.json')
    print(f'Готово: {best_path}')


if __name__ == '__main__':
    main()
