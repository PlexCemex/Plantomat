#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import ImageCSVDataset, build_image_model  # noqa: E402
from plantomat.utils import ensure_dir, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Оценка image-only модели.')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def save_metrics_figure(out_path: Path, accuracy: float, precision: float, recall: float, f1: float, split: str) -> None:
    labels = ['Accuracy', 'Precision', 'Recall', 'F1']
    values = [accuracy * 100.0, precision * 100.0, recall * 100.0, f1 * 100.0]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values)
    plt.ylim(0, 100)
    plt.ylabel('Проценты')
    plt.title(f'{split} metrics (%)')

    for bar, v in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, v + 1, f'{v:.2f}%', ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    df = pd.read_csv(args.csv)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    class_names = checkpoint['class_names']
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_size = int(checkpoint.get('image_size', 224))

    model = build_image_model(checkpoint['backbone'], num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    split_df = df[df['split'] == args.split].copy()
    ds = ImageCSVDataset(split_df, class_to_idx, image_size=image_size, augment=False)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == 'cuda',
    )

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    labels = list(range(len(class_names)))
    
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        average='macro',
        zero_division=0,
    )
    
    report = classification_report(
        targets,
        preds,
        labels=labels,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    
    cm = confusion_matrix(targets, preds, labels=labels)
    report['summary_metrics'] = {
        'accuracy': float(accuracy),
        'accuracy_pct': float(accuracy * 100.0),
        'macro_precision': float(precision),
        'macro_precision_pct': float(precision * 100.0),
        'macro_recall': float(recall),
        'macro_recall_pct': float(recall * 100.0),
        'macro_f1': float(f1),
        'macro_f1_pct': float(f1 * 100.0),
    }
    write_json(report, out_dir / f'{args.split}_classification_report.json')

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'{args.split} confusion matrix | accuracy={accuracy * 100.0:.2f}%')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.split}_confusion_matrix.png', dpi=150)
    plt.close()

    save_metrics_figure(
        out_dir / f'{args.split}_metrics.png',
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        split=args.split,
    )

    print(f'Готово: {out_dir}')
    print(f'Accuracy: {accuracy * 100.0:.2f}%')
    print(f'Macro Precision: {precision * 100.0:.2f}%')
    print(f'Macro Recall: {recall * 100.0:.2f}%')
    print(f'Macro F1: {f1 * 100.0:.2f}%')


if __name__ == '__main__':
    main()