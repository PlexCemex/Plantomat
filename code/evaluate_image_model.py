#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
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
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=device.type == 'cuda')

    preds, targets = [], []
    with torch.no_grad():
        for images, labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
            targets.extend(labels.tolist())

    report = classification_report(targets, preds, target_names=class_names, digits=4, output_dict=True)
    write_json(report, out_dir / f'{args.split}_classification_report.json')

    cm = confusion_matrix(targets, preds, labels=list(range(len(class_names))))
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest')
    plt.title(f'{args.split} confusion matrix')
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=90)
    plt.yticks(range(len(class_names)), class_names)
    plt.tight_layout()
    plt.savefig(out_dir / f'{args.split}_confusion_matrix.png', dpi=150)
    plt.close()
    print(f'Готово: {out_dir}')


if __name__ == '__main__':
    main()
