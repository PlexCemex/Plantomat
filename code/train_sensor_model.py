#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.sensor_pipeline import NUMERIC_SENSOR_COLUMNS, SensorAutoencoder, SensorPreprocessor  # noqa: E402
from plantomat.utils import ensure_dir, seed_everything, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Обучение sensor-only модели (автоэнкодер для оценки аномальности среды).')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--bottleneck-dim', type=int, default=16)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    losses = []
    for batch_x, in tqdm(loader, leave=False):
        batch_x = batch_x.to(device)
        with torch.set_grad_enabled(train):
            recon = model(batch_x)
            loss = criterion(recon, batch_x)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(float(loss.item()))
    return float(np.mean(losses)) if losses else 0.0


def reconstruction_errors(model, array: np.ndarray, device) -> np.ndarray:
    model.eval()
    x = torch.tensor(array, dtype=torch.float32, device=device)
    with torch.no_grad():
        recon = model(x)
        errors = torch.mean((recon - x) ** 2, dim=1).cpu().numpy()
    return errors


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    df = pd.read_csv(args.csv)
    train_df, tmp_df = train_test_split(df, test_size=0.30, random_state=args.seed)
    val_df, test_df = train_test_split(tmp_df, test_size=0.50, random_state=args.seed)

    preprocessor = SensorPreprocessor(NUMERIC_SENSOR_COLUMNS).fit(train_df)
    x_train = preprocessor.transform_df(train_df)
    x_val = preprocessor.transform_df(val_df)
    x_test = preprocessor.transform_df(test_df)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32)), batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(x_val, dtype=torch.float32)), batch_size=args.batch_size, shuffle=False)

    model = SensorAutoencoder(input_dim=x_train.shape[1], hidden_dim=args.hidden_dim, bottleneck_dim=args.bottleneck_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float('inf')
    best_path = out_dir / 'best_sensor_autoencoder.pt'
    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        row = {'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss}
        history.append(row)
        print(row)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'input_dim': int(x_train.shape[1]),
                'hidden_dim': args.hidden_dim,
                'bottleneck_dim': args.bottleneck_dim,
                'preprocessor_state': preprocessor.state_dict(),
                'config': vars(args),
            }, best_path)

    pd.DataFrame(history).to_csv(out_dir / 'sensor_history.csv', index=False)
    if history:
        plt.figure(figsize=(7, 4))
        plt.plot([r['epoch'] for r in history], [r['train_loss'] for r in history], label='train_loss')
        plt.plot([r['epoch'] for r in history], [r['val_loss'] for r in history], label='val_loss')
        plt.xlabel('epoch')
        plt.ylabel('mse')
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / 'sensor_training_curve.png', dpi=150)
        plt.close()

    checkpoint = torch.load(best_path, map_location='cpu')
    best_model = SensorAutoencoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        bottleneck_dim=checkpoint['bottleneck_dim'],
    ).to(device)
    best_model.load_state_dict(checkpoint['model_state_dict'])

    val_errors = reconstruction_errors(best_model, x_val, device)
    test_errors = reconstruction_errors(best_model, x_test, device)
    threshold = float(np.quantile(val_errors, 0.95))

    write_json(preprocessor.state_dict(), out_dir / 'sensor_preprocessor.json')
    write_json({
        'threshold': threshold,
        'val_mean_error': float(np.mean(val_errors)),
        'test_mean_error': float(np.mean(test_errors)),
        'train_size': int(len(train_df)),
        'val_size': int(len(val_df)),
        'test_size': int(len(test_df)),
        'numeric_cols': NUMERIC_SENSOR_COLUMNS,
        'best_checkpoint': str(best_path.resolve()),
    }, out_dir / 'sensor_model_summary.json')

    plt.figure(figsize=(7, 4))
    plt.hist(val_errors, bins=40, alpha=0.7, label='val')
    plt.hist(test_errors, bins=40, alpha=0.7, label='test')
    plt.axvline(threshold, linestyle='--', label='threshold')
    plt.xlabel('reconstruction error')
    plt.ylabel('count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'sensor_error_histogram.png', dpi=150)
    plt.close()

    print(f'Готово: {best_path}')
    print(f'Порог аномальности: {threshold:.6f}')


if __name__ == '__main__':
    main()
