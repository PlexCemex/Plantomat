#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.sensor_pipeline import SensorAutoencoder, SensorPreprocessor  # noqa: E402
from plantomat.utils import ensure_dir, read_json, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Оценка sensor-only модели.')
    parser.add_argument('--csv', type=str, required=True)
    parser.add_argument('--artifact-dir', type=str, required=True)
    parser.add_argument('--output-dir', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    return parser.parse_args()


def save_summary_figure(out_path: Path, mean_error: float, median_error: float, max_error: float, threshold: float, anomaly_rate: float) -> None:
    lines = [
        f'Mean reconstruction error: {mean_error:.6f}',
        f'Median reconstruction error: {median_error:.6f}',
        f'Max reconstruction error: {max_error:.6f}',
        f'Threshold: {threshold:.6f}',
        f'Anomaly rate: {anomaly_rate * 100.0:.2f}%',
    ]

    plt.figure(figsize=(8, 4.5))
    plt.axis('off')
    plt.title('Sensor model summary')
    y = 0.85
    for line in lines:
        plt.text(0.05, y, line, fontsize=12)
        y -= 0.14
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    artifact_dir = Path(args.artifact_dir).resolve()
    device = select_device(args.device)

    checkpoint = torch.load(artifact_dir / 'best_sensor_autoencoder.pt', map_location='cpu')
    preprocessor = SensorPreprocessor.from_state_dict(checkpoint['preprocessor_state'])
    model = SensorAutoencoder(
        input_dim=checkpoint['input_dim'],
        hidden_dim=checkpoint['hidden_dim'],
        bottleneck_dim=checkpoint['bottleneck_dim'],
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    df = pd.read_csv(args.csv)
    x = preprocessor.transform_df(df)
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        recon = model(xt)
        errors = torch.mean((recon - xt) ** 2, dim=1).cpu().numpy()

    summary_prev = read_json(artifact_dir / 'sensor_model_summary.json')
    threshold = float(summary_prev['threshold'])
    anomaly_flags = (errors > threshold).astype(int)

    summary = {
        'rows': int(len(df)),
        'mean_error': float(np.mean(errors)),
        'median_error': float(np.median(errors)),
        'max_error': float(np.max(errors)),
        'threshold': threshold,
        'anomaly_rate': float(np.mean(anomaly_flags)),
        'anomaly_rate_pct': float(np.mean(anomaly_flags) * 100.0),
    }
    write_json(summary, out_dir / 'sensor_eval_summary.json')
    pd.DataFrame(
        {
            'reconstruction_error': errors,
            'anomaly_flag': anomaly_flags,
        }
    ).to_csv(out_dir / 'sensor_eval_scores.csv', index=False)

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=40)
    plt.axvline(threshold, linestyle='--', linewidth=2, label=f'threshold={threshold:.6f}')
    plt.title('Sensor reconstruction error distribution')
    plt.xlabel('Reconstruction error')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / 'sensor_error_hist.png', dpi=150)
    plt.close()

    save_summary_figure(
        out_dir / 'sensor_summary.png',
        mean_error=summary['mean_error'],
        median_error=summary['median_error'],
        max_error=summary['max_error'],
        threshold=summary['threshold'],
        anomaly_rate=summary['anomaly_rate'],
    )

    print(f'Готово: {out_dir}')
    print(f'Anomaly rate: {summary["anomaly_rate_pct"]:.2f}%')
    print('Внимание: это не accuracy, а доля аномальных записей по порогу autoencoder.')


if __name__ == '__main__':
    main()