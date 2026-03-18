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
    parser = argparse.ArgumentParser(description="Расширенная оценка sensor-модели.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--artifact-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def save_sensor_summary(out_path: Path, summary: dict) -> None:
    lines = [
        f"Rows: {summary['rows']}",
        f"Mean error: {summary['mean_error']:.6f}",
        f"Median error: {summary['median_error']:.6f}",
        f"95th percentile: {summary['p95_error']:.6f}",
        f"99th percentile: {summary['p99_error']:.6f}",
        f"Threshold: {summary['threshold']:.6f}",
        f"Anomaly rate: {summary['anomaly_rate_pct']:.2f}%",
    ]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.axis("off")
    ax.set_title("Сводка по sensor-модели")
    y = 0.88
    for line in lines:
        ax.text(0.05, y, line, fontsize=12)
        y -= 0.12
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    artifact_dir = Path(args.artifact_dir).resolve()
    device = select_device(args.device)

    checkpoint = torch.load(artifact_dir / "best_sensor_autoencoder.pt", map_location="cpu")
    preprocessor = SensorPreprocessor.from_state_dict(checkpoint["preprocessor_state"])
    model = SensorAutoencoder(
        input_dim=checkpoint["input_dim"],
        hidden_dim=checkpoint["hidden_dim"],
        bottleneck_dim=checkpoint["bottleneck_dim"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    df = pd.read_csv(args.csv)
    x = preprocessor.transform_df(df)
    xt = torch.tensor(x, dtype=torch.float32, device=device)

    with torch.no_grad():
        recon = model(xt)
        errors = torch.mean((recon - xt) ** 2, dim=1).cpu().numpy()

    summary_prev = read_json(artifact_dir / "sensor_model_summary.json")
    threshold = float(summary_prev["threshold"])
    anomaly_flags = (errors > threshold).astype(int)

    summary = {
        "rows": int(len(df)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "p95_error": float(np.percentile(errors, 95)),
        "p99_error": float(np.percentile(errors, 99)),
        "max_error": float(np.max(errors)),
        "threshold": threshold,
        "anomaly_rate": float(np.mean(anomaly_flags)),
        "anomaly_rate_pct": float(np.mean(anomaly_flags) * 100.0),
    }
    write_json(summary, out_dir / "sensor_eval_summary.json")

    scores = pd.DataFrame(
        {
            "row_id": np.arange(len(errors)),
            "reconstruction_error": errors,
            "anomaly_flag": anomaly_flags,
        }
    )
    scores.to_csv(out_dir / "sensor_eval_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(errors, bins=40)
    ax.axvline(threshold, linestyle="--", linewidth=2, label=f"threshold={threshold:.6f}")
    ax.set_title("Распределение reconstruction error")
    ax.set_xlabel("Ошибка реконструкции")
    ax.set_ylabel("Количество")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "sensor_error_hist.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    save_sensor_summary(out_dir / "sensor_summary.png", summary)

    summary_md = [
        "# Расширенный отчёт по sensor-модели",
        "",
        f"- Rows: {summary['rows']}",
        f"- Mean error: {summary['mean_error']:.6f}",
        f"- Median error: {summary['median_error']:.6f}",
        f"- P95 error: {summary['p95_error']:.6f}",
        f"- P99 error: {summary['p99_error']:.6f}",
        f"- Threshold: {summary['threshold']:.6f}",
        f"- Anomaly rate: {summary['anomaly_rate_pct']:.2f}%",
        "",
        "Это не accuracy-классификатор. Здесь оценивается степень отклонения записи от распределения обучающего сенсорного набора.",
    ]
    (out_dir / "sensor_summary_report.md").write_text("\n".join(summary_md), encoding="utf-8")

    print(f"Готово: {out_dir}")
    print(f"Anomaly rate: {summary['anomaly_rate_pct']:.2f}%")
    print(f"Scores: {out_dir / 'sensor_eval_scores.csv'}")


if __name__ == "__main__":
    main()
