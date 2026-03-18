#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import build_eval_transform, build_image_model  # noqa: E402
from plantomat.utils import ensure_dir, write_json  # noqa: E402


class EvalImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], image_size: int):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, torch.tensor(label, dtype=torch.long), row["image_path"], row["label"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Расширенная оценка image-модели.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--max-error-examples", type=int, default=24)
    return parser.parse_args()


def save_confusion_counts(cm: np.ndarray, class_names: list[str], supports: list[int], out_path: Path, accuracy: float, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"Матрица ошибок ({title_suffix}) | accuracy={accuracy * 100:.2f}%")
    fig.colorbar(im, ax=ax)

    xticklabels = [f"{name}\n(n={supports[i]})" for i, name in enumerate(class_names)]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(xticklabels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    max_v = cm.max() if cm.size else 0
    thresh = max_v / 2.0 if max_v > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = int(cm[i, j])
            text = str(v)
            ax.text(j, i, text, ha="center", va="center", color="white" if v > thresh else "black", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_confusion_normalized(cm: np.ndarray, class_names: list[str], supports: list[int], out_path: Path) -> None:
    cm_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, np.maximum(cm_sum, 1), where=np.maximum(cm_sum, 1) != 0)

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title("Нормированная матрица ошибок")
    fig.colorbar(im, ax=ax)

    xticklabels = [f"{name}\n(n={supports[i]})" for i, name in enumerate(class_names)]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(xticklabels, rotation=90)
    ax.set_yticklabels(xticklabels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    thresh = 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            v = float(cm_norm[i, j])
            text = f"{v * 100:.1f}%"
            ax.text(j, i, text, ha="center", va="center", color="white" if v > thresh else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_summary_metrics(out_path: Path, accuracy: float, precision: float, recall: float, f1: float) -> None:
    labels = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    values = [accuracy * 100.0, precision * 100.0, recall * 100.0, f1 * 100.0]

    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Проценты")
    ax.set_title("Сводные метрики модели")

    for bar, v in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.2f}%", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_per_class_metrics(out_path: Path, class_names: list[str], report: dict) -> None:
    prec = [report[name]["precision"] * 100.0 for name in class_names]
    rec = [report[name]["recall"] * 100.0 for name in class_names]
    f1 = [report[name]["f1-score"] * 100.0 for name in class_names]
    x = np.arange(len(class_names))
    w = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - w, prec, width=w, label="Precision")
    ax.bar(x, rec, width=w, label="Recall")
    ax.bar(x + w, f1, width=w, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Проценты")
    ax.set_title("Метрики по классам")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_top_confusions(out_path: Path, cm: np.ndarray, class_names: list[str], top_n: int = 10) -> list[dict]:
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j:
                continue
            v = int(cm[i, j])
            if v > 0:
                rows.append({
                    "true_class": class_names[i],
                    "pred_class": class_names[j],
                    "count": v,
                })
    rows = sorted(rows, key=lambda x: x["count"], reverse=True)[:top_n]

    if not rows:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis("off")
        ax.set_title("Топ перепутываний")
        ax.text(0.05, 0.5, "Ошибочных пар не найдено.", fontsize=12)
        plt.tight_layout()
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        return rows

    labels = [f'{r["true_class"]} → {r["pred_class"]}' for r in rows]
    counts = [r["count"] for r in rows]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.barh(range(len(rows))[::-1], counts)
    ax.set_yticks(range(len(rows))[::-1])
    ax.set_yticklabels(labels)
    ax.set_xlabel("Количество")
    ax.set_title("Топ перепутываний классов")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return rows


def write_text_report(out_path: Path, split: str, summary: dict, report: dict, top_confusions: list[dict], supports: dict[str, int]) -> None:
    lines = []
    lines.append(f"# Расширенный отчёт по image-модели ({split})")
    lines.append("")
    lines.append("## Сводные метрики")
    lines.append(f"- Accuracy: {summary['accuracy_pct']:.2f}%")
    lines.append(f"- Macro Precision: {summary['macro_precision_pct']:.2f}%")
    lines.append(f"- Macro Recall: {summary['macro_recall_pct']:.2f}%")
    lines.append(f"- Macro F1: {summary['macro_f1_pct']:.2f}%")
    lines.append("")
    lines.append("## Метрики по классам")
    for class_name, support in supports.items():
        cls = report[class_name]
        lines.append(
            f"- {class_name}: support={support}, precision={cls['precision']:.4f}, "
            f"recall={cls['recall']:.4f}, f1={cls['f1-score']:.4f}"
        )
    lines.append("")
    lines.append("## Топ перепутываний")
    if top_confusions:
        for row in top_confusions:
            lines.append(f"- {row['true_class']} → {row['pred_class']}: {row['count']}")
    else:
        lines.append("- Ошибочных пар не найдено.")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)

    df = pd.read_csv(args.csv)
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = [str(x) for x in checkpoint["class_names"]]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    image_size = int(checkpoint.get("image_size", 224))

    model = build_image_model(checkpoint["backbone"], num_classes=len(class_names), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    split_df = df[df["split"] == args.split].copy()
    ds = EvalImageDataset(split_df, class_to_idx, image_size=image_size)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    preds, targets = [], []
    pred_rows = []
    with torch.no_grad():
        for images, labels, paths, true_labels in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            prob = torch.softmax(logits, dim=1)
            pred_idx = torch.argmax(prob, dim=1).cpu().tolist()
            pred_conf = prob.max(dim=1).values.cpu().tolist()

            preds.extend(pred_idx)
            targets.extend(labels.tolist())

            for path, true_label, p_idx, conf in zip(paths, true_labels, pred_idx, pred_conf):
                pred_rows.append({
                    "image_path": str(path),
                    "true_label": str(true_label),
                    "pred_label": class_names[int(p_idx)],
                    "confidence": float(conf),
                    "is_correct": int(class_names[int(p_idx)] == str(true_label)),
                })

    labels = list(range(len(class_names)))
    supports = {name: int((np.array(targets) == idx).sum()) for idx, name in enumerate(class_names)}

    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels,
        average="macro",
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
    report["summary_metrics"] = {
        "accuracy": float(accuracy),
        "accuracy_pct": float(accuracy * 100.0),
        "macro_precision": float(precision),
        "macro_precision_pct": float(precision * 100.0),
        "macro_recall": float(recall),
        "macro_recall_pct": float(recall * 100.0),
        "macro_f1": float(f1),
        "macro_f1_pct": float(f1 * 100.0),
    }
    write_json(report, out_dir / f"{args.split}_classification_report.json")

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)

    errors_df = pred_df[pred_df["is_correct"] == 0].sort_values(["true_label", "pred_label", "confidence"], ascending=[True, True, False])
    errors_df.head(args.max_error_examples).to_csv(out_dir / f"{args.split}_top_errors.csv", index=False)

    cm = confusion_matrix(targets, preds, labels=labels)
    support_list = [supports[name] for name in class_names]

    save_confusion_counts(cm, class_names, support_list, out_dir / f"{args.split}_confusion_matrix_counts.png", accuracy, title_suffix="с количествами")
    save_confusion_normalized(cm, class_names, support_list, out_dir / f"{args.split}_confusion_matrix_normalized.png")
    save_summary_metrics(out_dir / f"{args.split}_metrics.png", accuracy, precision, recall, f1)
    save_per_class_metrics(out_dir / f"{args.split}_per_class_metrics.png", class_names, report)
    top_confusions = save_top_confusions(out_dir / f"{args.split}_top_confusions.png", cm, class_names, top_n=10)
    write_text_report(
        out_dir / f"{args.split}_summary_report.md",
        args.split,
        report["summary_metrics"],
        report,
        top_confusions,
        supports,
    )

    print(f"Готово: {out_dir}")
    print(f"Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Macro Precision: {precision * 100.0:.2f}%")
    print(f"Macro Recall: {recall * 100.0:.2f}%")
    print(f"Macro F1: {f1 * 100.0:.2f}%")
    print(f"Предсказания: {out_dir / f'{args.split}_predictions.csv'}")
    print(f'Ошибки: {out_dir / f"{args.split}_top_errors.csv"}')


if __name__ == "__main__":
    main()
