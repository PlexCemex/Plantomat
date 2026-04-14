#!/usr/bin/env python3

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import build_eval_transform, build_image_model  # noqa: E402
from plantomat.utils import ensure_dir, write_json  # noqa: E402


class EvalImageDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        class_to_idx: dict[str, int],
        image_size: int,
        eval_mode: str,
        eval_crop_scale: float,
    ):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = build_eval_transform(image_size, mode=eval_mode, crop_scale=eval_crop_scale)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        source = str(row["source"]) if "source" in row.index else "unknown"
        return image, torch.tensor(label, dtype=torch.long), row["image_path"], row["label"], source


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
    parser.add_argument("--source", type=str, default="all", help="Оценивать только указанный source из CSV. По умолчанию all.")
    parser.add_argument(
        "--eval-mode",
        type=str,
        default="auto",
        choices=["auto", "resize", "center-crop"],
        help="Режим eval-preprocessing. Для robust checkpoint обычно нужен center-crop.",
    )
    parser.add_argument("--eval-crop-scale", type=float, default=1.15)
    parser.add_argument("--representative-per-class", type=int, default=0)
    parser.add_argument(
        "--representative-source",
        type=str,
        default="all",
        help="Источник для representative examples. По умолчанию all.",
    )
    parser.add_argument(
        "--balanced-count-per-class",
        type=int,
        default=100,
        help="Дополнительная матрица ошибок в абсолютных значениях: ровно N изображений на класс.",
    )
    parser.add_argument(
        "--relative-percent-per-class",
        type=float,
        default=10.0,
        help="Дополнительная нормированная матрица: взять X%% каждого класса и показать результат в процентах.",
    )
    parser.add_argument("--sampling-seed", type=int, default=42)
    return parser.parse_args()


def resolve_eval_mode(args: argparse.Namespace, checkpoint: dict) -> str:
    if args.eval_mode != "auto":
        return args.eval_mode

    ckpt_mode = str(checkpoint.get("eval_transform_mode", "")).strip().lower()
    if ckpt_mode in {"resize", "center-crop", "center_crop", "crop"}:
        return "center-crop" if ckpt_mode in {"center-crop", "center_crop", "crop"} else "resize"

    hint = " ".join(
        [
            str(args.checkpoint),
            str(checkpoint.get("training_script", "")),
            str(checkpoint.get("recipe", "")),
            str(checkpoint.get("notes", "")),
        ]
    ).lower()
    if any(token in hint for token in ["robust", "stage2", "stage3", "stage4", "polish", "realworld"]):
        return "center-crop"
    return "resize"


def save_confusion_counts(
    cm: np.ndarray,
    class_names: list[str],
    supports: list[int],
    out_path: Path,
    accuracy: float,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title(f"{title} | accuracy={accuracy * 100:.2f}%")
    fig.colorbar(im, ax=ax)

    ticklabels = [f"{name}\n(n={supports[i]})" for i, name in enumerate(class_names)]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    max_v = cm.max() if cm.size else 0
    thresh = max_v / 2.0 if max_v > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = int(cm[i, j])
            ax.text(j, i, str(value), ha="center", va="center", color="white" if value > thresh else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_confusion_relative(
    cm: np.ndarray,
    class_names: list[str],
    supports: list[int],
    out_path: Path,
    title: str,
) -> np.ndarray:
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm = (cm / row_sums) * 100.0

    fig, ax = plt.subplots(figsize=(14, 11))
    im = ax.imshow(cm_norm, interpolation="nearest", vmin=0, vmax=100)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)

    ticklabels = [f"{name}\n(n={supports[i]})" for i, name in enumerate(class_names)]
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(ticklabels, rotation=90)
    ax.set_yticklabels(ticklabels)
    ax.set_xlabel("Предсказанный класс")
    ax.set_ylabel("Истинный класс")

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            value = float(cm_norm[i, j])
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color="white" if value >= 50 else "black", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return cm_norm


def save_summary_metrics(out_path: Path, accuracy: float, precision: float, recall: float, f1: float) -> None:
    names = ["Accuracy", "Macro Precision", "Macro Recall", "Macro F1"]
    values = [accuracy * 100.0, precision * 100.0, recall * 100.0, f1 * 100.0]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(names, values)
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Сводные метрики")
    for i, value in enumerate(values):
        ax.text(i, min(99.0, value + 1.0), f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_per_class_metrics(out_path: Path, class_names: list[str], report: dict) -> None:
    prec = [report[name]["precision"] * 100.0 for name in class_names]
    rec = [report[name]["recall"] * 100.0 for name in class_names]
    f1 = [report[name]["f1-score"] * 100.0 for name in class_names]

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(16, 7))
    ax.bar(x - width, prec, width=width, label="Precision")
    ax.bar(x, rec, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.set_title("Метрики по классам")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def save_top_confusions(out_path: Path, cm: np.ndarray, class_names: list[str], top_n: int = 10) -> list[dict[str, int | str]]:
    rows: list[dict[str, int | str]] = []
    for i, true_name in enumerate(class_names):
        for j, pred_name in enumerate(class_names):
            if i == j:
                continue
            count = int(cm[i, j])
            if count <= 0:
                continue
            rows.append({"true_class": true_name, "pred_class": pred_name, "count": count})

    rows.sort(key=lambda item: int(item["count"]), reverse=True)
    top_rows = rows[:top_n]

    fig, ax = plt.subplots(figsize=(12, 6))
    if top_rows:
        labels = [f"{row['true_class']} → {row['pred_class']}" for row in top_rows]
        values = [int(row["count"]) for row in top_rows]
        ax.barh(labels[::-1], values[::-1])
        ax.set_xlabel("Количество")
        ax.set_title("Топ перепутываний")
    else:
        ax.text(0.5, 0.5, "Ошибочных пар не найдено", ha="center", va="center")
        ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)

    return top_rows


def compute_source_metrics(pred_df: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    if "source" not in pred_df.columns or pred_df.empty:
        return pd.DataFrame()

    labels = list(range(len(class_names)))
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    rows = []
    for source_name, chunk in pred_df.groupby("source"):
        targets = [class_to_idx[x] for x in chunk["true_label"].astype(str)]
        preds = [class_to_idx[x] for x in chunk["pred_label"].astype(str)]
        accuracy = accuracy_score(targets, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            targets,
            preds,
            labels=labels,
            average="macro",
            zero_division=0,
        )
        rows.append(
            {
                "source": str(source_name),
                "rows": int(len(chunk)),
                "accuracy": float(accuracy),
                "accuracy_pct": float(accuracy * 100.0),
                "macro_precision": float(precision),
                "macro_precision_pct": float(precision * 100.0),
                "macro_recall": float(recall),
                "macro_recall_pct": float(recall * 100.0),
                "macro_f1": float(f1),
                "macro_f1_pct": float(f1 * 100.0),
            }
        )
    return pd.DataFrame(rows).sort_values(["macro_f1_pct", "accuracy_pct", "rows"], ascending=[False, False, False])


def build_representative_examples(
    pred_df: pd.DataFrame,
    class_names: list[str],
    representative_per_class: int,
    representative_source: str,
) -> pd.DataFrame:
    if representative_per_class <= 0 or pred_df.empty:
        return pd.DataFrame()

    rep_df = pred_df.copy()
    if representative_source != "all":
        if "source" not in rep_df.columns:
            raise RuntimeError("В CSV нет колонки source, поэтому representative-source использовать нельзя.")
        rep_df = rep_df[rep_df["source"] == representative_source].copy()

    result_parts = []
    for class_name in class_names:
        chunk = rep_df[rep_df["true_label"] == class_name].copy()
        if chunk.empty:
            continue
        chunk = chunk.sort_values(["is_correct", "confidence", "margin"], ascending=[False, False, False])
        chunk.insert(0, "representative_class", class_name)
        result_parts.append(chunk.head(representative_per_class))

    if not result_parts:
        return pd.DataFrame()
    return pd.concat(result_parts, ignore_index=True)


def build_confusion_from_pred_df(pred_df: pd.DataFrame, class_names: list[str]) -> tuple[np.ndarray, list[int], float]:
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    targets = [class_to_idx[x] for x in pred_df["true_label"].astype(str)]
    preds = [class_to_idx[x] for x in pred_df["pred_label"].astype(str)]
    labels_idx = list(range(len(class_names)))
    cm = confusion_matrix(targets, preds, labels=labels_idx)
    supports = [int((np.array(targets) == idx).sum()) for idx in labels_idx]
    accuracy = accuracy_score(targets, preds) if len(targets) > 0 else 0.0
    return cm, supports, float(accuracy)


def sample_fixed_count_per_class(
    pred_df: pd.DataFrame,
    class_names: list[str],
    per_class: int,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, int | str | bool]]]:
    if per_class <= 0:
        return pd.DataFrame(), []

    sampled_parts = []
    notes: list[dict[str, int | str | bool]] = []
    for idx, class_name in enumerate(class_names):
        chunk = pred_df[pred_df["true_label"] == class_name].copy()
        if chunk.empty:
            continue
        replace = len(chunk) < per_class
        sampled = chunk.sample(n=per_class, replace=replace, random_state=seed + idx)
        sampled = sampled.copy()
        sampled.insert(0, "sampling_rule", f"fixed_{per_class}_per_class")
        sampled_parts.append(sampled)
        notes.append(
            {
                "class_name": class_name,
                "source_rows": int(len(chunk)),
                "sampled_rows": int(len(sampled)),
                "with_replacement": bool(replace),
            }
        )

    if not sampled_parts:
        return pd.DataFrame(), notes
    return pd.concat(sampled_parts, ignore_index=True), notes


def sample_percent_per_class(
    pred_df: pd.DataFrame,
    class_names: list[str],
    percent: float,
    seed: int,
) -> tuple[pd.DataFrame, list[dict[str, int | str]]]:
    if percent <= 0:
        return pd.DataFrame(), []

    sampled_parts = []
    notes: list[dict[str, int | str]] = []
    fraction = percent / 100.0
    for idx, class_name in enumerate(class_names):
        chunk = pred_df[pred_df["true_label"] == class_name].copy()
        if chunk.empty:
            continue
        n_take = max(1, int(round(len(chunk) * fraction)))
        n_take = min(n_take, len(chunk))
        sampled = chunk.sample(n=n_take, replace=False, random_state=seed + 1000 + idx)
        sampled = sampled.copy()
        sampled.insert(0, "sampling_rule", f"relative_{percent:g}_percent_per_class")
        sampled_parts.append(sampled)
        notes.append(
            {
                "class_name": class_name,
                "source_rows": int(len(chunk)),
                "sampled_rows": int(len(sampled)),
                "percent_requested": float(percent),
            }
        )

    if not sampled_parts:
        return pd.DataFrame(), notes
    return pd.concat(sampled_parts, ignore_index=True), notes


def write_matrix_csv(out_path: Path, matrix: np.ndarray, class_names: list[str], value_name: str) -> None:
    df = pd.DataFrame(matrix, index=class_names, columns=class_names)
    df.index.name = f"true_label_{value_name}"
    df.to_csv(out_path, encoding="utf-8")


def write_text_report(
    out_path: Path,
    split_name: str,
    source_name: str,
    eval_mode: str,
    summary: dict,
    report: dict,
    top_confusions: list[dict[str, int | str]],
    supports: dict[str, int],
    source_metrics_df: pd.DataFrame,
    representative_count: int,
    balanced_count_per_class: int,
    relative_percent_per_class: float,
    balanced_notes: list[dict[str, int | str | bool]],
    relative_notes: list[dict[str, int | str]],
) -> None:
    lines = []
    lines.append(f"# Отчёт по image-модели ({split_name})")
    lines.append("")
    lines.append(f"- Split: `{split_name}`")
    lines.append(f"- Source filter: `{source_name}`")
    lines.append(f"- Eval mode: `{eval_mode}`")
    lines.append(f"- Accuracy: {summary['accuracy_pct']:.2f}%")
    lines.append(f"- Macro Precision: {summary['macro_precision_pct']:.2f}%")
    lines.append(f"- Macro Recall: {summary['macro_recall_pct']:.2f}%")
    lines.append(f"- Macro F1: {summary['macro_f1_pct']:.2f}%")
    if representative_count > 0:
        lines.append(f"- Representative examples per class: {representative_count}")
    if balanced_count_per_class > 0:
        lines.append(f"- Доп. матрица 1: ровно {balanced_count_per_class} изображений на класс, абсолютные значения")
    if relative_percent_per_class > 0:
        lines.append(f"- Доп. матрица 2: {relative_percent_per_class:g}% каждого класса, относительные значения")
    lines.append("")

    lines.append("## Метрики по классам")
    lines.append("")
    for class_name, support in supports.items():
        cls = report[class_name]
        lines.append(
            f"- {class_name}: support={support}, precision={cls['precision']:.4f}, "
            f"recall={cls['recall']:.4f}, f1={cls['f1-score']:.4f}"
        )
    lines.append("")

    lines.append("## Топ перепутываний")
    lines.append("")
    if top_confusions:
        for row in top_confusions:
            lines.append(f"- {row['true_class']} → {row['pred_class']}: {row['count']}")
    else:
        lines.append("- Ошибочных пар не найдено.")
    lines.append("")

    if balanced_notes:
        lines.append("## Сэмплирование для матрицы 100 на класс")
        lines.append("")
        for row in balanced_notes:
            lines.append(
                f"- {row['class_name']}: source_rows={row['source_rows']}, sampled_rows={row['sampled_rows']}, "
                f"with_replacement={row['with_replacement']}"
            )
        lines.append("")

    if relative_notes:
        lines.append("## Сэмплирование для матрицы 10% на класс")
        lines.append("")
        for row in relative_notes:
            lines.append(
                f"- {row['class_name']}: source_rows={row['source_rows']}, sampled_rows={row['sampled_rows']}, "
                f"percent_requested={row['percent_requested']}"
            )
        lines.append("")

    if not source_metrics_df.empty:
        lines.append("## Метрики по источникам")
        lines.append("")
        for _, row in source_metrics_df.iterrows():
            lines.append(
                f"- {row['source']}: rows={int(row['rows'])}, accuracy={row['accuracy_pct']:.2f}%, "
                f"macro_f1={row['macro_f1_pct']:.2f}%"
            )
        lines.append("")

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
    eval_mode = resolve_eval_mode(args, checkpoint)

    model = build_image_model(checkpoint["backbone"], num_classes=len(class_names), pretrained=False).to(device)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    split_df = df[df["split"] == args.split].copy()
    if split_df.empty:
        raise RuntimeError(f"В CSV нет строк со split={args.split!r}.")

    if args.source != "all":
        if "source" not in split_df.columns:
            raise RuntimeError("В CSV нет колонки source, поэтому --source использовать нельзя.")
        split_df = split_df[split_df["source"].astype(str) == args.source].copy()
        if split_df.empty:
            raise RuntimeError(f"После фильтрации source={args.source!r} не осталось ни одной строки.")

    ds = EvalImageDataset(
        split_df,
        class_to_idx,
        image_size=image_size,
        eval_mode=eval_mode,
        eval_crop_scale=args.eval_crop_scale,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    preds: list[int] = []
    targets: list[int] = []
    pred_rows: list[dict] = []

    with torch.no_grad():
        for images, labels, paths, true_labels, sources in tqdm(loader):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            topk = torch.topk(probs, k=min(2, probs.shape[1]), dim=1)
            pred_idx = topk.indices[:, 0].cpu().tolist()
            pred_conf = topk.values[:, 0].cpu().tolist()
            if topk.values.shape[1] > 1:
                pred_margin = (topk.values[:, 0] - topk.values[:, 1]).cpu().tolist()
            else:
                pred_margin = topk.values[:, 0].cpu().tolist()

            preds.extend(pred_idx)
            targets.extend(labels.tolist())

            for path, true_label, source, p_idx, conf, margin in zip(paths, true_labels, sources, pred_idx, pred_conf, pred_margin):
                pred_label = class_names[int(p_idx)]
                pred_rows.append(
                    {
                        "image_path": str(path),
                        "source": str(source),
                        "true_label": str(true_label),
                        "pred_label": pred_label,
                        "confidence": float(conf),
                        "margin": float(margin),
                        "is_correct": int(pred_label == str(true_label)),
                    }
                )

    labels_idx = list(range(len(class_names)))
    supports = {name: int((np.array(targets) == idx).sum()) for idx, name in enumerate(class_names)}
    accuracy = accuracy_score(targets, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        targets,
        preds,
        labels=labels_idx,
        average="macro",
        zero_division=0,
    )
    report = classification_report(
        targets,
        preds,
        labels=labels_idx,
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
    report["run_config"] = {
        "csv": str(Path(args.csv).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "split": args.split,
        "source": args.source,
        "eval_mode": eval_mode,
        "eval_crop_scale": float(args.eval_crop_scale),
        "image_size": image_size,
        "rows_evaluated": int(len(split_df)),
        "balanced_count_per_class": int(args.balanced_count_per_class),
        "relative_percent_per_class": float(args.relative_percent_per_class),
        "sampling_seed": int(args.sampling_seed),
    }
    write_json(report, out_dir / f"{args.split}_classification_report.json")

    pred_df = pd.DataFrame(pred_rows)
    pred_df.to_csv(out_dir / f"{args.split}_predictions.csv", index=False)

    errors_df = pred_df[pred_df["is_correct"] == 0].sort_values(
        ["true_label", "pred_label", "confidence"],
        ascending=[True, True, False],
    )
    errors_df.head(args.max_error_examples).to_csv(out_dir / f"{args.split}_top_errors.csv", index=False)

    representative_df = build_representative_examples(
        pred_df=pred_df,
        class_names=class_names,
        representative_per_class=args.representative_per_class,
        representative_source=args.representative_source,
    )
    if not representative_df.empty:
        representative_df.to_csv(out_dir / f"{args.split}_representative_examples.csv", index=False)

    source_metrics_df = compute_source_metrics(pred_df, class_names)
    if not source_metrics_df.empty:
        source_metrics_df.to_csv(out_dir / f"{args.split}_source_metrics.csv", index=False)

    cm = confusion_matrix(targets, preds, labels=labels_idx)
    support_list = [supports[name] for name in class_names]

    save_confusion_counts(
        cm,
        class_names,
        support_list,
        out_dir / f"{args.split}_confusion_matrix_counts.png",
        accuracy,
        title="Матрица ошибок (полный тест, абсолютные значения)",
    )
    cm_norm_full = save_confusion_relative(
        cm,
        class_names,
        support_list,
        out_dir / f"{args.split}_confusion_matrix_normalized.png",
        title="Нормированная матрица ошибок (полный тест, %)"
    )
    write_matrix_csv(out_dir / f"{args.split}_confusion_matrix_counts.csv", cm, class_names, "counts")
    write_matrix_csv(out_dir / f"{args.split}_confusion_matrix_normalized.csv", cm_norm_full, class_names, "percent")

    balanced_df, balanced_notes = sample_fixed_count_per_class(
        pred_df=pred_df,
        class_names=class_names,
        per_class=args.balanced_count_per_class,
        seed=args.sampling_seed,
    )
    if not balanced_df.empty:
        balanced_df.to_csv(out_dir / f"{args.split}_balanced_{args.balanced_count_per_class}_predictions.csv", index=False)
        cm_balanced, balanced_supports, balanced_accuracy = build_confusion_from_pred_df(balanced_df, class_names)
        save_confusion_counts(
            cm_balanced,
            class_names,
            balanced_supports,
            out_dir / f"{args.split}_balanced_{args.balanced_count_per_class}_confusion_counts.png",
            balanced_accuracy,
            title=f"Матрица ошибок ({args.balanced_count_per_class} изображений на класс, абсолютные значения)",
        )
        write_matrix_csv(
            out_dir / f"{args.split}_balanced_{args.balanced_count_per_class}_confusion_counts.csv",
            cm_balanced,
            class_names,
            "counts",
        )

    relative_df, relative_notes = sample_percent_per_class(
        pred_df=pred_df,
        class_names=class_names,
        percent=args.relative_percent_per_class,
        seed=args.sampling_seed,
    )
    if not relative_df.empty:
        relative_df.to_csv(out_dir / f"{args.split}_relative_{str(args.relative_percent_per_class).replace('.', '_')}_predictions.csv", index=False)
        cm_relative, relative_supports, relative_accuracy = build_confusion_from_pred_df(relative_df, class_names)
        cm_relative_norm = save_confusion_relative(
            cm_relative,
            class_names,
            relative_supports,
            out_dir / f"{args.split}_relative_{str(args.relative_percent_per_class).replace('.', '_')}_confusion_percent.png",
            title=(
                f"Нормированная матрица ошибок ({args.relative_percent_per_class:g}% каждого класса, %)")
        )
        write_matrix_csv(
            out_dir / f"{args.split}_relative_{str(args.relative_percent_per_class).replace('.', '_')}_confusion_percent.csv",
            cm_relative_norm,
            class_names,
            "percent",
        )
        write_json(
            {
                "accuracy": float(relative_accuracy),
                "accuracy_pct": float(relative_accuracy * 100.0),
                "sample_rows": int(len(relative_df)),
                "relative_percent_per_class": float(args.relative_percent_per_class),
            },
            out_dir / f"{args.split}_relative_{str(args.relative_percent_per_class).replace('.', '_')}_summary.json",
        )

    save_summary_metrics(out_dir / f"{args.split}_metrics.png", accuracy, precision, recall, f1)
    save_per_class_metrics(out_dir / f"{args.split}_per_class_metrics.png", class_names, report)
    top_confusions = save_top_confusions(out_dir / f"{args.split}_top_confusions.png", cm, class_names, top_n=10)

    write_text_report(
        out_dir / f"{args.split}_summary_report.md",
        args.split,
        args.source,
        eval_mode,
        report["summary_metrics"],
        report,
        top_confusions,
        supports,
        source_metrics_df,
        args.representative_per_class,
        args.balanced_count_per_class,
        args.relative_percent_per_class,
        balanced_notes,
        relative_notes,
    )

    write_json(
        {
            "balanced_count_per_class": int(args.balanced_count_per_class),
            "relative_percent_per_class": float(args.relative_percent_per_class),
            "balanced_sampling": balanced_notes,
            "relative_sampling": relative_notes,
        },
        out_dir / f"{args.split}_sampling_metadata.json",
    )

    print(f"Готово: {out_dir}")
    print(f"Оценено изображений: {len(split_df)}")
    print(f"Eval mode: {eval_mode}")
    print(f"Accuracy: {accuracy * 100.0:.2f}%")
    print(f"Macro Precision: {precision * 100.0:.2f}%")
    print(f"Macro Recall: {recall * 100.0:.2f}%")
    print(f"Macro F1: {f1 * 100.0:.2f}%")
    if args.source != "all":
        print(f"Source filter: {args.source}")
    if args.balanced_count_per_class > 0:
        print(f"Доп. матрица 1: {args.balanced_count_per_class} изображений на класс, абсолютные значения")
    if args.relative_percent_per_class > 0:
        print(f"Доп. матрица 2: {args.relative_percent_per_class:g}% каждого класса, относительные значения")


if __name__ == "__main__":
    main()
