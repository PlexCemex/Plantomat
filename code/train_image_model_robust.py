#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import IMAGENET_MEAN, IMAGENET_STD, build_image_model  # noqa: E402
from plantomat.utils import ensure_dir, seed_everything, write_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robust fine-tuning модели по изображениям томата.")
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="efficientnet_b0", choices=["resnet18", "efficientnet_b0"])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--init-checkpoint", type=str, default=None, help="Стартовать с существующего checkpoint.")
    parser.add_argument("--realworld-boost", type=float, default=2.5, help="Во сколько раз усилить реальные фото в sampler.")
    return parser.parse_args()

def resolve_class_names(df: pd.DataFrame, init_checkpoint: str | None) -> list[str]:
    csv_classes = sorted(df["label"].unique().tolist())

    if not init_checkpoint:
        return csv_classes

    payload = torch.load(init_checkpoint, map_location="cpu")
    ckpt_classes = payload.get("class_names")

    if not ckpt_classes:
        return csv_classes

    ckpt_classes = [str(x) for x in ckpt_classes]

    missing_in_ckpt = sorted(set(csv_classes) - set(ckpt_classes))
    if missing_in_ckpt:
        raise RuntimeError(
            f"В текущем CSV есть классы, которых нет в checkpoint: {missing_in_ckpt}"
        )

    absent_in_csv = sorted(set(ckpt_classes) - set(csv_classes))
    if absent_in_csv:
        print("Классы из checkpoint, которых нет в текущем CSV:")
        for cls in absent_in_csv:
            print(f"  - {cls}")

    return ckpt_classes

def build_strong_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.55, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.RandomRotation(degrees=35),
        transforms.RandomPerspective(distortion_scale=0.35, p=0.35),
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08), scale=(0.85, 1.15), shear=12),
        transforms.ColorJitter(brightness=0.28, contrast=0.28, saturation=0.22, hue=0.06),
        transforms.RandomApply([transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 1.8))], p=0.20),
        transforms.RandomAutocontrast(p=0.15),
        transforms.ToTensor(),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value="random"),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(int(image_size * 1.15)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class CSVImageDataset(Dataset):
    def __init__(self, df: pd.DataFrame, class_to_idx: dict[str, int], image_size: int, augment: bool):
        self.df = df.reset_index(drop=True)
        self.class_to_idx = class_to_idx
        self.transform = build_strong_train_transform(image_size) if augment else build_eval_transform(image_size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        image = self.transform(image)
        label = self.class_to_idx[row["label"]]
        return image, torch.tensor(label, dtype=torch.long)


def build_sampler(train_df: pd.DataFrame, class_to_idx: dict[str, int], realworld_boost: float) -> WeightedRandomSampler:
    class_counts = train_df["label"].value_counts().to_dict()
    weights = []
    for _, row in train_df.iterrows():
        w = 1.0 / class_counts[row["label"]]
        if row.get("source", "plantvillage") != "plantvillage":
            w *= realworld_boost
        weights.append(w)
    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights_t, num_samples=len(weights), replacement=True)


def run_epoch(model, loader, criterion, optimizer, device, train: bool, use_amp: bool):
    losses, preds, targets = [], [], []
    model.train(train)
    scaler = getattr(run_epoch, "_scaler", None)
    if scaler is None and use_amp:
        scaler = GradScaler()
        run_epoch._scaler = scaler

    iterator = tqdm(loader, leave=False)
    for images, labels in iterator:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with autocast(enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, labels)

            if train:
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        losses.append(float(loss.item()))
        preds.extend(torch.argmax(logits, dim=1).detach().cpu().tolist())
        targets.extend(labels.detach().cpu().tolist())
        iterator.set_description(f"loss={sum(losses)/max(1, len(losses)):.4f}")

    acc = accuracy_score(targets, preds)
    macro_f1 = f1_score(targets, preds, average="macro")
    return sum(losses) / max(1, len(losses)), acc, macro_f1


def maybe_load_init_checkpoint(model: torch.nn.Module, ckpt_path: str | None) -> None:
    if not ckpt_path:
        return

    payload = torch.load(ckpt_path, map_location="cpu")
    state = payload["model_state_dict"] if isinstance(payload, dict) and "model_state_dict" in payload else payload

    missing, unexpected = model.load_state_dict(state, strict=False)

    print(f"Инициализация из checkpoint: {ckpt_path}")
    print(f"  missing keys: {len(missing)}")
    print(f"  unexpected keys: {len(unexpected)}")

    if missing:
        print("  missing sample:", missing[:10])
    if unexpected:
        print("  unexpected sample:", unexpected[:10])


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    out_dir = ensure_dir(args.output_dir)
    device = select_device(args.device)
    use_amp = device.type == "cuda"

    df = pd.read_csv(args.csv)
    class_names = resolve_class_names(df, args.init_checkpoint)
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "val"].copy()

    train_ds = CSVImageDataset(train_df, class_to_idx, args.image_size, augment=True)
    val_ds = CSVImageDataset(val_df, class_to_idx, args.image_size, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=build_sampler(train_df, class_to_idx, args.realworld_boost),
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = build_image_model(args.backbone, len(class_names), pretrained=args.pretrained).to(device)
    maybe_load_init_checkpoint(model, args.init_checkpoint)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = {"train_loss": [], "train_acc": [], "train_f1": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1 = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device, True, use_amp)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, optimizer, device, False, use_amp)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["train_f1"].append(train_f1)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "backbone": args.backbone,
                    "image_size": args.image_size,
                    "class_names": class_names,
                    "best_val_f1": best_f1,
                },
                out_dir / "best_image_model_robust.pt",
            )

    plt.figure(figsize=(9, 5))
    plt.plot(history["train_f1"], label="train_f1")
    plt.plot(history["val_f1"], label="val_f1")
    plt.plot(history["train_acc"], label="train_acc", linestyle="--")
    plt.plot(history["val_acc"], label="val_acc", linestyle="--")
    plt.legend()
    plt.title("Robust image fine-tuning")
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()

    write_json(
        {
            "class_names": class_names,
            "backbone": args.backbone,
            "image_size": args.image_size,
            "epochs": args.epochs,
            "best_val_f1": best_f1,
            "rows_total": int(len(df)),
            "rows_train": int(len(train_df)),
            "rows_val": int(len(val_df)),
            "rows_test": int((df["split"] == "test").sum()),
            "sources": df["source"].value_counts().to_dict() if "source" in df.columns else {},
            "realworld_boost": args.realworld_boost,
        },
        out_dir / "training_summary_robust.json",
    )

    print(f"Готово: {out_dir}")


if __name__ == "__main__":
    main()
