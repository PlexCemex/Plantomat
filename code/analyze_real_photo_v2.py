#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import IMAGENET_MEAN, IMAGENET_STD, build_image_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Устойчивый инференс по реальным фото листьев томата.")
    parser.add_argument("--image-checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--conf-threshold", type=float, default=0.65)
    parser.add_argument("--margin-threshold", type=float, default=0.12)
    return parser.parse_args()


def build_norm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_temp_crop_path(image_path: str) -> Path:
    temp_dir = PROJECT_ROOT / "results" / "_temp_crops"
    temp_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(image_path).stem
    return temp_dir / f"{stem}_crop.jpg"


def cleanup_temp_crop(crop_path: Path) -> None:
    try:
        if crop_path.exists():
            crop_path.unlink()
        if crop_path.parent.exists() and not any(crop_path.parent.iterdir()):
            crop_path.parent.rmdir()
    except Exception:
        pass


def auto_crop_leaf(image: Image.Image, min_area_ratio: float = 0.05) -> Image.Image:
    rgb = np.array(image.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    lower = np.array([20, 25, 25], dtype=np.uint8)
    upper = np.array([100, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return image

    h, w = mask.shape
    img_area = float(h * w)
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)

    if area / img_area < min_area_ratio:
        return image

    x, y, bw, bh = cv2.boundingRect(cnt)
    pad_x = int(0.12 * bw)
    pad_y = int(0.12 * bh)

    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(w, x + bw + pad_x)
    y2 = min(h, y + bh + pad_y)

    cropped = rgb[y1:y2, x1:x2]
    return Image.fromarray(cropped)


def build_tta_batch(image: Image.Image, image_size: int) -> torch.Tensor:
    norm = build_norm()
    base = image.resize((image_size + 48, image_size + 48))
    w, h = base.size
    s = image_size

    center = base.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    crops = [
        center,
        ImageOps.mirror(center),
        base.crop((0, 0, s, s)),
        base.crop((w - s, 0, w, s)),
        base.crop((0, h - s, s, h)),
        base.crop((w - s, h - s, w, h)),
    ]
    tensors = [norm(c.convert("RGB")) for c in crops]
    return torch.stack(tensors, dim=0)


def run_model(model: torch.nn.Module, image: Image.Image, image_size: int, device: torch.device) -> np.ndarray:
    batch = build_tta_batch(image, image_size).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()
    return probs


def summarize_prediction(class_names: list[str], probs: np.ndarray, topk: int, conf_thr: float, margin_thr: float) -> tuple[str, list[int], str]:
    top_idx = np.argsort(-probs)[:max(1, min(topk, len(probs)))]
    top1 = int(top_idx[0])
    top2 = int(top_idx[1]) if len(top_idx) > 1 else top1

    conf = float(probs[top1])
    margin = float(probs[top1] - probs[top2]) if len(top_idx) > 1 else conf

    note = "уверенно"
    if conf < conf_thr or margin < margin_thr:
        note = "неуверенно"

    healthy_idx = class_names.index("healthy") if "healthy" in class_names else None
    mosaic_idx = class_names.index("tomato_mosaic_virus") if "tomato_mosaic_virus" in class_names else None
    if mosaic_idx is not None and healthy_idx is not None and top1 == mosaic_idx:
        healthy_prob = float(probs[healthy_idx])
        if conf < 0.90 or healthy_prob > 0.20 or margin < 0.18:
            note = "неуверенно"

    return class_names[top1], list(top_idx), note


def main() -> None:
    args = parse_args()
    device = select_device(args.device)
    temp_crop_path = get_temp_crop_path(args.image)

    try:
        ckpt = torch.load(args.image_checkpoint, map_location="cpu")
        class_names = ckpt["class_names"]
        image_size = int(ckpt.get("image_size", 300))
        model = build_image_model(ckpt["backbone"], len(class_names), pretrained=False).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        image = Image.open(args.image).convert("RGB")
        cropped = auto_crop_leaf(image)
        cropped.save(temp_crop_path)

        probs_orig = run_model(model, image, image_size, device)
        probs_crop = run_model(model, cropped, image_size, device)

        probs = 0.35 * probs_orig + 0.65 * probs_crop
        pred_label, top_idx, confidence_note = summarize_prediction(
            class_names, probs, args.topk, args.conf_threshold, args.margin_threshold
        )

        print(f"Изображение: {Path(args.image).resolve()}")
        print(f"Временный crop: {temp_crop_path}")
        print(f"Предсказанный класс: {pred_label}")
        print(f"Статус уверенности: {confidence_note}")
        print("Top-k:")
        for idx in top_idx:
            print(f"  - {class_names[int(idx)]}: {float(probs[int(idx)]):.6f}")

        if confidence_note == "неуверенно":
            print("\nКомментарий:")
            print("  - Результат неуверенный. Лучше сделать более крупный снимок одного листа без водяных знаков и сложного фона.")

    finally:
        cleanup_temp_crop(temp_crop_path)


if __name__ == "__main__":
    main()