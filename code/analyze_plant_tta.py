#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import IMAGENET_MEAN, IMAGENET_STD, build_image_model  # noqa: E402
from plantomat.recommendations import analyze_sensor_snapshot  # noqa: E402
from plantomat.sensor_pipeline import SensorAutoencoder, SensorPreprocessor  # noqa: E402
from plantomat.utils import read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Анализ растения: TTA по фото + датчики.")
    parser.add_argument("--image-checkpoint", type=str, required=True)
    parser.add_argument("--sensor-artifact-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sensor-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--ae-threshold-multiplier", type=float, default=6000.0)
    parser.add_argument("--ae-warning-multiplier", type=float, default=2.5)
    return parser.parse_args()


def build_norm():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_tta_batch(image: Image.Image, image_size: int) -> torch.Tensor:
    norm = build_norm()
    big = image.resize((image_size + 48, image_size + 48))
    w, h = big.size
    s = image_size

    crops = [
        big.crop((0, 0, s, s)),
        big.crop((w - s, 0, w, s)),
        big.crop((0, h - s, s, h)),
        big.crop((w - s, h - s, w, h)),
        big.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2)),
    ]
    crops.append(ImageOps.mirror(crops[-1]))
    tensors = [norm(c.convert("RGB")) for c in crops]
    return torch.stack(tensors, dim=0)


def format_image_diagnosis(disease_label: str) -> str:
    if disease_label == "healthy":
        return "По изображению лист выглядит здоровым."
    return f"По изображению обнаружены признаки заболевания: {disease_label}."


def format_sensor_rule_text(rules_state: str) -> str:
    if rules_state == "риск":
        return "По данным датчиков есть отклонения условий выращивания."
    return "По данным датчиков условия выращивания близки к норме."


def interpret_autoencoder_error(error: float, base_threshold: float, threshold_multiplier: float, warning_multiplier: float):
    soft_threshold = base_threshold * threshold_multiplier
    strong_threshold = soft_threshold * warning_multiplier
    if error <= soft_threshold:
        return "обычный профиль", soft_threshold, strong_threshold
    if error <= strong_threshold:
        return "умеренно необычный профиль", soft_threshold, strong_threshold
    return "сильно необычный профиль", soft_threshold, strong_threshold


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    image_ckpt = torch.load(args.image_checkpoint, map_location="cpu")
    class_names = image_ckpt["class_names"]
    image_model = build_image_model(image_ckpt["backbone"], len(class_names), pretrained=False).to(device)
    image_model.load_state_dict(image_ckpt["model_state_dict"])
    image_model.eval()
    image_size = int(image_ckpt.get("image_size", 224))

    image = Image.open(args.image).convert("RGB")
    batch = build_tta_batch(image, image_size).to(device)
    with torch.no_grad():
        logits = image_model(batch)
        probs = torch.softmax(logits, dim=1).mean(dim=0).cpu().numpy()

    top_idx = np.argsort(-probs)[: max(1, min(args.topk, len(probs)))]
    disease_label = class_names[int(top_idx[0])]
    diagnosis_text = format_image_diagnosis(disease_label)

    sensor_dir = Path(args.sensor_artifact_dir).resolve()
    sensor_ckpt = torch.load(sensor_dir / "best_sensor_autoencoder.pt", map_location="cpu")
    preprocessor = SensorPreprocessor.from_state_dict(sensor_ckpt["preprocessor_state"])
    sensor_model = SensorAutoencoder(
        input_dim=sensor_ckpt["input_dim"],
        hidden_dim=sensor_ckpt["hidden_dim"],
        bottleneck_dim=sensor_ckpt["bottleneck_dim"],
    ).to(device)
    sensor_model.load_state_dict(sensor_ckpt["model_state_dict"])
    sensor_model.eval()

    sensor_summary = read_json(sensor_dir / "sensor_model_summary.json")
    base_threshold = float(sensor_summary["threshold"])
    sensor_snapshot = read_json(args.sensor_json)
    x = preprocessor.transform_row(sensor_snapshot)
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)

    with torch.no_grad():
        recon = sensor_model(xt)
        error = torch.mean((recon - xt) ** 2, dim=1).item()

    ae_state, soft_threshold, strong_threshold = interpret_autoencoder_error(
        error, base_threshold, args.ae_threshold_multiplier, args.ae_warning_multiplier
    )

    rules_state, issues, recommendations = analyze_sensor_snapshot(sensor_snapshot, sensor_snapshot.get("growth_stage"))
    sensor_text = format_sensor_rule_text(rules_state)

    print(f"Изображение: {Path(args.image).resolve()}")
    print(f"JSON датчиков: {Path(args.sensor_json).resolve()}")

    print("\nДиагноз по изображению (multi-crop TTA):")
    print(f"  {diagnosis_text}")
    print("  Top-k:")
    for idx in top_idx:
        print(f"    - {class_names[int(idx)]}: {probs[int(idx)]:.6f}")

    print("\nОценка датчиков:")
    print(
        f"  Состояние по автоэнкодеру: {ae_state} "
        f"(ошибка реконструкции: {error:.6f}, базовый порог: {base_threshold:.6f}, "
        f"мягкий порог: {soft_threshold:.6f}, сильный порог: {strong_threshold:.6f})"
    )
    print(f"  Состояние по правилам: {rules_state}")
    if issues:
        print("  Найденные отклонения:")
        for issue in issues:
            print(f"    - {issue}")

    print("\nРекомендации:")
    for rec in recommendations:
        print(f"  - {rec}")

    print("\nИтог:")
    if rules_state == "риск":
        print(f"  {diagnosis_text} {sensor_text}")
    else:
        print(f"  {diagnosis_text} {sensor_text}")


if __name__ == "__main__":
    main()
