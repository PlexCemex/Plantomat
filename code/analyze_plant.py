#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.device import select_device  # noqa: E402
from plantomat.image_pipeline import build_eval_transform, build_image_model  # noqa: E402
from plantomat.recommendations import analyze_sensor_snapshot  # noqa: E402
from plantomat.sensor_pipeline import SensorAutoencoder, SensorPreprocessor  # noqa: E402
from plantomat.utils import read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Раздельный анализ: болезнь по фото + состояние среды по датчикам."
    )
    parser.add_argument("--image-checkpoint", type=str, required=True)
    parser.add_argument("--sensor-artifact-dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--sensor-json", type=str, required=True)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--topk", type=int, default=3)

    # Новый мягкий режим для автоэнкодера
    parser.add_argument(
        "--ae-threshold-multiplier",
        type=float,
        default=6000.0,
        help="Во сколько раз увеличить исходный порог автоэнкодера для мягкой интерпретации.",
    )
    parser.add_argument(
        "--ae-warning-multiplier",
        type=float,
        default=2.5,
        help="Во сколько раз выше мягкого порога считать профиль уже сильно необычным.",
    )
    return parser.parse_args()


def format_image_diagnosis(disease_label: str) -> str:
    if disease_label == "healthy":
        return "По изображению лист выглядит здоровым."
    return f"По изображению обнаружены признаки заболевания: {disease_label}."


def format_sensor_rule_text(rules_state: str) -> str:
    if rules_state == "риск":
        return "По данным датчиков есть отклонения условий выращивания."
    return "По данным датчиков условия выращивания близки к норме."


def interpret_autoencoder_error(
    error: float,
    base_threshold: float,
    threshold_multiplier: float,
    warning_multiplier: float,
) -> tuple[str, float, float]:
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

    # -------- image model --------
    image_ckpt = torch.load(args.image_checkpoint, map_location="cpu")
    class_names = image_ckpt["class_names"]
    image_model = build_image_model(
        image_ckpt["backbone"],
        len(class_names),
        pretrained=False,
    ).to(device)
    image_model.load_state_dict(image_ckpt["model_state_dict"])
    image_model.eval()
    image_size = int(image_ckpt.get("image_size", 224))

    image = Image.open(args.image).convert("RGB")
    image_tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = image_model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    top_idx = np.argsort(-probs)[: max(1, min(args.topk, len(probs)))]
    disease_label = class_names[int(top_idx[0])]
    diagnosis_text = format_image_diagnosis(disease_label)

    # -------- sensor model --------
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
        error=error,
        base_threshold=base_threshold,
        threshold_multiplier=args.ae_threshold_multiplier,
        warning_multiplier=args.ae_warning_multiplier,
    )

    # -------- rules-based sensor analysis --------
    rules_state, issues, recommendations = analyze_sensor_snapshot(
        sensor_snapshot,
        sensor_snapshot.get("growth_stage"),
    )
    sensor_text = format_sensor_rule_text(rules_state)

    # -------- console output --------
    print(f"Изображение: {Path(args.image).resolve()}")
    print(f"JSON датчиков: {Path(args.sensor_json).resolve()}")

    print("\nДиагноз по изображению:")
    print(f"  {diagnosis_text}")
    print("  Top-k:")
    for idx in top_idx:
        print(f"    - {class_names[int(idx)]}: {probs[int(idx)]:.6f}")

    print("\nОценка датчиков:")
    print(
        f"  Состояние по автоэнкодеру: {ae_state} "
        f"(ошибка реконструкции: {error:.6f}, "
        f"базовый порог: {base_threshold:.6f}, "
        f"мягкий порог: {soft_threshold:.6f}, "
        f"сильный порог: {strong_threshold:.6f})"
    )
    print(f"  Состояние по правилам: {rules_state}")

    if ae_state in {"умеренно необычный профиль", "сильно необычный профиль"} and rules_state == "норма":
        print(
            "  Примечание: профиль датчиков отличается от обучающей выборки автоэнкодера, "
            "но по правилам критичных отклонений не найдено."
        )

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
        if ae_state == "сильно необычный профиль":
            print(
                f"  {diagnosis_text} {sensor_text} "
                "Дополнительно: сенсорный профиль заметно отличается от обучающей выборки."
            )
        else:
            print(f"  {diagnosis_text} {sensor_text}")


if __name__ == "__main__":
    main()