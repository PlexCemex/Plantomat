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
    parser = argparse.ArgumentParser(description='Раздельный анализ: болезнь по фото + состояние среды по датчикам.')
    parser.add_argument('--image-checkpoint', type=str, required=True)
    parser.add_argument('--sensor-artifact-dir', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--sensor-json', type=str, required=True)
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--topk', type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = select_device(args.device)

    image_ckpt = torch.load(args.image_checkpoint, map_location='cpu')
    class_names = image_ckpt['class_names']
    image_model = build_image_model(image_ckpt['backbone'], len(class_names), pretrained=False).to(device)
    image_model.load_state_dict(image_ckpt['model_state_dict'])
    image_model.eval()
    image_size = int(image_ckpt.get('image_size', 224))

    image = Image.open(args.image).convert('RGB')
    image_tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = image_model(image_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    top_idx = np.argsort(-probs)[: max(1, min(args.topk, len(probs)))]
    disease_label = class_names[int(top_idx[0])]

    sensor_dir = Path(args.sensor_artifact_dir).resolve()
    sensor_ckpt = torch.load(sensor_dir / 'best_sensor_autoencoder.pt', map_location='cpu')
    preprocessor = SensorPreprocessor.from_state_dict(sensor_ckpt['preprocessor_state'])
    sensor_model = SensorAutoencoder(
        input_dim=sensor_ckpt['input_dim'],
        hidden_dim=sensor_ckpt['hidden_dim'],
        bottleneck_dim=sensor_ckpt['bottleneck_dim'],
    ).to(device)
    sensor_model.load_state_dict(sensor_ckpt['model_state_dict'])
    sensor_model.eval()
    sensor_summary = read_json(sensor_dir / 'sensor_model_summary.json')
    threshold = float(sensor_summary['threshold'])

    sensor_snapshot = read_json(args.sensor_json)
    x = preprocessor.transform_row(sensor_snapshot)
    xt = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        recon = sensor_model(xt)
        error = torch.mean((recon - xt) ** 2, dim=1).item()
    sensor_state = 'риск' if error > threshold else 'норма'

    rules_state, issues, recommendations = analyze_sensor_snapshot(sensor_snapshot, sensor_snapshot.get('growth_stage'))

    if disease_label == 'healthy':
        diagnosis_text = 'По изображению лист выглядит здоровым.'
    else:
        diagnosis_text = f'По изображению обнаружены признаки заболевания: {disease_label}.'

    if sensor_state == 'риск' or rules_state == 'риск':
        sensor_text = 'По данным датчиков есть отклонения условий выращивания.'
    else:
        sensor_text = 'По данным датчиков условия выращивания близки к норме.'

    print(f'Изображение: {Path(args.image).resolve()}')
    print(f'JSON датчиков: {Path(args.sensor_json).resolve()}')
    print('\nДиагноз по изображению:')
    print(f'  {diagnosis_text}')
    print('  Top-k:')
    for idx in top_idx:
        print(f'    - {class_names[int(idx)]}: {probs[int(idx)]:.6f}')

    print('\nОценка датчиков:')
    print(f'  Состояние по автоэнкодеру: {sensor_state} (ошибка реконструкции: {error:.6f}, порог: {threshold:.6f})')
    print(f'  Состояние по правилам: {rules_state}')
    if issues:
        print('  Найденные отклонения:')
        for issue in issues:
            print(f'    - {issue}')

    print('\nРекомендации:')
    for rec in recommendations:
        print(f'  - {rec}')

    print('\nИтог:')
    print(f'  {diagnosis_text} {sensor_text}')


if __name__ == '__main__':
    main()
