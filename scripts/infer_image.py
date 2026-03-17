#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.features import TabularFeatureBuilder  # noqa: E402
from tomato_ai.models import MultimodalClassifier  # noqa: E402
from tomato_ai.recommender import generate_recommendations, load_rules  # noqa: E402
from tomato_ai.transforms import build_eval_transform  # noqa: E402
from tomato_ai.utils import read_json  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Инференс по одному изображению и одному сенсорному слепку.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--image', type=str, required=True)
    parser.add_argument('--sensor-json', type=str, default=None, help='JSON со значениями датчиков.')
    parser.add_argument('--growth-stage', type=str, default='vegetative')
    parser.add_argument('--rules', type=str, default='configs/recommendation_rules.yaml')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--topk', type=int, default=3)
    return parser.parse_args()


def load_model(checkpoint_path: str, device: str):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = checkpoint.get('config', {})
    feature_builder = TabularFeatureBuilder.from_state_dict(checkpoint['feature_builder_state'])
    model = MultimodalClassifier(
        num_classes=int(checkpoint['num_classes']),
        sensor_dim=feature_builder.feature_dim,
        backbone=checkpoint.get('backbone', cfg.get('backbone', 'efficientnet_b0')),
        pretrained=False,
        dropout=float(cfg.get('dropout', 0.30)),
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    idx_to_label = {int(k): v for k, v in checkpoint['idx_to_label'].items()}
    image_size = int(checkpoint.get('image_size', cfg.get('image_size', 224)))
    return model, feature_builder, idx_to_label, image_size


def main() -> None:
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, feature_builder, idx_to_label, image_size = load_model(args.checkpoint, device)
    sensor_snapshot: Dict[str, Any] = read_json(args.sensor_json) if args.sensor_json else {}
    sensor_snapshot['growth_stage'] = args.growth_stage

    image = Image.open(args.image).convert('RGB')
    image_tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device)
    if feature_builder.feature_dim > 0:
        sensor_tensor = torch.from_numpy(feature_builder.transform_row(sensor_snapshot)).unsqueeze(0).to(device)
    else:
        sensor_tensor = torch.zeros((1, 0), dtype=torch.float32, device=device)

    with torch.no_grad():
        logits = model(image_tensor, sensor_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    topk = min(args.topk, len(probs))
    top_idx = np.argsort(-probs)[:topk]
    predicted_label = idx_to_label[int(top_idx[0])]

    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = (ROOT / rules_path).resolve()
    rules = load_rules(str(rules_path))
    recommendations = generate_recommendations(sensor_snapshot, args.growth_stage, predicted_label, rules)

    print(f'Изображение: {Path(args.image).resolve()}')
    print(f'Предсказанный класс: {predicted_label}')
    print('Top-k вероятности:')
    for idx in top_idx:
        print(f'  - {idx_to_label[int(idx)]}: {probs[int(idx)]:.4f}')

    print('\nРекомендации:')
    for rec in recommendations:
        print(f'  - {rec}')


if __name__ == '__main__':
    main()
