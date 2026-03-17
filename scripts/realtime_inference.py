#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List

import cv2
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
    parser = argparse.ArgumentParser(description='Онлайн-инференс по камере и текущим показаниям датчиков.')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--sensor-json', type=str, default=None)
    parser.add_argument('--growth-stage', type=str, default='vegetative')
    parser.add_argument('--rules', type=str, default='configs/recommendation_rules.yaml')
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--topk', type=int, default=3)
    parser.add_argument('--every-n-frames', type=int, default=5)
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


def predict_frame(
    model,
    feature_builder,
    idx_to_label,
    image_size: int,
    frame_bgr: np.ndarray,
    sensor_snapshot: Dict[str, Any],
    device: str,
    topk: int,
):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    image_tensor = build_eval_transform(image_size)(image).unsqueeze(0).to(device)
    if feature_builder.feature_dim > 0:
        sensor_tensor = torch.from_numpy(feature_builder.transform_row(sensor_snapshot)).unsqueeze(0).to(device)
    else:
        sensor_tensor = torch.zeros((1, 0), dtype=torch.float32, device=device)

    with torch.no_grad():
        probs = torch.softmax(model(image_tensor, sensor_tensor), dim=1).cpu().numpy()[0]

    topk = min(topk, len(probs))
    top_idx = np.argsort(-probs)[:topk]
    top_items = [(idx_to_label[int(idx)], float(probs[int(idx)])) for idx in top_idx]
    return top_items


def draw_overlay(frame: np.ndarray, lines: List[str]) -> np.ndarray:
    out = frame.copy()
    x, y = 12, 28
    line_height = 24
    for i, line in enumerate(lines):
        cv2.putText(
            out,
            line,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return out


def main() -> None:
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model, feature_builder, idx_to_label, image_size = load_model(args.checkpoint, device)

    rules_path = Path(args.rules)
    if not rules_path.is_absolute():
        rules_path = (ROOT / rules_path).resolve()
    rules = load_rules(str(rules_path))

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f'Не удалось открыть камеру {args.camera}')

    frame_count = 0
    cached_lines = ['Инициализация...']

    print('Нажми q, чтобы выйти.')
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame_count += 1
            if frame_count % max(1, args.every_n_frames) == 0:
                sensor_snapshot = read_json(args.sensor_json) if args.sensor_json else {}
                sensor_snapshot['growth_stage'] = args.growth_stage
                top_items = predict_frame(
                    model,
                    feature_builder,
                    idx_to_label,
                    image_size,
                    frame,
                    sensor_snapshot,
                    device,
                    args.topk,
                )
                predicted_label = top_items[0][0]
                recommendations = generate_recommendations(sensor_snapshot, args.growth_stage, predicted_label, rules, max_items=3)
                cached_lines = [
                    f'Pred: {predicted_label} ({top_items[0][1]:.2%})',
                    *[f'{name}: {prob:.2%}' for name, prob in top_items[1:]],
                    *[f'Advice: {rec}' for rec in recommendations[:2]],
                    'Press q to quit',
                ]

            frame_overlay = draw_overlay(frame, cached_lines)
            cv2.imshow('Tomato realtime inference', frame_overlay)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
