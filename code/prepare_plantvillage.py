#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.utils import ensure_dir, seed_everything, slugify_label, stratified_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Подготовка CSV по изображениям томатов из PlantVillage.')
    parser.add_argument('--dataset-root', type=str, required=True, help='Путь к корню PlantVillage-Dataset или к папке, внутри которой есть color/.')
    parser.add_argument('--color-subdir', type=str, default='color', help='Подпапка с цветными изображениями.')
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def resolve_color_dir(dataset_root: Path, color_subdir: str) -> Path:
    candidates = [
        dataset_root / color_subdir,
        dataset_root / 'raw' / color_subdir,
        dataset_root,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f'Не удалось найти папку с изображениями: {dataset_root}')


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    dataset_root = Path(args.dataset_root).resolve()
    color_dir = resolve_color_dir(dataset_root, args.color_subdir)

    rows = []
    for class_dir in sorted(color_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.startswith('Tomato___'):
            continue
        label_slug = slugify_label(class_dir.name)
        for image_path in class_dir.rglob('*'):
            if image_path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                continue
            rows.append({
                'image_path': str(image_path.resolve()),
                'label': label_slug,
                'source_class_dir': class_dir.name,
            })

    if not rows:
        raise RuntimeError('Не найдено ни одного изображения томата.')

    df = pd.DataFrame(rows)
    df = stratified_split(df, label_col='label', val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)
    output_csv = Path(args.output_csv).resolve()
    ensure_dir(output_csv.parent)
    df.to_csv(output_csv, index=False)
    counts = df.groupby(['split', 'label']).size().unstack(fill_value=0)
    print(f'Готово: {output_csv}')
    print(counts)


if __name__ == '__main__':
    main()
