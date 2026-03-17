#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.utils import ensure_dir, save_json, seed_everything, slugify_label, stratified_split  # noqa: E402


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Подготовка tomato subset из PlantVillage.')
    parser.add_argument('--dataset-root', type=str, required=True, help='Путь к корню PlantVillage-Dataset.')
    parser.add_argument('--color-subdir', type=str, default='raw/color', help='Подкаталог с RGB-изображениями.')
    parser.add_argument('--output-csv', type=str, required=True, help='Куда сохранить итоговый CSV.')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def normalize_folder_label(folder_name: str) -> str:
    cleaned = folder_name.replace('Tomato___', '')
    return slugify_label(cleaned)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    root = Path(args.dataset_root).resolve()
    color_root = (root / args.color_subdir).resolve()
    if not color_root.exists():
        raise FileNotFoundError(f'Не найден каталог с цветными изображениями: {color_root}')

    rows = []
    for class_dir in sorted(color_root.iterdir()):
        if not class_dir.is_dir():
            continue
        if not class_dir.name.lower().startswith('tomato___'):
            continue

        label = normalize_folder_label(class_dir.name)
        for image_path in sorted(class_dir.rglob('*')):
            if image_path.suffix.lower() not in IMAGE_EXTS:
                continue
            rows.append(
                {
                    'image_path': str(image_path.resolve()),
                    'label': label,
                    'source_dataset': 'plantvillage',
                }
            )

    if not rows:
        raise RuntimeError('Не найдено ни одного изображения томатов в указанной директории.')

    df = pd.DataFrame(rows)
    df = stratified_split(
        df,
        label_col='label',
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        split_col='split',
    )

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)

    summary = {
        'dataset_root': str(root),
        'color_root': str(color_root),
        'num_images': int(len(df)),
        'num_classes': int(df['label'].nunique()),
        'classes': sorted(df['label'].unique().tolist()),
        'split_counts': {k: int(v) for k, v in df['split'].value_counts().to_dict().items()},
        'label_counts': {k: int(v) for k, v in df['label'].value_counts().to_dict().items()},
    }
    save_json(summary, out_csv.with_suffix('.summary.json'))

    print('Готово.')
    print(f'CSV: {out_csv}')
    print(f"Изображений: {summary['num_images']}")
    print(f"Классов: {summary['num_classes']}")
    print(f"Split counts: {summary['split_counts']}")


if __name__ == '__main__':
    main()
