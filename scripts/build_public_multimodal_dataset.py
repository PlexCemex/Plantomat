#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.utils import ensure_dir, normalize_stage, save_json, stratified_split  # noqa: E402


RANGE_RULES = {
    'seedling': {
        'air_temp_c': (22.0, 26.0),
        'air_humidity_pct': (60.0, 75.0),
        'light_lux': (8000.0, 18000.0),
    },
    'vegetative': {
        'air_temp_c': (21.0, 27.0),
        'air_humidity_pct': (60.0, 70.0),
        'light_lux': (12000.0, 25000.0),
    },
    'flowering': {
        'air_temp_c': (20.0, 26.0),
        'air_humidity_pct': (60.0, 70.0),
        'light_lux': (15000.0, 30000.0),
    },
    'fruiting': {
        'air_temp_c': (20.0, 27.0),
        'air_humidity_pct': (60.0, 70.0),
        'light_lux': (15000.0, 35000.0),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Сборка публичного мультимодального датасета из image-only и sensor-only источников.')
    parser.add_argument('--images-csv', type=str, required=True, help='CSV, подготовленный prepare_plantvillage.py')
    parser.add_argument('--sensor-csv', type=str, required=True, help='CSV, подготовленный prepare_udea_sensor_dataset.py')
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()



def temporal_split_sensor(sensor_df: pd.DataFrame, val_ratio: float, test_ratio: float) -> pd.DataFrame:
    out = sensor_df.sort_values('timestamp').reset_index(drop=True).copy()
    n = len(out)
    if n == 0:
        raise RuntimeError('sensor_df пустой.')
    train_end = max(1, int(round(n * (1.0 - val_ratio - test_ratio))))
    val_end = max(train_end + 1, int(round(n * (1.0 - test_ratio)))) if val_ratio > 0 else train_end
    train_end = min(train_end, n)
    val_end = min(max(val_end, train_end), n)
    out['split'] = 'train'
    if train_end < n:
        out.loc[train_end:val_end - 1, 'split'] = 'val'
        out.loc[val_end:, 'split'] = 'test'
    return out



def derive_sensor_context_label(row: pd.Series) -> str:
    stage = normalize_stage(row.get('growth_stage'), default='vegetative')
    rules = RANGE_RULES.get(stage, RANGE_RULES['vegetative'])
    deviations: List[str] = []
    for metric, (min_value, max_value) in rules.items():
        if metric not in row or pd.isna(row[metric]):
            continue
        value = float(row[metric])
        if value < min_value or value > max_value:
            if metric == 'air_temp_c':
                deviations.append('temperature')
            elif metric == 'air_humidity_pct':
                deviations.append('humidity')
            elif metric == 'light_lux':
                deviations.append('light')
            else:
                deviations.append(metric)
    if not deviations:
        return 'normal'
    unique = sorted(set(deviations))
    if len(unique) == 1:
        return f'{unique[0]}_stress'
    return 'mixed_stress'



def sample_sensor_rows(sensor_split_df: pd.DataFrame, sample_size: int, random_state: int) -> pd.DataFrame:
    replace = len(sensor_split_df) < sample_size
    return sensor_split_df.sample(n=sample_size, replace=replace, random_state=random_state).reset_index(drop=True)



def merge_split(images_split_df: pd.DataFrame, sensor_split_df: pd.DataFrame, random_state: int) -> pd.DataFrame:
    sampled_sensor = sample_sensor_rows(sensor_split_df, len(images_split_df), random_state=random_state)
    merged = images_split_df.reset_index(drop=True).copy()

    keep_sensor_columns = [col for col in sampled_sensor.columns if col not in {'split'}]
    rename_map: Dict[str, str] = {}
    for col in keep_sensor_columns:
        if col in merged.columns and col not in {'timestamp', 'growth_stage'}:
            rename_map[col] = f'sensor_{col}'
    sampled_sensor = sampled_sensor[keep_sensor_columns].rename(columns=rename_map)

    if 'growth_stage' not in sampled_sensor.columns:
        sampled_sensor['growth_stage'] = 'vegetative'

    sampled_sensor['sensor_context_label'] = sampled_sensor.apply(derive_sensor_context_label, axis=1)
    sampled_sensor['pairing_strategy'] = 'public_real_datasets_random_same_split'

    return pd.concat([merged, sampled_sensor], axis=1)



def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    image_df = pd.read_csv(args.images_csv)
    sensor_df = pd.read_csv(args.sensor_csv)

    image_df = stratified_split(
        image_df,
        label_col='label',
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
        split_col='split',
    )

    if 'timestamp' not in sensor_df.columns:
        raise ValueError('В sensor-csv должна быть колонка timestamp. Сначала запусти prepare_udea_sensor_dataset.py')
    sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'], errors='coerce')
    sensor_df = sensor_df.dropna(subset=['timestamp']).reset_index(drop=True)
    sensor_df = temporal_split_sensor(sensor_df, val_ratio=float(args.val_ratio), test_ratio=float(args.test_ratio))

    outputs: List[pd.DataFrame] = []
    split_counts: Dict[str, Dict[str, int]] = {}
    for offset, split_name in enumerate(['train', 'val', 'test']):
        images_split = image_df[image_df['split'] == split_name].reset_index(drop=True)
        if images_split.empty:
            continue
        sensor_split = sensor_df[sensor_df['split'] == split_name].reset_index(drop=True)
        if sensor_split.empty:
            sensor_split = sensor_df.copy()
        merged_split = merge_split(images_split, sensor_split, random_state=int(args.seed) + offset)
        outputs.append(merged_split)
        split_counts[split_name] = {
            'images': int(len(images_split)),
            'sensor_rows_pool': int(len(sensor_split)),
        }

    if not outputs:
        raise RuntimeError('Не удалось собрать итоговый мультимодальный датасет.')

    out_df = pd.concat(outputs, axis=0, ignore_index=True)
    out_df['growth_stage'] = out_df['growth_stage'].fillna('vegetative').map(lambda value: normalize_stage(value, default='vegetative'))

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    summary = {
        'warning': (
            'В этом CSV используются реальные изображения PlantVillage и реальные сенсорные строки UdeA, '
            'но пары image-sensor создаются автоматически, потому что открытого общедоступного набора с '
            'синхронными диагнозами и IoT-телеметрией для томата не найдено.'
        ),
        'images_csv': str(Path(args.images_csv).resolve()),
        'sensor_csv': str(Path(args.sensor_csv).resolve()),
        'output_csv': str(out_csv),
        'num_rows': int(len(out_df)),
        'split_counts': split_counts,
        'feature_columns': [
            col
            for col in out_df.columns
            if col not in {'image_path', 'label', 'split', 'source_dataset', 'source_dataset_sensor', 'pairing_strategy'}
        ],
        'label_counts': {k: int(v) for k, v in out_df['label'].value_counts().to_dict().items()},
        'growth_stage_counts': {k: int(v) for k, v in out_df['growth_stage'].value_counts().to_dict().items()},
        'sensor_context_counts': {k: int(v) for k, v in out_df['sensor_context_label'].value_counts().to_dict().items()},
    }
    save_json(summary, out_csv.with_suffix('.summary.json'))

    print('Готово.')
    print(f'Публичный мультимодальный CSV: {out_csv}')
    print(summary['warning'])


if __name__ == '__main__':
    main()
