#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import timedelta
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.utils import ensure_dir, save_json, stratified_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Сборка мультимодального CSV по изображениям и сенсорным логам.')
    parser.add_argument('--images-csv', type=str, required=True, help='CSV с image_path, label, timestamp и при желании growth_stage/plant_id.')
    parser.add_argument('--sensor-csv', type=str, required=True, help='CSV с timestamp и сенсорными колонками.')
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--window-minutes', type=int, default=360, help='Окно истории сенсоров перед моментом съёмки.')
    parser.add_argument('--nearest-tolerance-minutes', type=int, default=120)
    parser.add_argument('--timestamp-column', type=str, default='timestamp')
    parser.add_argument('--plant-id-column', type=str, default='plant_id')
    parser.add_argument('--val-ratio', type=float, default=0.15)
    parser.add_argument('--test-ratio', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


def infer_sensor_columns(sensor_df: pd.DataFrame, timestamp_col: str, plant_id_col: str) -> List[str]:
    cols: List[str] = []
    for col in sensor_df.columns:
        if col in {timestamp_col, plant_id_col}:
            continue
        if pd.api.types.is_numeric_dtype(sensor_df[col]):
            cols.append(col)
    return cols


def compute_window_features(window_df: pd.DataFrame, sensor_cols: List[str]) -> Dict[str, float]:
    features: Dict[str, float] = {}
    if window_df.empty:
        for col in sensor_cols:
            for suffix in ['last', 'mean', 'min', 'max', 'std', 'slope']:
                features[f'{col}_{suffix}'] = np.nan
        return features

    time_index = window_df.index
    if len(time_index) > 1:
        hours = (time_index.view('int64') - time_index.view('int64')[0]) / 3.6e12
    else:
        hours = np.array([0.0])

    for col in sensor_cols:
        series = pd.to_numeric(window_df[col], errors='coerce').dropna()
        if series.empty:
            for suffix in ['last', 'mean', 'min', 'max', 'std', 'slope']:
                features[f'{col}_{suffix}'] = np.nan
            continue

        features[f'{col}_last'] = float(series.iloc[-1])
        features[f'{col}_mean'] = float(series.mean())
        features[f'{col}_min'] = float(series.min())
        features[f'{col}_max'] = float(series.max())
        features[f'{col}_std'] = float(series.std(ddof=0)) if len(series) > 1 else 0.0

        if len(series) > 1 and hours[-1] > 0:
            y = series.to_numpy(dtype=np.float64)
            x = hours[: len(y)]
            slope = (y[-1] - y[0]) / max(hours[-1], 1e-6)
            features[f'{col}_slope'] = float(slope)
        else:
            features[f'{col}_slope'] = 0.0

    return features


def get_grouped_sensor_frames(sensor_df: pd.DataFrame, plant_id_col: str) -> Dict[str, pd.DataFrame]:
    if plant_id_col in sensor_df.columns:
        grouped = {}
        for plant_id, group in sensor_df.groupby(plant_id_col):
            grouped[str(plant_id)] = group.sort_values('timestamp').set_index('timestamp')
        return grouped

    return {'__all__': sensor_df.sort_values('timestamp').set_index('timestamp')}


def select_group(sensor_groups: Dict[str, pd.DataFrame], image_row: pd.Series, plant_id_col: str) -> pd.DataFrame:
    if plant_id_col in image_row and pd.notna(image_row[plant_id_col]):
        key = str(image_row[plant_id_col])
        if key in sensor_groups:
            return sensor_groups[key]
    return sensor_groups['__all__']


def main() -> None:
    args = parse_args()

    image_df = pd.read_csv(args.images_csv)
    sensor_df = pd.read_csv(args.sensor_csv)

    if args.timestamp_column not in image_df.columns:
        raise ValueError(f'В images-csv отсутствует колонка {args.timestamp_column}')
    if args.timestamp_column not in sensor_df.columns:
        raise ValueError(f'В sensor-csv отсутствует колонка {args.timestamp_column}')

    image_df = image_df.copy()
    sensor_df = sensor_df.copy()

    image_df[args.timestamp_column] = pd.to_datetime(image_df[args.timestamp_column], errors='coerce')
    sensor_df[args.timestamp_column] = pd.to_datetime(sensor_df[args.timestamp_column], errors='coerce')
    image_df = image_df.dropna(subset=[args.timestamp_column]).reset_index(drop=True)
    sensor_df = sensor_df.dropna(subset=[args.timestamp_column]).reset_index(drop=True)

    if image_df.empty:
        raise RuntimeError('После парсинга времени images-csv оказался пустым.')
    if sensor_df.empty:
        raise RuntimeError('После парсинга времени sensor-csv оказался пустым.')

    sensor_cols = infer_sensor_columns(sensor_df, args.timestamp_column, args.plant_id_column)
    if not sensor_cols:
        raise RuntimeError('Не найдено числовых сенсорных колонок в sensor-csv.')

    sensor_df = sensor_df.rename(columns={args.timestamp_column: 'timestamp'})
    image_df = image_df.rename(columns={args.timestamp_column: 'timestamp'})

    sensor_groups = get_grouped_sensor_frames(sensor_df, args.plant_id_column)
    if '__all__' not in sensor_groups:
        sensor_groups['__all__'] = sensor_df.sort_values('timestamp').set_index('timestamp')

    window_delta = timedelta(minutes=args.window_minutes)
    tolerance_delta = timedelta(minutes=args.nearest_tolerance_minutes)

    rows = []
    for _, image_row in tqdm(image_df.iterrows(), total=len(image_df), desc='Pairing'):
        group_df = select_group(sensor_groups, image_row, args.plant_id_column)
        ts = image_row['timestamp']
        start_ts = ts - window_delta
        window_df = group_df.loc[start_ts:ts]

        if window_df.empty:
            try:
                loc = group_df.index.get_indexer([ts], method='pad')[0]
            except Exception:
                loc = -1
            if loc >= 0:
                candidate = group_df.iloc[[loc]]
                if abs(ts - candidate.index[0]) <= tolerance_delta:
                    window_df = candidate

        features = compute_window_features(window_df, sensor_cols)
        base = image_row.to_dict()
        base['timestamp'] = pd.Timestamp(base['timestamp']).isoformat()
        base.update(features)
        rows.append(base)

    out_df = pd.DataFrame(rows)
    if 'split' not in out_df.columns and 'label' in out_df.columns:
        out_df = stratified_split(
            out_df,
            label_col='label',
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            split_col='split',
        )

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    feature_cols = [c for c in out_df.columns if c.endswith(('_last', '_mean', '_min', '_max', '_std', '_slope'))]
    summary = {
        'images_csv': str(Path(args.images_csv).resolve()),
        'sensor_csv': str(Path(args.sensor_csv).resolve()),
        'output_csv': str(out_csv),
        'num_rows': int(len(out_df)),
        'sensor_columns_used': sensor_cols,
        'generated_feature_columns': feature_cols,
        'window_minutes': args.window_minutes,
        'nearest_tolerance_minutes': args.nearest_tolerance_minutes,
    }
    save_json(summary, out_csv.with_suffix('.summary.json'))

    print('Готово.')
    print(f'CSV: {out_csv}')
    print(f'Сенсорных колонок: {len(sensor_cols)}')
    print(f'Сгенерировано признаков: {len(feature_cols)}')


if __name__ == '__main__':
    main()
