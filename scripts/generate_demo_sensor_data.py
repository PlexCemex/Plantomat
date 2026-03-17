#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.utils import ensure_dir, normalize_stage, save_json, seed_everything, slugify_label  # noqa: E402


STAGES = ['germination', 'seedling', 'vegetative', 'flowering', 'fruiting']

STAGE_BASE_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    'germination': {
        'air_temp_c': (24.0, 28.0),
        'air_humidity_pct': (70.0, 85.0),
        'soil_moisture_pct': (75.0, 90.0),
        'solution_ph': (5.6, 6.3),
        'ec_ms_cm': (1.2, 1.8),
        'light_lux': (6000.0, 12000.0),
    },
    'seedling': {
        'air_temp_c': (22.0, 26.0),
        'air_humidity_pct': (60.0, 75.0),
        'soil_moisture_pct': (70.0, 85.0),
        'solution_ph': (5.6, 6.3),
        'ec_ms_cm': (1.5, 2.2),
        'light_lux': (8000.0, 15000.0),
    },
    'vegetative': {
        'air_temp_c': (21.0, 27.0),
        'air_humidity_pct': (60.0, 70.0),
        'soil_moisture_pct': (55.0, 75.0),
        'solution_ph': (5.5, 6.3),
        'ec_ms_cm': (1.8, 3.0),
        'light_lux': (12000.0, 25000.0),
    },
    'flowering': {
        'air_temp_c': (20.0, 26.0),
        'air_humidity_pct': (60.0, 70.0),
        'soil_moisture_pct': (55.0, 70.0),
        'solution_ph': (5.5, 6.3),
        'ec_ms_cm': (2.0, 3.5),
        'light_lux': (15000.0, 30000.0),
    },
    'fruiting': {
        'air_temp_c': (20.0, 27.0),
        'air_humidity_pct': (60.0, 70.0),
        'soil_moisture_pct': (50.0, 70.0),
        'solution_ph': (5.5, 6.3),
        'ec_ms_cm': (2.0, 4.0),
        'light_lux': (15000.0, 35000.0),
    },
}


def sample_uniform(bounds: Tuple[float, float]) -> float:
    return random.uniform(bounds[0], bounds[1])


def adjust_by_label(label: str, values: Dict[str, float]) -> Dict[str, float]:
    label = slugify_label(label)
    v = dict(values)

    if label == 'healthy':
        return v

    if 'late_blight' in label:
        v['air_humidity_pct'] += random.uniform(10, 18)
        v['air_temp_c'] -= random.uniform(1, 4)
        v['soil_moisture_pct'] += random.uniform(5, 12)
    elif 'leaf_mold' in label:
        v['air_humidity_pct'] += random.uniform(12, 20)
        v['soil_moisture_pct'] += random.uniform(4, 10)
    elif 'septoria' in label or 'target_spot' in label or 'bacterial_spot' in label:
        v['air_humidity_pct'] += random.uniform(8, 16)
        v['air_temp_c'] += random.uniform(1, 4)
        v['soil_moisture_pct'] += random.uniform(3, 8)
    elif 'spider_mites' in label:
        v['air_temp_c'] += random.uniform(4, 8)
        v['air_humidity_pct'] -= random.uniform(12, 20)
        v['soil_moisture_pct'] -= random.uniform(4, 12)
    elif 'yellow_leaf_curl' in label:
        v['air_temp_c'] += random.uniform(4, 7)
        v['air_humidity_pct'] -= random.uniform(5, 12)
        v['ec_ms_cm'] += random.uniform(0.1, 0.6)
    elif 'mosaic_virus' in label:
        v['ec_ms_cm'] += random.uniform(-0.2, 0.4)
        v['solution_ph'] += random.uniform(-0.2, 0.3)
    elif 'early_blight' in label:
        v['air_temp_c'] += random.uniform(2, 6)
        v['air_humidity_pct'] += random.uniform(4, 10)

    return v


def clip_values(values: Dict[str, float]) -> Dict[str, float]:
    v = dict(values)
    v['air_humidity_pct'] = max(20.0, min(98.0, v['air_humidity_pct']))
    v['soil_moisture_pct'] = max(10.0, min(100.0, v['soil_moisture_pct']))
    v['solution_ph'] = max(4.8, min(7.2, v['solution_ph']))
    v['ec_ms_cm'] = max(0.5, min(5.5, v['ec_ms_cm']))
    v['light_lux'] = max(1000.0, min(45000.0, v['light_lux']))
    return v


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Генерация демонстрационных сенсорных признаков.')
    parser.add_argument('--input-csv', type=str, required=True, help='CSV с image_path, label, split.')
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--default-stage', type=str, default='vegetative')
    parser.add_argument('--start-time', type=str, default='2026-01-01T08:00:00')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    random.seed(args.seed)

    df = pd.read_csv(args.input_csv)
    if 'growth_stage' not in df.columns:
        df['growth_stage'] = [random.choice(STAGES) for _ in range(len(df))]
    else:
        df['growth_stage'] = df['growth_stage'].fillna(args.default_stage).apply(lambda x: normalize_stage(str(x), args.default_stage))

    if 'timestamp' not in df.columns:
        start_dt = datetime.fromisoformat(args.start_time)
        df['timestamp'] = [(start_dt + timedelta(minutes=10 * i)).isoformat() for i in range(len(df))]

    rows = []
    for _, row in df.iterrows():
        stage = normalize_stage(row.get('growth_stage'), args.default_stage)
        base_ranges = STAGE_BASE_RANGES.get(stage, STAGE_BASE_RANGES['vegetative'])
        sample = {metric: sample_uniform(bounds) for metric, bounds in base_ranges.items()}
        sample = adjust_by_label(row['label'], sample)
        sample = clip_values(sample)
        sample['tds_ppm'] = round(sample['ec_ms_cm'] * 500.0, 2)
        sample['co2_ppm'] = round(random.uniform(380, 750), 2)
        sample['leaf_wetness'] = round(random.uniform(0.0, 1.0), 3)
        rows.append(sample)

    sensor_df = pd.DataFrame(rows)
    out_df = pd.concat([df.reset_index(drop=True), sensor_df], axis=1)

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    summary = {
        'warning': 'ЭТО ДЕМО-ДАННЫЕ. Они нужны только для отладки мультимодального пайплайна, а не для научных выводов.',
        'input_csv': str(Path(args.input_csv).resolve()),
        'output_csv': str(out_csv),
        'num_rows': int(len(out_df)),
        'stages': sorted(out_df['growth_stage'].unique().tolist()),
        'features': [
            'air_temp_c',
            'air_humidity_pct',
            'soil_moisture_pct',
            'solution_ph',
            'ec_ms_cm',
            'tds_ppm',
            'light_lux',
            'co2_ppm',
            'leaf_wetness',
        ],
    }
    save_json(summary, out_csv.with_suffix('.summary.json'))

    print('Готово. Демонстрационный мультимодальный CSV сохранён в:')
    print(out_csv)
    print(summary['warning'])


if __name__ == '__main__':
    main()
