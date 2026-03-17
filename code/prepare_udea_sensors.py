#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.utils import detect_csv_separator, ensure_dir, write_json  # noqa: E402

CANONICAL_COLUMNS = [
    'air_temp_c',
    'air_humidity_pct',
    'soil_moisture_pct',
    'solution_ph',
    'ec_ms_cm',
    'tds_ppm',
    'light_lux',
    'co2_ppm',
    'leaf_wetness',
    'growth_stage',
]

PATTERNS = {
    'air_temp_c': [r'air.*temp', r'temp.*air', r'temperature', r'temperatura'],
    'air_humidity_pct': [r'air.*hum', r'hum.*air', r'relative.*humidity', r'humidity', r'humedad'],
    'soil_moisture_pct': [r'soil.*moist', r'substrate.*moist', r'sustrat.*hum', r'suelo.*hum'],
    'solution_ph': [r'(^|_)ph($|_)'],
    'ec_ms_cm': [r'(^|_)ec($|_)', r'conductiv', r'conductividad'],
    'tds_ppm': [r'tds'],
    'light_lux': [r'light', r'lux', r'lum', r'illum'],
    'co2_ppm': [r'co2'],
    'leaf_wetness': [r'leaf.*wet', r'wetness'],
    'growth_stage': [r'growth.*stage', r'phenolog', r'phase', r'stage', r'etapa'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Очистка и канонизация CSV датчиков томата.')
    parser.add_argument('--input-csv', type=str, required=True)
    parser.add_argument('--output-csv', type=str, required=True)
    parser.add_argument('--summary-json', type=str, default=None)
    return parser.parse_args()


def normalize_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r'[^a-z0-9а-яё]+', '_', name)
    return name.strip('_')


def guess_mapping(columns: List[str]) -> Dict[str, Optional[str]]:
    norm_cols = {col: normalize_name(col) for col in columns}
    mapping: Dict[str, Optional[str]] = {k: None for k in CANONICAL_COLUMNS}
    for canonical, patterns in PATTERNS.items():
        for original, norm in norm_cols.items():
            if any(re.search(pattern, norm) for pattern in patterns):
                mapping[canonical] = original
                break
    return mapping


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    sep = detect_csv_separator(input_csv)
    df = pd.read_csv(input_csv, sep=sep)
    mapping = guess_mapping(df.columns.tolist())

    out = pd.DataFrame()
    for canonical in CANONICAL_COLUMNS:
        source = mapping.get(canonical)
        if source is None or source not in df.columns:
            out[canonical] = np.nan if canonical != 'growth_stage' else 'unknown'
        else:
            out[canonical] = df[source]

    for col in CANONICAL_COLUMNS:
        if col != 'growth_stage':
            out[col] = pd.to_numeric(out[col], errors='coerce')

    out['growth_stage'] = out['growth_stage'].fillna('unknown').astype(str).str.strip().str.lower()
    out['row_id'] = range(len(out))

    output_csv = Path(args.output_csv).resolve()
    ensure_dir(output_csv.parent)
    out.to_csv(output_csv, index=False)

    summary = {
        'input_csv': str(input_csv),
        'output_csv': str(output_csv),
        'separator': sep,
        'rows': int(len(out)),
        'column_mapping': mapping,
        'missing_ratio': {col: float(out[col].isna().mean()) for col in CANONICAL_COLUMNS if col != 'growth_stage'},
        'stage_values': sorted(out['growth_stage'].dropna().astype(str).unique().tolist()),
    }
    summary_path = Path(args.summary_json).resolve() if args.summary_json else output_csv.with_suffix('.summary.json')
    write_json(summary, summary_path)
    print(f'Готово: {output_csv}')
    print(f'Сводка: {summary_path}')
    print('Сопоставление колонок:')
    for k, v in mapping.items():
        print(f'  {k}: {v}')


if __name__ == '__main__':
    main()
