#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tomato_ai.utils import ensure_dir, normalize_stage, save_json  # noqa: E402


DEFAULT_START_TS = pd.Timestamp('2023-11-09T08:00:00')
STAGE_BINS = [
    (0.00, 0.20, 'seedling'),
    (0.20, 0.60, 'vegetative'),
    (0.60, 0.80, 'flowering'),
    (0.80, 1.01, 'fruiting'),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Подготовка открытого сенсорного датасета UdeA Tomato к мультимодальному пайплайну.')
    parser.add_argument('--input-csv', type=str, required=True, help='Путь к DB_Mobile_Manual_Tomato.csv')
    parser.add_argument('--output-csv', type=str, required=True, help='Куда сохранить очищенный CSV')
    parser.add_argument('--dataset-name', type=str, default='udea_tomato')
    return parser.parse_args()


def normalize_name(text: str) -> str:
    text = unicodedata.normalize('NFKD', str(text))
    text = ''.join(ch for ch in text if not unicodedata.combining(ch))
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]+', '_', text)
    text = re.sub(r'_+', '_', text)
    return text.strip('_')


def parse_numeric_value(value: object) -> float:
    if value is None:
        return np.nan
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    text = str(value).strip()
    if text == '' or text.lower() in {'nan', 'none', 'null'}:
        return np.nan
    text = text.replace('\u00a0', '').replace(' ', '')
    if ',' in text and '.' in text:
        if text.rfind(',') > text.rfind('.'):
            text = text.replace('.', '').replace(',', '.')
        else:
            text = text.replace(',', '')
    elif ',' in text:
        text = text.replace(',', '.')
    text = re.sub(r'[^0-9eE+\-.]', '', text)
    try:
        return float(text)
    except Exception:
        return np.nan



def coerce_numeric_series(series: pd.Series) -> pd.Series:
    return series.map(parse_numeric_value).astype(float)



def read_csv_flexible(path: str) -> Tuple[pd.DataFrame, str]:
    csv_path = Path(path)
    attempts: List[Tuple[str, pd.DataFrame]] = []
    for sep in [None, ';', ',', '\t', '|']:
        try:
            if sep is None:
                df = pd.read_csv(csv_path, sep=None, engine='python')
                used_sep = 'auto'
            else:
                df = pd.read_csv(csv_path, sep=sep)
                used_sep = sep
            attempts.append((used_sep, df))
            if df.shape[1] > 1:
                return df, used_sep
        except Exception:
            continue
    if attempts:
        return max(attempts, key=lambda item: item[1].shape[1])
    raise RuntimeError(f'Не удалось прочитать CSV: {csv_path}')



def find_best_column(normalized_columns: Dict[str, str], positive: Iterable[str], negative: Iterable[str] | None = None) -> Optional[str]:
    positive = list(positive)
    negative = list(negative or [])
    best_col = None
    best_score = 0
    for original, norm in normalized_columns.items():
        score = 0
        for token in positive:
            if token in norm:
                score += 2 if norm == token else 1
        for token in negative:
            if token in norm:
                score -= 2 if norm == token else 1
        if score > best_score:
            best_score = score
            best_col = original
    return best_col



def derive_timestamp(df: pd.DataFrame, normalized_columns: Dict[str, str]) -> Tuple[pd.Series, str]:
    timestamp_col = find_best_column(
        normalized_columns,
        positive=['timestamp', 'datetime', 'fecha_hora', 'date_time', 'fecha', 'date', 'time'],
        negative=['update'],
    )
    if timestamp_col is not None:
        ts = pd.to_datetime(df[timestamp_col], errors='coerce', dayfirst=True)
        if ts.notna().mean() >= 0.30:
            return ts, timestamp_col

    date_col = find_best_column(normalized_columns, positive=['fecha', 'date', 'dia'])
    time_col = find_best_column(normalized_columns, positive=['hora', 'time'])
    if date_col is not None and time_col is not None and date_col != time_col:
        ts = pd.to_datetime(df[date_col].astype(str).str.strip() + ' ' + df[time_col].astype(str).str.strip(), errors='coerce', dayfirst=True)
        if ts.notna().mean() >= 0.30:
            return ts, f'{date_col}+{time_col}'

    session_col = find_best_column(normalized_columns, positive=['session', 'sesion', 'week', 'semana', 'sampling'])
    if session_col is not None:
        session_num = coerce_numeric_series(df[session_col]).fillna(0)
        session_num = (session_num - session_num.min()).fillna(0)
        ts = DEFAULT_START_TS + pd.to_timedelta(session_num.astype(int) * 7, unit='D')
        return ts, session_col

    ts = DEFAULT_START_TS + pd.to_timedelta(np.arange(len(df)) * 5, unit='m')
    return pd.Series(ts), '__synthetic_order__'



def derive_growth_stage(df: pd.DataFrame, normalized_columns: Dict[str, str], timestamp_series: pd.Series) -> Tuple[pd.Series, str]:
    stage_col = find_best_column(normalized_columns, positive=['growth_stage', 'stage', 'phenology', 'phase', 'etapa'])
    if stage_col is not None:
        out = df[stage_col].fillna('vegetative').map(lambda value: normalize_stage(str(value), default='vegetative'))
        return out, stage_col

    ordered_index = pd.Series(timestamp_series).rank(method='first', pct=True).fillna(0.5)
    stage_values: List[str] = []
    for pct in ordered_index:
        assigned = 'vegetative'
        for left, right, stage in STAGE_BINS:
            if left <= float(pct) < right:
                assigned = stage
                break
        stage_values.append(assigned)
    return pd.Series(stage_values), '__derived_from_time__'



def derive_plant_id(df: pd.DataFrame, normalized_columns: Dict[str, str]) -> Tuple[pd.Series, str]:
    plant_id_col = find_best_column(normalized_columns, positive=['plant_id', 'planta', 'plant', 'id_plant'], negative=['treatment'])
    if plant_id_col is not None:
        values = df[plant_id_col].astype(str).str.strip()
        values = values.replace({'': np.nan})
        if values.notna().any():
            return values.fillna(method='ffill').fillna(method='bfill').fillna('plant_unknown'), plant_id_col

    return pd.Series([f'plant_{idx:05d}' for idx in range(len(df))]), '__row_id__'



def canonical_sensor_mapping(normalized_columns: Dict[str, str]) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    mapping_specs = {
        'air_temp_c': (['air_temp', 'temp_air', 'temperature_air', 'ambient_temp', 'temperature', 'temp'], ['soil', 'water', 'fruit', 'stem', 'leaf']),
        'air_humidity_pct': (['air_humidity', 'relative_humidity', 'humidity', 'rh'], ['soil', 'substrate']),
        'light_lux': (['light', 'lux', 'par', 'radiation', 'irradiance'], []),
        'soil_moisture_pct': (['soil_moisture', 'substrate_moisture', 'soil_humidity', 'moisture', 'substrate'], ['air']),
        'solution_ph': (['solution_ph', 'water_ph', 'ph'], []),
        'ec_ms_cm': (['ec', 'electrical_conductivity', 'conductivity'], []),
        'tds_ppm': (['tds', 'ppm'], []),
        'co2_ppm': (['co2'], []),
        'leaf_wetness': (['leaf_wetness', 'wetness'], []),
    }
    used_cols: set[str] = set()
    for canonical, (positive, negative) in mapping_specs.items():
        col = find_best_column(normalized_columns, positive=positive, negative=negative)
        if col is not None and col not in used_cols:
            mapping[canonical] = col
            used_cols.add(col)
    return mapping



def is_identifier_like(column_name: str) -> bool:
    norm = normalize_name(column_name)
    bad_tokens = ['id', 'index', 'row', 'treatment', 'group', 'bloque', 'block', 'replicate', 'rep', 'session', 'week', 'semana']
    return any(token in norm for token in bad_tokens)



def main() -> None:
    args = parse_args()

    raw_df, used_separator = read_csv_flexible(args.input_csv)
    raw_df = raw_df.copy()
    normalized_columns = {col: normalize_name(col) for col in raw_df.columns}

    timestamp_series, timestamp_source = derive_timestamp(raw_df, normalized_columns)
    plant_id_series, plant_id_source = derive_plant_id(raw_df, normalized_columns)
    growth_stage_series, growth_stage_source = derive_growth_stage(raw_df, normalized_columns, timestamp_series)
    sensor_mapping = canonical_sensor_mapping(normalized_columns)

    out_df = pd.DataFrame(
        {
            'timestamp': pd.to_datetime(timestamp_series, errors='coerce'),
            'plant_id': plant_id_series.astype(str),
            'growth_stage': growth_stage_series.astype(str),
            'source_dataset_sensor': args.dataset_name,
        }
    )

    for canonical, source_col in sensor_mapping.items():
        out_df[canonical] = coerce_numeric_series(raw_df[source_col])

    extra_numeric_columns: List[str] = []
    reserved_source_columns = set(sensor_mapping.values())
    reserved_source_columns.update({timestamp_source, plant_id_source, growth_stage_source})

    for source_col in raw_df.columns:
        if source_col in reserved_source_columns:
            continue
        if is_identifier_like(source_col):
            continue
        numeric_series = coerce_numeric_series(raw_df[source_col])
        if numeric_series.notna().mean() < 0.60:
            continue
        target_col = f'extra_{normalize_name(source_col)}'
        if target_col in out_df.columns:
            continue
        out_df[target_col] = numeric_series
        extra_numeric_columns.append(target_col)

    out_df = out_df.dropna(subset=['timestamp']).sort_values(['timestamp', 'plant_id']).reset_index(drop=True)
    out_df['sensor_row_id'] = np.arange(len(out_df), dtype=np.int64)

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    out_df.to_csv(out_csv, index=False)

    summary = {
        'input_csv': str(Path(args.input_csv).resolve()),
        'output_csv': str(out_csv),
        'used_separator': used_separator,
        'num_rows': int(len(out_df)),
        'num_columns': int(out_df.shape[1]),
        'timestamp_source': timestamp_source,
        'plant_id_source': plant_id_source,
        'growth_stage_source': growth_stage_source,
        'canonical_sensor_mapping': sensor_mapping,
        'extra_numeric_columns': extra_numeric_columns,
        'all_output_columns': out_df.columns.tolist(),
    }
    save_json(summary, out_csv.with_suffix('.summary.json'))

    print('Готово.')
    print(f'Очищенный сенсорный CSV: {out_csv}')
    print(f'Строк: {len(out_df)}')
    print(f'Канонических сенсоров найдено: {len(sensor_mapping)}')
    print(f'Доп. числовых признаков: {len(extra_numeric_columns)}')


if __name__ == '__main__':
    main()
