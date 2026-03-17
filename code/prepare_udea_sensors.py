#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.utils import detect_csv_separator, ensure_dir, write_json  # noqa: E402


CANONICAL_COLUMNS = [
    "date",
    "plant_id",
    "treatment",
    "air_temp_c",
    "air_humidity_pct",
    "soil_moisture_pct",
    "solution_ph",
    "ec_ms_cm",
    "tds_ppm",
    "light_lux",
    "co2_ppm",
    "leaf_wetness",
    "growth_stage",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Очистка и канонизация CSV датчиков томата UdeA.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--summary-json", type=str, default=None)
    return parser.parse_args()


def to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def build_mapping(df: pd.DataFrame) -> Dict[str, str]:
    cols = set(df.columns)

    def pick(*candidates: str) -> Optional[str]:
        for c in candidates:
            if c in cols:
                return c
        return None

    mapping: Dict[str, str] = {
        "date": pick("Date") or "derived_missing",
        "plant_id": pick("Plant") or "derived_row_id",
        "treatment": pick("Treatment") or "derived_unknown",
        "air_temp_c": pick("Air_sensor_Temperature[C]", "7in1_S_Temperature[C]") or "derived_nan",
        "air_humidity_pct": pick("Air_sensor_Humidity[%RH]") or "derived_nan",
        "soil_moisture_pct": pick("7in1_Moisture[%RH]") or "derived_from_humidity",
        "solution_ph": pick("Sap pH", "Hanna Soil pH", "Horiba Soil pH", "7in1_Ph[pH]") or "derived_nan",
        "ec_ms_cm": pick("Sap EC (mS/cm)", "Horiba Soil EC (mS/cm)", "7in1_EC[uS/cm]") or "derived_nan",
        "tds_ppm": pick("Sap NO3 (ppm)", "Horiba Soil NO3 (ppm)") or "derived_from_ec",
        "light_lux": pick("Pynamometer_Radiation[W/m2]") or "derived_from_treatment",
        "co2_ppm": "derived_constant_400ppm",
        "leaf_wetness": "derived_from_humidity_and_moisture",
        "growth_stage": "derived_from_flowers_fruits_harvest",
    }
    return mapping


def infer_growth_stage(df: pd.DataFrame) -> pd.Series:
    flowers = to_numeric(df.get("Number of Flowers", pd.Series(index=df.index, dtype=float))).fillna(0)
    fruits = to_numeric(df.get("Number of Fruits", pd.Series(index=df.index, dtype=float))).fillna(0)
    harvested = to_numeric(df.get("Numer of Harvested Fruits", pd.Series(index=df.index, dtype=float))).fillna(0)

    stage = np.full(len(df), "vegetative", dtype=object)
    stage = np.where(flowers > 0, "flowering", stage)
    stage = np.where(fruits > 0, "fruiting", stage)
    stage = np.where(harvested > 0, "harvest", stage)
    return pd.Series(stage, index=df.index, dtype="object")


def derive_soil_moisture(df: pd.DataFrame, out: pd.DataFrame) -> pd.Series:
    if "7in1_Moisture[%RH]" in df.columns:
        return to_numeric(df["7in1_Moisture[%RH]"])

    air_h = out["air_humidity_pct"].fillna(out["air_humidity_pct"].median())
    return (air_h * 0.75).clip(20, 95)


def derive_tds(df: pd.DataFrame, out: pd.DataFrame) -> pd.Series:
    if "Sap NO3 (ppm)" in df.columns:
        return to_numeric(df["Sap NO3 (ppm)"])
    if "Horiba Soil NO3 (ppm)" in df.columns:
        return to_numeric(df["Horiba Soil NO3 (ppm)"])

    ec = out["ec_ms_cm"].fillna(out["ec_ms_cm"].median())
    return (ec * 640.0).clip(lower=0)


def derive_light(df: pd.DataFrame, out: pd.DataFrame) -> pd.Series:
    if "Pynamometer_Radiation[W/m2]" in df.columns:
        rad = to_numeric(df["Pynamometer_Radiation[W/m2]"])
        return (rad * 120.0).clip(lower=0)

    treatment = out["treatment"].astype(str).str.lower()
    base = np.where(treatment.str.contains("shade|shadow"), 9000.0, 18000.0)
    return pd.Series(base, index=df.index, dtype=float)


def derive_co2(df: pd.DataFrame) -> pd.Series:
    return pd.Series(np.full(len(df), 400.0), index=df.index, dtype=float)


def derive_leaf_wetness(out: pd.DataFrame) -> pd.Series:
    hum = out["air_humidity_pct"].fillna(out["air_humidity_pct"].median())
    moist = out["soil_moisture_pct"].fillna(out["soil_moisture_pct"].median())

    wet = ((hum - 60.0) / 40.0) * 0.7 + ((moist - 40.0) / 60.0) * 0.3
    wet = wet.clip(0.0, 1.0)
    return wet.astype(float)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    sep = detect_csv_separator(input_csv)
    df = pd.read_csv(input_csv, sep=sep)

    mapping = build_mapping(df)
    out = pd.DataFrame(index=df.index)

    if mapping["date"] != "derived_missing":
        out["date"] = pd.to_datetime(df[mapping["date"]], errors="coerce", dayfirst=True)
    else:
        out["date"] = pd.NaT

    if mapping["plant_id"] != "derived_row_id":
        out["plant_id"] = df[mapping["plant_id"]]
    else:
        out["plant_id"] = [f"plant_{i}" for i in range(len(df))]

    if mapping["treatment"] != "derived_unknown":
        out["treatment"] = df[mapping["treatment"]].astype(str).str.strip()
    else:
        out["treatment"] = "unknown"

    if mapping["air_temp_c"] != "derived_nan":
        out["air_temp_c"] = to_numeric(df[mapping["air_temp_c"]])
    else:
        out["air_temp_c"] = np.nan

    if mapping["air_humidity_pct"] != "derived_nan":
        out["air_humidity_pct"] = to_numeric(df[mapping["air_humidity_pct"]])
    else:
        out["air_humidity_pct"] = np.nan

    if mapping["solution_ph"] != "derived_nan":
        out["solution_ph"] = to_numeric(df[mapping["solution_ph"]])
    else:
        out["solution_ph"] = np.nan

    if mapping["ec_ms_cm"] == "7in1_EC[uS/cm]":
        out["ec_ms_cm"] = to_numeric(df["7in1_EC[uS/cm]"]) / 1000.0
    elif mapping["ec_ms_cm"] != "derived_nan":
        out["ec_ms_cm"] = to_numeric(df[mapping["ec_ms_cm"]])
    else:
        out["ec_ms_cm"] = np.nan

    out["soil_moisture_pct"] = derive_soil_moisture(df, out)
    out["tds_ppm"] = derive_tds(df, out)
    out["light_lux"] = derive_light(df, out)
    out["co2_ppm"] = derive_co2(df)
    out["leaf_wetness"] = derive_leaf_wetness(out)
    out["growth_stage"] = infer_growth_stage(df)

    numeric_cols = [
        "air_temp_c",
        "air_humidity_pct",
        "soil_moisture_pct",
        "solution_ph",
        "ec_ms_cm",
        "tds_ppm",
        "light_lux",
        "co2_ppm",
        "leaf_wetness",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["growth_stage"] = out["growth_stage"].fillna("vegetative").astype(str).str.strip().str.lower()
    out["row_id"] = range(len(out))

    output_csv = Path(args.output_csv).resolve()
    ensure_dir(output_csv.parent)
    out.to_csv(output_csv, index=False)

    summary = {
        "input_csv": str(input_csv),
        "output_csv": str(output_csv),
        "separator": sep,
        "rows": int(len(out)),
        "column_mapping": mapping,
        "derived_fields": {
            "co2_ppm": "constant 400.0 ppm baseline proxy",
            "leaf_wetness": "derived from air_humidity_pct and soil_moisture_pct, range 0..1",
            "growth_stage": "derived from Number of Flowers / Number of Fruits / Numer of Harvested Fruits",
            "light_lux": "derived from Pynamometer_Radiation[W/m2] * 120 when radiation exists",
            "tds_ppm": "derived from ec_ms_cm * 640 if NO3 ppm columns absent",
        },
        "missing_ratio": {
            col: float(out[col].isna().mean())
            for col in numeric_cols
        },
        "growth_stage_values": sorted(out["growth_stage"].dropna().astype(str).unique().tolist()),
        "date_min": None if out["date"].isna().all() else str(out["date"].min()),
        "date_max": None if out["date"].isna().all() else str(out["date"].max()),
    }

    summary_path = Path(args.summary_json).resolve() if args.summary_json else output_csv.with_suffix(".summary.json")
    write_json(summary, summary_path)

    print(f"Готово: {output_csv}")
    print(f"Сводка: {summary_path}")
    print("Сопоставление колонок:")
    for k, v in mapping.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()