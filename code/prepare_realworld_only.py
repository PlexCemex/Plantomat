#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.utils import ensure_dir, seed_everything, stratified_split  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собирает CSV только из real-world изображений.")
    parser.add_argument("--mixed-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    df = pd.read_csv(args.mixed_csv).copy()
    if "source" not in df.columns:
        raise RuntimeError("В mixed CSV нет колонки source.")

    real_df = df[df["source"] != "plantvillage"].copy()
    if real_df.empty:
        raise RuntimeError("Не найдено real-world строк. Проверь mixed CSV.")

    real_df = stratified_split(
        real_df,
        label_col="label",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    real_df.to_csv(out_csv, index=False)

    print(f"Готово: {out_csv}")
    print("\nРаспределение по источникам:")
    print(real_df.groupby(['source', 'label']).size().unstack(fill_value=0))
    print("\nРаспределение по split:")
    print(real_df.groupby(['split', 'label']).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()