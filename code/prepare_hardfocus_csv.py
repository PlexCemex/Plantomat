#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


DEFAULT_FOCUS_CLASSES = [
    "target_spot",
    "leaf_mold",
    "spider_mites_two_spotted_spider_mite",
    "tomato_mosaic_virus",
    "early_blight",
    "septoria_leaf_spot",
]

DEFAULT_FOCUS_SOURCES = [
    "plantdoc",
    "pakistan_real",
    "realworld_tomato",
]


def parse_csv_list(value: str) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собрать hard-focus CSV для дообучения проблемных классов.")
    parser.add_argument("--input-csv", type=str, required=True)
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--focus-classes", type=str, default=",".join(DEFAULT_FOCUS_CLASSES))
    parser.add_argument("--focus-sources", type=str, default=",".join(DEFAULT_FOCUS_SOURCES))
    parser.add_argument("--focus-class-multiplier", type=int, default=3)
    parser.add_argument("--focus-source-multiplier", type=int, default=2)
    parser.add_argument("--intersection-multiplier", type=int, default=6)
    parser.add_argument("--max-multiplier", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if "split" not in df.columns:
        raise RuntimeError("В CSV нет колонки split.")
    if "label" not in df.columns:
        raise RuntimeError("В CSV нет колонки label.")
    if "source" not in df.columns:
        raise RuntimeError("В CSV нет колонки source.")

    focus_classes = set(parse_csv_list(args.focus_classes))
    focus_sources = set(parse_csv_list(args.focus_sources))

    train_df = df[df["split"].astype(str) == "train"].copy()
    rest_df = df[df["split"].astype(str) != "train"].copy()

    duplicated_parts = []
    summary_rows = []
    for _, row in train_df.iterrows():
        row_label = str(row["label"])
        row_source = str(row["source"])

        class_hit = row_label in focus_classes
        source_hit = row_source in focus_sources

        multiplier = 1
        if class_hit and source_hit:
            multiplier = args.intersection_multiplier
        elif class_hit:
            multiplier = args.focus_class_multiplier
        elif source_hit:
            multiplier = args.focus_source_multiplier

        multiplier = max(1, min(int(multiplier), int(args.max_multiplier)))
        duplicated_parts.extend([row.to_dict()] * multiplier)
        summary_rows.append(
            {
                "label": row_label,
                "source": row_source,
                "multiplier": multiplier,
                "class_hit": int(class_hit),
                "source_hit": int(source_hit),
            }
        )

    train_out = pd.DataFrame(duplicated_parts)
    out_df = pd.concat([train_out, rest_df], ignore_index=True)
    out_df.to_csv(output_path, index=False)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_path.with_name(output_path.stem + "_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    class_before = train_df["label"].value_counts().sort_index()
    class_after = train_out["label"].value_counts().sort_index()
    compare_df = pd.DataFrame({
        "train_rows_before": class_before,
        "train_rows_after": class_after,
    }).fillna(0)
    compare_df["boost_ratio"] = compare_df["train_rows_after"] / compare_df["train_rows_before"].replace(0, 1)
    compare_path = output_path.with_name(output_path.stem + "_class_compare.csv")
    compare_df.to_csv(compare_path)

    source_before = train_df["source"].value_counts().sort_index()
    source_after = train_out["source"].value_counts().sort_index()
    source_compare_df = pd.DataFrame({
        "train_rows_before": source_before,
        "train_rows_after": source_after,
    }).fillna(0)
    source_compare_df["boost_ratio"] = source_compare_df["train_rows_after"] / source_compare_df["train_rows_before"].replace(0, 1)
    source_compare_path = output_path.with_name(output_path.stem + "_source_compare.csv")
    source_compare_df.to_csv(source_compare_path)

    print(f"Готово: {output_path}")
    print(f"Фокус-классы: {sorted(focus_classes)}")
    print(f"Фокус-источники: {sorted(focus_sources)}")
    print(f"Train rows before: {len(train_df)}")
    print(f"Train rows after: {len(train_out)}")
    print(f"Val/Test rows kept unchanged: {len(rest_df)}")
    print(f"Summary: {summary_path}")
    print(f"Class compare: {compare_path}")
    print(f"Source compare: {source_compare_path}")


if __name__ == "__main__":
    main()
