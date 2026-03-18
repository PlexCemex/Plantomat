#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from plantomat.utils import ensure_dir, seed_everything, stratified_split  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


SUPPORTED = {
    "healthy",
    "bacterial_spot",
    "early_blight",
    "late_blight",
    "leaf_mold",
    "septoria_leaf_spot",
    "tomato_mosaic_virus",
    "tomato_yellow_leaf_curl_virus",
    "spider_mites_two_spotted_spider_mite",
    "target_spot",
}


SKIP_PATTERNS = [
    "leaf miner",
    "leaf_miner",
    "insect damage",
    "insect_damage",
    "magnesium deficiency",
    "magnesium_deficiency",
    "gray mold",
    "bacterial wilt",
    "powdery mildew",
    "powdery_mildew",
    "gray leaf spot",
    "grey leaf spot",
    "black spot",
    "uncategorized",
    "other",
    "unhealthy",
]


ALIASES = [
    ("tomato_yellow_leaf_curl_virus", ["yellow leaf curl", "yellow_leaf_curl", "leaf yellow virus", "leaf yellow", "yellow virus", "tylcv", "leaf curl virus", "tomato leaf curl virus", "curl virus"]),
    ("tomato_mosaic_virus", ["mosaic virus", "mosaic_virus", "leaf mosaic", "tomato mosaic", "mosaic"]),
    ("septoria_leaf_spot", ["septoria leaf spot", "septoria_leaf_spot", "septoria"]),
    ("spider_mites_two_spotted_spider_mite", ["two spotted spider mite", "two_spotted_spider_mite", "two-spotted spider mite", "spider mite", "spider_mites"]),
    ("bacterial_spot", ["bacterial spot", "baterial spot", "bacterial_spot", "leaf bacterial spot"]),
    ("early_blight", ["early blight", "early_blight"]),
    ("late_blight", ["late blight", "late_blight", "blight leaf"]),
    ("leaf_mold", ["leaf mold", "leaf_mold", "mold leaf", "tomato mold", "black leaf mold", "cercospora leaf mold", "cercospora"]),
    ("target_spot", ["target spot", "target_spot"]),
    ("healthy", ["tomato healthy", "healthy leaf", "tomato leaf", "healthy"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Собирает общий CSV из PlantVillage и реальных датасетов.")
    parser.add_argument("--base-csv", type=str, required=True, help="CSV PlantVillage из prepare_plantvillage.py")
    parser.add_argument("--plantdoc-root", type=str, default=None)
    parser.add_argument("--pakistan-root", type=str, default=None)
    parser.add_argument("--realworld-root", type=str, default=None, help="Например Mendeley rnbsw72zb5")
    parser.add_argument("--extra-root", action="append", default=[], help="Дополнительный корень с реальными изображениями. Можно указывать несколько раз.")
    parser.add_argument("--output-csv", type=str, required=True)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def detect_label(path: Path) -> str | None:
    rel = " / ".join(path.parts).lower()

    for pat in SKIP_PATTERNS:
        if pat in rel:
            return None

    for canonical, aliases in ALIASES:
        for alias in aliases:
            if alias in rel:
                return canonical
    return None


def collect_images(root: Path, source_name: str) -> list[dict]:
    rows: list[dict] = []
    if not root or not root.exists():
        return rows

    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        label = detect_label(p.relative_to(root))
        if label is None or label not in SUPPORTED:
            continue
        rows.append(
            {
                "image_path": str(p.resolve()),
                "label": label,
                "source": source_name,
                "source_relpath": str(p.relative_to(root)),
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    base_df = pd.read_csv(args.base_csv).copy()
    base_df["source"] = "plantvillage"
    if "source_relpath" not in base_df.columns:
        base_df["source_relpath"] = base_df.get("source_class_dir", "plantvillage")

    extra_rows: list[dict] = []
    roots = [
        ("plantdoc", args.plantdoc_root),
        ("pakistan_real", args.pakistan_root),
        ("realworld_tomato", args.realworld_root),
    ] + [(f"extra_{i+1}", p) for i, p in enumerate(args.extra_root)]

    for source_name, root_str in roots:
        if not root_str:
            continue
        extra_rows.extend(collect_images(Path(root_str).resolve(), source_name))

    extra_df = pd.DataFrame(extra_rows) if extra_rows else pd.DataFrame(columns=["image_path", "label", "source", "source_relpath"])
    df = pd.concat(
        [
            base_df[["image_path", "label", "source", "source_relpath"]],
            extra_df[["image_path", "label", "source", "source_relpath"]],
        ],
        ignore_index=True,
    )

    if df.empty:
        raise RuntimeError("Не найдено ни одного изображения. Проверь пути к датасетам.")

    df = stratified_split(
        df,
        label_col="label",
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    out_csv = Path(args.output_csv).resolve()
    ensure_dir(out_csv.parent)
    df.to_csv(out_csv, index=False)

    print(f"Готово: {out_csv}")
    print("\nРаспределение по источникам:")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0))
    print("\nРаспределение по split:")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))


if __name__ == "__main__":
    main()
