from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils.common import ensure_dir, project_path


def model_size_mb(weight_path: str) -> float:
    path = project_path(weight_path)
    return round(path.stat().st_size / (1024 * 1024), 2) if path.exists() else 0.0


def make_note(row: pd.Series) -> str:
    fps = float(row.get("fps", 0))
    map50_value = float(row.get("map50", 0))
    if fps >= 20 and map50_value >= 0.5:
        return "Can bang tot giua toc do va do chinh xac"
    if fps >= 20:
        return "Nhanh, phu hop demo real-time"
    if map50_value >= 0.5:
        return "Do chinh xac kha, can toi uu toc do"
    return "Can train them hoac tinh chinh tham so"


def compare_results(input_csv: str = "results/metrics.csv", output_csv: str = "results/comparison_table.csv") -> Path:
    input_path = project_path(input_csv)
    if not input_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {input_path}")
    df = pd.read_csv(input_path)
    if df.empty:
        raise ValueError(f"Metrics CSV is empty: {input_path}")
    required = {"model", "precision", "recall", "map50", "fps", "weight_path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Metrics CSV is missing required columns: {', '.join(missing)}")
    numeric_columns = ["precision", "recall", "map50", "fps"]
    if df[numeric_columns].isna().any().any() or not np.isfinite(df[numeric_columns].to_numpy()).all():
        raise ValueError("Metrics CSV contains non-finite metric values.")
    if df["model"].duplicated().any():
        raise ValueError("Metrics CSV contains duplicate model rows.")

    protocol_columns = [column for column in ("confidence", "samples", "data_path", "device") if column in df]
    mismatched = [column for column in protocol_columns if df[column].nunique(dropna=False) > 1]
    if mismatched:
        raise ValueError(
            "Cannot compare runs with different evaluation protocols: " + ", ".join(mismatched)
        )

    table = pd.DataFrame(
        {
            "Model": df["model"],
            "Precision": df["precision"],
            "Recall": df["recall"],
            "mAP@50": df["map50"],
            "FPS": df["fps"],
            "Model Size": df["weight_path"].apply(lambda p: f"{model_size_mb(p)} MB"),
            "Nhan xet": df.apply(make_note, axis=1),
        }
    )
    output_path = project_path(output_csv)
    ensure_dir(output_path.parent)
    table.to_csv(output_path, index=False)

    figures_dir = ensure_dir(output_path.parent / "figures")
    metrics = ["precision", "recall", "map50", "fps"]
    labels = ["Precision", "Recall", "mAP@50", "FPS"]
    for metric, label in zip(metrics, labels):
        plt.figure(figsize=(7, 4))
        plt.bar(df["model"], df[metric], color=["#2d9cdb", "#27ae60", "#f2994a"][: len(df)])
        plt.title(f"So sanh {label}")
        plt.xlabel("Model")
        plt.ylabel(label)
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_comparison.png", dpi=160)
        plt.close()
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create comparison table and charts from metrics.csv.")
    parser.add_argument("--input", default="results/metrics.csv")
    parser.add_argument("--output", default="results/comparison_table.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = compare_results(args.input, args.output)
    print(f"Comparison table saved to: {output}")


if __name__ == "__main__":
    main()
