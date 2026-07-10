from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.dataset.road_damage_dataset import read_yolo_label
from src.demo.demo_image import load_detector
from src.evaluation.metrics import aggregate_precision_recall, map50
from src.utils.common import (
    ensure_dir,
    list_images,
    portable_path,
    project_path,
    resolve_device,
    resolve_inference_weights,
    validate_confidence,
)


def resolve_image_and_label_dirs(data: str | Path) -> tuple[Path, Path]:
    data_path = project_path(data)
    candidates: list[tuple[Path, Path]] = []
    candidates.append((data_path / "images" / "test", data_path / "labels" / "test"))
    candidates.append((data_path / "images", data_path / "labels"))
    if data_path.name in {"train", "val", "test"}:
        split = data_path.name
        candidates.append((data_path.parent / "images" / split, data_path.parent / "labels" / split))
        if data_path.parent.name == "images":
            candidates.append((data_path, data_path.parent.parent / "labels" / split))

    for image_dir, label_dir in candidates:
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    checked = "\n".join(f"  images={images}, labels={labels}" for images, labels in candidates)
    raise FileNotFoundError(f"Could not resolve matching image/label directories from --data {data_path}.\n{checked}")


def evaluate_model(
    model_name: str,
    weights: str,
    data: str,
    output_csv: str = "results/metrics.csv",
    conf: float = 0.25,
    device: str = "auto",
) -> dict[str, float | str]:
    conf = validate_confidence(conf)
    image_dir, label_dir = resolve_image_and_label_dirs(data)
    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No test images found in {image_dir}")

    missing_labels = 0
    has_ground_truth = False
    for image_path in images:
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        if not label_path.is_file():
            missing_labels += 1
            continue
        if not has_ground_truth:
            boxes, _ = read_yolo_label(label_path, image_width=1, image_height=1)
            has_ground_truth = len(boxes) > 0
    if missing_labels:
        raise FileNotFoundError(f"{missing_labels} test image(s) have no matching label file in {label_dir}")
    if not has_ground_truth:
        raise ValueError(
            f"Evaluation split {label_dir} contains no ground-truth boxes; "
            "use an annotated split or rebuild train/val/test from annotated data."
        )

    weight_locations = {
        "yolo": ("runs/yolo", ".pt"),
        "faster_rcnn": ("runs/faster_rcnn", ".pth"),
        "rtdetr": ("runs/rtdetr", ".pt"),
    }
    model_dir, extension = weight_locations[model_name]
    resolved_weights = resolve_inference_weights(weights, model_dir, extension)
    detector, predict_fn = load_detector(model_name, str(resolved_weights), device=device)
    predictions: list[dict] = []
    thresholded_predictions: list[dict] = []
    ground_truths: list[dict] = []
    total_time = 0.0
    warmed_up = False

    for image_path in tqdm(images, desc=f"Evaluating {model_name}"):
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
        gt_boxes, gt_labels = read_yolo_label(label_path, width, height)
        gt_labels = gt_labels - 1

        if not warmed_up:
            predict_fn(detector, image, conf=conf, device=device)
            warmed_up = True

        start = time.perf_counter()
        threshold_boxes, threshold_labels, threshold_scores = predict_fn(
            detector, image, conf=conf, device=device
        )
        total_time += time.perf_counter() - start

        if conf > 0.001:
            boxes, labels, scores = predict_fn(detector, image, conf=0.001, device=device)
        else:
            boxes, labels, scores = threshold_boxes, threshold_labels, threshold_scores

        predictions.append({"boxes": boxes, "labels": labels.astype(np.int64), "scores": scores})
        thresholded_predictions.append(
            {
                "boxes": threshold_boxes,
                "labels": threshold_labels.astype(np.int64),
                "scores": threshold_scores,
            }
        )
        ground_truths.append({"boxes": gt_boxes, "labels": gt_labels.astype(np.int64)})

    if not predictions:
        raise ValueError(f"No readable test images found in {image_dir}")

    precision, recall = aggregate_precision_recall(thresholded_predictions, ground_truths)
    result = {
        "model": model_name,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "map50": round(float(map50(predictions, ground_truths)), 6),
        "fps": round(len(predictions) / max(total_time, 1e-9), 3),
        "confidence": conf,
        "samples": len(predictions),
        "device": resolve_device(device),
        "data_path": portable_path(image_dir),
        "weight_path": portable_path(resolved_weights),
    }
    output_path = project_path(output_csv)
    ensure_dir(output_path.parent)
    if output_path.exists():
        df = pd.read_csv(output_path)
        df = df[df["model"] != model_name] if "model" in df.columns else df
        df = pd.concat([df, pd.DataFrame([result])], ignore_index=True)
    else:
        df = pd.DataFrame([result])
    df.to_csv(output_path, index=False)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a detector on the processed test split.")
    parser.add_argument("--model", choices=["yolo", "faster_rcnn", "rtdetr"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--data", default="data/processed/test", help="Test folder or data/processed root/test alias.")
    parser.add_argument("--output", default="results/metrics.csv")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = evaluate_model(args.model, args.weights, args.data, args.output, args.conf, args.device)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Error: {exc}") from exc
    print(result)


if __name__ == "__main__":
    main()
