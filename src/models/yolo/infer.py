from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from ultralytics import YOLO

from src.utils.common import (
    ensure_dir,
    project_path,
    resolve_inference_weights,
    resolve_ultralytics_device,
    source_kind,
    validate_confidence,
)
from src.utils.visualization import draw_detections, save_image
from src.utils.video import create_video_writer, managed_video_output, video_properties


def load_yolo(weights: str | Path) -> YOLO:
    weight_path = resolve_inference_weights(weights, "runs/yolo", ".pt")
    return YOLO(str(weight_path))


def predict_yolo_image(
    model: YOLO,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    device: str = "auto",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    conf = validate_confidence(conf)
    result = model.predict(image_bgr, conf=conf, device=resolve_ultralytics_device(device), verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return empty_predictions()
    boxes = result.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    labels = result.boxes.cls.detach().cpu().numpy().astype(np.int64)
    scores = result.boxes.conf.detach().cpu().numpy().astype(np.float32)
    return boxes, labels, scores


def empty_predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((0, 4), dtype=np.float32),
        np.zeros((0,), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
    )


def infer_yolo_image(
    weights: str,
    source: str,
    output: str = "results",
    conf: float = 0.25,
    device: str = "auto",
) -> Path:
    source_path = project_path(source)
    image = cv2.imread(str(source_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source_path}")
    model = load_yolo(weights)
    boxes, labels, scores = predict_yolo_image(model, image, conf=conf, device=device)
    result = draw_detections(image, boxes, labels, scores)
    out_path = ensure_dir(output) / f"{source_path.stem}_yolo_detected{source_path.suffix}"
    save_image(out_path, result)
    return out_path


def infer_yolo_video(
    weights: str,
    source: str,
    output: str = "results",
    conf: float = 0.25,
    device: str = "auto",
) -> Path:
    source_path = project_path(source)
    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {source_path}")

    try:
        width, height, input_fps = video_properties(cap)
        out_path = ensure_dir(output) / f"{source_path.stem}_yolo_detected.mp4"
        writer = create_video_writer(out_path, input_fps, (width, height))
    except Exception:
        cap.release()
        raise

    with managed_video_output(cap, writer, out_path):
        model = load_yolo(weights)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            boxes, labels, scores = predict_yolo_image(model, frame, conf=conf, device=device)
            fps = 1.0 / max(time.perf_counter() - start, 1e-9)
            result = draw_detections(frame, boxes, labels, scores)
            cv2.putText(result, f"FPS: {fps:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            writer.write(result)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO inference on an image or video.")
    parser.add_argument("--weights", default="runs/yolo/best.pt")
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if source_kind(args.source) == "video":
        out = infer_yolo_video(args.weights, args.source, args.output, args.conf, args.device)
    else:
        out = infer_yolo_image(args.weights, args.source, args.output, args.conf, args.device)
    print(f"Saved result to: {out}")


if __name__ == "__main__":
    main()
