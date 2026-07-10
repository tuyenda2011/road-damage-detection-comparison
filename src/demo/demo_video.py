from __future__ import annotations

import argparse
from contextlib import nullcontext
import sys
import time
from pathlib import Path
from threading import Lock
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.demo.demo_image import load_detector
from src.utils.common import ensure_dir, project_path
from src.utils.visualization import draw_detections
from src.utils.video import create_video_writer, managed_video_output, video_properties


def run_video_demo(
    model: str,
    weights: str,
    source: str,
    output: str = "results",
    conf: float = 0.25,
    device: str = "auto",
    detector_bundle: tuple[object, Callable] | None = None,
    prediction_lock: Lock | None = None,
) -> Path:
    source_path = project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source video not found: {source_path}")

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source_path}")

    try:
        width, height, input_fps = video_properties(cap)
        output_dir = ensure_dir(output)
        out_path = output_dir / f"{source_path.stem}_{model}_detected.mp4"
        writer = create_video_writer(out_path, input_fps, (width, height))
    except Exception:
        cap.release()
        raise

    with managed_video_output(cap, writer, out_path):
        detector, predict_fn = detector_bundle or load_detector(model, weights, device=device)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            with prediction_lock if prediction_lock is not None else nullcontext():
                boxes, labels, scores = predict_fn(detector, frame, conf=conf, device=device)
            elapsed = max(time.perf_counter() - start, 1e-9)
            fps = 1.0 / elapsed
            result = draw_detections(frame, boxes, labels, scores)
            cv2.putText(result, f"FPS: {fps:.1f}", (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (30, 255, 30), 2)
            writer.write(result)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo object detection on one video.")
    parser.add_argument("--model", choices=["yolo", "faster_rcnn", "rtdetr"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_video_demo(args.model, args.weights, args.source, args.output, args.conf, args.device)
    print(f"Saved video result to: {out_path}")


if __name__ == "__main__":
    main()
