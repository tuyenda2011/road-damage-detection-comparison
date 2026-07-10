from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2

from src.utils.common import ensure_dir, project_path
from src.utils.visualization import draw_detections, save_image


def load_detector(model: str, weights: str, device: str = "auto"):
    model = model.lower()
    if model == "yolo":
        from src.models.yolo.infer import load_yolo, predict_yolo_image

        detector = load_yolo(weights)
        return detector, predict_yolo_image
    if model == "faster_rcnn":
        from src.models.faster_rcnn.infer import predict_faster_rcnn_image
        from src.models.faster_rcnn.model import load_faster_rcnn
        from src.utils.common import resolve_device

        detector = load_faster_rcnn(project_path(weights), resolve_device(device))
        return detector, predict_faster_rcnn_image
    if model == "rtdetr":
        from src.models.rtdetr.infer import load_rtdetr, predict_rtdetr_image

        detector = load_rtdetr(weights)
        return detector, predict_rtdetr_image
    raise ValueError(f"Unsupported model: {model}")


def run_image_demo(
    model: str,
    weights: str,
    source: str,
    output: str = "results",
    conf: float = 0.25,
    device: str = "auto",
) -> Path:
    source_path = project_path(source)
    if not source_path.exists():
        raise FileNotFoundError(f"Source image not found: {source_path}")
    image = cv2.imread(str(source_path))
    if image is None:
        raise ValueError(f"Could not decode image: {source_path}")

    detector, predict_fn = load_detector(model, weights, device=device)
    boxes, labels, scores = predict_fn(detector, image, conf=conf, device=device)
    result = draw_detections(image, boxes, labels, scores)

    output_dir = ensure_dir(output)
    out_path = output_dir / f"{source_path.stem}_{model}_detected{source_path.suffix}"
    save_image(out_path, result)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Demo object detection on one image.")
    parser.add_argument("--model", choices=["yolo", "faster_rcnn", "rtdetr"], required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", default="results")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = run_image_demo(args.model, args.weights, args.source, args.output, args.conf, args.device)
    print(f"Saved image result to: {out_path}")


if __name__ == "__main__":
    main()
