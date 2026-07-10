from __future__ import annotations

from typing import Iterable

import numpy as np


def xyxy_to_yolo(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    xmin, ymin, xmax, ymax = map(float, box)
    if not np.isfinite([xmin, ymin, xmax, ymax]).all():
        raise ValueError("Bounding-box coordinates must be finite.")
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("Expected an xyxy box with positive width and height.")
    x_center = ((xmin + xmax) / 2.0) / width
    y_center = ((ymin + ymax) / 2.0) / height
    box_width = (xmax - xmin) / width
    box_height = (ymax - ymin) / height
    return x_center, y_center, box_width, box_height


def yolo_to_xyxy(box: Iterable[float], width: int, height: int) -> tuple[float, float, float, float]:
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")
    x_center, y_center, box_width, box_height = map(float, box)
    if not np.isfinite([x_center, y_center, box_width, box_height]).all():
        raise ValueError("Bounding-box coordinates must be finite.")
    bw = box_width * width
    bh = box_height * height
    cx = x_center * width
    cy = y_center * height
    xmin = cx - bw / 2.0
    ymin = cy - bh / 2.0
    xmax = cx + bw / 2.0
    ymax = cy + bh / 2.0
    return xmin, ymin, xmax, ymax


def clip_boxes_xyxy(boxes: np.ndarray, width: int, height: int) -> np.ndarray:
    if boxes.size == 0:
        return boxes.reshape(0, 4)
    boxes = boxes.copy()
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
    return boxes


def box_area_xyxy(boxes: np.ndarray) -> np.ndarray:
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    return widths * heights


def main() -> None:
    print(xyxy_to_yolo([10, 20, 110, 120], width=200, height=200))


if __name__ == "__main__":
    main()
