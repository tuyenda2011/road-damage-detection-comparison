from __future__ import annotations

from pathlib import Path
from typing import Sequence
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.utils.common import CLASS_NAMES


COLORS = {
    0: (0, 180, 255),
    1: (255, 120, 60),
    2: (70, 220, 120),
    3: (230, 70, 180),
}


def draw_detections(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Sequence[int],
    scores: Sequence[float] | None = None,
    class_names: Sequence[str] = CLASS_NAMES,
    thickness: int = 2,
) -> np.ndarray:
    boxes_array = np.asarray(boxes, dtype=np.float32)
    if boxes_array.size == 0:
        boxes_array = boxes_array.reshape(0, 4)
    if boxes_array.ndim != 2 or boxes_array.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4).")
    labels_array = np.asarray(labels).reshape(-1)
    scores_array = np.asarray(scores if scores is not None else [1.0] * len(boxes_array), dtype=np.float32).reshape(-1)
    if not (len(boxes_array) == len(labels_array) == len(scores_array)):
        raise ValueError("boxes, labels and scores must have the same length.")
    if not np.isfinite(boxes_array).all() or not np.isfinite(scores_array).all():
        raise ValueError("Detection coordinates and scores must be finite.")
    if thickness <= 0:
        raise ValueError("thickness must be positive.")

    output = image.copy()

    for box, label, score in zip(boxes_array, labels_array, scores_array):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = COLORS.get(int(label), (255, 255, 255))
        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)
        caption = f"{name} {float(score):.2f}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        (tw, th), baseline = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        y_text = max(y1, th + baseline + 4)
        cv2.rectangle(output, (x1, y_text - th - baseline - 4), (x1 + tw + 6, y_text), color, -1)
        cv2.putText(
            output,
            caption,
            (x1 + 3, y_text - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return output


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise IOError(f"Could not save image to {path}")


def main() -> None:
    print("Visualization helpers for drawing road-damage detections.")


if __name__ == "__main__":
    main()
