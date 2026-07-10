from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np

from src.dataset.road_damage_dataset import read_yolo_label
from src.utils.common import CLASS_NAMES, ensure_dir, list_images, project_path


COLORS = {
    0: (0, 180, 255),
    1: (255, 120, 60),
    2: (70, 220, 120),
    3: (230, 70, 180),
}


def read_yolo_txt(label_path: Path, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes, labels = read_yolo_label(label_path, width, height)
    return boxes, labels - 1


def draw_boxes(
    image: np.ndarray,
    boxes: np.ndarray,
    labels: np.ndarray,
    class_names: list[str],
    thickness: int,
    plainbox: bool = False,
) -> np.ndarray:
    output = image.copy()

    for box, label in zip(boxes, labels):
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        color = COLORS.get(int(label), (255, 255, 255))
        cv2.rectangle(output, (x1, y1), (x2, y2), color, thickness)
        if plainbox:
            continue

        name = class_names[int(label)] if 0 <= int(label) < len(class_names) else str(label)

        (text_w, text_h), baseline = cv2.getTextSize(name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y_text = max(y1, text_h + baseline + 4)
        cv2.rectangle(output, (x1, y_text - text_h - baseline - 5), (x1 + text_w + 8, y_text), color, -1)
        cv2.putText(
            output,
            name,
            (x1 + 4, y_text - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )

    return output


def draw_folder(
    image_dir: str | Path,
    label_dir: str | Path,
    output_dir: str | Path,
    class_names: list[str],
    thickness: int = 2,
    limit: int | None = None,
    plainbox: bool = False,
    require_labels: bool = False,
) -> int:
    image_dir = project_path(image_dir)
    label_dir = project_path(label_dir)
    output_dir = ensure_dir(output_dir)

    images = list_images(image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")
    if require_labels:
        labeled_images = []
        for image_path in images:
            label_path = label_dir / image_path.relative_to(image_dir).with_suffix(".txt")
            if label_path.exists() and label_path.stat().st_size > 0:
                labeled_images.append(image_path)
        images = labeled_images
        if not images:
            raise FileNotFoundError(f"No non-empty label files found in {label_dir}")
    if limit is not None:
        images = images[:limit]

    count = 0
    for image_path in images:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        rel_path = image_path.relative_to(image_dir)
        label_path = label_dir / rel_path.with_suffix(".txt")
        boxes, labels = read_yolo_txt(label_path, width, height)

        output = draw_boxes(image, boxes, labels, class_names, thickness, plainbox)
        save_path = output_dir / rel_path
        save_path.parent.mkdir(parents=True, exist_ok=True)
        ok = cv2.imwrite(str(save_path), output)
        if not ok:
            raise IOError(f"Could not save image: {save_path}")
        count += 1

    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw bounding boxes from YOLO .txt labels on an image folder.")
    parser.add_argument("--images", required=True, help="Folder containing images.")
    parser.add_argument("--labels", required=True, help="Folder containing YOLO .txt labels.")
    parser.add_argument("--output", default="results/label_visualization", help="Folder for annotated images.")
    parser.add_argument("--class-names", default=",".join(CLASS_NAMES), help="Comma-separated class names.")
    parser.add_argument("--thickness", type=int, default=2, help="Bounding box line thickness.")
    parser.add_argument("--limit", type=int, default=None, help="Only draw this many images.")
    parser.add_argument("--plainbox", action="store_true", help="Draw boxes only, without class labels.")
    parser.add_argument("--require-labels", action="store_true", help="Only process images with non-empty label files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_names = [name.strip() for name in args.class_names.split(",") if name.strip()]
    total = draw_folder(
        args.images,
        args.labels,
        args.output,
        class_names,
        args.thickness,
        args.limit,
        args.plainbox,
        args.require_labels,
    )
    print(f"Saved {total} annotated images to {project_path(args.output)}")


if __name__ == "__main__":
    main()
