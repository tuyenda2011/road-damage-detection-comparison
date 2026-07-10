from __future__ import annotations

import xml.etree.ElementTree as ET
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as F

from src.utils.bbox import clip_boxes_xyxy, yolo_to_xyxy
from src.utils.common import CLASS_NAMES, CLASS_TO_ID, project_path


def parse_voc_xml(xml_path: str | Path, allowed_classes: set[str] | None = None) -> dict[str, Any]:
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    width = int(size.findtext("width", "0")) if size is not None else 0
    height = int(size.findtext("height", "0")) if size is not None else 0

    boxes: list[list[float]] = []
    labels: list[int] = []
    names: list[str] = []
    for obj in root.findall("object"):
        name = obj.findtext("name", "").strip()
        if allowed_classes is not None and name not in allowed_classes:
            continue
        if name not in CLASS_TO_ID:
            continue
        bnd = obj.find("bndbox")
        if bnd is None:
            continue
        xmin = float(bnd.findtext("xmin", "0"))
        ymin = float(bnd.findtext("ymin", "0"))
        xmax = float(bnd.findtext("xmax", "0"))
        ymax = float(bnd.findtext("ymax", "0"))
        if not np.isfinite([xmin, ymin, xmax, ymax]).all():
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_ID[name] + 1)  # Faster R-CNN uses 0 for background.
        names.append(name)
    return {
        "filename": root.findtext("filename", "").strip(),
        "width": width,
        "height": height,
        "boxes": boxes,
        "labels": labels,
        "names": names,
    }


def read_yolo_label(label_path: str | Path, image_width: int, image_height: int) -> tuple[np.ndarray, np.ndarray]:
    label_path = Path(label_path)
    boxes: list[tuple[float, float, float, float]] = []
    labels: list[int] = []
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive.")

    with label_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_no}: expected 5 values.")
            try:
                class_value = float(parts[0])
                cls_id = int(class_value)
                yolo_box = [float(v) for v in parts[1:]]
            except ValueError as exc:
                raise ValueError(f"Invalid numeric value at {label_path}:{line_no}.") from exc
            if class_value != cls_id or not 0 <= cls_id < len(CLASS_NAMES):
                raise ValueError(f"Invalid class id at {label_path}:{line_no}: {parts[0]}")
            if (
                not np.isfinite(yolo_box).all()
                or any(value < 0.0 or value > 1.0 for value in yolo_box)
                or yolo_box[2] <= 0
                or yolo_box[3] <= 0
            ):
                raise ValueError(f"Invalid bounding box at {label_path}:{line_no}.")
            box = np.asarray([yolo_to_xyxy(yolo_box, image_width, image_height)], dtype=np.float32)
            box = clip_boxes_xyxy(box, image_width, image_height)[0]
            if box[2] <= box[0] or box[3] <= box[1]:
                raise ValueError(f"Bounding box is outside the image at {label_path}:{line_no}.")
            boxes.append(tuple(float(v) for v in box))
            labels.append(cls_id + 1)
    return np.asarray(boxes, dtype=np.float32).reshape((-1, 4)), np.asarray(labels, dtype=np.int64)


class RoadDamageDataset(Dataset):
    """Torchvision detection dataset reading YOLO labels from data/processed."""

    def __init__(self, root: str | Path = "data/processed", split: str = "train") -> None:
        self.root = project_path(root)
        self.split = split
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        if not self.image_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {self.image_dir}. "
                "Run split_dataset.py or place processed data first."
            )
        if not self.label_dir.exists():
            raise FileNotFoundError(f"Label directory not found: {self.label_dir}")
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        self.images = sorted(p for p in self.image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_path = self.images[idx]
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        rel = image_path.relative_to(self.image_dir).with_suffix(".txt")
        label_path = self.label_dir / rel
        boxes, labels = read_yolo_label(label_path, width, height)

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)
        area = (boxes_tensor[:, 2] - boxes_tensor[:, 0]).clamp(min=0) * (
            boxes_tensor[:, 3] - boxes_tensor[:, 1]
        ).clamp(min=0)
        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([idx]),
            "area": area,
            "iscrowd": torch.zeros((len(boxes_tensor),), dtype=torch.int64),
        }
        return F.to_tensor(image), target


def collate_fn(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]) -> tuple[tuple, tuple]:
    return tuple(zip(*batch))


def read_image_bgr(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return image


NUM_DETECTION_CLASSES = len(CLASS_NAMES) + 1


def main() -> None:
    print("RoadDamageDataset reads YOLO labels from data/processed/images/<split> and labels/<split>.")


if __name__ == "__main__":
    main()
