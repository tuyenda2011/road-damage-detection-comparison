from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from src.dataset.road_damage_dataset import read_yolo_label
from src.utils.common import CLASS_NAMES, list_images, project_path


def validate_dataset(
    root: str | Path = "data/processed",
    splits: tuple[str, ...] = ("train", "val", "test"),
    check_images: bool = False,
) -> dict[str, object]:
    dataset_root = project_path(root)
    report: dict[str, object] = {"root": str(dataset_root), "splits": {}}
    errors: list[str] = []
    seen_names: dict[str, str] = {}

    for split in splits:
        image_dir = dataset_root / "images" / split
        label_dir = dataset_root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            errors.append(f"{split}: missing image or label directory")
            continue

        images = list_images(image_dir)
        image_labels = {image.relative_to(image_dir).with_suffix(".txt") for image in images}
        label_files = {label.relative_to(label_dir) for label in label_dir.rglob("*.txt") if label.is_file()}
        missing_labels = sorted(image_labels - label_files)
        orphan_labels = sorted(label_files - image_labels)
        class_counts: Counter[str] = Counter()
        invalid_labels = 0
        unreadable_images = 0

        for image in images:
            relative_image = image.relative_to(image_dir)
            key = relative_image.as_posix().casefold()
            if key in seen_names:
                errors.append(f"{split}: image path also appears in {seen_names[key]}: {relative_image}")
            else:
                seen_names[key] = split

            if check_images:
                try:
                    with Image.open(image) as opened:
                        opened.verify()
                except Exception as exc:
                    unreadable_images += 1
                    if unreadable_images <= 10:
                        errors.append(f"{split}: unreadable image {relative_image}: {exc}")

            label = label_dir / relative_image.with_suffix(".txt")
            if not label.is_file():
                continue
            try:
                _, labels = read_yolo_label(label, image_width=1, image_height=1)
                class_counts.update(CLASS_NAMES[int(class_id) - 1] for class_id in labels)
            except ValueError as exc:
                invalid_labels += 1
                if invalid_labels <= 10:
                    errors.append(str(exc))

        split_report = {
            "images": len(images),
            "labels": len(label_files),
            "boxes": sum(class_counts.values()),
            "classes": {name: class_counts[name] for name in CLASS_NAMES},
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
            "invalid_labels": invalid_labels,
            "unreadable_images": unreadable_images,
        }
        report["splits"][split] = split_report
        if missing_labels:
            errors.append(f"{split}: {len(missing_labels)} image(s) have no label file")
        if orphan_labels:
            errors.append(f"{split}: {len(orphan_labels)} orphan label file(s)")
        if images and sum(class_counts.values()) == 0:
            errors.append(f"{split}: no annotated bounding boxes were found")

    report["valid"] = not errors
    report["errors"] = errors
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate processed YOLO dataset structure and labels.")
    parser.add_argument("--root", default="data/processed")
    parser.add_argument("--splits", default="train,val,test", help="Comma-separated split names.")
    parser.add_argument("--check-images", action="store_true", help="Decode-check every image (slower).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    splits = tuple(split.strip() for split in args.splits.split(",") if split.strip())
    if not splits:
        raise ValueError("At least one split is required.")
    report = validate_dataset(args.root, splits, args.check_images)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
