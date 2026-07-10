from __future__ import annotations

import argparse
import math
import random
import shutil
import sys
import tempfile
from contextlib import ExitStack
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.dataset.convert_rdd_to_yolo import convert_voc_to_yolo
from src.utils.common import ensure_dir, project_path, set_seed


SPLITS = ("train", "val", "test")


def collect_yolo_images(input_dir: Path) -> list[Path]:
    image_dir = input_dir / "images"
    search_root = image_dir if image_dir.exists() else input_dir
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in search_root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def collect_images_from_dir(image_dir: Path) -> list[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in image_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def detect_existing_yolo_splits(input_dir: Path) -> dict[str, list[Path]]:
    split_aliases = {"train": "train", "val": "val", "valid": "val", "test": "test"}
    detected: dict[str, list[Path]] = {}

    for raw_split, target_split in split_aliases.items():
        if target_split in detected:
            continue
        candidates = [
            input_dir / raw_split / "images",
            input_dir / "images" / raw_split,
        ]
        for image_dir in candidates:
            if image_dir.exists():
                images = collect_images_from_dir(image_dir)
                if images:
                    detected[target_split] = images
                    break
    return detected


def split_has_annotations(images: list[Path], input_dir: Path) -> bool:
    return any(
        (label := label_for_image(image, input_dir)).is_file() and label.stat().st_size > 0
        for image in images
    )


def label_for_image(image_path: Path, input_dir: Path) -> Path:
    relative = image_path.relative_to(input_dir)
    if "images" in relative.parts:
        parts = list(relative.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        return (input_dir / Path(*parts)).with_suffix(".txt")
    return (input_dir / "labels" / relative).with_suffix(".txt")


def relative_output_path(image_path: Path, input_dir: Path) -> Path:
    """Keep source subdirectories so equal filenames cannot overwrite each other."""
    relative = image_path.relative_to(input_dir)
    parts = list(relative.parts)
    if "images" in parts:
        parts = parts[parts.index("images") + 1 :]
    if parts and parts[0].lower() in {"train", "val", "valid", "test"}:
        parts = parts[1:]
    return Path(*parts) if parts else Path(image_path.name)


def copy_split(split: str, split_images: list[Path], output_dir: Path, working_input: Path) -> int:
    image_out = ensure_dir(output_dir / "images" / split)
    label_out = ensure_dir(output_dir / "labels" / split)
    for image_path in tqdm(split_images, desc=f"Copying {split}"):
        label_path = label_for_image(image_path, working_input)
        relative = relative_output_path(image_path, working_input)
        target_image = image_out / relative
        target_label = (label_out / relative).with_suffix(".txt")
        if target_image.exists() or target_label.exists():
            raise FileExistsError(f"Output collision while splitting dataset: {relative}")
        target_image.parent.mkdir(parents=True, exist_ok=True)
        target_label.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(image_path, target_image)
        if label_path.exists():
            shutil.copy2(label_path, target_label)
        else:
            target_label.write_text("", encoding="utf-8")
    return len(split_images)


def publish_split_output(staging_dir: Path, output_dir: Path) -> None:
    """Publish a fully built staging dataset without leaving stale split files."""
    resolved_output = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    for kind in ("images", "labels"):
        for split in SPLITS:
            target = (resolved_output / kind / split).resolve()
            if target.parent.parent != resolved_output:
                raise ValueError(f"Unsafe generated dataset path: {target}")
            source = staging_dir / kind / split
            if not source.is_dir():
                raise FileNotFoundError(f"Incomplete staged dataset: {source}")
            if target.exists():
                shutil.rmtree(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))


def split_dataset(
    input_dir: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict[str, int]:
    input_dir = project_path(input_dir)
    output_dir = project_path(output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Dataset input not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output dataset directories must be different.")
    if not all(math.isfinite(value) for value in (train_ratio, val_ratio)):
        raise ValueError("Split ratios must be finite.")
    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio and val_ratio must leave a positive test split.")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        staging_dir = Path(
            stack.enter_context(
                tempfile.TemporaryDirectory(prefix=f".{output_dir.name}-staging-", dir=output_dir.parent)
            )
        )
        working_input = input_dir
        if list(input_dir.rglob("*.xml")) and not (input_dir / "labels").exists():
            working_input = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="rdd-yolo-")))
            convert_voc_to_yolo(input_dir, working_input)

        existing_splits = detect_existing_yolo_splits(working_input)
        images_to_resplit: list[Path] | None = None
        if all(split in existing_splits for split in SPLITS):
            if split_has_annotations(existing_splits["test"], working_input):
                counts = {
                    split: copy_split(split, existing_splits[split], staging_dir, working_input)
                    for split in SPLITS
                }
                publish_split_output(staging_dir, output_dir)
                print("Detected annotated YOLO train/val/test structure; preserved original splits.")
                return counts
            images_to_resplit = existing_splits["train"] + existing_splits["val"]
            print("Existing test split has no annotations; rebuilding all splits from annotated train/val data.")

        images = images_to_resplit or collect_yolo_images(working_input)
        if not images:
            raise FileNotFoundError(
                f"No images found in {working_input}. Expected YOLO images/labels or RDD XML data."
            )

        set_seed(seed)
        random.shuffle(images)
        n = len(images)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if min(n_train, n_val, n - n_train - n_val) <= 0:
            raise ValueError(f"Dataset with {n} image(s) is too small for the requested split ratios.")
        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }

        counts = {
            split: copy_split(split, split_images, staging_dir, working_input)
            for split, split_images in splits.items()
        }
        publish_split_output(staging_dir, output_dir)
        return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split RDD/YOLO dataset into train, val and test sets.")
    parser.add_argument("--input", default="data/raw", help="Raw RDD folder or YOLO-format folder.")
    parser.add_argument("--output", default="data/processed", help="Processed dataset output folder.")
    parser.add_argument("--train", type=float, default=0.7, help="Train split ratio.")
    parser.add_argument("--val", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.train <= 0 or args.val < 0 or args.train + args.val >= 1:
        raise ValueError("--train and --val must leave a positive test split.")
    counts = split_dataset(args.input, args.output, args.train, args.val, args.seed)
    print(f"Dataset split completed: {counts}")


if __name__ == "__main__":
    main()
