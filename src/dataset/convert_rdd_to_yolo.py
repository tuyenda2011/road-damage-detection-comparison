from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image
from tqdm import tqdm

from src.dataset.road_damage_dataset import parse_voc_xml
from src.utils.bbox import xyxy_to_yolo
from src.utils.common import CLASS_TO_ID, ensure_dir, project_path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def build_image_index(input_dir: Path) -> dict[str, list[Path]]:
    """Index images once instead of recursively searching for every XML file."""
    index: dict[str, list[Path]] = {}
    for path in sorted(input_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index.setdefault(path.stem.casefold(), []).append(path)
    return index


def find_image_for_xml(
    xml_path: Path,
    image_index: dict[str, list[Path]],
    annotated_filename: str = "",
) -> Path | None:
    lookup_stem = Path(annotated_filename).stem if annotated_filename else xml_path.stem
    candidates = image_index.get(lookup_stem.casefold(), [])
    if annotated_filename:
        exact = [path for path in candidates if path.name.casefold() == Path(annotated_filename).name.casefold()]
        candidates = exact or candidates
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    xml_parts = xml_path.parent.parts

    def shared_parent_depth(path: Path) -> int:
        depth = 0
        for left, right in zip(xml_parts, path.parent.parts):
            if left.casefold() != right.casefold():
                break
            depth += 1
        return depth

    scores = [(shared_parent_depth(path), -len(path.parts), path) for path in candidates]
    best_score = max((depth, length) for depth, length, _ in scores)
    best = [path for depth, length, path in scores if (depth, length) == best_score]
    if len(best) > 1:
        choices = ", ".join(str(path) for path in best[:3])
        raise ValueError(f"Ambiguous image match for {xml_path}: {choices}")
    return best[0]


def unique_output_name(image_path: Path, input_dir: Path) -> str:
    rel = image_path.relative_to(input_dir)
    safe_stem = "_".join(rel.with_suffix("").parts)
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{safe_stem}_{digest}{image_path.suffix.lower()}"


def convert_voc_to_yolo(
    input_dir: str | Path,
    output_dir: str | Path,
    copy_images: bool = True,
    clean_output: bool = True,
) -> int:
    input_dir = project_path(input_dir)
    output_dir = project_path(output_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset directory not found: {input_dir}")
    if input_dir.resolve() == output_dir.resolve():
        raise ValueError("Input and output conversion directories must be different.")
    xml_files = sorted(input_dir.rglob("*.xml"))
    if not xml_files:
        raise FileNotFoundError(f"No XML annotation files found under {input_dir}")
    image_index = build_image_index(input_dir)
    if not image_index:
        raise FileNotFoundError(f"No supported image files found under {input_dir}")

    staging_context: tempfile.TemporaryDirectory[str] | None = None
    write_root = output_dir
    if clean_output:
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        staging_context = tempfile.TemporaryDirectory(
            prefix=f".{output_dir.name}-conversion-",
            dir=output_dir.parent,
        )
        write_root = Path(staging_context.name)
    image_out = ensure_dir(write_root / "images")
    label_out = ensure_dir(write_root / "labels")

    converted = 0
    missing_images = 0
    written_boxes = 0
    converted_images: set[Path] = set()

    try:
        for xml_path in tqdm(xml_files, desc="Converting XML to YOLO"):
            ann = parse_voc_xml(xml_path, allowed_classes=set(CLASS_TO_ID))
            image_path = find_image_for_xml(xml_path, image_index, ann["filename"])
            if image_path is None:
                missing_images += 1
                continue
            resolved_image = image_path.resolve()
            if resolved_image in converted_images:
                raise ValueError(f"More than one XML annotation maps to image: {image_path}")
            converted_images.add(resolved_image)
            with Image.open(image_path) as img:
                width, height = img.size

            rows: list[str] = []
            for name, box in zip(ann["names"], ann["boxes"]):
                xmin, ymin, xmax, ymax = box
                clipped_box = [
                    min(max(xmin, 0.0), float(width)),
                    min(max(ymin, 0.0), float(height)),
                    min(max(xmax, 0.0), float(width)),
                    min(max(ymax, 0.0), float(height)),
                ]
                if clipped_box[2] <= clipped_box[0] or clipped_box[3] <= clipped_box[1]:
                    continue
                cls_id = CLASS_TO_ID[name]
                x, y, w, h = xyxy_to_yolo(clipped_box, width, height)
                rows.append(f"{cls_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
                written_boxes += 1

            rel_name = unique_output_name(image_path, input_dir)
            if copy_images:
                shutil.copy2(image_path, image_out / rel_name)
            (label_out / f"{Path(rel_name).stem}.txt").write_text("\n".join(rows), encoding="utf-8")
            converted += 1
        if missing_images:
            print(f"Warning: skipped {missing_images} XML file(s) without a matching image.")
        if converted == 0:
            raise RuntimeError("No annotations were converted; check image/XML filenames.")
        if written_boxes == 0:
            raise RuntimeError(f"Converted {converted} image(s), but found no supported damage annotations.")

        if clean_output:
            generated_kinds = ["labels"] + (["images"] if copy_images else [])
            for kind in generated_kinds:
                source = write_root / kind
                target = output_dir / kind
                if target.exists():
                    shutil.rmtree(target)
                output_dir.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(target))
        return converted
    finally:
        if staging_context is not None:
            staging_context.cleanup()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RDD Pascal VOC XML annotations to YOLO format.")
    parser.add_argument("--input", default="data/raw", help="Folder containing RDD images and XML annotations.")
    parser.add_argument("--output", default="data/processed/all", help="Output folder with images/ and labels/.")
    parser.add_argument("--no-copy-images", action="store_true", help="Only write labels; do not copy image files.")
    parser.add_argument("--no-clean", action="store_true", help="Keep existing output files (may leave stale data).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = convert_voc_to_yolo(
        args.input,
        args.output,
        copy_images=not args.no_copy_images,
        clean_output=not args.no_clean,
    )
    print(f"Converted {count} images to YOLO format at {project_path(args.output)}")


if __name__ == "__main__":
    main()
