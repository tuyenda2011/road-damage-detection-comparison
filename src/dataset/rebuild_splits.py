from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix
from tqdm import tqdm

from src.dataset.road_damage_dataset import read_yolo_label
from src.utils.common import CLASS_NAMES, list_images, project_path


TARGET_SPLITS = ("train", "val", "test")
SOURCE_SPLITS = ("train", "val")


@dataclass(frozen=True)
class ImageRecord:
    image: Path
    label: Path
    source_split: str
    relative_path: Path
    source_name: str
    sequence_index: int
    class_counts: tuple[int, ...]


@dataclass
class ImageGroup:
    key: str
    source_name: str
    records: list[ImageRecord]
    class_counts: tuple[int, ...]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def positive_images(self) -> int:
        return sum(sum(record.class_counts) > 0 for record in self.records)


def parse_sequence(stem: str) -> tuple[str, int]:
    match = re.fullmatch(r"(.+?)[_-]?([0-9]+)", stem)
    if match is None:
        raise ValueError(f"Filename does not end in a numeric sequence id: {stem}")
    return match.group(1).rstrip("_-"), int(match.group(2))


def collect_annotated_records(input_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []
    seen_paths: set[str] = set()
    for source_split in SOURCE_SPLITS:
        image_dir = input_root / "images" / source_split
        label_dir = input_root / "labels" / source_split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source split: {image_dir} or {label_dir}")

        for image in tqdm(list_images(image_dir), desc=f"Reading {source_split} labels"):
            relative = image.relative_to(image_dir)
            normalized_path = relative.as_posix().casefold()
            if normalized_path in seen_paths:
                raise ValueError(f"Duplicate image path across annotated splits: {relative}")
            seen_paths.add(normalized_path)

            label = label_dir / relative.with_suffix(".txt")
            if not label.is_file():
                raise FileNotFoundError(f"Missing label for {image}: {label}")
            _, labels = read_yolo_label(label, image_width=1, image_height=1)
            counts = Counter(int(label_id) - 1 for label_id in labels)
            source_name, sequence_index = parse_sequence(image.stem)
            records.append(
                ImageRecord(
                    image=image,
                    label=label,
                    source_split=source_split,
                    relative_path=relative,
                    source_name=source_name,
                    sequence_index=sequence_index,
                    class_counts=tuple(counts[class_id] for class_id in range(len(CLASS_NAMES))),
                )
            )
    if not records:
        raise ValueError("No annotated train/val images were found.")
    return records


def build_groups(records: list[ImageRecord], block_size: int) -> list[ImageGroup]:
    grouped: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        block = record.sequence_index // block_size
        grouped[f"{record.source_name}:{block:06d}"].append(record)

    groups: list[ImageGroup] = []
    for key, group_records in grouped.items():
        class_counts = tuple(
            sum(record.class_counts[class_id] for record in group_records)
            for class_id in range(len(CLASS_NAMES))
        )
        groups.append(
            ImageGroup(
                key=key,
                source_name=group_records[0].source_name,
                records=sorted(group_records, key=lambda record: record.sequence_index),
                class_counts=class_counts,
            )
        )
    return groups


def group_features(group: ImageGroup, source_names: tuple[str, ...]) -> tuple[int, ...]:
    source_features = tuple(group.size if name == group.source_name else 0 for name in source_names)
    return (group.size, group.positive_images, *group.class_counts, *source_features)


def assign_groups(
    groups: list[ImageGroup],
    ratios: tuple[float, float, float],
    seed: int,
) -> dict[str, list[ImageGroup]]:
    """Assign intact sequence groups with exact image counts via MILP."""
    source_names = tuple(sorted({group.source_name for group in groups}))
    features = {group.key: group_features(group, source_names) for group in groups}
    feature_count = len(next(iter(features.values())))
    feature_totals = np.asarray(
        [sum(features[group.key][index] for group in groups) for index in range(feature_count)],
        dtype=np.float64,
    )

    raw_image_targets = feature_totals[0] * np.asarray(ratios)
    image_targets = np.floor(raw_image_targets).astype(int)
    remainder = int(feature_totals[0]) - int(image_targets.sum())
    fractional_order = sorted(
        range(len(TARGET_SPLITS)),
        key=lambda index: (raw_image_targets[index] - image_targets[index], index),
        reverse=True,
    )
    for index in fractional_order[:remainder]:
        image_targets[index] += 1

    binary_count = len(groups) * len(TARGET_SPLITS)
    balanced_features = feature_count - 1
    deviation_count = len(TARGET_SPLITS) * balanced_features * 2
    variable_count = binary_count + deviation_count
    objective = np.zeros(variable_count, dtype=np.float64)
    rng = random.Random(seed)
    objective[:binary_count] = [rng.random() * 1e-9 for _ in range(binary_count)]
    integrality = np.zeros(variable_count, dtype=np.uint8)
    integrality[:binary_count] = 1
    lower_bounds = np.zeros(variable_count, dtype=np.float64)
    upper_bounds = np.full(variable_count, np.inf, dtype=np.float64)
    upper_bounds[:binary_count] = 1.0

    feature_weights = np.asarray(
        [3.0, *([5.0] * len(CLASS_NAMES)), *([2.0] * len(source_names))],
        dtype=np.float64,
    )
    row_count = len(groups) + len(TARGET_SPLITS) + len(TARGET_SPLITS) * balanced_features
    matrix = lil_matrix((row_count, variable_count), dtype=np.float64)
    constraint_lower = np.zeros(row_count, dtype=np.float64)
    constraint_upper = np.zeros(row_count, dtype=np.float64)
    row = 0

    for group_index in range(len(groups)):
        for split_index in range(len(TARGET_SPLITS)):
            matrix[row, group_index * len(TARGET_SPLITS) + split_index] = 1.0
        constraint_lower[row] = constraint_upper[row] = 1.0
        row += 1

    for split_index in range(len(TARGET_SPLITS)):
        for group_index, group in enumerate(groups):
            matrix[row, group_index * len(TARGET_SPLITS) + split_index] = features[group.key][0]
        constraint_lower[row] = constraint_upper[row] = image_targets[split_index]
        row += 1

    deviation_offset = binary_count
    for split_index, ratio in enumerate(ratios):
        for feature_index in range(1, feature_count):
            for group_index, group in enumerate(groups):
                matrix[row, group_index * len(TARGET_SPLITS) + split_index] = features[group.key][feature_index]
            local_index = split_index * balanced_features + feature_index - 1
            positive_deviation = deviation_offset + local_index * 2
            negative_deviation = positive_deviation + 1
            matrix[row, positive_deviation] = -1.0
            matrix[row, negative_deviation] = 1.0
            target = feature_totals[feature_index] * ratio
            constraint_lower[row] = constraint_upper[row] = target
            normalized_weight = feature_weights[feature_index - 1] / max(target, 1.0)
            objective[positive_deviation] = normalized_weight
            objective[negative_deviation] = normalized_weight
            row += 1

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower_bounds, upper_bounds),
        constraints=LinearConstraint(matrix.tocsr(), constraint_lower, constraint_upper),
        options={"time_limit": 180.0, "mip_rel_gap": 0.001},
    )
    if result.x is None or result.status not in {0, 1}:
        raise RuntimeError(f"MILP split optimization failed: {result.message}")

    assignment = {split: [] for split in TARGET_SPLITS}
    for group_index, group in enumerate(groups):
        values = result.x[
            group_index * len(TARGET_SPLITS) : (group_index + 1) * len(TARGET_SPLITS)
        ]
        selected_index = int(np.argmax(values))
        if values[selected_index] < 0.5:
            raise RuntimeError(f"MILP returned a fractional assignment for {group.key}")
        assignment[TARGET_SPLITS[selected_index]].append(group)
    return assignment


def link_or_copy(source: Path, target: Path, hardlink_images: bool) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if hardlink_images:
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def materialize_dataset(
    input_root: Path,
    output_root: Path,
    assignment: dict[str, list[ImageGroup]],
    ratios: tuple[float, float, float],
    seed: int,
    block_size: int,
    hardlink_images: bool,
) -> dict[str, object]:
    if output_root.exists():
        raise FileExistsError(f"Output already exists: {output_root}")
    if output_root.resolve() == input_root.resolve():
        raise ValueError("Input and output roots must be different.")
    output_root.mkdir(parents=True)

    manifest_rows: list[dict[str, object]] = []
    try:
        for split in TARGET_SPLITS:
            records = [record for group in assignment[split] for record in group.records]
            for record in tqdm(records, desc=f"Building {split}"):
                image_target = output_root / "images" / split / record.relative_path
                label_target = output_root / "labels" / split / record.relative_path.with_suffix(".txt")
                link_or_copy(record.image, image_target, hardlink_images)
                label_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(record.label, label_target)
                manifest_rows.append(
                    {
                        "target_split": split,
                        "source_split": record.source_split,
                        "relative_path": record.relative_path.as_posix(),
                        "source_name": record.source_name,
                        "sequence_index": record.sequence_index,
                        "sequence_group": f"{record.source_name}:{record.sequence_index // block_size:06d}",
                        **{
                            class_name: record.class_counts[index]
                            for index, class_name in enumerate(CLASS_NAMES)
                        },
                    }
                )

        challenge_images = list_images(input_root / "images" / "test")
        for image in tqdm(challenge_images, desc="Preserving challenge"):
            relative = image.relative_to(input_root / "images" / "test")
            link_or_copy(image, output_root / "images" / "challenge" / relative, hardlink_images)

        manifest_path = output_root / "split_manifest.csv"
        fieldnames = [
            "target_split",
            "source_split",
            "relative_path",
            "source_name",
            "sequence_index",
            "sequence_group",
            *CLASS_NAMES,
        ]
        with manifest_path.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(manifest_rows)

        summary: dict[str, object] = {
            "seed": seed,
            "block_size": block_size,
            "requested_ratios": dict(zip(TARGET_SPLITS, ratios)),
            "challenge_images": len(challenge_images),
            "splits": {},
        }
        total_images = len(manifest_rows)
        for split in TARGET_SPLITS:
            rows = [row for row in manifest_rows if row["target_split"] == split]
            summary["splits"][split] = {
                "images": len(rows),
                "ratio": len(rows) / total_images,
                "groups": len(assignment[split]),
                "classes": {
                    class_name: sum(int(row[class_name]) for row in rows)
                    for class_name in CLASS_NAMES
                },
            }
        (output_root / "split_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return summary
    except Exception:
        resolved_output = output_root.resolve()
        if resolved_output.parent == input_root.resolve().parent and resolved_output != input_root.resolve():
            shutil.rmtree(resolved_output, ignore_errors=True)
        raise


def optimize_existing_dataset(
    root: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    block_size: int = 100,
) -> dict[str, object]:
    """Reassign an existing generated manifest without recreating image data."""
    dataset_root = project_path(root)
    manifest_path = dataset_root / "split_manifest.csv"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Split manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"Split manifest is empty: {manifest_path}")

    records: list[ImageRecord] = []
    row_groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        old_split = row["target_split"]
        relative = Path(row["relative_path"])
        record = ImageRecord(
            image=dataset_root / "images" / old_split / relative,
            label=dataset_root / "labels" / old_split / relative.with_suffix(".txt"),
            source_split=row["source_split"],
            relative_path=relative,
            source_name=row["source_name"],
            sequence_index=int(row["sequence_index"]),
            class_counts=tuple(int(row[class_name]) for class_name in CLASS_NAMES),
        )
        if not record.image.is_file() or not record.label.is_file():
            raise FileNotFoundError(f"Manifest file is missing on disk: {record.image} or {record.label}")
        records.append(record)
        row_groups[row["sequence_group"]].append(row)

    groups = build_groups(records, block_size)
    assignment = assign_groups(groups, ratios, seed)
    target_by_group = {
        group.key: split for split, split_groups in assignment.items() for group in split_groups
    }
    if set(target_by_group) != set(row_groups):
        raise ValueError("Manifest group keys do not match reconstructed sequence groups.")

    planned_moves: list[tuple[Path, Path]] = []
    for row in rows:
        old_split = row["target_split"]
        new_split = target_by_group[row["sequence_group"]]
        if old_split == new_split:
            continue
        relative = Path(row["relative_path"])
        for kind, suffix in (("images", relative.suffix), ("labels", ".txt")):
            source_relative = relative if kind == "images" else relative.with_suffix(suffix)
            source = dataset_root / kind / old_split / source_relative
            target = dataset_root / kind / new_split / source_relative
            if target.exists():
                raise FileExistsError(f"Rebalance destination already exists: {target}")
            planned_moves.append((source, target))

    completed_moves: list[tuple[Path, Path]] = []
    try:
        for source, target in tqdm(planned_moves, desc="Rebalancing staged files"):
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            completed_moves.append((source, target))
    except Exception:
        for source, target in reversed(completed_moves):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                target.replace(source)
        raise

    for row in rows:
        row["target_split"] = target_by_group[row["sequence_group"]]
    temporary_manifest = manifest_path.with_suffix(".csv.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary_manifest.replace(manifest_path)

    summary: dict[str, object] = {
        "seed": seed,
        "block_size": block_size,
        "requested_ratios": dict(zip(TARGET_SPLITS, ratios)),
        "challenge_images": len(list_images(dataset_root / "images" / "challenge")),
        "splits": {},
    }
    for split in TARGET_SPLITS:
        split_rows = [row for row in rows if row["target_split"] == split]
        summary["splits"][split] = {
            "images": len(split_rows),
            "ratio": len(split_rows) / len(rows),
            "positive_images": sum(
                any(int(row[class_name]) > 0 for class_name in CLASS_NAMES) for row in split_rows
            ),
            "groups": len({row["sequence_group"] for row in split_rows}),
            "classes": {
                class_name: sum(int(row[class_name]) for row in split_rows)
                for class_name in CLASS_NAMES
            },
            "sources": dict(Counter(row["source_name"] for row in split_rows)),
        }
    (dataset_root / "split_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def rebuild_splits(
    input_root: str | Path,
    output_root: str | Path,
    ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: int = 42,
    block_size: int = 100,
    hardlink_images: bool = True,
) -> dict[str, object]:
    if len(ratios) != 3 or any(ratio <= 0 for ratio in ratios) or abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError("Ratios must be three positive values that sum to 1.")
    if block_size <= 0:
        raise ValueError("block_size must be positive.")
    input_path = project_path(input_root)
    output_path = project_path(output_root)
    records = collect_annotated_records(input_path)
    groups = build_groups(records, block_size)
    assignment = assign_groups(groups, ratios, seed)
    return materialize_dataset(
        input_path,
        output_path,
        assignment,
        ratios,
        seed,
        block_size,
        hardlink_images,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebuild labeled train/val/test splits and preserve the original blind test as challenge."
    )
    parser.add_argument("--input", default="data/processed")
    parser.add_argument("--output", default="data/processed_rebuilt")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.1)
    parser.add_argument("--test", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=100)
    parser.add_argument(
        "--optimize-existing",
        default=None,
        help="Re-optimize a previously generated output in place using its manifest.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy image bytes instead of using same-volume hard links.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ratios = (args.train, args.val, args.test)
    if args.optimize_existing:
        summary = optimize_existing_dataset(
            args.optimize_existing,
            ratios=ratios,
            seed=args.seed,
            block_size=args.block_size,
        )
    else:
        summary = rebuild_splits(
            args.input,
            args.output,
            ratios=ratios,
            seed=args.seed,
            block_size=args.block_size,
            hardlink_images=not args.copy_images,
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
