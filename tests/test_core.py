from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
import yaml
from PIL import Image

from src.dataset.convert_rdd_to_yolo import convert_voc_to_yolo
from src.dataset.rebuild_splits import rebuild_splits
from src.dataset.road_damage_dataset import read_yolo_label
from src.dataset.split_dataset import label_for_image, relative_output_path, split_dataset
from src.evaluation.evaluate import evaluate_model, resolve_image_and_label_dirs
from src.evaluation.metrics import aggregate_precision_recall, calculate_iou, map50
from src.models.faster_rcnn.train import train_faster_rcnn
from src.models.yolo.train import train_yolo
from src.utils.bbox import clip_boxes_xyxy, xyxy_to_yolo, yolo_to_xyxy
from src.utils.checkpoint import (
    atomic_torch_save,
    checksum_path,
    load_torch_checkpoint,
    previous_checkpoint_path,
    verify_checkpoint,
)
from src.utils.common import resolve_device, resolve_inference_weights, write_ultralytics_dataset


class BoundingBoxTests(unittest.TestCase):
    def test_round_trip(self) -> None:
        xyxy = (10.0, 20.0, 110.0, 120.0)
        yolo = xyxy_to_yolo(xyxy, width=200, height=160)
        np.testing.assert_allclose(yolo_to_xyxy(yolo, 200, 160), xyxy)

    def test_clip_does_not_mutate_input(self) -> None:
        boxes = np.array([[-1.0, -2.0, 20.0, 30.0]], dtype=np.float32)
        clipped = clip_boxes_xyxy(boxes, width=10, height=10)
        np.testing.assert_array_equal(boxes, [[-1.0, -2.0, 20.0, 30.0]])
        np.testing.assert_array_equal(clipped, [[0.0, 0.0, 10.0, 10.0]])

    def test_rejects_invalid_image_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            xyxy_to_yolo((0, 0, 1, 1), width=0, height=10)


class LabelTests(unittest.TestCase):
    def test_reads_valid_label_and_offsets_faster_rcnn_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text("2 0.5 0.5 0.4 0.2\n", encoding="utf-8")
            boxes, labels = read_yolo_label(label, image_width=100, image_height=200)
        np.testing.assert_allclose(boxes, [[30.0, 80.0, 70.0, 120.0]])
        np.testing.assert_array_equal(labels, [3])

    def test_rejects_out_of_range_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text("0 1.2 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid bounding box"):
                read_yolo_label(label, image_width=100, image_height=100)

    def test_rejects_unknown_class(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            label = Path(tmp) / "sample.txt"
            label.write_text("9 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Invalid class id"):
                read_yolo_label(label, image_width=100, image_height=100)


class VocConversionTests(unittest.TestCase):
    @staticmethod
    def write_sample(root: Path, xmin: str = "10") -> None:
        image_path = root / "images" / "sample.jpg"
        xml_path = root / "annotations" / "sample.xml"
        image_path.parent.mkdir(parents=True)
        xml_path.parent.mkdir(parents=True)
        Image.new("RGB", (100, 80), "white").save(image_path)
        xml_path.write_text(
            "<annotation><filename>sample.jpg</filename><size><width>100</width>"
            "<height>80</height></size><object><name>D00</name><bndbox>"
            f"<xmin>{xmin}</xmin><ymin>10</ymin><xmax>50</xmax><ymax>40</ymax>"
            "</bndbox></object></annotation>",
            encoding="utf-8",
        )

    def test_converts_voc_and_uses_annotated_filename(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            self.write_sample(source)
            count = convert_voc_to_yolo(source, output)
            labels = list((output / "labels").glob("*.txt"))
        self.assertEqual(count, 1)
        self.assertEqual(len(labels), 1)

    def test_failed_conversion_preserves_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            self.write_sample(source, xmin="nan")
            previous = output / "labels" / "keep.txt"
            previous.parent.mkdir(parents=True)
            previous.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no supported damage"):
                convert_voc_to_yolo(source, output)
            self.assertEqual(previous.read_text(encoding="utf-8"), "keep")


class DatasetSplitTests(unittest.TestCase):
    def test_nested_paths_are_preserved(self) -> None:
        root = Path("dataset")
        image = root / "images" / "train" / "country" / "same.jpg"
        self.assertEqual(relative_output_path(image, root), Path("country/same.jpg"))
        self.assertEqual(label_for_image(image, root), root / "labels" / "train" / "country" / "same.txt")

    def test_split_keeps_duplicate_basenames_in_separate_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for group in ("a", "b"):
                for index in range(2):
                    image = source / "images" / group / f"same{index}.jpg"
                    label = source / "labels" / group / f"same{index}.txt"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"image")
                    label.write_text("", encoding="utf-8")

            counts = split_dataset(source, output, train_ratio=0.5, val_ratio=0.25, seed=42)
            copied = list((output / "images").rglob("*.jpg"))

        self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})
        self.assertEqual(len(copied), 4)
        self.assertEqual(len({path.relative_to(output / "images") for path in copied}), 4)

    def test_rerun_removes_stale_split_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for index in range(10):
                image = source / "images" / f"{index}.jpg"
                label = source / "labels" / f"{index}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"image")
                label.write_text("", encoding="utf-8")

            split_dataset(source, output, train_ratio=0.6, val_ratio=0.2, seed=1)
            counts = split_dataset(source, output, train_ratio=0.6, val_ratio=0.2, seed=2)
            actual = {
                split: len(list((output / "images" / split).rglob("*.jpg")))
                for split in ("train", "val", "test")
            }

        self.assertEqual(actual, counts)

    def test_unannotated_public_test_is_not_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for split, count, annotated in (("train", 3, True), ("val", 1, True), ("test", 2, False)):
                for index in range(count):
                    image = source / "images" / split / f"{split}{index}.jpg"
                    label = source / "labels" / split / f"{split}{index}.txt"
                    image.parent.mkdir(parents=True, exist_ok=True)
                    label.parent.mkdir(parents=True, exist_ok=True)
                    image.write_bytes(b"image")
                    label.write_text("0 0.5 0.5 0.2 0.2\n" if annotated else "", encoding="utf-8")

            counts = split_dataset(source, output, train_ratio=0.5, val_ratio=0.25, seed=42)
            copied_names = {path.name for path in (output / "images").rglob("*.jpg")}

        self.assertEqual(counts, {"train": 2, "val": 1, "test": 1})
        self.assertFalse(any(name.startswith("test") for name in copied_names))

    def test_failed_staging_keeps_previous_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            output = root / "output"
            for index in range(4):
                image = source / "images" / f"{index}.jpg"
                label = source / "labels" / f"{index}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(b"new")
                label.write_text("", encoding="utf-8")
            previous = output / "images" / "train" / "keep.jpg"
            previous.parent.mkdir(parents=True)
            previous.write_bytes(b"previous")

            with patch("src.dataset.split_dataset.copy_split", side_effect=RuntimeError("copy failed")):
                with self.assertRaisesRegex(RuntimeError, "copy failed"):
                    split_dataset(source, output, train_ratio=0.5, val_ratio=0.25)

            self.assertEqual(previous.read_bytes(), b"previous")


class BalancedRebuildTests(unittest.TestCase):
    def test_rebuilds_labeled_splits_and_preserves_challenge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "processed"
            output = root / "rebuilt"
            for index in range(60):
                source_split = "train" if index < 50 else "val"
                source_name = ("CountryA", "CountryB", "CountryC")[index % 3]
                filename = f"{source_name}_{index:06d}"
                image = source / "images" / source_split / f"{filename}.jpg"
                label = source / "labels" / source_split / f"{filename}.txt"
                image.parent.mkdir(parents=True, exist_ok=True)
                label.parent.mkdir(parents=True, exist_ok=True)
                image.write_bytes(f"image-{index}".encode())
                label.write_text(f"{index % 4} 0.5 0.5 0.2 0.2\n", encoding="utf-8")
            for index in range(3):
                challenge = source / "images" / "test" / f"Blind_{index:06d}.jpg"
                challenge.parent.mkdir(parents=True, exist_ok=True)
                challenge.write_bytes(b"blind")

            summary = rebuild_splits(
                source,
                output,
                ratios=(0.8, 0.1, 0.1),
                seed=42,
                block_size=5,
            )

            split_total = sum(summary["splits"][split]["images"] for split in ("train", "val", "test"))
            self.assertEqual(split_total, 60)
            self.assertEqual(summary["challenge_images"], 3)
            self.assertEqual(len(list((output / "images" / "challenge").glob("*.jpg"))), 3)
            self.assertTrue((output / "split_manifest.csv").is_file())


class MetricsTests(unittest.TestCase):
    def test_iou(self) -> None:
        self.assertAlmostEqual(calculate_iou([0, 0, 10, 10], [5, 5, 15, 15]), 25 / 175)

    def test_map_does_not_penalize_absent_classes(self) -> None:
        prediction = {
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
            "scores": np.array([0.9], dtype=np.float32),
        }
        ground_truth = {
            "boxes": np.array([[0, 0, 10, 10]], dtype=np.float32),
            "labels": np.array([2], dtype=np.int64),
        }
        self.assertAlmostEqual(map50([prediction], [ground_truth]), 1.0)

    def test_rejects_mismatched_batch_lengths(self) -> None:
        with self.assertRaises(ValueError):
            aggregate_precision_recall([], [{}])


class RuntimeConfigTests(unittest.TestCase):
    def test_runtime_dataset_path_is_absolute(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "run"
            runtime = write_ultralytics_dataset("configs/dataset.yaml", destination)
            config = yaml.safe_load(runtime.read_text(encoding="utf-8"))
        self.assertTrue(Path(config["path"]).is_absolute())
        self.assertTrue(config["path"].endswith("data\\processed") or config["path"].endswith("data/processed"))

    @patch("src.utils.common.torch.cuda.is_available", return_value=True)
    @patch("src.utils.common.torch.cuda.device_count", return_value=2)
    def test_numeric_torch_device_is_normalized(self, _count, _available) -> None:
        self.assertEqual(resolve_device("1"), "cuda:1")

    @patch("src.utils.common.torch.cuda.is_available", return_value=True)
    @patch("src.utils.common.torch.cuda.device_count", return_value=1)
    def test_out_of_range_device_is_rejected(self, _count, _available) -> None:
        with self.assertRaises(ValueError):
            resolve_device("2")

    @patch("src.models.yolo.train.YOLO")
    def test_yolo_train_passes_runtime_yaml_path_and_copies_best(self, yolo_class) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            trainer_best = root / "trainer" / "best.pt"
            trainer_best.parent.mkdir()
            trainer_best.write_bytes(b"weights")
            model = yolo_class.return_value
            model.trainer = SimpleNamespace(best=trainer_best)

            final_best = train_yolo(
                data="configs/dataset.yaml",
                model_name="yolov8n.pt",
                epochs=1,
                imgsz=64,
                batch=1,
                lr0=0.001,
                device="cpu",
                project=str(root / "run"),
                workers=0,
            )

            train_data = Path(model.train.call_args.kwargs["data"])
            self.assertTrue(train_data.is_file())
            self.assertTrue(train_data.name.endswith(".resolved.yaml"))
            self.assertEqual(final_best.read_bytes(), b"weights")
            self.assertEqual(final_best, root / "run" / "checkpoints" / "best.pt")
            self.assertTrue(checksum_path(final_best).is_file())
            self.assertTrue((final_best.parent / "manifest.json").is_file())


class CheckpointSafetyTests(unittest.TestCase):
    def test_atomic_save_keeps_verified_previous_generation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoints" / "best.pth"
            atomic_torch_save({"epoch": 1}, checkpoint)
            atomic_torch_save({"epoch": 2}, checkpoint)

            self.assertTrue(checksum_path(checkpoint).is_file())
            previous = previous_checkpoint_path(checkpoint)
            self.assertTrue(previous.is_file())
            verify_checkpoint(checkpoint, require_checksum=True)
            verify_checkpoint(previous, require_checksum=True)

            checkpoint.write_bytes(b"corrupted")
            payload, loaded_path = load_torch_checkpoint(checkpoint, require_checksum=True)
            self.assertEqual(payload["epoch"], 1)
            self.assertEqual(loaded_path, previous)

            atomic_torch_save({"epoch": 3}, checkpoint)
            previous_payload, _ = load_torch_checkpoint(previous, require_checksum=True)
            self.assertEqual(previous_payload["epoch"], 1)

    def test_inference_resolver_rolls_back_corrupted_managed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_root = Path(tmp) / "model"
            checkpoint = model_root / "checkpoints" / "best.pth"
            atomic_torch_save({"epoch": 1}, checkpoint)
            atomic_torch_save({"epoch": 2}, checkpoint)
            checkpoint.write_bytes(b"corrupted")

            resolved = resolve_inference_weights(checkpoint, model_root, ".pth")
            self.assertEqual(resolved, previous_checkpoint_path(checkpoint))

    @patch("src.models.faster_rcnn.train.validate_map50", return_value=0.5)
    @patch("src.models.faster_rcnn.train.train_one_epoch", return_value=1.0)
    @patch("src.models.faster_rcnn.train.build_faster_rcnn")
    def test_faster_rcnn_resume_restores_last_training_state(
        self,
        build_model,
        _train_epoch,
        _validate,
    ) -> None:
        class TinyDetector(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.backbone = torch.nn.Linear(2, 2)
                self.head = torch.nn.Linear(2, 1)

        build_model.side_effect = lambda pretrained: TinyDetector()

        def fake_train_epoch(model, _loader, optimizer, *_args) -> float:
            optimizer.zero_grad(set_to_none=True)
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.grad = torch.zeros_like(parameter)
            optimizer.step()
            return 1.0

        _train_epoch.side_effect = fake_train_epoch
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            for split in ("train", "val"):
                (data / "images" / split).mkdir(parents=True)
                (data / "labels" / split).mkdir(parents=True)
                (data / "images" / split / "sample.jpg").write_bytes(b"unused")
                (data / "labels" / split / "sample.txt").write_text("", encoding="utf-8")
            (data / "split_manifest.csv").write_text("path,split\nsample,train\n", encoding="utf-8")
            output = root / "run"

            train_faster_rcnn(
                str(data),
                epochs=1,
                batch=1,
                lr=0.01,
                device="cpu",
                output_dir=str(output),
                num_workers=0,
            )
            train_faster_rcnn(
                str(data),
                epochs=2,
                batch=1,
                lr=0.01,
                device="cpu",
                output_dir=str(output),
                num_workers=0,
                resume="auto",
            )

            last = output / "checkpoints" / "last.pth"
            payload, _ = load_torch_checkpoint(last, require_checksum=True)
            previous_payload, _ = load_torch_checkpoint(
                previous_checkpoint_path(last), require_checksum=True
            )
            self.assertEqual(payload["epoch"], 2)
            self.assertEqual(previous_payload["epoch"], 1)

            (data / "split_manifest.csv").write_text("path,split\nsample,test\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Dataset split manifest changed"):
                train_faster_rcnn(
                    str(data),
                    epochs=3,
                    batch=1,
                    lr=0.01,
                    device="cpu",
                    output_dir=str(output),
                    num_workers=0,
                    resume="auto",
                )


class EvaluationPreflightTests(unittest.TestCase):
    def test_resolves_direct_image_split(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "test"
            label_dir = root / "labels" / "test"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            self.assertEqual(resolve_image_and_label_dirs(image_dir), (image_dir, label_dir))

    @patch("src.evaluation.evaluate.load_detector")
    def test_empty_ground_truth_fails_before_model_load(self, load_detector_mock) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images" / "test"
            label_dir = root / "labels" / "test"
            image_dir.mkdir(parents=True)
            label_dir.mkdir(parents=True)
            (image_dir / "sample.jpg").write_bytes(b"image")
            (label_dir / "sample.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "no ground-truth boxes"):
                evaluate_model("yolo", "missing.pt", str(root))
        load_detector_mock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
