from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import RTDETR

from src.utils.common import (
    ensure_dir,
    load_yaml,
    project_path,
    resolve_model_reference,
    resolve_ultralytics_device,
    write_ultralytics_dataset,
)
from src.utils.checkpoint import UltralyticsCheckpointPublisher, resolve_checkpoint_with_fallback


def train_rtdetr(
    data: str,
    model_name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    lr0: float,
    device: str,
    project: str = "runs/rtdetr",
    name: str = "train",
    freeze: int = 10,
    seed: int = 42,
    workers: int = 4,
    resume: str | None = None,
) -> Path:
    data_path = project_path(data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    dataset_path = write_ultralytics_dataset(data_path, project)

    if epochs <= 0 or imgsz <= 0 or batch == 0 or lr0 <= 0 or workers < 0 or freeze < 0:
        raise ValueError(
            "epochs, imgsz and lr0 must be positive; batch must be non-zero; "
            "workers and freeze cannot be negative."
        )

    checkpoint_dir = ensure_dir(project_path(project) / "checkpoints")
    resume_path: Path | None = None
    if resume:
        requested_resume = checkpoint_dir / "last.pt" if resume.lower() == "auto" else project_path(resume)
        resume_path = resolve_checkpoint_with_fallback(
            requested_resume,
            require_checksum=requested_resume.parent.name == "checkpoints",
        )

    model = RTDETR(str(resume_path) if resume_path else resolve_model_reference(model_name))
    publisher = UltralyticsCheckpointPublisher(
        checkpoint_dir,
        "rtdetr",
        metadata={
            "data": str(data_path.resolve()),
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "seed": seed,
            "resume_from": str(resume_path) if resume_path else None,
        },
    )
    model.add_callback("on_model_save", publisher.publish)

    train_options = dict(
        data=str(dataset_path),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        lr0=lr0,
        device=resolve_ultralytics_device(device),
        project=str(project_path(project)),
        name=name,
        freeze=freeze,
        seed=seed,
        deterministic=True,
        workers=workers,
        exist_ok=resume_path is not None,
    )
    if resume_path is not None:
        train_options["resume"] = str(resume_path)
    model.train(**train_options)

    publisher.publish(model.trainer, names=("best",), force=True)
    if not (checkpoint_dir / "last.pt").is_file():
        publisher.publish(model.trainer, names=("last",), force=True)
    final_best = checkpoint_dir / "best.pt"
    if not final_best.is_file():
        raise FileNotFoundError(f"Ultralytics did not produce a best checkpoint: {model.trainer.best}")
    publisher.write_manifest()
    return final_best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune RT-DETR on Road Damage Detection.")
    parser.add_argument("--config", default="configs/rtdetr.yaml")
    parser.add_argument("--data", default=None)
    parser.add_argument("--model", default=None, help="Example: rtdetr-l.pt or rtdetr-x.pt")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr0", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="Number of early layers to freeze. Use 0 for full fine-tune.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint path, or use 'auto' for runs/rtdetr/checkpoints/last.pt.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    best = train_rtdetr(
        data=args.data or cfg.get("data", "configs/dataset.yaml"),
        model_name=args.model or cfg.get("model", "rtdetr-l.pt"),
        epochs=args.epochs if args.epochs is not None else int(cfg.get("epochs", 50)),
        imgsz=args.imgsz if args.imgsz is not None else int(cfg.get("imgsz", 640)),
        batch=args.batch if args.batch is not None else int(cfg.get("batch", 2)),
        lr0=args.lr0 if args.lr0 is not None else float(cfg.get("lr0", 0.0001)),
        device=args.device or cfg.get("device", "auto"),
        project=args.project or cfg.get("project", "runs/rtdetr"),
        name=args.name or cfg.get("name", "train"),
        freeze=args.freeze if args.freeze is not None else int(cfg.get("freeze", 10)),
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 42)),
        workers=args.workers if args.workers is not None else int(cfg.get("workers", 4)),
        resume=args.resume,
    )
    print(f"Best RT-DETR checkpoint: {best}")


if __name__ == "__main__":
    main()
