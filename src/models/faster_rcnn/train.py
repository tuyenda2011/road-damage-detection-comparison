from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset.road_damage_dataset import RoadDamageDataset, collate_fn
from src.evaluation.metrics import map50
from src.models.faster_rcnn.model import build_faster_rcnn
from src.utils.checkpoint import (
    atomic_torch_save,
    load_torch_checkpoint,
    resolve_checkpoint_with_fallback,
    sha256_file,
    write_checkpoint_manifest,
)
from src.utils.common import ensure_dir, load_yaml, project_path, resolve_device, set_seed


def freeze_backbone(model: torch.nn.Module) -> None:
    for param in model.backbone.parameters():
        param.requires_grad = False


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def dataset_manifest_sha256(data_root: str | Path) -> str | None:
    manifest = project_path(data_root) / "split_manifest.csv"
    return sha256_file(manifest) if manifest.is_file() else None


def restore_rng_state(checkpoint: dict) -> None:
    rng_state = checkpoint.get("rng_state")
    if not isinstance(rng_state, dict):
        return
    torch_state = rng_state.get("torch")
    if isinstance(torch_state, torch.Tensor):
        torch.set_rng_state(torch_state.cpu())
    cuda_states = rng_state.get("cuda")
    if torch.cuda.is_available() and isinstance(cuda_states, list):
        torch.cuda.set_rng_state_all([state.cpu() for state in cuda_states])


def train_one_epoch(
    model,
    loader,
    optimizer,
    scaler: torch.amp.GradScaler,
    device: str,
    epoch: int,
    amp_enabled: bool,
    grad_clip: float,
) -> float:
    model.train()
    total_loss = 0.0
    progress = tqdm(loader, desc=f"Epoch {epoch} train")
    for images, targets in progress:
        images = [img.to(device, non_blocking=True) for img in images]
        targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
        with torch.autocast(device_type=torch.device(device).type, enabled=amp_enabled):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = float(losses.detach().cpu())
        if not math.isfinite(loss_value):
            raise FloatingPointError(f"Non-finite training loss at epoch {epoch}: {loss_value}")
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(losses).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss_value
        progress.set_postfix(loss=f"{loss_value:.4f}")
    return total_loss / max(len(loader), 1)


@torch.inference_mode()
def validate_map50(model, loader, device: str, epoch: int) -> float:
    model.eval()
    predictions: list[dict] = []
    ground_truths: list[dict] = []
    previous_threshold = float(model.roi_heads.score_thresh)
    model.roi_heads.score_thresh = 0.001
    try:
        for images, targets in tqdm(loader, desc=f"Epoch {epoch} val"):
            device_images = [image.to(device, non_blocking=True) for image in images]
            outputs = model(device_images)
            for output, target in zip(outputs, targets):
                predictions.append(
                    {
                        "boxes": output["boxes"].detach().cpu().numpy(),
                        "labels": output["labels"].detach().cpu().numpy(),
                        "scores": output["scores"].detach().cpu().numpy(),
                    }
                )
                ground_truths.append(
                    {
                        "boxes": target["boxes"].numpy(),
                        "labels": target["labels"].numpy(),
                    }
                )
    finally:
        model.roi_heads.score_thresh = previous_threshold
    return map50(predictions, ground_truths, num_classes=5)


def train_faster_rcnn(
    data_root: str,
    epochs: int,
    batch: int,
    lr: float,
    device: str,
    output_dir: str,
    num_workers: int = 2,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    freeze_backbone_params: bool = True,
    seed: int = 42,
    amp: bool = True,
    grad_clip: float = 10.0,
    resume: str | None = None,
) -> Path:
    if epochs <= 0 or batch <= 0 or lr <= 0 or num_workers < 0 or grad_clip <= 0:
        raise ValueError(
            "epochs, batch, lr and grad_clip must be positive; num_workers cannot be negative."
        )
    set_seed(seed)
    device = resolve_device(device)
    data_root_path = project_path(data_root)
    manifest_digest = dataset_manifest_sha256(data_root_path)
    output_dir_path = ensure_dir(output_dir)
    checkpoint_dir = ensure_dir(output_dir_path / "checkpoints")
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"

    resume_checkpoint: dict | None = None
    loaded_resume_path: Path | None = None
    if resume:
        requested_resume = last_path if resume.lower() == "auto" else project_path(resume)
        resume_checkpoint, loaded_resume_path = load_torch_checkpoint(
            requested_resume,
            map_location="cpu",
            weights_only=True,
            require_checksum=requested_resume.parent.name == "checkpoints",
        )
        if not isinstance(resume_checkpoint, dict) or "model_state_dict" not in resume_checkpoint:
            raise ValueError(f"Resume checkpoint has no training state: {loaded_resume_path}")
        stored_manifest = resume_checkpoint.get("dataset_manifest_sha256")
        if stored_manifest is not None and stored_manifest != manifest_digest:
            raise ValueError(
                "Dataset split manifest changed since the checkpoint was saved; "
                "refusing an unsafe resume. Start a fresh run instead."
            )
        stored_freeze = resume_checkpoint.get("freeze_backbone")
        if stored_freeze is not None and bool(stored_freeze) != freeze_backbone_params:
            raise ValueError(
                "freeze_backbone differs from the resume checkpoint; use the original setting or start fresh."
            )

    train_ds = RoadDamageDataset(data_root_path, split="train")
    val_ds = RoadDamageDataset(data_root_path, split="val")
    use_pinned_memory = device.startswith("cuda")
    loader_options = {
        "num_workers": num_workers,
        "collate_fn": collate_fn,
        "pin_memory": use_pinned_memory,
        "persistent_workers": num_workers > 0,
    }
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, **loader_options)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, **loader_options)

    model = build_faster_rcnn(pretrained=resume_checkpoint is None).to(device)
    if freeze_backbone_params:
        freeze_backbone(model)
        print("Frozen Faster R-CNN backbone; training RPN and ROI heads only.")

    total_params, trainable_params = count_parameters(model)
    print(f"Trainable parameters: {trainable_params:,}/{total_params:,}")

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=8, gamma=0.1)
    amp_enabled = amp and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    print(f"Automatic mixed precision: {'enabled' if amp_enabled else 'disabled'}")

    best_map = -1.0
    start_epoch = 1
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if isinstance(scaler_state, dict):
            scaler.load_state_dict(scaler_state)
        start_epoch = int(resume_checkpoint.get("epoch", 0)) + 1
        best_map = float(resume_checkpoint.get("best_map50", resume_checkpoint.get("val_map50", -1.0)))
        restore_rng_state(resume_checkpoint)
        print(
            f"Resuming Faster R-CNN from {loaded_resume_path} at epoch {start_epoch}/{epochs} "
            f"(best mAP@50={best_map:.4f})."
        )

    if start_epoch > epochs:
        resolved_best = resolve_checkpoint_with_fallback(best_path, require_checksum=True)
        print(f"Checkpoint already reached epoch {start_epoch - 1}; no additional training required.")
        return resolved_best

    for epoch in range(start_epoch, epochs + 1):
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            epoch,
            amp_enabled,
            grad_clip,
        )
        val_map = validate_map50(model, val_loader, device, epoch)
        if not math.isfinite(val_map):
            raise FloatingPointError(f"Non-finite validation mAP@50 at epoch {epoch}: {val_map}")
        scheduler.step()
        print(f"Epoch {epoch}/{epochs}: train_loss={train_loss:.4f}, val_map50={val_map:.4f}")
        if val_map > best_map:
            best_map = val_map
            inference_checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_map50": val_map,
                "dataset_manifest_sha256": manifest_digest,
            }
            atomic_torch_save(inference_checkpoint, best_path)
            print(f"Saved new best checkpoint: {best_path}")
        training_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "train_loss": train_loss,
            "val_map50": val_map,
            "best_map50": best_map,
            "dataset_manifest_sha256": manifest_digest,
            "freeze_backbone": freeze_backbone_params,
            "rng_state": {
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
        }
        atomic_torch_save(training_checkpoint, last_path)
    resolved_best = resolve_checkpoint_with_fallback(best_path, require_checksum=True)
    write_checkpoint_manifest(
        checkpoint_dir,
        "faster_rcnn",
        {"best": resolved_best, "last": last_path},
        metadata={
            "data": str(data_root_path.resolve()),
            "dataset_manifest_sha256": manifest_digest,
            "epochs": epochs,
            "batch": batch,
            "seed": seed,
            "best_map50": best_map,
            "resume_from": str(loaded_resume_path) if loaded_resume_path else None,
        },
    )
    return resolved_best


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Faster R-CNN on Road Damage Detection.")
    parser.add_argument("--config", default="configs/faster_rcnn.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--freeze-backbone", dest="freeze_backbone", action="store_true", default=None)
    parser.add_argument("--no-freeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument(
        "--resume",
        default=None,
        help="Resume from a checkpoint path, or use 'auto' for runs/faster_rcnn/checkpoints/last.pth.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    best = train_faster_rcnn(
        data_root=args.data_root or cfg.get("data_root", "data/processed"),
        epochs=args.epochs if args.epochs is not None else int(cfg.get("epochs", 30)),
        batch=args.batch if args.batch is not None else int(cfg.get("batch", 2)),
        lr=args.lr if args.lr is not None else float(cfg.get("lr", 0.0025)),
        device=args.device or cfg.get("device", "auto"),
        output_dir=args.output_dir or cfg.get("output_dir", "runs/faster_rcnn"),
        num_workers=args.num_workers if args.num_workers is not None else int(cfg.get("num_workers", 2)),
        momentum=float(cfg.get("momentum", 0.9)),
        weight_decay=float(cfg.get("weight_decay", 0.0005)),
        freeze_backbone_params=(
            args.freeze_backbone if args.freeze_backbone is not None else bool(cfg.get("freeze_backbone", True))
        ),
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 42)),
        amp=args.amp if args.amp is not None else bool(cfg.get("amp", True)),
        grad_clip=args.grad_clip if args.grad_clip is not None else float(cfg.get("grad_clip", 10.0)),
        resume=args.resume,
    )
    print(f"Best Faster R-CNN checkpoint: {project_path(best)}")


if __name__ == "__main__":
    main()
