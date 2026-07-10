from __future__ import annotations

import hashlib
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[2]


def add_project_root_to_path() -> None:
    """Allow direct execution of scripts under src/ from the project root."""
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


def project_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def portable_path(path: str | Path) -> str:
    resolved = project_path(path).resolve()
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def ensure_dir(path: str | Path) -> Path:
    path = project_path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = project_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def resolve_device(device: str | int | None = "auto") -> str:
    value = "auto" if device is None else str(device).strip().lower()
    if value == "auto":
        return f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
    requests_cuda = value == "cuda" or value.startswith("cuda:") or value.isdigit()
    if requests_cuda and not torch.cuda.is_available():
        print("Warning: CUDA is not available. Falling back to CPU; training/inference will be slow.")
        return "cpu"
    if value == "cuda":
        value = f"cuda:{torch.cuda.current_device()}"
    if value.isdigit():
        value = f"cuda:{value}"
    if value.startswith("cuda:"):
        try:
            index = int(value.split(":", 1)[1])
        except ValueError as exc:
            raise ValueError(f"Invalid CUDA device: {device}") from exc
        if index < 0 or index >= torch.cuda.device_count():
            raise ValueError(f"CUDA device index {index} is unavailable; found {torch.cuda.device_count()} device(s).")
    try:
        torch.device(value)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(f"Invalid device: {device}") from exc
    return value


def resolve_ultralytics_device(device: str | int | None = "auto") -> str | int:
    value = "auto" if device is None else str(device).strip()
    if "," in value:
        indices = [part.strip() for part in value.split(",")]
        if not indices or any(not index.isdigit() for index in indices):
            raise ValueError(f"Invalid Ultralytics device list: {device}")
        if len(indices) != len(set(indices)):
            raise ValueError(f"Duplicate device index in Ultralytics device list: {device}")
        if not torch.cuda.is_available():
            print("Warning: CUDA is not available. Falling back to CPU; training/inference will be slow.")
            return "cpu"
        for index in indices:
            resolve_device(index)
        return ",".join(indices)
    resolved = resolve_device(device)
    if resolved == "cuda":
        return 0
    if resolved.startswith("cuda:"):
        return int(resolved.split(":", 1)[1])
    return resolved


def validate_confidence(confidence: float) -> float:
    value = float(confidence)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Confidence threshold must be a finite value between 0 and 1.")
    return value


def resolve_model_reference(model: str | Path) -> str:
    """Resolve an existing local model while preserving downloadable model aliases."""
    path = Path(model)
    if path.is_absolute():
        return str(path)
    local_path = project_path(path)
    return str(local_path) if local_path.exists() else str(model)


def resolve_inference_weights(weights: str | Path, model_dir: str | Path, extension: str) -> Path:
    """Resolve canonical weights, falling back to the newest completed run."""
    requested = project_path(weights)
    if requested.is_file():
        return requested
    canonical = project_path(model_dir) / f"best{extension}"
    if requested.resolve() == canonical.resolve():
        candidates = [path for path in project_path(model_dir).glob(f"*/weights/best{extension}") if path.is_file()]
        if candidates:
            fallback = max(candidates, key=lambda path: path.stat().st_mtime_ns)
            print(f"Warning: canonical checkpoint is missing; using latest run checkpoint: {fallback}")
            return fallback
    raise FileNotFoundError(f"Model checkpoint not found: {requested}")


def write_ultralytics_dataset(path: str | Path, output_dir: str | Path) -> Path:
    """Write a portable runtime YAML with an absolute dataset root."""
    dataset = load_yaml(path)
    dataset["path"] = str(project_path(dataset.get("path", "data/processed")).resolve())
    fingerprint = hashlib.sha1(
        yaml.safe_dump(dataset, sort_keys=True, allow_unicode=True).encode("utf-8")
    ).hexdigest()[:12]
    output_path = ensure_dir(output_dir) / f"dataset.{fingerprint}.resolved.yaml"
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(dataset, file, sort_keys=False, allow_unicode=True)
    temporary_path.replace(output_path)
    return output_path


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def list_images(path: str | Path) -> list[Path]:
    path = project_path(path)
    if not path.exists():
        return []
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def is_video_file(path: str | Path) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


CLASS_NAMES = ["D00", "D10", "D20", "D40"]
CLASS_TO_ID = {name: idx for idx, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {idx: name for idx, name in enumerate(CLASS_NAMES)}


def count_by_class(labels: np.ndarray) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for label in labels:
        idx = int(label)
        if 0 <= idx < len(CLASS_NAMES):
            counts[CLASS_NAMES[idx]] += 1
    return counts


def source_kind(path: str | Path) -> str:
    return "video" if is_video_file(path) else "image"


def main() -> None:
    print(f"Project root: {ROOT}")
    print(f"Classes: {', '.join(CLASS_NAMES)}")


if __name__ == "__main__":
    main()
