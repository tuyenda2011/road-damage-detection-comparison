from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


class CheckpointIntegrityError(RuntimeError):
    """Raised when a checkpoint does not match its integrity metadata."""


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def checksum_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(path.name + ".sha256")


def previous_checkpoint_path(path: str | Path) -> Path:
    path = Path(path)
    return path.with_name(f"{path.stem}.previous{path.suffix}")


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as file:
            file.write(text)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_checksum(path: Path, digest: str | None = None) -> str:
    digest = digest or sha256_file(path)
    _atomic_write_text(checksum_path(path), f"{digest}  {path.name}\n")
    return digest


def verify_checkpoint(path: str | Path, require_checksum: bool = True) -> str:
    checkpoint = Path(path)
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        raise CheckpointIntegrityError(f"Checkpoint is missing or empty: {checkpoint}")
    sidecar = checksum_path(checkpoint)
    if not sidecar.is_file():
        if require_checksum:
            raise CheckpointIntegrityError(f"Checkpoint checksum is missing: {sidecar}")
        return sha256_file(checkpoint)
    expected = sidecar.read_text(encoding="utf-8").strip().split(maxsplit=1)[0]
    actual = sha256_file(checkpoint)
    if expected != actual:
        raise CheckpointIntegrityError(
            f"Checkpoint checksum mismatch: {checkpoint} (expected {expected}, got {actual})"
        )
    return actual


def _backup_current(path: Path) -> None:
    if not path.is_file():
        return
    sidecar = checksum_path(path)
    if sidecar.is_file():
        try:
            verify_checkpoint(path, require_checksum=True)
        except CheckpointIntegrityError:
            # Never replace a known-good previous generation with a corrupted
            # current file. The new temporary checkpoint is still publishable.
            return
    previous = previous_checkpoint_path(path)
    temporary_previous = previous.with_name(f".{previous.name}.{os.getpid()}.tmp")
    temporary_previous.unlink(missing_ok=True)
    try:
        os.link(path, temporary_previous)
    except OSError:
        shutil.copy2(path, temporary_previous)
    os.replace(temporary_previous, previous)
    _write_checksum(previous, sha256_file(path))


def atomic_torch_save(payload: Any, path: str | Path, keep_previous: bool = True) -> Path:
    """Serialize, reload-verify and atomically publish a PyTorch checkpoint."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        # Windows requires a writable descriptor for fsync.
        with temporary.open("r+b") as file:
            os.fsync(file.fileno())
        torch.load(temporary, map_location="cpu", weights_only=True)
        digest = sha256_file(temporary)
        if keep_previous:
            _backup_current(destination)
        os.replace(temporary, destination)
        _write_checksum(destination, digest)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def atomic_copy_checkpoint(source: str | Path, destination: str | Path, keep_previous: bool = True) -> Path:
    """Copy a framework checkpoint atomically and attach SHA-256 integrity metadata."""
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise FileNotFoundError(f"Source checkpoint is missing or empty: {source_path}")
    if source_path.resolve() == destination_path.resolve():
        _write_checksum(destination_path)
        return destination_path

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.", suffix=".tmp", dir=destination_path.parent
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        source_digest = sha256_file(source_path)
        shutil.copyfile(source_path, temporary)
        with temporary.open("r+b") as file:
            os.fsync(file.fileno())
        if sha256_file(temporary) != source_digest:
            raise CheckpointIntegrityError(f"Checkpoint copy verification failed: {source_path}")
        if keep_previous:
            _backup_current(destination_path)
        os.replace(temporary, destination_path)
        _write_checksum(destination_path, source_digest)
        return destination_path
    finally:
        temporary.unlink(missing_ok=True)


def resolve_checkpoint_with_fallback(path: str | Path, require_checksum: bool = False) -> Path:
    checkpoint = Path(path)
    errors: list[str] = []
    for candidate in (checkpoint, previous_checkpoint_path(checkpoint)):
        if not candidate.is_file():
            continue
        try:
            verify_checkpoint(candidate, require_checksum=require_checksum)
            if candidate != checkpoint:
                print(f"Warning: using previous verified checkpoint after primary failure: {candidate}")
            return candidate
        except CheckpointIntegrityError as exc:
            errors.append(str(exc))
    detail = "; ".join(errors) if errors else "no checkpoint file found"
    raise CheckpointIntegrityError(f"No valid checkpoint for {checkpoint}: {detail}")


def load_torch_checkpoint(
    path: str | Path,
    *,
    map_location: str | torch.device = "cpu",
    weights_only: bool = True,
    require_checksum: bool = False,
) -> tuple[Any, Path]:
    """Load the primary checkpoint, falling back to its verified previous generation."""
    checkpoint = Path(path)
    errors: list[str] = []
    for candidate in (checkpoint, previous_checkpoint_path(checkpoint)):
        if not candidate.is_file():
            continue
        try:
            verify_checkpoint(candidate, require_checksum=require_checksum)
            payload = torch.load(candidate, map_location=map_location, weights_only=weights_only)
            if candidate != checkpoint:
                print(f"Warning: loaded previous checkpoint generation: {candidate}")
            return payload, candidate
        except (CheckpointIntegrityError, OSError, RuntimeError, ValueError, EOFError, pickle.UnpicklingError) as exc:
            errors.append(f"{candidate}: {exc}")
    detail = "; ".join(errors) if errors else "no checkpoint file found"
    raise CheckpointIntegrityError(f"Could not load a valid checkpoint for {checkpoint}: {detail}")


def write_checkpoint_manifest(
    checkpoint_dir: str | Path,
    model_name: str,
    checkpoints: dict[str, str | Path],
    metadata: dict[str, Any] | None = None,
) -> Path:
    directory = Path(checkpoint_dir)
    records: dict[str, Any] = {}
    for name, checkpoint in checkpoints.items():
        path = Path(checkpoint)
        if path.is_file():
            records[name] = {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": verify_checkpoint(path, require_checksum=True),
            }
    payload = {
        "model": model_name,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoints": records,
        "metadata": metadata or {},
    }
    manifest = directory / "manifest.json"
    _atomic_write_text(manifest, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return manifest


class UltralyticsCheckpointPublisher:
    """Publish trainer checkpoints safely after each Ultralytics save event."""

    def __init__(
        self,
        checkpoint_dir: str | Path,
        model_name: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.directory = Path(checkpoint_dir)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.metadata = metadata or {}
        self._published_signatures: dict[str, tuple[int, int]] = {}

    def publish(
        self,
        trainer: Any,
        names: tuple[str, ...] = ("best", "last"),
        *,
        force: bool = False,
    ) -> dict[str, Path]:
        published: dict[str, Path] = {}
        for name in names:
            source = Path(getattr(trainer, name, ""))
            if not source.is_file():
                continue
            stat = source.stat()
            signature = (stat.st_size, stat.st_mtime_ns)
            if not force and self._published_signatures.get(name) == signature:
                continue
            destination = self.directory / source.name
            atomic_copy_checkpoint(source, destination)
            self._published_signatures[name] = signature
            published[name] = destination
        return published

    def write_manifest(self) -> Path:
        checkpoints = {
            name: self.directory / f"{name}.pt"
            for name in ("best", "last")
            if (self.directory / f"{name}.pt").is_file()
        }
        return write_checkpoint_manifest(
            self.directory,
            self.model_name,
            checkpoints,
            metadata=self.metadata,
        )
