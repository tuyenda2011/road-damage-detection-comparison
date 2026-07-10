from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import math
from pathlib import Path

import cv2


def video_properties(capture: cv2.VideoCapture) -> tuple[int, int, float]:
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        raise ValueError("Input video has invalid frame dimensions.")
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    return width, height, fps


def create_video_writer(path: str | Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        writer.release()
        raise IOError(f"Could not create output video: {path}")
    return writer


@contextmanager
def managed_video_output(
    capture: cv2.VideoCapture,
    writer: cv2.VideoWriter,
    path: str | Path,
) -> Iterator[None]:
    completed = False
    try:
        yield
        completed = True
    finally:
        capture.release()
        writer.release()
        if not completed:
            Path(path).unlink(missing_ok=True)
