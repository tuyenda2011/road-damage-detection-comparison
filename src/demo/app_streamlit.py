"""Streamlit control center for road-damage detection and model comparison.

Run from the project root:
    streamlit run src/demo/app_streamlit.py --server.address 127.0.0.1
"""
from __future__ import annotations

import json
import importlib
import sys
import tempfile
import threading
import time
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch

from src.demo.demo_image import load_detector
from src.demo.demo_video import run_video_demo
import src.demo.streamlit_styles as streamlit_styles
from src.utils.checkpoint import CheckpointIntegrityError
from src.utils.common import CLASS_NAMES, count_by_class, project_path, resolve_inference_weights
from src.utils.visualization import draw_detections


MODEL_OPTIONS: dict[str, str] = {
    "YOLO · One-stage": "yolo",
    "Faster R-CNN · Two-stage": "faster_rcnn",
    "RT-DETR · Transformer": "rtdetr",
}

MODEL_INFO: dict[str, dict[str, str]] = {
    "yolo": {
        "name": "YOLO",
        "family": "One-stage detector",
        "description": "Pipeline gọn, phù hợp kiểm tra ảnh và luồng video thời gian thực.",
    },
    "faster_rcnn": {
        "name": "Faster R-CNN",
        "family": "Two-stage detector",
        "description": "Region proposal và ROI head, dùng làm mốc so sánh detector hai giai đoạn.",
    },
    "rtdetr": {
        "name": "RT-DETR",
        "family": "End-to-end Transformer",
        "description": "Detector Transformer end-to-end, cân bằng cấu trúc và tốc độ suy luận.",
    },
}

MODEL_COLOR: dict[str, str] = {
    "yolo": "#0F9F8F",
    "faster_rcnn": "#A96B00",
    "rtdetr": "#7C3AED",
}

CLASS_FULL_NAME: dict[str, str] = {
    "D00": "Nứt dọc",
    "D10": "Nứt ngang",
    "D20": "Nứt lưới",
    "D40": "Ổ gà",
}

CLASS_COLOR_HEX: dict[str, str] = {
    "D00": "#D95D2A",
    "D10": "#A96B00",
    "D20": "#7C3AED",
    "D40": "#D94C4C",
}

WEIGHT_DEFAULTS: dict[str, tuple[str, str, str]] = {
    "yolo": ("runs/yolo/checkpoints/best.pt", "runs/yolo", ".pt"),
    "faster_rcnn": ("runs/faster_rcnn/checkpoints/best.pth", "runs/faster_rcnn", ".pth"),
    "rtdetr": ("runs/rtdetr/checkpoints/best.pt", "runs/rtdetr", ".pt"),
}

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov", "mkv", "m4v"]


def inject_css() -> None:
    # Streamlit keeps imported modules alive between reruns. Reloading here makes
    # style edits visible immediately instead of retaining an older dark theme.
    current_styles = importlib.reload(streamlit_styles)
    st.markdown(current_styles.APP_CSS, unsafe_allow_html=True)


def resolve_weight(model_key: str, weights: str | Path) -> Path:
    _, model_dir, extension = WEIGHT_DEFAULTS[model_key]
    return resolve_inference_weights(weights, model_dir, extension)


def default_weight(model_key: str) -> str:
    requested, _, _ = WEIGHT_DEFAULTS[model_key]
    try:
        return str(resolve_weight(model_key, requested))
    except (FileNotFoundError, CheckpointIntegrityError):
        return requested


def weight_health(model_key: str, weights: str | Path) -> tuple[bool, Path | None, str]:
    try:
        resolved = resolve_weight(model_key, weights)
        return True, resolved, "Checkpoint đã xác minh"
    except FileNotFoundError:
        return False, None, "Chưa có checkpoint"
    except CheckpointIntegrityError as exc:
        return False, None, f"Checkpoint không hợp lệ: {exc}"


def checkpoint_readiness() -> dict[str, tuple[bool, Path | None, str]]:
    return {
        key: weight_health(key, WEIGHT_DEFAULTS[key][0])
        for key in MODEL_INFO
    }


@st.cache_data(show_spinner=False)
def load_dataset_snapshot(path_text: str, mtime_ns: int) -> dict[str, Any]:
    del mtime_ns
    path = Path(path_text)
    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def dataset_snapshot() -> dict[str, Any]:
    path = project_path("data/processed/split_summary.json")
    mtime_ns = path.stat().st_mtime_ns if path.is_file() else 0
    return load_dataset_snapshot(str(path), mtime_ns)


def available_devices() -> list[str]:
    devices = ["auto", "cpu"]
    if torch.cuda.is_available():
        devices.extend(f"cuda:{index}" for index in range(torch.cuda.device_count()))
    return devices


def encode_image_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Không thể mã hóa ảnh kết quả.")
    return buffer.tobytes()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def decode_uploaded_image(uploaded_file) -> np.ndarray | None:
    data = np.frombuffer(uploaded_file.getvalue(), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


@st.cache_resource(show_spinner=False)
def cached_detector(model_key: str, resolved_weights: str, device: str, mtime_ns: int):
    """Cache one detector and invalidate it when the checkpoint changes."""
    del mtime_ns
    detector, predict_fn = load_detector(model_key, resolved_weights, device=device)
    return detector, predict_fn, threading.Lock()


def run_inference(
    model_key: str,
    resolved_weights: str,
    device: str,
    image: np.ndarray,
    confidence: float,
) -> tuple[np.ndarray, Any, Any, Any, float]:
    checkpoint = Path(resolved_weights)
    mtime_ns = checkpoint.stat().st_mtime_ns if checkpoint.is_file() else 0
    detector, predict_fn, lock = cached_detector(model_key, resolved_weights, device, mtime_ns)
    started = time.perf_counter()
    with lock:
        boxes, labels, scores = predict_fn(
            detector,
            image,
            conf=confidence,
            device=device,
        )
    inference_ms = (time.perf_counter() - started) * 1000
    result = draw_detections(image, boxes, labels, scores)
    return result, boxes, labels, scores, inference_ms


def build_detection_df(boxes, labels, scores) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, (box, label, score) in enumerate(zip(boxes, labels, scores), start=1):
        label_id = int(label)
        class_name = CLASS_NAMES[label_id] if 0 <= label_id < len(CLASS_NAMES) else str(label_id)
        x1, y1, x2, y2 = [int(round(float(value))) for value in box]
        rows.append(
            {
                "#": index,
                "Lớp": class_name,
                "Loại hư hỏng": CLASS_FULL_NAME.get(class_name, class_name),
                "Confidence": round(float(score), 4),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "Rộng": x2 - x1,
                "Cao": y2 - y1,
            }
        )
    return pd.DataFrame(rows)


def section_header(eyebrow: str, title: str, description: str) -> None:
    st.markdown(
        f"""
<div class="section-heading">
  <div class="section-eyebrow">{escape(eyebrow)}</div>
  <div class="section-title">{escape(title)}</div>
  <div class="section-copy">{escape(description)}</div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_app_header() -> None:
    snapshot = dataset_snapshot()
    splits = snapshot.get("splits", {}) if isinstance(snapshot.get("splits"), dict) else {}
    train_images = int(splits.get("train", {}).get("images", 0))
    test_images = int(splits.get("test", {}).get("images", 0))
    ready = sum(status[0] for status in checkpoint_readiness().values())
    st.markdown(
        f"""
<div class="app-hero">
  <div class="hero-grid">
    <div>
      <div class="hero-eyebrow">Road intelligence workspace</div>
      <div class="hero-title">Quan sát mặt đường.<br>Ra quyết định từ dữ liệu.</div>
      <div class="hero-copy">
        Một không gian thống nhất để chạy phát hiện, kiểm tra checkpoint và so sánh
        YOLO, Faster R-CNN, RT-DETR trên cùng giao thức đánh giá.
      </div>
    </div>
    <div class="hero-stats">
      <div class="hero-stat">
        <div class="hero-stat-value">{ready}/3</div>
        <div class="hero-stat-label">checkpoint sẵn sàng</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">{train_images:,}</div>
        <div class="hero-stat-label">ảnh train</div>
      </div>
      <div class="hero-stat">
        <div class="hero-stat-value">{test_images:,}</div>
        <div class="hero-stat-label">ảnh test có nhãn</div>
      </div>
    </div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )


def render_checkpoint_cards() -> None:
    readiness = checkpoint_readiness()
    columns = st.columns(3)
    for column, model_key in zip(columns, MODEL_INFO):
        info = MODEL_INFO[model_key]
        ready, resolved, _ = readiness[model_key]
        color = MODEL_COLOR[model_key]
        state_class = "ready" if ready else "missing"
        state_text = "Sẵn sàng" if ready else "Chưa train"
        path_text = resolved.name if resolved else Path(WEIGHT_DEFAULTS[model_key][0]).name
        column.markdown(
            f"""
<div class="checkpoint-card">
  <div class="checkpoint-top">
    <div class="checkpoint-name">{escape(info['name'])}</div>
    <div class="checkpoint-state {state_class}">● {state_text}</div>
  </div>
  <div class="checkpoint-copy">{escape(info['family'])}<br>{escape(path_text)}</div>
  <div class="checkpoint-line" style="background:{color};"></div>
</div>
""",
            unsafe_allow_html=True,
        )


def render_sidebar() -> dict[str, Any]:
    with st.sidebar:
        st.markdown(
            """
<div class="brand-lockup">
  <div class="brand-mark">⌁</div>
  <div>
    <div class="brand-name">Road Damage Lab</div>
    <div class="brand-meta">DETECTION CONTROL CENTER · V3</div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown('<div class="side-label">Mô hình đang dùng</div>', unsafe_allow_html=True)
        model_label = st.radio(
            "Mô hình",
            list(MODEL_OPTIONS),
            label_visibility="collapsed",
            key="selected_model",
        )
        model_key = MODEL_OPTIONS[model_label]
        info = MODEL_INFO[model_key]
        st.markdown(
            f"""
<div class="model-context">
  <div class="model-context-title" style="color:{MODEL_COLOR[model_key]};">{escape(info['family'])}</div>
  <div class="model-context-copy">{escape(info['description'])}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="side-label">Ngưỡng & thiết bị</div>', unsafe_allow_html=True)
        confidence = st.slider(
            "Confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            format="%.2f",
            help="0.25 là ngưỡng khởi đầu cân bằng cho demo.",
        )
        device = st.selectbox(
            "Thiết bị suy luận",
            available_devices(),
            help="Auto ưu tiên GPU khả dụng và tự rơi về CPU.",
        )

        with st.expander("Checkpoint & runtime", expanded=False):
            weights = st.text_input(
                "Đường dẫn checkpoint",
                value=default_weight(model_key),
                key=f"weights_{model_key}",
            )
            ready, resolved, status_message = weight_health(model_key, weights)
            state_class = "ready" if ready else ""
            st.markdown(
                f"""
<div class="runtime-row">
  <span class="runtime-label"><span class="status-dot {state_class}"></span>Checkpoint</span>
  <span class="runtime-value">{'Đã xác minh' if ready else 'Chưa sẵn sàng'}</span>
</div>
<div class="runtime-row">
  <span class="runtime-label">Dataset</span>
  <span class="runtime-value">80 / 10 / 10</span>
</div>
<div class="runtime-row">
  <span class="runtime-label">Accelerator</span>
  <span class="runtime-value">{escape(device)}</span>
</div>
""",
                unsafe_allow_html=True,
            )
            if not ready:
                st.caption(status_message)
            if st.button("Làm mới model cache", width="stretch"):
                cached_detector.clear()
                st.toast("Đã xóa model cache.")

        st.divider()
        st.caption("Dataset: RDD2022 · Classes: D00, D10, D20, D40")

    ready, resolved, status_message = weight_health(model_key, weights)
    return {
        "model_label": model_label,
        "model_key": model_key,
        "weights": weights,
        "resolved_weights": str(resolved) if resolved else None,
        "weight_ok": ready,
        "weight_status": status_message,
        "confidence": confidence,
        "device": device,
    }


def render_demo_tab(config: dict[str, Any]) -> None:
    section_header(
        "Detection workbench",
        "Phân tích ảnh và video",
        "Chọn nguồn dữ liệu, chạy một mô hình hoặc đặt ba kiến trúc cạnh nhau trên cùng một ảnh.",
    )

    source_mode = st.radio(
        "Loại dữ liệu",
        ["Ảnh tĩnh", "Video"],
        horizontal=True,
        label_visibility="collapsed",
        key="source_mode",
    )
    is_image = source_mode == "Ảnh tĩnh"
    uploaded = st.file_uploader(
        "Kéo thả ảnh cần kiểm tra" if is_image else "Kéo thả video cần kiểm tra",
        type=IMAGE_TYPES if is_image else VIDEO_TYPES,
        key="analysis_upload_image" if is_image else "analysis_upload_video",
    )

    action_run, action_compare, action_clear = st.columns([1.2, 1.1, 0.7])
    with action_run:
        run_single = st.button(
            "Chạy phát hiện",
            type="primary",
            width="stretch",
            disabled=uploaded is None or not config["weight_ok"],
        )
    with action_compare:
        run_all = st.button(
            "So sánh 3 mô hình",
            type="secondary",
            width="stretch",
            disabled=uploaded is None or not is_image,
            help="Mỗi mô hình phải có checkpoint hợp lệ.",
        )
    with action_clear:
        clear_result = st.button(
            "Xóa kết quả",
            width="stretch",
            disabled="last_result" not in st.session_state and "compare_result" not in st.session_state,
        )

    if clear_result:
        st.session_state.pop("last_result", None)
        st.session_state.pop("compare_result", None)
        st.rerun()

    if not config["weight_ok"]:
        st.warning(
            f"Checkpoint {MODEL_INFO[config['model_key']]['name']} chưa sẵn sàng. "
            "Train mô hình hoặc mở **Checkpoint & runtime** để chọn file khác."
        )

    if uploaded is None:
        st.markdown(
            """
<div class="workbench-empty">
  <div class="workbench-icon">⌁</div>
  <div class="workbench-title">Workbench đang chờ dữ liệu</div>
  <div class="workbench-copy">Tải một ảnh hoặc video đường bộ để bắt đầu phân tích.</div>
  <div class="step-row">
    <span class="step-chip">01 · Chọn model</span>
    <span class="step-chip">02 · Tải dữ liệu</span>
    <span class="step-chip">03 · Kiểm tra kết quả</span>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown("<br>", unsafe_allow_html=True)
        render_checkpoint_cards()
        return

    size_mb = len(uploaded.getvalue()) / (1024 * 1024)
    st.markdown(
        f"""
<div class="file-summary">
  <div>
    <div class="file-name">{escape(uploaded.name)}</div>
    <div class="file-meta">{escape(source_mode)} · {size_mb:.2f} MB</div>
  </div>
  <div class="file-meta">{escape(MODEL_INFO[config['model_key']]['name'])} · conf {config['confidence']:.2f}</div>
</div>
""",
        unsafe_allow_html=True,
    )

    if run_all and is_image:
        run_compare_all(uploaded, config["confidence"], config["device"])
        return

    if run_single:
        if not config["resolved_weights"]:
            st.error("Không thể chạy vì checkpoint chưa hợp lệ.")
            return
        if is_image:
            run_image_inference(uploaded, config)
        else:
            run_video_inference(uploaded, config)
        return

    previous_compare = st.session_state.get("compare_result")
    if is_image and previous_compare and previous_compare.get("filename") == uploaded.name:
        display_compare_result(previous_compare)
        return

    previous = st.session_state.get("last_result")
    if previous and previous.get("filename") == uploaded.name:
        if previous.get("kind") == "image" and is_image:
            display_image_result(previous)
        elif previous.get("kind") == "video" and not is_image:
            display_video_result(previous)
        return

    if is_image:
        image = decode_uploaded_image(uploaded)
        if image is None:
            st.error("Không thể giải mã ảnh. Hãy thử JPG, PNG, BMP hoặc WebP khác.")
            return
        preview, context = st.columns([1.6, 0.8])
        preview.image(bgr_to_rgb(image), caption="Ảnh đầu vào", width="stretch")
        with context:
            st.metric("Kích thước", f"{image.shape[1]} × {image.shape[0]}")
            st.metric("Dung lượng", f"{size_mb:.2f} MB")
            st.info("Nhấn **Chạy phát hiện** để tạo bounding box và thống kê theo lớp.")
    else:
        st.video(uploaded.getvalue())
        st.info("Video được xử lý tuần tự theo frame. Thời gian chạy phụ thuộc độ dài và model.")


def run_image_inference(uploaded, config: dict[str, Any]) -> None:
    image = decode_uploaded_image(uploaded)
    if image is None:
        st.error("Không thể giải mã ảnh đầu vào.")
        return
    with st.spinner(f"Đang chạy {MODEL_INFO[config['model_key']]['name']}…"):
        try:
            result_bgr, boxes, labels, scores, inference_ms = run_inference(
                config["model_key"],
                config["resolved_weights"],
                config["device"],
                image,
                config["confidence"],
            )
        except Exception as exc:
            show_model_error(exc, config["model_key"])
            return

    counts = count_by_class(labels)
    result = {
        "kind": "image",
        "filename": uploaded.name,
        "model_key": config["model_key"],
        "model_name": MODEL_INFO[config["model_key"]]["name"],
        "confidence": config["confidence"],
        "result_rgb": bgr_to_rgb(result_bgr),
        "result_bgr": result_bgr,
        "boxes": boxes,
        "labels": labels,
        "scores": scores,
        "counts": counts,
        "inference_ms": inference_ms,
        "image_size": (image.shape[1], image.shape[0]),
    }
    st.session_state["last_result"] = result
    st.session_state.pop("compare_result", None)
    st.toast(f"Hoàn tất: {sum(counts.values())} phát hiện", icon="✅")
    display_image_result(result)


def display_image_result(result: dict[str, Any]) -> None:
    boxes = result["boxes"]
    scores = result["scores"]
    counts = result["counts"]
    total = sum(counts.values())
    width, height = result["image_size"]
    st.markdown(
        f"""
<div class="result-toolbar">
  <span class="result-chip"><strong>{escape(result['model_name'])}</strong></span>
  <span class="result-chip">conf <strong>{result['confidence']:.2f}</strong></span>
  <span class="result-chip">latency <strong>{result['inference_ms']:.0f} ms</strong></span>
  <span class="result-chip">detections <strong>{total}</strong></span>
  <span class="result-chip">input <strong>{width} × {height}</strong></span>
</div>
""",
        unsafe_allow_html=True,
    )

    image_column, insight_column = st.columns([1.7, 0.8])
    with image_column:
        st.image(result["result_rgb"], caption=f"Kết quả · {result['filename']}", width="stretch")
        st.download_button(
            "Tải ảnh đã đánh dấu",
            data=encode_image_to_png_bytes(result["result_bgr"]),
            file_name=f"detected_{Path(result['filename']).stem}.png",
            mime="image/png",
            width="stretch",
        )
    with insight_column:
        st.markdown("#### Phân bổ hư hỏng")
        maximum = max(counts.values(), default=1)
        for class_name in CLASS_NAMES:
            count = int(counts.get(class_name, 0))
            width_percent = (count / maximum * 100) if maximum else 0
            color = CLASS_COLOR_HEX[class_name]
            st.markdown(
                f"""
<div style="margin:0 0 .8rem;">
  <div style="display:flex;justify-content:space-between;font-size:.76rem;margin-bottom:.28rem;">
    <span style="color:{color};font-weight:700;">{class_name} · {escape(CLASS_FULL_NAME[class_name])}</span>
    <strong>{count}</strong>
  </div>
  <div style="height:6px;background:#dce6eb;border-radius:99px;overflow:hidden;">
    <div style="height:100%;width:{width_percent:.1f}%;background:{color};border-radius:99px;"></div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )
        average_confidence = float(np.mean(scores)) if len(scores) else 0.0
        maximum_confidence = float(np.max(scores)) if len(scores) else 0.0
        left, right = st.columns(2)
        left.metric("Conf trung bình", f"{average_confidence:.2f}")
        right.metric("Conf cao nhất", f"{maximum_confidence:.2f}")

    if len(boxes):
        with st.expander(f"Chi tiết {total} bounding box"):
            st.dataframe(
                build_detection_df(boxes, result["labels"], scores),
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence",
                        min_value=0,
                        max_value=1,
                        format="%.3f",
                    )
                },
                width="stretch",
                hide_index=True,
            )


def run_compare_all(uploaded, confidence: float, device: str) -> None:
    image = decode_uploaded_image(uploaded)
    if image is None:
        st.error("Không thể giải mã ảnh đầu vào.")
        return
    results: dict[str, dict[str, Any] | None] = {}
    progress = st.progress(0, text="Đang chuẩn bị so sánh…")
    for index, model_key in enumerate(MODEL_INFO, start=1):
        ready, resolved, status = weight_health(model_key, default_weight(model_key))
        progress.progress((index - 1) / len(MODEL_INFO), text=f"Đang chạy {MODEL_INFO[model_key]['name']}…")
        if not ready or resolved is None:
            results[model_key] = {"error": status}
            continue
        try:
            result_bgr, boxes, labels, scores, inference_ms = run_inference(
                model_key,
                str(resolved),
                device,
                image,
                confidence,
            )
            results[model_key] = {
                "result_rgb": bgr_to_rgb(result_bgr),
                "result_bgr": result_bgr,
                "boxes": boxes,
                "labels": labels,
                "scores": scores,
                "counts": count_by_class(labels),
                "inference_ms": inference_ms,
            }
        except Exception as exc:
            results[model_key] = {"error": str(exc)}
    progress.progress(1.0, text="Đã hoàn tất so sánh.")
    progress.empty()
    payload = {
        "filename": uploaded.name,
        "confidence": confidence,
        "results": results,
    }
    st.session_state["compare_result"] = payload
    st.session_state.pop("last_result", None)
    display_compare_result(payload)


def display_compare_result(payload: dict[str, Any]) -> None:
    section_header(
        "Side-by-side review",
        f"So sánh trên {payload['filename']}",
        "Số detection và confidence chỉ mô tả output demo; dùng test có nhãn để kết luận chất lượng model.",
    )
    columns = st.columns(3)
    for column, model_key in zip(columns, MODEL_INFO):
        result = payload["results"].get(model_key)
        color = MODEL_COLOR[model_key]
        with column:
            st.markdown(
                f"<div class='section-eyebrow' style='color:{color};'>{escape(MODEL_INFO[model_key]['name'])}</div>",
                unsafe_allow_html=True,
            )
            if not result or "error" in result:
                st.warning(result.get("error", "Không có kết quả") if result else "Không có kết quả")
                continue
            st.image(result["result_rgb"], width="stretch")
            total = sum(result["counts"].values())
            average_confidence = float(np.mean(result["scores"])) if len(result["scores"]) else 0.0
            metric_detection, metric_latency = st.columns(2)
            metric_detection.metric("Detection", total)
            metric_latency.metric("Latency", f"{result['inference_ms']:.0f} ms")
            st.caption(f"Confidence trung bình: {average_confidence:.3f}")


def run_video_inference(uploaded, config: dict[str, Any]) -> None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
            temporary.write(uploaded.getvalue())
            temporary_path = Path(temporary.name)
        checkpoint = Path(config["resolved_weights"])
        mtime_ns = checkpoint.stat().st_mtime_ns if checkpoint.is_file() else 0
        with st.spinner(f"Đang xử lý video bằng {MODEL_INFO[config['model_key']]['name']}…"):
            detector, predict_fn, lock = cached_detector(
                config["model_key"],
                config["resolved_weights"],
                config["device"],
                mtime_ns,
            )
            output_path = run_video_demo(
                config["model_key"],
                config["resolved_weights"],
                str(temporary_path),
                output="results",
                conf=config["confidence"],
                device=config["device"],
                detector_bundle=(detector, predict_fn),
                prediction_lock=lock,
            )
        capture = cv2.VideoCapture(str(temporary_path))
        fps = float(capture.get(cv2.CAP_PROP_FPS)) if capture.isOpened() else 0.0
        frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) if capture.isOpened() else 0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) if capture.isOpened() else 0
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) if capture.isOpened() else 0
        capture.release()
        result = {
            "kind": "video",
            "filename": uploaded.name,
            "output_name": output_path.name,
            "video_bytes": output_path.read_bytes(),
            "model_name": MODEL_INFO[config["model_key"]]["name"],
            "fps": fps,
            "frames": frames,
            "size": (width, height),
            "duration": frames / fps if fps > 0 else 0.0,
        }
        st.session_state["last_result"] = result
        st.session_state.pop("compare_result", None)
        display_video_result(result)
    except Exception as exc:
        show_model_error(exc, config["model_key"])
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def display_video_result(result: dict[str, Any]) -> None:
    st.markdown(
        f"""
<div class="result-toolbar">
  <span class="result-chip"><strong>{escape(result['model_name'])}</strong></span>
  <span class="result-chip">frames <strong>{result['frames']:,}</strong></span>
  <span class="result-chip">duration <strong>{result['duration']:.1f}s</strong></span>
  <span class="result-chip">input FPS <strong>{result['fps']:.1f}</strong></span>
</div>
""",
        unsafe_allow_html=True,
    )
    video_column, info_column = st.columns([1.7, 0.8])
    with video_column:
        st.video(result["video_bytes"])
        st.download_button(
            "Tải video kết quả",
            data=result["video_bytes"],
            file_name=result["output_name"],
            mime="video/mp4",
            width="stretch",
        )
    with info_column:
        width, height = result["size"]
        st.metric("Độ phân giải", f"{width} × {height}")
        st.metric("Tổng frame", f"{result['frames']:,}")
        st.metric("Thời lượng", f"{result['duration']:.1f} giây")


def show_model_error(exc: Exception, model_key: str) -> None:
    message = str(exc)
    st.error(f"Không thể chạy {MODEL_INFO[model_key]['name']}: {message}")
    if "cuda" in message.lower():
        st.info("Thử chọn **cpu** hoặc **auto** trong sidebar.")
    if model_key == "rtdetr" and "rtdetr" in message.lower():
        st.info("Kiểm tra phiên bản Ultralytics và checkpoint RT-DETR.")


def load_metrics() -> tuple[pd.DataFrame | None, Path]:
    path = project_path("results/metrics.csv")
    if not path.is_file():
        return None, path
    try:
        metrics = pd.read_csv(path)
    except (OSError, pd.errors.ParserError):
        return None, path
    if metrics.empty:
        return None, path
    for column in ("precision", "recall", "map50", "fps"):
        if column in metrics.columns:
            metrics[column] = pd.to_numeric(metrics[column], errors="coerce")
    return metrics, path


def render_metrics_empty() -> None:
    st.markdown(
        """
<div class="workbench-empty">
  <div class="workbench-icon">∿</div>
  <div class="workbench-title">Chưa có phiên đánh giá hoàn chỉnh</div>
  <div class="workbench-copy">Train model trên split hiện tại, sau đó chạy evaluate cho từng checkpoint.</div>
</div>
""",
        unsafe_allow_html=True,
    )
    st.code(
        "python src/evaluation/evaluate.py --model yolo "
        "--weights runs/yolo/checkpoints/best.pt --data data/processed/test",
        language="bash",
    )


def render_comparison_tab() -> None:
    section_header(
        "Evaluation ledger",
        "So sánh trên cùng giao thức",
        "Chỉ ghép các run có cùng test split, confidence, thiết bị và số mẫu để tránh kết luận sai.",
    )
    metrics, metrics_path = load_metrics()
    if metrics is None:
        render_metrics_empty()
        return

    updated = datetime.fromtimestamp(metrics_path.stat().st_mtime).strftime("%d/%m/%Y · %H:%M")
    st.caption(f"Nguồn: results/metrics.csv · cập nhật {updated}")
    render_metric_cards(metrics)
    st.divider()
    render_recommendation(metrics)
    st.markdown("#### Bảng kết quả")
    display_columns = [
        column
        for column in (
            "model",
            "precision",
            "recall",
            "map50",
            "fps",
            "confidence",
            "samples",
            "device",
            "data_path",
            "weight_path",
        )
        if column in metrics.columns
    ]
    display = metrics[display_columns].copy()
    column_config: dict[str, Any] = {}
    for column in ("precision", "recall", "map50"):
        if column in display:
            column_config[column] = st.column_config.ProgressColumn(
                "mAP@50" if column == "map50" else column.title(),
                min_value=0,
                max_value=1,
                format="%.3f",
            )
    if "fps" in display:
        column_config["fps"] = st.column_config.NumberColumn("FPS", format="%.1f")
    st.dataframe(display, column_config=column_config, width="stretch", hide_index=True)
    st.download_button(
        "Xuất bảng CSV",
        data=display.to_csv(index=False).encode("utf-8"),
        file_name="model_comparison.csv",
        mime="text/csv",
    )


def best_metric_row(metrics: pd.DataFrame, column: str) -> pd.Series | None:
    if column not in metrics or "model" not in metrics or metrics[column].dropna().empty:
        return None
    return metrics.loc[metrics[column].idxmax()]


def render_metric_cards(metrics: pd.DataFrame) -> None:
    definitions = [
        ("map50", "Best mAP@50", lambda value: f"{value:.3f}"),
        ("fps", "Nhanh nhất", lambda value: f"{value:.1f} FPS"),
        ("precision", "Precision cao nhất", lambda value: f"{value:.3f}"),
        ("recall", "Recall cao nhất", lambda value: f"{value:.3f}"),
    ]
    available = [(column, label, formatter, best_metric_row(metrics, column)) for column, label, formatter in definitions]
    available = [item for item in available if item[3] is not None]
    if not available:
        return
    columns = st.columns(len(available))
    for ui_column, (metric_name, label, formatter, row) in zip(columns, available):
        ui_column.metric(
            f"{label} · {row['model']}",
            formatter(float(row[metric_name])),
        )


def render_recommendation(metrics: pd.DataFrame) -> None:
    map_row = best_metric_row(metrics, "map50")
    fps_row = best_metric_row(metrics, "fps")
    if map_row is None or fps_row is None:
        return
    balanced_name = "—"
    valid = metrics.dropna(subset=[column for column in ("map50", "fps") if column in metrics]).copy()
    if not valid.empty and {"map50", "fps", "model"} <= set(valid.columns):
        max_fps = float(valid["fps"].max())
        valid["balanced_score"] = valid["map50"] + (valid["fps"] / max_fps if max_fps > 0 else 0)
        balanced_name = str(valid.loc[valid["balanced_score"].idxmax(), "model"])
    st.info(
        f"**Theo số liệu hiện có:** độ chính xác → **{map_row['model']}** · "
        f"tốc độ → **{fps_row['model']}** · cân bằng mAP/FPS → **{balanced_name}**. "
        "Khuyến nghị chỉ có giá trị khi các run cùng giao thức."
    )


def model_color(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "yolo": "yolo",
        "faster_rcnn": "faster_rcnn",
        "faster_r_cnn": "faster_rcnn",
        "rtdetr": "rtdetr",
        "rt_detr": "rtdetr",
    }
    return MODEL_COLOR.get(aliases.get(normalized, normalized), "#0F9F8F")


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red, green, blue = (int(value[index:index + 2], 16) for index in (0, 2, 4))
    return f"rgba({red},{green},{blue},{alpha})"


def render_charts_tab() -> None:
    section_header(
        "Visual analytics",
        "Đọc trade-off giữa chất lượng và tốc độ",
        "Biểu đồ dùng trực tiếp results/metrics.csv; FPS được chuẩn hóa riêng trong radar chart.",
    )
    metrics, _ = load_metrics()
    if metrics is None or "model" not in metrics:
        render_metrics_empty()
        return
    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("Cài plotly để hiển thị biểu đồ tương tác.")
        return

    chart_columns = [column for column in ("precision", "recall", "map50", "fps") if column in metrics]
    ui_columns = st.columns(2)
    titles = {"precision": "Precision", "recall": "Recall", "map50": "mAP@50", "fps": "FPS"}
    for index, metric_name in enumerate(chart_columns):
        valid = metrics.dropna(subset=[metric_name])
        figure = go.Figure(
            go.Bar(
                x=valid["model"],
                y=valid[metric_name],
                marker_color=[model_color(str(name)) for name in valid["model"]],
                text=[f"{value:.1f}" if metric_name == "fps" else f"{value:.3f}" for value in valid[metric_name]],
                textposition="outside",
            )
        )
        figure.update_layout(
            title=dict(text=titles[metric_name], font=dict(size=14, color="#14303f", family="Inter")),
            height=300,
            margin=dict(l=8, r=8, t=44, b=8),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#f8fafb",
            font=dict(color="#6b8a97", family="Inter", size=11),
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color="#3a5561", size=11)),
            yaxis=dict(showgrid=True, gridcolor="#e2eaed", zeroline=False, tickfont=dict(color="#6b8a97")),
            bargap=0.35,
        )
        figure.update_traces(marker_line_width=0, textfont=dict(color="#14303f", size=11))
        ui_columns[index % 2].plotly_chart(figure, use_container_width=True)

    radar_metrics = [column for column in ("precision", "recall", "map50") if column in metrics]
    if not radar_metrics:
        return
    radar_data = metrics.dropna(subset=radar_metrics).copy()
    if "fps" in radar_data and not radar_data["fps"].dropna().empty:
        maximum_fps = float(radar_data["fps"].max())
        radar_data["fps_norm"] = radar_data["fps"] / maximum_fps if maximum_fps > 0 else 0
        radar_metrics.append("fps_norm")
    labels = ["mAP@50" if name == "map50" else "FPS chuẩn hóa" if name == "fps_norm" else name.title() for name in radar_metrics]
    radar = go.Figure()
    for _, row in radar_data.iterrows():
        values = [float(row[name]) for name in radar_metrics]
        color = model_color(str(row["model"]))
        radar.add_trace(
            go.Scatterpolar(
                r=values + [values[0]],
                theta=labels + [labels[0]],
                fill="toself",
                name=str(row["model"]),
                line_color=color,
                fillcolor=hex_to_rgba(color, 0.13),
            )
        )
    radar.update_layout(
        height=440,
        margin=dict(l=30, r=30, t=30, b=60),
        paper_bgcolor="#ffffff",
        font=dict(color="#3a5561", family="Inter", size=11),
        polar=dict(
            bgcolor="#f8fafb",
            radialaxis=dict(
                range=[0, 1],
                gridcolor="#e2eaed",
                tickfont=dict(color="#6b8a97", size=9),
            ),
            angularaxis=dict(
                gridcolor="#e2eaed",
                tickfont=dict(color="#3a5561", size=11, family="Inter"),
            ),
        ),
        legend=dict(
            orientation="h",
            y=-0.12,
            font=dict(color="#3a5561", size=11),
        ),
    )
    st.plotly_chart(radar, use_container_width=True)


def render_help_tab() -> None:
    section_header(
        "Operator guide",
        "Từ checkpoint đến báo cáo",
        "Luồng thao tác ngắn gọn để demo không bị lẫn với quy trình đánh giá khoa học.",
    )
    setup, operate, evaluate = st.columns(3)
    setup.markdown("#### 01 · Chuẩn bị")
    setup.write("Train model và kiểm tra checkpoint an toàn trong `runs/<model>/checkpoints/`.")
    operate.markdown("#### 02 · Phân tích")
    operate.write("Chọn model, confidence, thiết bị rồi tải ảnh hoặc video vào workbench.")
    evaluate.markdown("#### 03 · Đánh giá")
    evaluate.write("Chạy test có nhãn cho từng model và chỉ so sánh các run cùng giao thức.")

    st.divider()
    left, right = st.columns([1.1, 0.9])
    with left:
        st.markdown("### Các lớp RDD2022")
        class_descriptions = {
            "D00": "Vết nứt chạy dọc theo hướng tuyến đường.",
            "D10": "Vết nứt ngang, gần vuông góc với hướng tuyến.",
            "D20": "Mạng nứt dạng lưới, thường phản ánh suy yếu kết cấu.",
            "D40": "Vùng bong vỡ tạo hố trên bề mặt đường.",
        }
        for class_name in CLASS_NAMES:
            color = CLASS_COLOR_HEX[class_name]
            st.markdown(
                f"""
<div class="checkpoint-card" style="min-height:auto;margin-bottom:.65rem;border-left:3px solid {color};">
  <div class="checkpoint-name" style="color:{color};">{class_name} · {escape(CLASS_FULL_NAME[class_name])}</div>
  <div class="checkpoint-copy">{escape(class_descriptions[class_name])}</div>
</div>
""",
                unsafe_allow_html=True,
            )
    with right:
        st.markdown("### Lệnh thường dùng")
        st.code("python src/models/yolo/train.py --config configs/yolo.yaml", language="bash")
        st.code(
            "python src/evaluation/evaluate.py --model yolo "
            "--weights runs/yolo/checkpoints/best.pt --data data/processed/test",
            language="bash",
        )
        st.markdown("### Nguyên tắc")
        st.info(
            "Demo ảnh/video không thay thế đánh giá. Precision, Recall và mAP@50 phải được tính trên test có nhãn."
        )


def main() -> None:
    st.set_page_config(
        page_title="Road Damage Lab",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "About": "Road Damage Detection · YOLO · Faster R-CNN · RT-DETR",
        },
    )
    inject_css()
    config = render_sidebar()
    render_app_header()
    demo_tab, comparison_tab, charts_tab, help_tab = st.tabs(
        ["Workbench", "Model benchmark", "Visual analytics", "Hướng dẫn"]
    )
    with demo_tab:
        render_demo_tab(config)
    with comparison_tab:
        render_comparison_tab()
    with charts_tab:
        render_charts_tab()
    with help_tab:
        render_help_tab()


if __name__ == "__main__":
    main()
