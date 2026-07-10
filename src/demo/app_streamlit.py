"""Road Damage Detection — Streamlit Dashboard.

Chạy:
    streamlit run src/demo/app_streamlit.py --server.address 127.0.0.1
"""
from __future__ import annotations

import io
import os
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from src.demo.demo_image import load_detector
from src.demo.demo_video import run_video_demo
from src.utils.common import (
    CLASS_NAMES,
    count_by_class,
    project_path,
    resolve_inference_weights,
)
from src.utils.visualization import draw_detections

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MODEL_OPTIONS: dict[str, str] = {
    "🔵 YOLO": "yolo",
    "🟢 Faster R-CNN": "faster_rcnn",
    "🟣 RT-DETR": "rtdetr",
}

MODEL_BADGE: dict[str, str] = {
    "yolo": "Nhanh nhất · 6 MB",
    "faster_rcnn": "Chính xác nhất · 160 MB",
    "rtdetr": "Cân bằng · 120 MB",
}

MODEL_COLOR: dict[str, str] = {
    "yolo": "#4A90E2",
    "faster_rcnn": "#7ED321",
    "rtdetr": "#BD10E0",
}

CLASS_FULL_NAME: dict[str, str] = {
    "D00": "D00 — Nứt dọc (Longitudinal crack)",
    "D10": "D10 — Nứt ngang (Transverse crack)",
    "D20": "D20 — Nứt lưới (Alligator crack)",
    "D40": "D40 — Ổ gà (Pothole)",
}

CLASS_COLOR_HEX: dict[str, str] = {
    "D00": "#FF6B35",
    "D10": "#FFD60A",
    "D20": "#7B68EE",
    "D40": "#FF4757",
}

WEIGHT_DEFAULTS: dict[str, tuple[str, str]] = {
    "yolo":        ("runs/yolo/best.pt",         "runs/yolo",         ".pt"),
    "faster_rcnn": ("runs/faster_rcnn/best.pth", "runs/faster_rcnn",  ".pth"),
    "rtdetr":      ("runs/rtdetr/best.pt",        "runs/rtdetr",       ".pt"),
}

IMAGE_TYPES = ["jpg", "jpeg", "png", "bmp", "webp"]
VIDEO_TYPES = ["mp4", "avi", "mov"]

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def default_weight(model_key: str) -> str:
    requested, model_dir, ext = WEIGHT_DEFAULTS[model_key]
    try:
        return str(resolve_inference_weights(requested, model_dir, ext))
    except FileNotFoundError:
        return requested


def weight_exists(path: str) -> bool:
    return project_path(path).is_file()


def encode_image_to_png_bytes(bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise RuntimeError("Failed to encode image")
    return buf.tobytes()


def bgr_to_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_resource(show_spinner=False)
def cached_detector(model_key: str, weights: str, device: str, mtime_ns: int):
    """Cache detector; reloads automatically when the checkpoint file changes."""
    del mtime_ns  # only used as cache key
    detector, predict_fn = load_detector(model_key, weights, device=device)
    return detector, predict_fn, threading.Lock()


def run_inference(
    model_key: str, weights: str, device: str, image: np.ndarray, conf: float
) -> tuple[np.ndarray, list, list, list, float]:
    """Return (result_bgr, boxes, labels, scores, inference_ms)."""
    resolved = project_path(weights)
    mtime_ns = resolved.stat().st_mtime_ns if resolved.is_file() else 0
    detector, predict_fn, lock = cached_detector(model_key, weights, device, mtime_ns)
    t0 = time.perf_counter()
    with lock:
        boxes, labels, scores = predict_fn(detector, image, conf=conf, device=device)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    result = draw_detections(image, boxes, labels, scores)
    return result, boxes, labels, scores, elapsed_ms


def build_detection_df(boxes, labels, scores) -> pd.DataFrame:
    rows = []
    for i, (box, lbl, sc) in enumerate(zip(boxes, labels, scores), start=1):
        class_name = CLASS_NAMES[int(lbl)] if 0 <= int(lbl) < len(CLASS_NAMES) else str(lbl)
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        rows.append({
            "#": i,
            "Lớp": class_name,
            "Tên đầy đủ": CLASS_FULL_NAME.get(class_name, class_name),
            "Confidence": round(float(sc), 4),
            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
            "w": x2 - x1, "h": y2 - y1,
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# CSS injection
# ─────────────────────────────────────────────────────────────────────────────

def inject_css() -> None:
    st.markdown(
        """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* ── Sidebar ─────────────────────────────── */
[data-testid="stSidebar"] {
    background-color: #1A1D27 !important;
    border-right: 1px solid #2D3142;
}
[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

/* ── Primary button ──────────────────────── */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B35 0%, #FF4757 100%) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px;
    box-shadow: 0 4px 16px rgba(255,107,53,0.35) !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 24px rgba(255,107,53,0.55) !important;
}
.stButton > button[kind="primary"]:active { transform: translateY(0) !important; }

/* ── Secondary button ────────────────────── */
.stButton > button[kind="secondary"] {
    background: transparent !important;
    border: 1.5px solid #FF6B35 !important;
    border-radius: 10px !important;
    color: #FF6B35 !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button[kind="secondary"]:hover {
    background: rgba(255,107,53,0.08) !important;
    transform: translateY(-1px) !important;
}

/* ── Metric cards ────────────────────────── */
[data-testid="stMetric"] {
    background: #1A1D27 !important;
    border: 1px solid #2D3142 !important;
    border-radius: 14px !important;
    padding: 1rem 1.25rem !important;
    transition: border-color 0.25s, box-shadow 0.25s;
}
[data-testid="stMetric"]:hover {
    border-color: #FF6B35 !important;
    box-shadow: 0 4px 18px rgba(255,107,53,0.18) !important;
}
[data-testid="stMetricValue"] { font-size: 1.6rem !important; font-weight: 700 !important; }
[data-testid="stMetricDelta"] { font-size: 0.78rem !important; }

/* ── Tabs ────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
    background: #1A1D27;
    border-radius: 12px;
    padding: 5px;
    border: 1px solid #2D3142;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    padding: 8px 18px !important;
    font-weight: 500 !important;
    transition: background 0.18s !important;
}
.stTabs [aria-selected="true"] {
    background: #FF6B35 !important;
    color: white !important;
}

/* ── Alerts ──────────────────────────────── */
[data-testid="stAlert"] { border-radius: 10px !important; }

/* ── Images ──────────────────────────────── */
[data-testid="stImage"] img {
    border-radius: 12px !important;
    box-shadow: 0 6px 28px rgba(0,0,0,0.45) !important;
}

/* ── Dataframe ───────────────────────────── */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden; }

/* ── Divider ─────────────────────────────── */
hr { border-color: #2D3142 !important; }

/* ── Slider track ────────────────────────── */
[data-testid="stSlider"] > div > div > div > div {
    background: linear-gradient(90deg, #FF6B35, #FFD60A) !important;
}

/* ── File uploader ───────────────────────── */
[data-testid="stFileUploader"] > section {
    border: 2px dashed #2D3142 !important;
    border-radius: 12px !important;
    background: #1A1D27 !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"] > section:hover { border-color: #FF6B35 !important; }

/* ── Spinner text ────────────────────────── */
.stSpinner > div { border-top-color: #FF6B35 !important; }

/* ── Scrollbar ───────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #1A1D27; }
::-webkit-scrollbar-thumb { background: #2D3142; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #FF6B35; }
</style>
""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

def render_sidebar() -> dict:
    """Render the sidebar and return a config dict with all user selections."""
    with st.sidebar:
        # ── Logo / Title ────────────────────────────────────────────────────
        st.markdown(
            """
<div style="text-align:center; padding:0.5rem 0 1rem 0;">
  <div style="font-size:2.4rem;">🛣️</div>
  <div style="font-size:1.05rem; font-weight:700; color:#FF6B35; letter-spacing:0.3px;">
    Road Damage Detection
  </div>
  <div style="font-size:0.72rem; color:#8899AA; margin-top:2px;">
    Comparison Dashboard · v2.0
  </div>
</div>
""",
            unsafe_allow_html=True,
        )
        st.divider()

        # ── Model selection ─────────────────────────────────────────────────
        st.markdown("#### ⚙️ Cài đặt mô hình")
        model_label = st.radio(
            "Chọn mô hình",
            list(MODEL_OPTIONS.keys()),
            index=0,
            help="YOLO: nhanh nhất | Faster R-CNN: chính xác nhất | RT-DETR: cân bằng",
        )
        model_key = MODEL_OPTIONS[model_label]

        badge = MODEL_BADGE[model_key]
        color = MODEL_COLOR[model_key]
        st.markdown(
            f'<div style="font-size:0.72rem; color:{color}; margin-top:-8px; margin-bottom:8px;">'
            f"📌 {badge}</div>",
            unsafe_allow_html=True,
        )

        # ── Weight path ─────────────────────────────────────────────────────
        weights = st.text_input(
            "Đường dẫn weight",
            value=default_weight(model_key),
            key=f"weights_{model_key}",
            help="Tuyệt đối hoặc tương đối so với thư mục gốc dự án",
        )
        ok = weight_exists(weights)
        if ok:
            st.markdown(
                '<span style="color:#00D4AA; font-size:0.8rem;">✅ Weight tồn tại</span>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<span style="color:#FF4757; font-size:0.8rem;">❌ Không tìm thấy file weight</span>',
                unsafe_allow_html=True,
            )

        # ── Confidence ──────────────────────────────────────────────────────
        conf = st.slider(
            "Ngưỡng Confidence",
            min_value=0.05,
            max_value=0.95,
            value=0.25,
            step=0.05,
            format="%.2f",
            help="Thấp → phát hiện nhiều, có thể nhiễu | Cao → chỉ kết quả chắc chắn",
        )

        # ── Device ──────────────────────────────────────────────────────────
        device = st.selectbox(
            "Thiết bị",
            ["auto", "cpu", "cuda:0", "cuda:1"],
            help="auto = tự chọn GPU nếu có, ngược lại CPU",
        )
        st.divider()

        # ── Upload ──────────────────────────────────────────────────────────
        st.markdown("#### 📂 Nguồn đầu vào")
        up_tab_img, up_tab_vid = st.tabs(["📸 Ảnh", "🎬 Video"])
        with up_tab_img:
            uploaded_img = st.file_uploader(
                "Tải ảnh lên",
                type=IMAGE_TYPES,
                label_visibility="collapsed",
                key="upload_img",
            )
        with up_tab_vid:
            uploaded_vid = st.file_uploader(
                "Tải video lên",
                type=VIDEO_TYPES,
                label_visibility="collapsed",
                key="upload_vid",
            )

        uploaded = uploaded_img or uploaded_vid
        is_image = uploaded_img is not None
        st.divider()

        # ── Action buttons ──────────────────────────────────────────────────
        run_single = st.button(
            "▶ RUN DETECTION",
            type="primary",
            use_container_width=True,
            disabled=(uploaded is None or not ok),
            key="btn_run_single",
        )
        run_all = st.button(
            "🆚 So sánh cả 3 mô hình",
            type="secondary",
            use_container_width=True,
            disabled=(uploaded_img is None or not ok),
            key="btn_run_all",
            help="Chỉ hoạt động với ảnh. Chạy cả YOLO, Faster R-CNN và RT-DETR.",
        )
        st.divider()

        # ── Quick stats (from session state) ────────────────────────────────
        result = st.session_state.get("last_result")
        if result and result.get("is_image"):
            st.markdown("#### 📋 Kết quả nhanh")
            counts = result.get("counts", {})
            total = sum(counts.values())
            st.markdown(f"**Tổng phát hiện:** `{total}`")
            for cls, cnt in counts.items():
                frac = cnt / total if total > 0 else 0
                color_hex = CLASS_COLOR_HEX.get(cls, "#AAAAAA")
                bar_w = int(frac * 80)
                st.markdown(
                    f'<div style="display:flex; align-items:center; gap:8px; margin-bottom:4px;">'
                    f'  <span style="width:32px; font-size:0.8rem; color:{color_hex}; font-weight:600;">{cls}</span>'
                    f'  <div style="height:8px; width:{bar_w}px; background:{color_hex}; border-radius:4px;"></div>'
                    f'  <span style="font-size:0.8rem; color:#CCCCCC;">{cnt}</span>'
                    f"</div>",
                    unsafe_allow_html=True,
                )
            inf_ms = result.get("inference_ms", 0)
            st.markdown(
                f'<div style="font-size:0.78rem; color:#8899AA; margin-top:6px;">'
                f"⏱ Inference: {inf_ms:.0f} ms</div>",
                unsafe_allow_html=True,
            )

    return {
        "model_label": model_label,
        "model_key": model_key,
        "weights": weights,
        "weight_ok": ok,
        "conf": conf,
        "device": device,
        "uploaded": uploaded,
        "uploaded_img": uploaded_img,
        "uploaded_vid": uploaded_vid,
        "is_image": is_image,
        "run_single": run_single,
        "run_all": run_all,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Tab 1 — Demo Ảnh / Video
# ─────────────────────────────────────────────────────────────────────────────

def render_demo_tab(cfg: dict) -> None:
    uploaded     = cfg["uploaded"]
    uploaded_img = cfg["uploaded_img"]
    uploaded_vid = cfg["uploaded_vid"]
    model_key    = cfg["model_key"]
    model_label  = cfg["model_label"]
    weights      = cfg["weights"]
    conf         = cfg["conf"]
    device       = cfg["device"]
    weight_ok    = cfg["weight_ok"]
    run_single   = cfg["run_single"]
    run_all      = cfg["run_all"]

    # ── Empty state ──────────────────────────────────────────────────────────
    if uploaded is None:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(
            """
<div style="text-align:center; padding:3rem 1rem;">
  <div style="font-size:4rem; margin-bottom:1rem;">🛣️</div>
  <h2 style="color:#FAFAFA; margin-bottom:0.5rem;">Road Damage Detection</h2>
  <p style="color:#8899AA; font-size:1rem;">
    Upload ảnh hoặc video ở sidebar để bắt đầu phân tích.
  </p>
</div>
""",
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        for col, (key, label, fps_label) in zip(
            [c1, c2, c3],
            [
                ("yolo",        "🔵 YOLO",        "~35 FPS"),
                ("faster_rcnn", "🟢 Faster R-CNN", "~8 FPS"),
                ("rtdetr",      "🟣 RT-DETR",      "~22 FPS"),
            ],
        ):
            clr = MODEL_COLOR[key]
            col.markdown(
                f"""<div style="background:#1A1D27; border:1px solid #2D3142; border-radius:14px;
                               padding:1.2rem; text-align:center; transition:border-color 0.2s;">
  <div style="font-size:1.6rem; margin-bottom:0.4rem;">{label}</div>
  <div style="font-size:0.78rem; color:{clr}; font-weight:600;">{MODEL_BADGE[key]}</div>
  <div style="font-size:0.75rem; color:#8899AA; margin-top:4px;">{fps_label}</div>
</div>""",
                unsafe_allow_html=True,
            )
        return

    # ── Run compare-all (3 columns) ──────────────────────────────────────────
    if run_all and uploaded_img is not None:
        _run_compare_all(uploaded_img, conf, weights, device)
        return

    # ── Run single model ─────────────────────────────────────────────────────
    if run_single and uploaded is not None:
        if not weight_ok:
            st.error(
                f"❌ Không tìm thấy file weight: `{weights}`\n\n"
                "Hãy train mô hình trước hoặc kiểm tra lại đường dẫn."
            )
            return

        if uploaded_img is not None:
            _run_image_inference(uploaded_img, model_key, model_label, weights, conf, device)
        elif uploaded_vid is not None:
            _run_video_inference(uploaded_vid, model_key, model_label, weights, conf, device)
        return

    # ── Idle (file uploaded but not run) ─────────────────────────────────────
    if uploaded is not None:
        st.info(
            f"📁 Đã tải: **{uploaded.name}**  |  "
            f"Mô hình: **{model_label}**  |  "
            f"Conf: **{conf:.2f}**\n\n"
            "Nhấn **▶ RUN DETECTION** ở sidebar để bắt đầu."
        )

    # ── Show previous result if available ────────────────────────────────────
    prev = st.session_state.get("last_result")
    if prev and prev.get("is_image") and prev.get("result_rgb") is not None:
        _display_image_result(prev)


def _run_image_inference(uploaded_img, model_key, model_label, weights, conf, device):
    data = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("❌ Không thể đọc file ảnh. Thử lại với định dạng khác.")
        return

    with st.spinner(f"Đang tải mô hình {model_label} và chạy inference…"):
        try:
            result_bgr, boxes, labels, scores, inf_ms = run_inference(
                model_key, weights, device, image, conf
            )
        except Exception as exc:
            _show_model_error(exc, model_key)
            return

    h, w = image.shape[:2]
    counts = count_by_class(labels)
    total  = sum(counts.values())

    st.session_state["last_result"] = {
        "is_image":     True,
        "result_rgb":   bgr_to_rgb(result_bgr),
        "result_bgr":   result_bgr,
        "boxes":        boxes,
        "labels":       labels,
        "scores":       scores,
        "counts":       counts,
        "inference_ms": inf_ms,
        "img_size":     (w, h),
        "model_label":  model_label,
        "conf":         conf,
        "filename":     uploaded_img.name,
    }

    st.success(f"✅ Phát hiện **{total}** vị trí hư hỏng trong **{inf_ms:.0f} ms**")
    _display_image_result(st.session_state["last_result"])


def _display_image_result(res: dict):
    result_rgb  = res["result_rgb"]
    result_bgr  = res["result_bgr"]
    boxes       = res["boxes"]
    labels      = res["labels"]
    scores      = res["scores"]
    counts      = res["counts"]
    inf_ms      = res["inference_ms"]
    w, h        = res["img_size"]
    model_label = res["model_label"]
    conf        = res["conf"]
    filename    = res.get("filename", "image")
    total       = sum(counts.values())

    # Header strip
    st.markdown(
        f'<div style="background:#1A1D27; border:1px solid #2D3142; border-radius:10px; '
        f'padding:0.6rem 1rem; margin-bottom:0.8rem; display:flex; gap:1.5rem; '
        f'align-items:center; flex-wrap:wrap;">'
        f'<span style="font-weight:600;">{model_label}</span>'
        f'<span style="color:#8899AA;">conf = {conf:.2f}</span>'
        f'<span style="color:#00D4AA;">⏱ {inf_ms:.0f} ms</span>'
        f'<span style="color:#FFD60A;">🎯 {total} detections</span>'
        f'<span style="color:#8899AA; font-size:0.82rem;">📐 {w}×{h}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )

    col_img, col_stats = st.columns([0.65, 0.35])

    with col_img:
        st.image(result_rgb, use_container_width=True, caption=f"Kết quả: {filename}")
        # Download button
        png_bytes = encode_image_to_png_bytes(result_bgr)
        st.download_button(
            label="📥 Tải ảnh kết quả",
            data=png_bytes,
            file_name=f"result_{filename}.png",
            mime="image/png",
            use_container_width=True,
        )

    with col_stats:
        st.markdown("**📊 Phân bổ theo lớp**")
        max_cnt = max(counts.values(), default=1)
        for cls, cnt in counts.items():
            frac = cnt / max_cnt if max_cnt > 0 else 0
            bar_w = max(int(frac * 100), 1) if cnt > 0 else 0
            clr   = CLASS_COLOR_HEX.get(cls, "#AAAAAA")
            full  = CLASS_FULL_NAME.get(cls, cls)
            st.markdown(
                f'<div style="margin-bottom:10px;">'
                f'  <div style="display:flex; justify-content:space-between; '
                f'              font-size:0.82rem; margin-bottom:3px;">'
                f'    <span style="color:{clr}; font-weight:600;">{full}</span>'
                f'    <span style="color:#FAFAFA; font-weight:700;">{cnt}</span>'
                f"  </div>"
                f'  <div style="height:8px; background:#2D3142; border-radius:4px; overflow:hidden;">'
                f'    <div style="height:100%; width:{bar_w}%; background:{clr}; '
                f'                border-radius:4px; transition:width 0.4s;"></div>'
                f"  </div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        st.divider()

        # Metrics
        avg_conf = float(np.mean(scores)) if len(scores) > 0 else 0.0
        max_conf = float(np.max(scores))  if len(scores) > 0 else 0.0
        m1, m2 = st.columns(2)
        m1.metric("Conf TB", f"{avg_conf:.2f}")
        m2.metric("Conf max", f"{max_conf:.2f}")

    # ── Detection detail table ───────────────────────────────────────────────
    if len(boxes) > 0:
        with st.expander(f"📋 Chi tiết {total} bounding box", expanded=False):
            df = build_detection_df(boxes, labels, scores)
            st.dataframe(
                df,
                column_config={
                    "Confidence": st.column_config.ProgressColumn(
                        "Confidence", min_value=0, max_value=1, format="%.3f"
                    ),
                },
                use_container_width=True,
                hide_index=True,
            )


def _run_compare_all(uploaded_img, conf, any_weights, device):
    """Run all 3 models on the same image and show side-by-side."""
    data  = np.frombuffer(uploaded_img.read(), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        st.error("❌ Không thể đọc file ảnh.")
        return

    st.markdown(
        f'<h3 style="color:#FF6B35; margin-bottom:0.8rem;">🆚 So sánh cả 3 mô hình — {uploaded_img.name}</h3>',
        unsafe_allow_html=True,
    )

    results: dict[str, dict] = {}
    col1, col2, col3 = st.columns(3)
    cols   = [col1, col2, col3]
    models = list(MODEL_OPTIONS.items())

    for col, (label, key) in zip(cols, models):
        w_path = default_weight(key)
        clr    = MODEL_COLOR[key]
        with col:
            st.markdown(
                f'<div style="background:{clr}22; border:1px solid {clr}55; '
                f'border-radius:10px; padding:0.5rem 0.75rem; text-align:center; '
                f'font-weight:600; margin-bottom:0.6rem; color:{clr};">{label}</div>',
                unsafe_allow_html=True,
            )
            if not weight_exists(w_path):
                st.warning(f"Weight không tìm thấy:\n`{w_path}`")
                results[key] = None
                continue

            with st.spinner(f"Đang chạy {label}…"):
                try:
                    res_bgr, boxes, lbls, scores, inf_ms = run_inference(
                        key, w_path, device, image, conf
                    )
                    results[key] = {
                        "result_rgb": bgr_to_rgb(res_bgr),
                        "result_bgr": res_bgr,
                        "boxes": boxes, "labels": lbls, "scores": scores,
                        "inf_ms": inf_ms,
                        "counts": count_by_class(lbls),
                        "total": sum(count_by_class(lbls).values()),
                    }
                except Exception as exc:
                    st.error(str(exc))
                    results[key] = None
                    continue

            r = results[key]
            st.image(r["result_rgb"], use_container_width=True)

            avg_c = float(np.mean(r["scores"])) if r["scores"] else 0.0
            st.markdown(
                f'<div style="font-size:0.8rem; color:#CCCCCC; margin-top:4px;">'
                f'Detections: <b>{r["total"]}</b> &nbsp;·&nbsp; '
                f'Conf avg: <b>{avg_c:.2f}</b> &nbsp;·&nbsp; '
                f'⏱ <b>{r["inf_ms"]:.0f} ms</b>'
                f"</div>",
                unsafe_allow_html=True,
            )

    # Legend
    st.divider()
    legend_parts = [
        f'<span style="background:{v}33; border:1px solid {v}; border-radius:5px; '
        f'padding:3px 10px; font-size:0.8rem; color:{v}; font-weight:600;">'
        f'■ {k} {CLASS_FULL_NAME.get(k, k).split("—")[1].strip()}</span>'
        for k, v in CLASS_COLOR_HEX.items()
    ]
    st.markdown(
        '<div style="display:flex; gap:10px; flex-wrap:wrap; margin-top:0.5rem;">'
        + " ".join(legend_parts)
        + "</div>",
        unsafe_allow_html=True,
    )


def _run_video_inference(uploaded_vid, model_key, model_label, weights, conf, device):
    suffix = Path(uploaded_vid.name).suffix
    tmp_path = None
    out_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_vid.read())
            tmp_path = tmp.name

        resolved = project_path(weights)
        mtime_ns = resolved.stat().st_mtime_ns if resolved.is_file() else 0

        with st.spinner(f"Đang xử lý video với {model_label}… (có thể mất vài phút)"):
            detector, predict_fn, lock = cached_detector(model_key, weights, device, mtime_ns)
            out_path = run_video_demo(
                model_key, weights, tmp_path,
                output="results", conf=conf, device=device,
                detector_bundle=(detector, predict_fn),
                prediction_lock=lock,
            )

        st.success(f"✅ Video đã xử lý xong: `{out_path.name}`")

        col_vid, col_info = st.columns([0.65, 0.35])
        with col_vid:
            st.video(out_path.read_bytes())
            st.download_button(
                "📥 Tải video kết quả",
                data=out_path.read_bytes(),
                file_name=out_path.name,
                mime="video/mp4",
                use_container_width=True,
            )
        with col_info:
            st.markdown("**🎬 Thông tin video**")
            cap = cv2.VideoCapture(tmp_path)
            if cap.isOpened():
                fps_in = cap.get(cv2.CAP_PROP_FPS)
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                vw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                vh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                dur_s = frames / fps_in if fps_in > 0 else 0
                cap.release()
                st.metric("Độ phân giải", f"{vw}×{vh}")
                st.metric("Tổng frame", str(frames))
                st.metric("Thời lượng", f"{dur_s:.1f}s")
                st.metric("FPS gốc", f"{fps_in:.1f}")

    except Exception as exc:
        _show_model_error(exc, model_key)
    finally:
        if tmp_path:
            try: os.unlink(tmp_path)
            except OSError: pass


def _show_model_error(exc: Exception, model_key: str) -> None:
    msg = str(exc)
    st.error(f"❌ Lỗi khi chạy mô hình: {msg}")
    if model_key == "rtdetr" and "RTDETR" in msg:
        st.info("💡 Thử: `pip install -U ultralytics` để cập nhật RT-DETR support.")
    if "CUDA" in msg.upper() or "cuda" in msg:
        st.info("💡 GPU không có sẵn. Thử chọn **cpu** ở phần Thiết bị trong sidebar.")


# ─────────────────────────────────────────────────────────────────────────────
# Tab 2 — So Sánh Mô Hình
# ─────────────────────────────────────────────────────────────────────────────

def render_comparison_tab() -> None:
    metrics_path = project_path("results/metrics.csv")

    if not metrics_path.exists():
        st.info(
            "📭 Chưa có dữ liệu so sánh.\n\n"
            "Chạy lệnh evaluate để tạo `results/metrics.csv`:\n"
            "```bash\n"
            "python src/evaluation/evaluate.py --model yolo --weights runs/yolo/best.pt "
            "--data data/processed/test\n"
            "```"
        )
        return

    try:
        metrics = pd.read_csv(metrics_path)
    except Exception as e:
        st.error(f"Không thể đọc metrics.csv: {e}")
        return

    if metrics.empty or len(metrics) == 0:
        st.info(
            "📭 `results/metrics.csv` hiện đang rỗng.\n\n"
            "Chạy evaluate.py cho từng mô hình để điền dữ liệu."
        )
        return

    # ── Last updated ─────────────────────────────────────────────────────────
    mtime = metrics_path.stat().st_mtime
    from datetime import datetime
    updated = datetime.fromtimestamp(mtime).strftime("%d/%m/%Y %H:%M")
    st.markdown(
        f'<div style="color:#8899AA; font-size:0.8rem; margin-bottom:0.5rem;">'
        f"📄 Nguồn: <code>results/metrics.csv</code> &nbsp;·&nbsp; Cập nhật: {updated}"
        f"</div>",
        unsafe_allow_html=True,
    )

    # ── Metric cards ──────────────────────────────────────────────────────────
    num_cols = ["precision", "recall", "map50", "fps"]
    existing = [c for c in num_cols if c in metrics.columns]

    if existing:
        _render_metric_cards(metrics, existing)

    st.divider()

    # ── Recommendation ───────────────────────────────────────────────────────
    if "fps" in metrics.columns and "map50" in metrics.columns and "model" in metrics.columns:
        _render_recommendation(metrics)
        st.divider()

    # ── Filter controls ───────────────────────────────────────────────────────
    st.markdown("#### 📊 Bảng so sánh chi tiết")
    display_cols = [c for c in ["model", "precision", "recall", "map50", "fps",
                                 "confidence", "samples", "device", "data_path", "weight_path"]
                    if c in metrics.columns]
    display_df = metrics[display_cols].copy()

    col_cfg = {}
    for c in ["precision", "recall", "map50"]:
        if c in display_df.columns:
            col_cfg[c] = st.column_config.ProgressColumn(
                c.upper().replace("MAP50", "mAP@50"),
                min_value=0, max_value=1, format="%.3f",
            )
    if "fps" in display_df.columns:
        col_cfg["fps"] = st.column_config.NumberColumn("FPS", format="%.1f ⚡")

    st.dataframe(display_df, column_config=col_cfg, use_container_width=True, hide_index=True)

    # Export
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Xuất CSV",
        data=csv_bytes,
        file_name="comparison_export.csv",
        mime="text/csv",
    )


def _render_metric_cards(metrics: pd.DataFrame, existing: list[str]) -> None:
    cards = []
    if "map50" in existing and "model" in metrics.columns:
        best_row = metrics.loc[metrics["map50"].idxmax()]
        cards.append(("🎯 Best mAP@50", best_row["model"], f'{best_row["map50"]:.3f}', None))
    if "fps" in existing and "model" in metrics.columns:
        best_row = metrics.loc[metrics["fps"].idxmax()]
        cards.append(("⚡ Fastest", best_row["model"], f'{best_row["fps"]:.1f} FPS', None))
    if "precision" in existing and "model" in metrics.columns:
        best_row = metrics.loc[metrics["precision"].idxmax()]
        cards.append(("🏆 Best Precision", best_row["model"], f'{best_row["precision"]:.3f}', None))
    if "recall" in existing and "model" in metrics.columns:
        best_row = metrics.loc[metrics["recall"].idxmax()]
        cards.append(("🔄 Best Recall", best_row["model"], f'{best_row["recall"]:.3f}', None))

    if not cards:
        return

    cols = st.columns(len(cards))
    for col, (title, model, value, delta) in zip(cols, cards):
        col.metric(label=f"{title} · {model}", value=value, delta=delta)


def _render_recommendation(metrics: pd.DataFrame) -> None:
    fastest  = metrics.loc[metrics["fps"].idxmax(), "model"]    if "fps"       in metrics.columns else "—"
    accurate = metrics.loc[metrics["map50"].idxmax(), "model"]  if "map50"     in metrics.columns else "—"
    balanced = metrics.loc[(metrics["map50"] + metrics["fps"] / metrics["fps"].max()).idxmax(), "model"] \
               if {"map50","fps"} <= set(metrics.columns) else "—"

    st.markdown(
        f"""<div style="background:#1A1D27; border:1px solid #2D3142; border-radius:12px;
                        padding:1rem 1.25rem;">
  <div style="font-weight:600; color:#FF6B35; margin-bottom:0.6rem;">💡 Khuyến nghị</div>
  <div style="font-size:0.87rem; line-height:1.8;">
    ✅ <b>Real-time / Edge device</b> → <span style="color:#4A90E2;">{fastest}</span>
       (tốc độ cao nhất)<br>
    ✅ <b>Cần độ chính xác cao</b> → <span style="color:#7ED321;">{accurate}</span>
       (mAP@50 cao nhất)<br>
    ✅ <b>Cân bằng tốc độ &amp; chính xác</b> → <span style="color:#BD10E0;">{balanced}</span>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Tab 3 — Biểu Đồ
# ─────────────────────────────────────────────────────────────────────────────

def render_charts_tab() -> None:
    metrics_path = project_path("results/metrics.csv")
    figures_dir  = project_path("results/figures")

    metrics = None
    if metrics_path.exists():
        try:
            df = pd.read_csv(metrics_path)
            if not df.empty:
                metrics = df
        except Exception:
            pass

    # ── Plotly charts (interactive) ───────────────────────────────────────────
    if metrics is not None:
        try:
            import plotly.graph_objects as go
            import plotly.express as px

            PLOTLY_COLORS = {
                "yolo":        "#4A90E2",
                "faster_rcnn": "#7ED321",
                "rtdetr":      "#BD10E0",
            }

            def _model_color(name: str) -> str:
                key = str(name).lower().replace("-", "_").replace(" ", "_")
                return PLOTLY_COLORS.get(key, "#FF6B35")

            colors = metrics["model"].apply(_model_color).tolist() if "model" in metrics.columns else None
            dark_layout = dict(
                plot_bgcolor="#1A1D27",
                paper_bgcolor="#0F1117",
                font_color="#FAFAFA",
                showlegend=False,
                margin=dict(l=10, r=10, t=40, b=10),
            )

            metric_cols = [c for c in ["precision", "recall", "map50", "fps"] if c in metrics.columns]
            if metric_cols and "model" in metrics.columns:
                c_left, c_right = st.columns(2)
                chart_pairs = list(zip(metric_cols[::2], metric_cols[1::2]))
                remainder   = [metric_cols[-1]] if len(metric_cols) % 2 == 1 else []

                col_cycle = [c_left, c_right] * 10
                for i, col_name in enumerate(metric_cols):
                    col = col_cycle[i]
                    label_map = {"precision": "Precision", "recall": "Recall",
                                 "map50": "mAP@50", "fps": "FPS ⚡"}
                    title = label_map.get(col_name, col_name)
                    fig = px.bar(
                        metrics, x="model", y=col_name,
                        title=title,
                        text_auto=".3f" if col_name != "fps" else ".1f",
                        color="model",
                        color_discrete_sequence=[_model_color(m) for m in metrics["model"]],
                    )
                    fig.update_layout(**dark_layout, title_font_size=15)
                    fig.update_traces(marker_line_width=0)
                    col.plotly_chart(fig, use_container_width=True)

                # Radar chart
                st.divider()
                st.markdown("#### 🕸️ Radar — So sánh đa chiều")
                radar_cols = [c for c in ["precision", "recall", "map50"] if c in metrics.columns]
                if "fps" in metrics.columns and len(metrics) > 0:
                    max_fps = metrics["fps"].max()
                    metrics_radar = metrics.copy()
                    metrics_radar["fps_norm"] = metrics_radar["fps"] / max_fps if max_fps > 0 else 0
                    radar_cols_full = radar_cols + ["fps_norm"]
                    cat_labels = [c.upper() for c in radar_cols] + ["FPS (norm)"]
                else:
                    metrics_radar = metrics.copy()
                    radar_cols_full = radar_cols
                    cat_labels = [c.upper() for c in radar_cols]

                fig_radar = go.Figure()
                for _, row in metrics_radar.iterrows():
                    model_name = str(row.get("model", ""))
                    clr = _model_color(model_name)
                    vals = [float(row[c]) for c in radar_cols_full]
                    fig_radar.add_trace(go.Scatterpolar(
                        r=vals + [vals[0]],
                        theta=cat_labels + [cat_labels[0]],
                        fill="toself",
                        name=model_name,
                        line_color=clr,
                        fillcolor=clr.replace("#", "rgba(") + ",0.12)",
                        opacity=0.85,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        bgcolor="#1A1D27",
                        radialaxis=dict(visible=True, range=[0, 1], color="#8899AA"),
                        angularaxis=dict(color="#FAFAFA"),
                    ),
                    showlegend=True,
                    legend=dict(font_color="#FAFAFA"),
                    paper_bgcolor="#0F1117",
                    font_color="#FAFAFA",
                    margin=dict(l=20, r=20, t=20, b=20),
                )
                st.plotly_chart(fig_radar, use_container_width=True)

        except ImportError:
            st.warning("💡 Cài `plotly` để xem biểu đồ interactive: `pip install plotly`")
            _show_static_figures(figures_dir)
        return

    # ── Fallback: static PNG ──────────────────────────────────────────────────
    _show_static_figures(figures_dir)

    if not figures_dir.exists():
        st.info(
            "📭 Chưa có biểu đồ.\n\n"
            "Chạy lệnh sau để tạo biểu đồ:\n"
            "```bash\npython src/evaluation/compare_results.py "
            "--input results/metrics.csv\n```"
        )


def _show_static_figures(figures_dir: Path) -> None:
    if not figures_dir.exists():
        return
    pngs = sorted(figures_dir.glob("*.png"))
    if not pngs:
        return
    st.markdown("#### 📷 Biểu đồ tĩnh từ `results/figures/`")
    c1, c2 = st.columns(2)
    for i, png in enumerate(pngs):
        col = c1 if i % 2 == 0 else c2
        col.image(str(png), caption=png.stem.replace("_", " ").title(),
                  use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tab 4 — Hướng Dẫn
# ─────────────────────────────────────────────────────────────────────────────

def render_help_tab() -> None:
    st.markdown("## 📖 Hướng dẫn sử dụng")

    steps = [
        (
            "Bước 1 · Chuẩn bị weight mô hình",
            "Đảm bảo đã train hoặc copy file weight vào đúng vị trí:\n\n"
            "```\nruns/yolo/best.pt\n"
            "runs/faster_rcnn/best.pth\n"
            "runs/rtdetr/best.pt\n```\n\n"
            "Hoặc nhập đường dẫn tuỳ chỉnh trong sidebar.",
        ),
        (
            "Bước 2 · Upload ảnh hoặc video",
            "Chọn tab **📸 Ảnh** hoặc **🎬 Video** trong sidebar rồi kéo thả hoặc nhấn Browse.\n\n"
            "**Ảnh hỗ trợ:** JPG, PNG, BMP, WebP\n\n"
            "**Video hỗ trợ:** MP4, AVI, MOV",
        ),
        (
            "Bước 3 · Điều chỉnh Confidence",
            "Slider **Ngưỡng Confidence** kiểm soát độ nhạy:\n\n"
            "- `0.05 – 0.20` → Phát hiện nhiều, có thể có nhiễu\n"
            "- `0.25 – 0.50` → **Cân bằng (khuyến nghị)**\n"
            "- `0.50+` → Chỉ giữ lại detection rất chắc chắn",
        ),
        (
            "Bước 4 · Chạy phát hiện",
            "- **▶ RUN DETECTION** — chạy mô hình đang chọn\n"
            "- **🆚 So sánh cả 3 mô hình** — chạy song song 3 model trên cùng ảnh "
            "(chỉ với ảnh, không hỗ trợ video)",
        ),
        (
            "Bước 5 · Xem kết quả và so sánh",
            "- **Tab Demo:** xem ảnh/video với bounding box, bảng chi tiết, tải kết quả\n"
            "- **Tab So Sánh:** bảng metrics từ `results/metrics.csv`\n"
            "- **Tab Biểu Đồ:** biểu đồ bar + radar interactive (cần `plotly`)\n\n"
            "Để cập nhật metrics, chạy:\n"
            "```bash\npython src/evaluation/evaluate.py --model yolo "
            "--weights runs/yolo/best.pt --data data/processed/test\n```",
        ),
    ]

    for title, body in steps:
        with st.expander(f"▸ {title}", expanded=False):
            st.markdown(body)

    st.divider()

    # ── Class descriptions ────────────────────────────────────────────────────
    st.markdown("### 🔍 Các lớp hư hỏng được phát hiện")
    classes = [
        ("D00", "#FF6B35", "Nứt dọc", "Longitudinal crack",
         "Vết nứt chạy dọc theo chiều mặt đường, thường do mỏi lớp nhựa hoặc lún nền."),
        ("D10", "#FFD60A", "Nứt ngang", "Transverse crack",
         "Vết nứt vuông góc với chiều đường, thường do co nhiệt độ hoặc mỏi liên kết."),
        ("D20", "#7B68EE", "Nứt lưới", "Alligator crack",
         "Mạng lưới vết nứt dày đặc giống vảy cá sấu — dấu hiệu nền đường suy yếu nghiêm trọng."),
        ("D40", "#FF4757", "Ổ gà", "Pothole",
         "Hố lõm sâu trên mặt đường do lớp nhựa bị bong vỡ, nguy hiểm cho xe cộ."),
    ]
    c1, c2 = st.columns(2)
    for i, (code, clr, vn, en, desc) in enumerate(classes):
        col = c1 if i % 2 == 0 else c2
        col.markdown(
            f"""<div style="background:#1A1D27; border-left:4px solid {clr};
                            border-radius:0 10px 10px 0; padding:0.8rem 1rem;
                            margin-bottom:0.8rem;">
  <div style="font-weight:700; color:{clr}; font-size:1rem;">{code} — {vn}</div>
  <div style="font-size:0.78rem; color:#8899AA; margin-bottom:4px;
              font-style:italic;">{en}</div>
  <div style="font-size:0.85rem; color:#CCCCCC;">{desc}</div>
</div>""",
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Links ─────────────────────────────────────────────────────────────────
    st.markdown("### 🔗 Liên kết hữu ích")
    links = [
        ("Dataset RDD2022", "https://github.com/sekilab/RoadDamageDetector"),
        ("YOLO (Ultralytics)", "https://docs.ultralytics.com"),
        ("Torchvision Faster R-CNN", "https://pytorch.org/vision/stable/models/faster_rcnn.html"),
        ("Plotly Python", "https://plotly.com/python/"),
    ]
    cols = st.columns(len(links))
    for col, (text, url) in zip(cols, links):
        col.markdown(
            f'<a href="{url}" target="_blank" style="text-decoration:none;">'
            f'<div style="background:#1A1D27; border:1px solid #2D3142; border-radius:10px; '
            f'padding:0.6rem; text-align:center; font-size:0.82rem; color:#4A90E2; '
            f'transition:border-color 0.2s;">{text} ↗</div></a>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    st.set_page_config(
        page_title="Road Damage Detection — Comparison Dashboard",
        page_icon="🛣️",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://github.com/sekilab/RoadDamageDetector",
            "About": "Road Damage Detection Comparison Dashboard v2.0\n"
                     "YOLO · Faster R-CNN · RT-DETR",
        },
    )

    inject_css()

    cfg = render_sidebar()

    tab_demo, tab_compare, tab_charts, tab_help = st.tabs([
        "🔍 Demo Ảnh / Video",
        "📊 So Sánh Mô Hình",
        "📈 Biểu Đồ Kết Quả",
        "ℹ️ Hướng Dẫn",
    ])

    with tab_demo:
        render_demo_tab(cfg)

    with tab_compare:
        render_comparison_tab()

    with tab_charts:
        render_charts_tab()

    with tab_help:
        render_help_tab()


if __name__ == "__main__":
    main()
