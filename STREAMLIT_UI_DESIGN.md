# 🛣️ Thiết Kế Giao Diện Streamlit — Road Damage Detection Comparison

> **Mục tiêu:** Giao diện trực quan, chuyên nghiệp, dễ dùng cho cả demo ảnh/video lẫn
> so sánh ba mô hình YOLO · Faster R-CNN · RT-DETR trên bài toán phát hiện hư hỏng
> mặt đường (D00 · D10 · D20 · D40).

---

## 1. Tổng quan kiến trúc trang

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  SIDEBAR (280 px)          │  MAIN CONTENT AREA (fluid)                     │
│  ─────────────────         │  ─────────────────────────                     │
│  Logo + Tiêu đề            │  Tab 1: 🔍 Demo Ảnh / Video                   │
│  ─────────────────         │  Tab 2: 📊 So Sánh Mô Hình                    │
│  ⚙️ Cài đặt chung          │  Tab 3: 📈 Biểu Đồ Kết Quả                   │
│    • Chọn mô hình          │  Tab 4: ℹ️ Hướng Dẫn Sử Dụng                 │
│    • Weight path           │                                                 │
│    • Confidence slider     │                                                 │
│    • Device (CPU/GPU)      │                                                 │
│  ─────────────────         │                                                 │
│  📂 Upload file            │                                                 │
│  ▶ Nút Run Detection       │                                                 │
│  ─────────────────         │                                                 │
│  📋 Thống kê nhanh         │                                                 │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Lý do layout này:**
- Sidebar cố định giúp user luôn thấy và điều chỉnh cài đặt mà không cuộn trang.
- 4 Tab phân tách rõ chức năng: Demo ↔ So sánh ↔ Biểu đồ ↔ Hướng dẫn.
- Main area rộng đủ để hiển thị ảnh/video kết quả to, rõ ràng.

---

## 2. Palette màu & Theme

```python
# .streamlit/config.toml
[theme]
primaryColor       = "#FF6B35"   # Cam đỏ — nổi bật, năng động
backgroundColor    = "#0F1117"   # Nền tối (dark mode)
secondaryBackgroundColor = "#1A1D27"  # Card / sidebar nền
textColor          = "#FAFAFA"   # Chữ sáng
font               = "sans serif"
```

| Màu | Hex | Dùng cho |
|---|---|---|
| Primary Orange | `#FF6B35` | CTA buttons, tiêu đề chính, highlight |
| Accent Teal | `#00D4AA` | Badge trạng thái OK, metric tốt |
| Warning Yellow | `#FFD60A` | Confidence thấp, cảnh báo |
| Danger Red | `#FF4757` | Lỗi, model chưa load |
| Surface Dark | `#1A1D27` | Card background |
| Border Gray | `#2D3142` | Đường viền card, divider |
| YOLO Blue | `#4A90E2` | Màu riêng cho YOLO |
| FRCNN Green | `#7ED321` | Màu riêng cho Faster R-CNN |
| RTDETR Purple | `#BD10E0` | Màu riêng cho RT-DETR |

**Bounding box màu theo lớp:**
```
D00 (Longitudinal crack) → đỏ cam  #FF6B35
D10 (Transverse crack)   → vàng    #FFD60A
D20 (Alligator crack)    → tím xanh #7B68EE
D40 (Pothole)            → đỏ đậm  #FF4757
```

---

## 3. SIDEBAR — Chi tiết từng thành phần

```
┌─────────────────────────────┐
│  🛣️ Road Damage Detection   │  ← st.image(logo) + st.markdown heading
│  Comparison Dashboard       │
├─────────────────────────────┤
│  ──── ⚙️ CÀI ĐẶT MÔ HÌNH ─ │
│                             │
│  Chọn mô hình               │  ← st.radio với icon (không dùng selectbox)
│  ○ 🔵 YOLO (YOLOv8n)        │    Hiển thị badge "Nhanh nhất" / "Chính xác nhất"
│  ○ 🟢 Faster R-CNN          │
│  ● 🟣 RT-DETR               │    ← selected
│                             │
│  Đường dẫn weight           │  ← st.text_input, auto-fill khi đổi model
│  [runs/rtdetr/best.pt     ] │    + icon ✅ nếu file tồn tại, ❌ nếu không
│                             │
│  Ngưỡng Confidence          │  ← st.slider 0.05–0.95, step 0.05
│  ━━━━━━━━◉━━━━━━━━━━  0.25  │    Hiển thị màu gradient: đỏ→vàng→xanh
│                             │
│  Thiết bị                   │  ← st.selectbox
│  [🖥️ CPU              ▼]   │    Options: auto | cpu | cuda:0 | cuda:1
│                             │
├─────────────────────────────┤
│  ──── 📂 NGUỒN ĐẦU VÀO ─── │
│                             │
│  [  📸 Ảnh  ] [  🎬 Video ] │  ← st.tabs nội bộ trong sidebar
│                             │
│  (Tab Ảnh)                  │
│  ┌─────────────────────┐    │  ← st.file_uploader, drag & drop zone
│  │  Kéo thả ảnh vào    │    │    type: jpg/jpeg/png/bmp/webp
│  │  hoặc Browse...     │    │
│  │  📁 Max 200MB       │    │
│  └─────────────────────┘    │
│                             │
│  (Tab Video)                │
│  ┌─────────────────────┐    │  ← type: mp4/avi/mov
│  │  Kéo thả video vào  │    │
│  │  hoặc Browse...     │    │
│  │  🎬 Max 200MB       │    │
│  └─────────────────────┘    │
│                             │
├─────────────────────────────┤
│                             │
│  [▶ RUN DETECTION       ]   │  ← st.button primary, full-width
│                             │    disabled nếu chưa upload hoặc weight lỗi
│                             │
│  [🆚 So sánh cả 3 model]    │  ← st.button secondary (chạy cả 3 model)
│                             │
├─────────────────────────────┤
│  ──── 📋 KẾT QUẢ NHANH ─── │
│                             │
│  (Hiện sau khi Run)         │
│  Tổng phát hiện: 7          │
│  D00 ▓▓▓▓░░░ 3              │  ← progress bar theo tỷ lệ
│  D10 ▓▓░░░░░ 2              │
│  D20 ▓░░░░░░ 1              │
│  D40 ▓░░░░░░ 1              │
│                             │
│  ⏱ Inference: 45ms          │  ← thời gian xử lý
│  📐 Ảnh: 1280×720           │  ← kích thước ảnh/video
│                             │
└─────────────────────────────┘
```

### 3.1 Logic sidebar

```python
# Pseudo-code cho sidebar
with st.sidebar:
    # Header
    st.markdown("## 🛣️ Road Damage Detection")
    st.caption("Comparison Dashboard v1.0")
    st.divider()

    # Model selection với radio + custom styling
    st.markdown("#### ⚙️ Cài đặt mô hình")
    model_label = st.radio(
        "Chọn mô hình",
        options=["🔵 YOLO", "🟢 Faster R-CNN", "🟣 RT-DETR"],
        help="YOLO: nhanh nhất | Faster R-CNN: chính xác nhất | RT-DETR: cân bằng"
    )

    # Weight path với validation live
    weights = st.text_input("Đường dẫn weight", value=default_weight(model_key))
    weight_ok = Path(weights).is_file()
    st.markdown("✅ Weight tồn tại" if weight_ok else "❌ Không tìm thấy file weight")

    # Confidence slider
    conf = st.slider("Ngưỡng Confidence", 0.05, 0.95, 0.25, 0.05,
                     format="%.2f")

    # Device
    device = st.selectbox("Thiết bị", ["auto", "cpu", "cuda:0"])
    st.divider()

    # Upload tabs
    tab_img, tab_vid = st.tabs(["📸 Ảnh", "🎬 Video"])
    with tab_img:
        uploaded_img = st.file_uploader("", type=["jpg","jpeg","png","bmp","webp"],
                                         label_visibility="collapsed")
    with tab_vid:
        uploaded_vid = st.file_uploader("", type=["mp4","avi","mov"],
                                         label_visibility="collapsed")
    st.divider()

    run_single = st.button("▶ RUN DETECTION", type="primary", use_container_width=True,
                            disabled=(uploaded_img is None and uploaded_vid is None) or not weight_ok)
    run_all    = st.button("🆚 So sánh cả 3 mô hình", use_container_width=True,
                            disabled=uploaded_img is None)
```

---

## 4. TAB 1 — 🔍 Demo Ảnh / Video

### 4.1 Trạng thái rỗng (chưa upload)

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│                    🛣️ Road Damage Detection                          │
│                                                                      │
│         Upload ảnh hoặc video ở sidebar để bắt đầu phân tích.       │
│                                                                      │
│         ┌──────────────┐  ┌──────────────┐  ┌──────────────┐        │
│         │  🔵 YOLO      │  │ 🟢 Faster R  │  │ 🟣 RT-DETR  │        │
│         │  One-stage    │  │ Two-stage    │  │ Transformer  │        │
│         │  ~35 FPS      │  │  ~8 FPS      │  │  ~22 FPS     │        │
│         └──────────────┘  └──────────────┘  └──────────────┘        │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Sau khi chạy inference — Ảnh đơn

```
┌──────────────────────────────────────────────────────────────────────┐
│  🟣 RT-DETR  │  conf=0.25  │  ⏱ 45ms  │  7 detections              │
├───────────────────────────────────────┬──────────────────────────────┤
│                                       │  📊 KẾT QUẢ PHÂN TÍCH       │
│   [ẢNH KẾT QUẢ VỚI BOUNDING BOXES]   │  ──────────────────────      │
│                                       │                              │
│   ██████████████████████              │  Tổng phát hiện: 7          │
│   █  D00 0.87  ██████  █              │                              │
│   ██████████████████████              │  Loại hư hỏng    Số lượng   │
│        ████████████                   │  ──────────────────────      │
│        █ D40  █                       │  D00 Nứt dọc        3  ████ │
│        █ 0.91 █                       │  D10 Nứt ngang      2  ███  │
│        ████████████                   │  D20 Nứt lưới       1  ██   │
│                                       │  D40 Ổ gà           1  ██   │
│   ███████████████████                 │                              │
│   █ D10  0.79 ███████                 │  ──────────────────────      │
│   ███████████████████                 │  📐 Kích thước ảnh           │
│                                       │  1280 × 720 px               │
│                                       │                              │
│                                       │  🎯 Confidence cao nhất      │
│                                       │  D40: 0.91                   │
│                                       │                              │
│                                       │  📥 [Tải ảnh kết quả]        │
├───────────────────────────────────────┴──────────────────────────────┤
│  ──── CHI TIẾT TỪNG BOUNDING BOX ────────────────────────────────── │
│                                                                      │
│  #  │ Lớp │ Tên đầy đủ          │ Confidence │ Tọa độ (x,y,w,h)    │
│  1  │ D40  │ Pothole             │    0.91    │ 320,280,150,120      │
│  2  │ D00  │ Longitudinal crack  │    0.87    │  45,100,400,  35     │
│  3  │ D10  │ Transverse crack    │    0.79    │  60,480,560,  28     │
│  4  │ D00  │ Longitudinal crack  │    0.76    │  10,210,380,  22     │
│  5  │ D00  │ Longitudinal crack  │    0.65    │  30,330,350,  18     │
│  6  │ D10  │ Transverse crack    │    0.61    │  80,560,500,  30     │
│  7  │ D20  │ Alligator crack     │    0.55    │ 200,150,200, 180     │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.3 Sau khi chạy inference — Video

```
┌──────────────────────────────────────────────────────────────────────┐
│  🟣 RT-DETR  │  conf=0.25  │  Video: road.mp4  │  720p               │
├────────────────────────────────────┬─────────────────────────────────┤
│                                    │  📊 THỐNG KÊ VIDEO               │
│  [VIDEO PLAYER — kết quả output]   │  ─────────────────────────────  │
│  ▶──────────────────────  0:12    │                                  │
│  FPS hiện tại: 22 fps              │  Tổng frame xử lý: 360          │
│                                    │  Thời lượng: 00:12              │
│                                    │  FPS trung bình: 22.3           │
│                                    │  FPS min/max: 18 / 26           │
│                                    │                                  │
│                                    │  Tổng detection: 2,847          │
│                                    │  D00: 1,245  ████████           │
│                                    │  D10:   892  ██████             │
│                                    │  D20:   410  ████               │
│                                    │  D40:   300  ███                │
│                                    │                                  │
│                                    │  📥 [Tải video kết quả]          │
└────────────────────────────────────┴─────────────────────────────────┘
```

### 4.4 Chế độ So sánh cả 3 mô hình (nút "So sánh cả 3")

```
┌──────────────────────────────────────────────────────────────────────┐
│  🆚 SO SÁNH 3 MÔ HÌNH — road_test.jpg                               │
├──────────────────┬────────────────────┬──────────────────────────────┤
│  🔵 YOLO         │  🟢 Faster R-CNN   │  🟣 RT-DETR                  │
│  ─────────────── │  ───────────────── │  ───────────────────         │
│  [Ảnh + bbox]    │  [Ảnh + bbox]      │  [Ảnh + bbox]                │
│                  │                    │                               │
│  Detections: 5   │  Detections: 8     │  Detections: 7               │
│  Conf avg: 0.72  │  Conf avg: 0.81    │  Conf avg: 0.76              │
│  Time: 28ms      │  Time: 125ms       │  Time: 46ms                  │
├──────────────────┴────────────────────┴──────────────────────────────┤
│  Chú thích: ■ D00 Nứt dọc   ■ D10 Nứt ngang   ■ D20 Lưới   ■ D40 Ổ│
└──────────────────────────────────────────────────────────────────────┘
```

### 4.5 Trạng thái loading / progress

```
Đang tải model RT-DETR...   ████████░░ 80%
Đang chạy inference...      ██████████ Done ✅
```
→ Dùng `st.progress()` + `st.spinner("Đang tải model...")`

---

## 5. TAB 2 — 📊 So Sánh Mô Hình

### 5.1 Bảng so sánh tổng hợp

```
┌──────────────────────────────────────────────────────────────────────┐
│  📊 BẢNG SO SÁNH KẾT QUẢ EVALUATION                                  │
│  Nguồn: results/metrics.csv  │  Cập nhật: 10/07/2026 17:30           │
│                                                                      │
│  [Lọc theo: Confidence ▼] [Dataset ▼] [Device ▼]  [↻ Refresh]       │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Mô hình      │ Precision │ Recall │ mAP@50 │  FPS  │ Thiết bị      │
│  ─────────────┼───────────┼────────┼────────┼───────┼───────────    │
│  🔵 YOLO      │  0.70 ███ │ 0.66██ │  0.72  │  35.0 │ CUDA:0        │
│  🟢 F-RCNN    │  0.74 ████│ 0.68██ │  0.75🥇│   8.0 │ CUDA:0        │
│  🟣 RT-DETR   │  0.72 ███ │ 0.67██ │  0.74  │  22.0 │ CUDA:0        │
│                                                                      │
│  Chú thích: 🥇 = tốt nhất theo cột                                    │
│                                                                      │
│                         [📥 Xuất CSV]  [📋 Copy bảng]                │
└──────────────────────────────────────────────────────────────────────┘
```

> **Kỹ thuật render:** `st.dataframe()` với `column_config` để thêm progress bar
> vào cột Precision, Recall, mAP@50 (dùng `st.column_config.ProgressColumn`).

```python
st.dataframe(
    metrics,
    column_config={
        "precision": st.column_config.ProgressColumn("Precision", min_value=0, max_value=1, format="%.3f"),
        "recall":    st.column_config.ProgressColumn("Recall",    min_value=0, max_value=1, format="%.3f"),
        "map50":     st.column_config.ProgressColumn("mAP@50",    min_value=0, max_value=1, format="%.3f"),
        "fps":       st.column_config.NumberColumn("FPS", format="%.1f ⚡"),
    },
    use_container_width=True,
    hide_index=True,
)
```

### 5.2 Metric cards tóm tắt

```
┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐
│  🎯 Best mAP   │  │  ⚡ Fastest     │  │  🏆 Best P     │  │  🔄 Best R     │
│                │  │                │  │                │  │                │
│  Faster R-CNN  │  │  YOLO          │  │  Faster R-CNN  │  │  Faster R-CNN  │
│    0.750       │  │  35.0 FPS      │  │    0.740       │  │    0.680       │
│  ↑ +0.02 vs YOLO│  │  4.4× vs FRCNN │  │  +0.04 vs YOLO │  │  +0.02 vs YOLO │
└────────────────┘  └────────────────┘  └────────────────┘  └────────────────┘
```

### 5.3 Gợi ý khuyến nghị sử dụng

```
┌──────────────────────────────────────────────────────────────────────┐
│  💡 KHUYẾN NGHỊ DỰA TRÊN KẾT QUẢ                                    │
│                                                                      │
│  ✅ Real-time / Edge device → YOLO (35 FPS, model nhỏ 6MB)          │
│  ✅ Cần độ chính xác cao   → Faster R-CNN (mAP@50: 0.75)           │
│  ✅ Cân bằng tốc độ/chính xác → RT-DETR (22 FPS, mAP 0.74)        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. TAB 3 — 📈 Biểu Đồ Kết Quả

### 6.1 Layout biểu đồ

```
┌──────────────────────────────────────────────────────────────────────┐
│  Hiển thị biểu đồ từ: ○ results/figures/  ○ Tạo trực tiếp           │
│                                                                      │
│  [Biểu đồ Precision]   [Biểu đồ Recall]                             │
│  ┌────────────────┐     ┌────────────────┐                           │
│  │  Bar chart     │     │  Bar chart     │                           │
│  │  3 màu model   │     │  3 màu model   │                           │
│  └────────────────┘     └────────────────┘                           │
│                                                                      │
│  [Biểu đồ mAP@50]      [Biểu đồ FPS]                                │
│  ┌────────────────┐     ┌────────────────┐                           │
│  │  Bar chart     │     │  Bar chart     │                           │
│  │  + annotation  │     │  + speed label │                           │
│  └────────────────┘     └────────────────┘                           │
│                                                                      │
│  [Radar Chart — so sánh đa chiều]                                    │
│  ┌───────────────────────────────────┐                               │
│  │    Precision                      │                               │
│  │       ▲                           │                               │
│  │  FPS ◄ · YOLO(🔵) ► Recall       │                               │
│  │       ▼  F-RCNN(🟢)              │                               │
│  │    mAP50  RT-DETR(🟣)            │                               │
│  └───────────────────────────────────┘                               │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Kỹ thuật render biểu đồ

```python
import plotly.graph_objects as go
import plotly.express as px

# Palette màu theo model
COLOR_MAP = {
    "yolo":        "#4A90E2",  # xanh dương
    "faster_rcnn": "#7ED321",  # xanh lá
    "rtdetr":      "#BD10E0",  # tím
}

# Bar chart với Plotly (thay vì ảnh PNG tĩnh)
fig = px.bar(
    metrics,
    x="model", y="map50",
    color="model",
    color_discrete_map=COLOR_MAP,
    title="mAP@50 So sánh",
    text_auto=".3f",
    template="plotly_dark",
)
fig.update_layout(
    plot_bgcolor="#1A1D27",
    paper_bgcolor="#1A1D27",
    font_color="#FAFAFA",
    showlegend=False,
)
st.plotly_chart(fig, use_container_width=True)

# Radar chart
categories = ["Precision", "Recall", "mAP@50", "FPS (norm)"]
fig_radar = go.Figure()
for _, row in metrics.iterrows():
    fig_radar.add_trace(go.Scatterpolar(
        r=[row.precision, row.recall, row.map50, row.fps / metrics.fps.max()],
        theta=categories,
        fill="toself",
        name=row.model,
    ))
st.plotly_chart(fig_radar, use_container_width=True)
```

> **Ưu tiên:** Dùng `plotly` (interactive, hover tooltips) thay vì hiển thị ảnh PNG tĩnh.
> Nếu `plotly` chưa cài, fallback sang `st.image()` từ `results/figures/`.

---

## 7. TAB 4 — ℹ️ Hướng Dẫn Sử Dụng

```
┌──────────────────────────────────────────────────────────────────────┐
│  📖 HƯỚNG DẪN SỬ DỤNG                                                │
│                                                                      │
│  ▼ Bước 1: Chuẩn bị weight mô hình                                   │
│    Chắc chắn đã train hoặc đặt file weight đúng đường dẫn:           │
│    • YOLO:        runs/yolo/best.pt                                   │
│    • Faster R-CNN: runs/faster_rcnn/best.pth                         │
│    • RT-DETR:     runs/rtdetr/best.pt                                 │
│                                                                      │
│  ▼ Bước 2: Upload ảnh hoặc video                                     │
│    Hỗ trợ: JPG, PNG, BMP, WebP (ảnh) | MP4, AVI, MOV (video)        │
│                                                                      │
│  ▼ Bước 3: Điều chỉnh ngưỡng Confidence                              │
│    • 0.05–0.20: Phát hiện nhiều, có thể nhiễu                        │
│    • 0.25–0.50: Cân bằng (khuyến nghị)                               │
│    • 0.50+:     Chỉ detection chắc chắn                              │
│                                                                      │
│  ▼ Bước 4: Nhấn Run Detection                                        │
│    Kết quả hiện ngay trong Tab "Demo Ảnh / Video"                    │
│                                                                      │
│  ▼ Bước 5: Xem so sánh                                               │
│    Tab "So Sánh" hiển thị bảng metrics từ results/metrics.csv        │
│    Chạy evaluate.py để cập nhật số liệu mới nhất.                    │
│                                                                      │
│  ──────────────────────────────────────────────────────────────────  │
│  📌 Các lớp hư hỏng được phát hiện                                    │
│                                                                      │
│  ■ D00 Nứt dọc (Longitudinal crack)                                  │
│    Vết nứt chạy theo chiều dọc mặt đường.                            │
│                                                                      │
│  ■ D10 Nứt ngang (Transverse crack)                                  │
│    Vết nứt vuông góc với chiều đường.                                 │
│                                                                      │
│  ■ D20 Nứt lưới (Alligator crack)                                    │
│    Mạng lưới vết nứt, dấu hiệu nền đường yếu.                        │
│                                                                      │
│  ■ D40 Ổ gà (Pothole)                                                │
│    Hố lõm trên mặt đường, nguy hiểm cho xe.                          │
│                                                                      │
│  ──────────────────────────────────────────────────────────────────  │
│  🔗 Liên kết hữu ích                                                  │
│  • Dataset RDD2022: https://github.com/sekilab/RoadDamageDetector    │
│  • YOLO Docs:       https://docs.ultralytics.com                     │
│  • Torchvision:     https://pytorch.org/vision/stable                │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 8. Xử lý trạng thái (State Management)

```python
# Session state keys cần quản lý
st.session_state:
  uploaded_file      : UploadedFile | None   # file đang upload
  input_type         : "image" | "video"     # loại file
  run_result         : dict | None           # kết quả inference gần nhất
    .model_key       : str                   # model đã chạy
    .image_result    : np.ndarray | None     # ảnh kết quả có bbox
    .boxes           : list[list[float]]
    .labels          : list[int]
    .scores          : list[float]
    .inference_ms    : float                 # thời gian inference
    .video_out_path  : Path | None           # video output nếu là video
  compare_results    : dict | None           # kết quả chạy cả 3 model
    .yolo            : dict                  # tương tự run_result
    .faster_rcnn     : dict
    .rtdetr          : dict
  metrics_df         : pd.DataFrame | None  # cache metrics.csv
```

---

## 9. Thông báo & Error handling

| Trường hợp | Loại thông báo | Nội dung |
|---|---|---|
| File weight không tồn tại | `st.error` + icon ❌ | "Không tìm thấy file weight tại [path]. Hãy train mô hình trước." |
| Không decode được ảnh | `st.error` | "Không thể đọc file ảnh. Thử lại với định dạng khác." |
| CUDA không có | `st.warning` | "Không tìm thấy GPU, tự động chuyển sang CPU." |
| Inference thành công | `st.success` | "Phát hiện 7 vị trí hư hỏng trong 45ms." |
| metrics.csv rỗng | `st.info` | "Chưa có dữ liệu so sánh. Chạy evaluate.py để tạo metrics." |
| Đang load model | `st.spinner` | "Đang tải mô hình RT-DETR..." |
| Import lỗi (RTDETR) | `st.error` + hướng dẫn | "pip install -U ultralytics" |

---

## 10. Custom CSS inject

```python
# Tích hợp vào đầu main()
st.markdown("""
<style>
/* Font */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: #1A1D27;
    border-right: 1px solid #2D3142;
}

/* Primary button */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #FF6B35, #FF4757);
    border: none;
    border-radius: 8px;
    font-weight: 600;
    letter-spacing: 0.5px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(255,107,53,0.3);
}
.stButton button[kind="primary"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(255,107,53,0.5);
}

/* Secondary button */
.stButton button[kind="secondary"] {
    background: #2D3142;
    border: 1px solid #FF6B35;
    border-radius: 8px;
    color: #FF6B35;
    font-weight: 500;
    transition: all 0.2s ease;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: #1A1D27;
    border: 1px solid #2D3142;
    border-radius: 12px;
    padding: 16px;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: #FF6B35;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #1A1D27;
    border-radius: 10px;
    padding: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 500;
}

/* Info/Success/Error boxes */
[data-testid="stAlert"] {
    border-radius: 10px;
    border-left-width: 4px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Image container */
[data-testid="stImage"] img {
    border-radius: 12px;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
</style>
""", unsafe_allow_html=True)
```

---

## 11. Cấu trúc file sau khi mở rộng

```
src/demo/
├── app_streamlit.py        ← file chính (entry point)
├── components/             ← (mới, tùy chọn — nếu muốn tách component)
│   ├── sidebar.py          ← render_sidebar()
│   ├── result_panel.py     ← render_image_result(), render_video_result()
│   ├── comparison_tab.py   ← render_comparison_table(), render_metric_cards()
│   ├── charts_tab.py       ← render_charts()
│   └── help_tab.py         ← render_help()
└── styles.py               ← CSS string và hàm inject_css()

.streamlit/
└── config.toml             ← theme, server settings
```

---

## 12. Checklist triển khai

- [ ] Tạo `.streamlit/config.toml` với theme dark + primaryColor orange
- [ ] Inject custom CSS vào đầu `main()`
- [ ] Chuyển model selector từ `st.selectbox` sang `st.radio` (trực quan hơn)
- [ ] Thêm icon ✅/❌ validate weight path live
- [ ] Tách upload thành 2 tab (Ảnh / Video) trong sidebar
- [ ] Thêm metric cards (Best mAP, Fastest, Best Precision, Best Recall)
- [ ] Thêm bảng chi tiết bounding box (sortable dataframe)
- [ ] Thêm nút "📥 Tải ảnh kết quả" (st.download_button)
- [ ] Thêm nút "🆚 So sánh cả 3 mô hình"
- [ ] Thêm biểu đồ Plotly interactive (bar + radar) cho Tab 3
- [ ] Thêm section Khuyến nghị sử dụng dựa trên kết quả metrics
- [ ] Thêm Tab Hướng dẫn với expander theo từng bước
- [ ] Xử lý đầy đủ các trường hợp lỗi với thông báo thân thiện
- [ ] Test với cả CPU và GPU
- [ ] Test upload ảnh PNG, JPG, BMP, WebP
- [ ] Test upload video MP4, AVI, MOV

---

## 13. Lệnh chạy

```bash
# Cài thêm plotly nếu chưa có
pip install plotly

# Chạy app
streamlit run src/demo/app_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

---

*Tài liệu thiết kế này dùng để hướng dẫn cài đặt/nâng cấp `src/demo/app_streamlit.py`.
Mọi thành phần đều bám sát code hiện tại và chỉ mở rộng — không phá vỡ logic cũ.*
