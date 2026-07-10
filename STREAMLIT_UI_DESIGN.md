# Streamlit UI — Control Center v3

## Mục tiêu

Giao diện phục vụ hai luồng tách biệt:

1. Demo vận hành: chạy phát hiện trên ảnh/video và tải kết quả.
2. Đánh giá nghiên cứu: đọc metric từ test có nhãn và so sánh cùng giao thức.

Giao diện không dùng số FPS, kích thước model hoặc tuyên bố “model tốt nhất” nếu chưa có dữ
liệu trong `results/metrics.csv`.

## Bố cục

```text
┌──────────────────────┬───────────────────────────────────────────────┐
│ Sidebar              │ Hero: trạng thái checkpoint và dataset       │
│                      ├───────────────────────────────────────────────┤
│ • Chọn model         │ Workbench | Benchmark | Analytics | Hướng dẫn│
│ • Confidence         │                                               │
│ • Thiết bị           │ Nội dung khu vực đang chọn                    │
│ • Checkpoint/runtime │                                               │
└──────────────────────┴───────────────────────────────────────────────┘
```

Sidebar chỉ chứa cấu hình bền vững. Upload, nút chạy và kết quả nằm trong workbench để không
chia nhỏ luồng thao tác giữa hai vùng màn hình.

## Design system

- Canvas: `#FFFFFF`
- Panel phụ: `#F7FAFC`
- Border: `#DCE6EB`
- Primary: `#0F9F8F`
- Warning: `#A96B00`
- Error: `#D94C4C`
- Text: `#16303F`
- Muted: `#607886`

Font ưu tiên system font (`Manrope`, `Inter`, `Segoe UI`) để giao diện không phụ thuộc Google
Fonts hoặc kết nối Internet. Component dùng border mảnh, nền trong nhẹ và khoảng trắng thay vì
shadow/gradient quá mạnh.

## Các khu vực

### Workbench

- Chuyển giữa ảnh tĩnh và video.
- Preview dữ liệu trước khi chạy.
- Chạy model đang chọn hoặc so sánh ba model trên cùng ảnh.
- Hiển thị latency, số detection, confidence, phân bổ D00/D10/D20/D40.
- Cho tải ảnh/video kết quả.
- Không dùng số detection hoặc confidence để kết luận model chính xác hơn.

### Model benchmark

- Đọc `results/metrics.csv`.
- Hiển thị best mAP@50, FPS, Precision và Recall theo dữ liệu thực tế.
- Nhắc rõ chỉ so sánh các run cùng test split, confidence, device và số mẫu.
- Cho xuất bảng CSV.

### Visual analytics

- Bar chart cho Precision, Recall, mAP@50 và FPS.
- Radar chart chuẩn hóa FPS về `[0, 1]`.
- Màu model nhất quán giữa chart và checkpoint card.

### Hướng dẫn

- Tách demo khỏi đánh giá.
- Mô tả bốn lớp RDD2022.
- Cung cấp lệnh train/evaluate đang dùng trong repository.

## Trạng thái và an toàn

- Checkpoint trong `runs/<model>/checkpoints/` được resolve qua checksum/rollback hiện có.
- Nút chạy bị khóa nếu checkpoint model đang chọn chưa hợp lệ.
- So sánh ba model vẫn mở để chỉ rõ model nào thiếu checkpoint.
- Detector cache tự invalid khi mtime checkpoint thay đổi; có nút xóa cache thủ công.
- Nội dung tên file đi vào HTML phải được escape.

## File liên quan

```text
src/demo/app_streamlit.py       # logic, state và page composition
src/demo/streamlit_styles.py    # design tokens và CSS
.streamlit/config.toml          # theme/server defaults
```

## Kiểm tra

```bash
python -m py_compile src/demo/app_streamlit.py src/demo/streamlit_styles.py
streamlit run src/demo/app_streamlit.py --server.address 127.0.0.1 --server.port 8501
```

Các trạng thái tối thiểu cần kiểm tra: không có checkpoint, checkpoint hợp lệ, ảnh hợp lệ, ảnh
không giải mã được, video, metrics rỗng và metrics đủ ba model.
