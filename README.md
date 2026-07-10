# So sánh YOLO, Faster R-CNN và RT-DETR trong phát hiện hư hỏng mặt đường

Project này xây dựng một pipeline Python hoàn chỉnh để fine-tune, chạy demo và so sánh ba mô hình object detection cho bài toán phát hiện hư hỏng mặt đường từ ảnh hoặc video.

## Giới thiệu đề tài

Hư hỏng mặt đường như nứt dọc, nứt ngang, nứt dạng lưới và ổ gà ảnh hưởng trực tiếp đến an toàn giao thông và chi phí bảo trì. Thay vì kiểm tra thủ công, thị giác máy tính có thể tự động phát hiện vị trí hư hỏng bằng bounding box, từ đó hỗ trợ khảo sát đường bộ nhanh hơn và nhất quán hơn.

## Lý do chọn đề tài

- Bài toán có ý nghĩa thực tế trong quản lý hạ tầng giao thông.
- Dataset Road Damage Detection có annotation phù hợp cho object detection.
- Ba mô hình YOLO, Faster R-CNN và RT-DETR đại diện cho ba hướng tiếp cận khác nhau: one-stage detector, two-stage detector và detector dựa trên Transformer.
- Kết quả có thể so sánh bằng cả độ chính xác và tốc độ suy luận, phù hợp cho bài cuối kỳ môn Thị giác máy tính.

## Mục tiêu nghiên cứu

- Phát hiện hư hỏng mặt đường từ ảnh hoặc video.
- Fine-tune và so sánh YOLO, Faster R-CNN và RT-DETR.
- Đánh giá bằng Precision, Recall, mAP@50 và FPS.
- Tạo demo CLI và Streamlit hiển thị bounding box, tên lớp và confidence.
- Xuất bảng so sánh kết quả giữa ba mô hình.

## Dataset

> **Quan trọng:** tập dùng để đánh giá phải có ground truth. Một số bản RDD2022 cung cấp
> thư mục `test` với file nhãn rỗng để nộp bài thi; tập đó không thể dùng để tính
> Precision/Recall/mAP. Pipeline sẽ dừng thay vì xuất số liệu sai khi toàn bộ test không
> có bounding box.

Project hướng đến dataset Road Damage Detection, ví dụ RDD2022. Nếu chưa có dataset, tải RDD2022 từ nguồn chính thức của cuộc thi/dataset rồi đặt dữ liệu vào:

```text
data/raw/
```

Script hỗ trợ hai dạng dữ liệu đầu vào:

- Pascal VOC XML: ảnh và file `.xml` annotation như nhiều bản RDD gốc.
- YOLO format: thư mục `images/` và `labels/`, mỗi ảnh có một file `.txt` cùng tên.

Sau xử lý, dữ liệu nên có dạng:

```text
data/processed/
├── split_manifest.csv
├── split_summary.json
├── images/
│   ├── train/
│   ├── val/
│   ├── test/       # Có nhãn, dùng tính Precision/Recall/mAP
│   └── challenge/  # Test mù gốc RDD2022, không có nhãn
└── labels/
    ├── train/
    ├── val/
    └── test/
```

## Các lớp phát hiện

| Mã lớp | Tên tiếng Anh | Mô tả |
|---|---|---|
| D00 | Longitudinal crack | Nứt dọc theo chiều đường |
| D10 | Transverse crack | Nứt ngang qua mặt đường |
| D20 | Alligator crack | Nứt dạng lưới, thường do kết cấu mặt đường suy yếu |
| D40 | Pothole | Ổ gà hoặc vùng mặt đường bị bong vỡ |

## Mô hình

- YOLO: mô hình phát hiện đối tượng nhanh, phù hợp real-time và demo video.
- Faster R-CNN: two-stage detector kinh điển, thường có độ chính xác tốt nhưng tốc độ thấp hơn.
- RT-DETR: object detector dựa trên Transformer, hướng đến real-time. Project dùng API `RTDETR` của Ultralytics; nếu môi trường Ultralytics đang cài chưa hỗ trợ RT-DETR, hãy cập nhật `ultralytics` hoặc đổi sang weight tương thích.

## Cài đặt môi trường Conda

Yêu cầu Python 3.10+.

File Conda tạo environment riêng tên `road-damage-detection` (Python 3.10,
PyTorch 2.5/Torchvision 0.20, CUDA 12.1):

```bash
conda env create -f environment.yml
conda activate road-damage-detection
conda config --env --set channel_priority strict
```

Nếu đã có environment này, cập nhật có kiểm soát và loại dependency thừa:

```bash
conda env update -n road-damage-detection -f environment.yml --prune
conda activate road-damage-detection
```

Hoặc cài nhanh bằng pip trong environment đang active:

```bash
pip install -r requirements.txt
```

`environment.yml` dành cho NVIDIA CUDA 12.1. Với CPU-only/macOS, tạo Python 3.10
environment, cài PyTorch theo hướng dẫn chính thức cho nền tảng đó rồi chạy
`pip install -r requirements.txt`. Lần train đầu cần Internet nếu dùng tên model
như `yolov8n.pt`, `rtdetr-l.pt` hoặc pretrained Faster R-CNN; có thể tải trước và
truyền đường dẫn local để chạy offline.

## Chuẩn bị dataset

Đặt dataset gốc vào `data/raw/`, sau đó chạy:

```bash
python src/dataset/split_dataset.py --input data/raw --output data/processed
```

Kiểm tra toàn bộ cấu trúc và nhãn trước khi train (thêm `--check-images` nếu muốn
decode-check mọi ảnh):

```bash
python -m src.dataset.validate_dataset --root data/processed
```

Lệnh trả exit code khác 0 nếu có label sai, file thiếu/mồ côi, ảnh trùng giữa các
split hoặc một split hoàn toàn không có bounding box. Khi chạy lại công cụ chia dữ
liệu, sáu thư mục `images|labels/{train,val,test}` cũ được dọn có kiểm soát để tránh
ảnh cũ xuất hiện đồng thời ở nhiều split.

Với RDD2022 có test mù, dùng công cụ chia cân bằng để gộp train/val có nhãn,
chia chính xác 80/10/10 và giữ test cũ thành `challenge`:

```bash
python -m src.dataset.rebuild_splits \
  --input data/processed_original \
  --output data/processed_rebuilt \
  --train 0.8 --val 0.1 --test 0.1 \
  --seed 42 --block-size 100
```

Công cụ giữ nguyên block ảnh liên tiếp, dùng tối ưu nguyên để cân bằng số ảnh,
ảnh dương, số box D00/D10/D20/D40 và từng nguồn quốc gia. Ảnh được hard-link trên
cùng ổ đĩa để không nhân đôi dung lượng. Kết quả chia được ghi tại
`split_manifest.csv` và `split_summary.json` để có thể kiểm tra/tái lập.

Workspace hiện tại đã được chia theo cấu hình trên. Bộ dữ liệu trước khi chia nằm
ở `data/processed_original` để rollback; `data/processed` là bộ đang dùng để train
và đánh giá.

Các checkpoint tạo từ split cũ không được dùng để báo cáo metric trên test mới vì
có thể gây data leakage. Các run cũ đã được xóa khỏi workspace; cần train lại cả ba
mô hình trên split mới trước khi chạy đánh giá chính thức.

### Dùng dataset Kaggle RDD2022 YOLO

Dataset `sreekaraditya/rdd2022-yolo-crackscan-v2` trên Kaggle là bản YOLO format, nên có thể dùng trực tiếp với project này. Tải bằng Kaggle CLI:

```bash
pip install kaggle
# Đặt API token tại ~/.kaggle/kaggle.json (Linux/macOS) hoặc %USERPROFILE%\.kaggle\kaggle.json (Windows)
kaggle datasets download -d sreekaraditya/rdd2022-yolo-crackscan-v2 -p data/raw --unzip
```

Sau đó chuẩn hóa về cấu trúc `data/processed`:

```bash
python src/dataset/split_dataset.py --input data/raw --output data/processed
```

Script sẽ tự nhận các cấu trúc YOLO phổ biến như `train/images`, `valid/images`, `test/images` hoặc `images/train`, `images/val`, `images/test`. Thư mục `valid` sẽ được map thành `val`.

Trước khi train, mở file `data.yaml` của dataset nếu có và kiểm tra class id phải khớp:

```text
0: D00
1: D10
2: D20
3: D40
```

Nếu dataset có thêm class khác hoặc thứ tự class khác, cần sửa `configs/dataset.yaml` và mapping trong `src/utils/common.py` cho thống nhất.

### Dùng dataset gốc Figshare RDD2022

Dataset gốc Figshare `RDD2022 - The multi-national Road Damage Dataset released through CRDDC'2022` thường dùng annotation Pascal VOC XML theo từng quốc gia/thư mục. Project này hỗ trợ đọc XML và convert sang YOLO trước khi train.

Sau khi tải file từ Figshare, giải nén vào:

```text
data/raw/
```

Ví dụ cấu trúc có thể là nhiều thư mục quốc gia hoặc tập con, bên trong có ảnh và XML annotation. Chạy:

```bash
python src/dataset/split_dataset.py --input data/raw --output data/processed
```

Nếu phát hiện XML, script sẽ:

1. Parse annotation Pascal VOC.
2. Chỉ giữ các lớp `D00`, `D10`, `D20`, `D40`.
3. Convert bbox sang YOLO format.
4. Gộp dữ liệu annotated và chia thành `train/val/test`.

Với bản gốc Figshare, nếu có thư mục test không có annotation thì project không dùng phần đó để evaluate vì không có ground truth. Nên dùng các ảnh có XML để tự chia train/val/test cho bài so sánh mô hình.

Nếu muốn chỉ convert XML sang YOLO format:

```bash
python src/dataset/convert_rdd_to_yolo.py --input data/raw --output data/processed/all
```

File `configs/dataset.yaml` đã trỏ đến `data/processed` và dùng bốn lớp `D00`, `D10`, `D20`, `D40`.

## Train YOLO

```bash
python src/models/yolo/train.py --data configs/dataset.yaml --model yolov8n.pt --epochs 60 --imgsz 640 --batch 16 --device 0
```

Kết quả framework được lưu trong `runs/yolo/train*/`, checkpoint chuẩn được xuất an toàn về:

```text
runs/yolo/checkpoints/best.pt
```

Có thể đổi model thành `yolov8s.pt` hoặc `yolo11n.pt` nếu môi trường Ultralytics hỗ trợ. Nên đặt weight tải sẵn trong `weights/` để thư mục gốc không bị lộn xộn.

Mặc định dùng `freeze: 0` để fine-tune toàn bộ YOLOv8n trên tập train hiện tại. Nếu cần
giảm thời gian hoặc VRAM có thể dùng `--freeze 10`. Dataset YAML được chuẩn hóa sang đường dẫn tuyệt đối trong
`runs/yolo/` nên lệnh train không phụ thuộc current working directory.

## Train Faster R-CNN

```bash
python src/models/faster_rcnn/train.py --config configs/faster_rcnn.yaml --epochs 30 --batch 2 --device cuda
```

Checkpoint tốt nhất:

```text
runs/faster_rcnn/checkpoints/best.pth
```

Mô hình dùng `torchvision.models.detection.fasterrcnn_resnet50_fpn`, load pretrained weights và thay classification head thành 5 lớp: background, D00, D10, D20, D40.

Backbone được freeze theo config mặc định; dùng `--no-freeze-backbone` để fine-tune
toàn bộ. Checkpoint `best.pth` được chọn theo validation mAP@50 và chỉ chứa model để
inference gọn hơn; `last.pth` giữ model, optimizer, scheduler, AMP scaler và RNG state để resume.

## Train RT-DETR

```bash
python src/models/rtdetr/train.py --data configs/dataset.yaml --model rtdetr-l.pt --epochs 50 --imgsz 640 --batch 2 --device 0
```

Kết quả framework được lưu trong `runs/rtdetr/train*/`, checkpoint chuẩn được xuất an toàn về:

```text
runs/rtdetr/checkpoints/best.pt
```

## Checkpoint an toàn và resume

Ba mô hình dùng chung cấu trúc checkpoint ổn định:

```text
runs/<model>/checkpoints/
├── best.pt hoặc best.pth
├── last.pt hoặc last.pth
├── best.previous.pt/pth
├── last.previous.pt/pth
├── *.sha256
└── manifest.json
```

Checkpoint được ghi qua file tạm rồi thay thế nguyên tử. Mỗi file có SHA-256; khi file hiện tại
bị thiếu, rỗng hoặc sai checksum, inference/resume tự dùng bản `previous` đã xác minh. Không xóa
checkpoint tốt cũ khi bắt đầu một lần train mới.

Resume từ `last` an toàn:

```bash
python src/models/yolo/train.py --resume auto
python src/models/rtdetr/train.py --resume auto
python src/models/faster_rcnn/train.py --resume auto --epochs 50
```

Với Faster R-CNN, `--epochs` là tổng số epoch đích. Resume sẽ bị từ chối nếu checksum lỗi mà
không có bản dự phòng, hoặc nếu `split_manifest.csv`/cấu hình freeze backbone đã thay đổi.

## Demo ảnh

YOLO:

```bash
python src/demo/demo_image.py --model yolo --weights runs/yolo/checkpoints/best.pt --source data/samples/test.jpg
```

Faster R-CNN:

```bash
python src/demo/demo_image.py --model faster_rcnn --weights runs/faster_rcnn/checkpoints/best.pth --source data/samples/test.jpg
```

RT-DETR:

```bash
python src/demo/demo_image.py --model rtdetr --weights runs/rtdetr/checkpoints/best.pt --source data/samples/test.jpg
```

Có thể chạy trực tiếp qua các file inference riêng:

```bash
python src/models/yolo/infer.py --weights runs/yolo/checkpoints/best.pt --source data/samples/test.jpg
python src/models/faster_rcnn/infer.py --weights runs/faster_rcnn/checkpoints/best.pth --source data/samples/test.jpg
python src/models/rtdetr/infer.py --weights runs/rtdetr/checkpoints/best.pt --source data/samples/test.jpg
```

## Demo video

```bash
python src/demo/demo_video.py --model yolo --weights runs/yolo/checkpoints/best.pt --source data/samples/road.mp4
```

Video kết quả được lưu vào `results/` và có hiển thị FPS theo từng frame.

## Giao diện Streamlit

```bash
streamlit run src/demo/app_streamlit.py --server.address 127.0.0.1
```

Giao diện cho phép upload ảnh/video, chọn mô hình, nhập đường dẫn weight, xem kết quả detect và thống kê số lượng lỗi theo từng lớp. Nếu có `results/metrics.csv`, giao diện sẽ hiển thị bảng so sánh.

## Đánh giá

YOLO:

```bash
python src/evaluation/evaluate.py --model yolo --weights runs/yolo/checkpoints/best.pt --data data/processed/test
```

Faster R-CNN:

```bash
python src/evaluation/evaluate.py --model faster_rcnn --weights runs/faster_rcnn/checkpoints/best.pth --data data/processed/test
```

RT-DETR:

```bash
python src/evaluation/evaluate.py --model rtdetr --weights runs/rtdetr/checkpoints/best.pt --data data/processed/test
```

Mỗi lần đánh giá sẽ cập nhật `results/metrics.csv` với các cột:

```text
model, precision, recall, map50, fps, confidence, samples, device, data_path, weight_path
```

mAP@50 được tính từ prediction confidence thấp (`0.001`), còn Precision/Recall và
FPS dùng đúng `--conf`. FPS có một lượt warm-up. Công cụ so sánh sẽ từ chối ghép các
run khác confidence, số mẫu, dataset hoặc device.

## So sánh kết quả

```bash
python src/evaluation/compare_results.py --input results/metrics.csv
```

Script tạo:

- `results/comparison_table.csv`
- `results/figures/precision_comparison.png`
- `results/figures/recall_comparison.png`
- `results/figures/map50_comparison.png`
- `results/figures/fps_comparison.png`

## Bảng kết quả mẫu

Các giá trị dưới đây chỉ là ví dụ minh họa. Cần train và evaluate trên dataset thật để có số liệu chính thức.

| Model | Precision | Recall | mAP@50 | FPS | Model Size | Nhận xét |
|---|---:|---:|---:|---:|---:|---|
| YOLO | 0.70 | 0.66 | 0.72 | 35.0 | 6 MB | Nhanh, phù hợp demo real-time |
| Faster R-CNN | 0.74 | 0.68 | 0.75 | 8.0 | 160 MB | Chính xác khá, tốc độ thấp hơn |
| RT-DETR | 0.72 | 0.67 | 0.74 | 22.0 | 120 MB | Cân bằng giữa Transformer và tốc độ |

## Phân công nhóm 3 người

- Người 1: YOLO, chuẩn bị cấu hình Ultralytics, train, demo ảnh/video.
- Người 2: Faster R-CNN, Dataset class PyTorch, training loop, checkpoint.
- Người 3: RT-DETR, đánh giá, so sánh kết quả, Streamlit và báo cáo.

## Cấu trúc project

```text
road-damage-detection/
├── README.md
├── requirements.txt
├── configs/
├── data/
├── src/
│   ├── dataset/
│   ├── models/
│   │   ├── yolo/
│   │   │   ├── train.py
│   │   │   └── infer.py
│   │   ├── faster_rcnn/
│   │   │   ├── model.py
│   │   │   ├── train.py
│   │   │   └── infer.py
│   │   └── rtdetr/
│   │       ├── train.py
│   │       └── infer.py
│   ├── evaluation/
│   ├── demo/
│   └── utils/
├── runs/
└── results/
```

## Ghi chú khi chạy

Chạy kiểm thử hồi quy:

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

- Không hard-code đường dẫn tuyệt đối; các script mặc định chạy từ root project.
- Nếu không có GPU, script sẽ dùng CPU và chạy chậm hơn.
- Với dataset thật, cần kiểm tra lại annotation XML có đúng tên lớp `D00`, `D10`, `D20`, `D40` hay không.
- Faster R-CNN dùng label `0` cho background, còn YOLO/RT-DETR dùng class id `0..3`; code đã tự chuyển đổi khi đọc label.
- RT-DETR phụ thuộc vào phiên bản `ultralytics`. Nếu import `RTDETR` lỗi, hãy chạy `pip install -U ultralytics`.

## Kết luận và hướng phát triển

Project cung cấp đầy đủ pipeline cho bài toán phát hiện hư hỏng mặt đường: chuẩn bị dữ liệu, fine-tune ba mô hình, inference ảnh/video, đánh giá định lượng và demo bằng Streamlit. Hướng phát triển tiếp theo gồm tăng kích thước dataset, thử augmentation phù hợp với điều kiện đường Việt Nam, đánh giá thêm mAP@50:95, tối ưu model bằng TensorRT/ONNX và triển khai thử nghiệm trên camera hành trình hoặc thiết bị edge.
