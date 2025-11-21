# 🚦 Hệ Thống Nhận Diện Biển Báo Giao Thông Việt Nam

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.38.0-FF4B4B.svg)](https://streamlit.io)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Hệ thống nhận diện biển báo giao thông Việt Nam sử dụng YOLOv8 với giao diện web thân thiện. Hỗ trợ xử lý hình ảnh, video và webcam realtime với 58 loại biển báo.

<!-- TODO: Add demo GIF here -->
<!-- ![Demo](docs/demo.gif) -->

## 🔄 PROJECT STATUS (November 5, 2025)

**Current Status:** 🟡 57/58 classes (98.3% complete)

### ✅ Completed:
- ✅ Model with 27 classes trained (mAP50: 92.92%)
- ✅ Dataset collected for 57 classes (5,020 images total)
- ✅ Data merged and organized
- ✅ Web application fully functional

### 🚧 In Progress:
- 🔄 Training model for 57 classes (previous attempt failed due to Virtual Memory)
- 📋 Missing 1 class: **W.221b** (Đường không bằng phẳng - Uneven Road)

### 📖 Next Steps:
**📄 Start here:** [`START_NOW.md`](START_NOW.md) - Quick action guide  
**📄 Full guide:** [`NEXT_STEPS.md`](NEXT_STEPS.md) - Detailed roadmap  
**📄 Fix training:** [`FIX_TRAINING_ERROR.md`](FIX_TRAINING_ERROR.md) - Troubleshooting

**Quick Actions:**
```bash
# Check system resources
python check_system_resources.py

# Start training (auto-optimized)
python train_57_classes_safe.py
```

---

## ✨ Tính Năng

### 🎯 Nhận Diện
- **📷 Hình Ảnh**: Upload và nhận diện biển báo từ ảnh
- **🎥 Video**: Xử lý video với thống kê chi tiết
- **📹 Webcam**: Nhận diện realtime từ webcam
- **58 Loại Biển Báo**: Hỗ trợ đầy đủ các biển báo Việt Nam (P, R, W, S, DP)

### ⚡ Hiệu Năng
- **GPU/CPU Auto-detect**: Tự động sử dụng GPU nếu có
- **Half Precision (FP16)**: Tăng tốc inference trên GPU
- **Model Caching**: Cache model để tránh load lại
- **Frame Skipping**: Tối ưu FPS cho webcam/video

### 🎨 Giao Diện
- **Dashboard Thống Kê**: Biểu đồ và thống kê chi tiết
- **Multi-tab UI**: Giao diện đa tab dễ sử dụng
- **Responsive Design**: Tương thích mọi kích thước màn hình
- **Vietnamese UI**: Giao diện hoàn toàn tiếng Việt

### 💾 Dữ Liệu
- **Database SQLite**: Lưu lịch sử phát hiện
- **Export CSV**: Xuất dữ liệu ra file CSV
- **Video Recording**: Ghi lại video từ webcam
- **Capture Images**: Chụp ảnh từ webcam

### 🔊 Cảnh Báo
- **Audio Alerts**: Cảnh báo âm thanh khi phát hiện
- **Cross-platform**: Hỗ trợ Windows, Linux, macOS
- **Customizable**: Tùy chỉnh cooldown và enable/disable

## 🚀 Quick Start

### 📋 Yêu Cầu Hệ Thống

- **Python**: 3.8 trở lên
- **RAM**: Tối thiểu 4GB (khuyến nghị 8GB+)
- **GPU**: NVIDIA GPU với CUDA (tùy chọn, tăng tốc inference)
- **OS**: Windows, Linux, hoặc macOS

### 📦 Cài Đặt

#### 1. Clone Repository

```bash
git clone https://github.com/yourusername/NhanDienBienBaoGiaoThong.git
cd NhanDienBienBaoGiaoThong
```

#### 2. Tạo Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

#### 3. Cài Đặt Dependencies

```bash
pip install -r requirements.txt
```

#### 4. Chuẩn Bị Model

Đảm bảo file model YOLO đã được train nằm tại:
```
runs/train/exp/weights/best.pt
```

Hoặc đặt file `best.pt` ở thư mục root của project.

### ▶️ Chạy Ứng Dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

### 🎬 Setup Nhanh (Windows)

```bash
setup.bat
```

Script này sẽ:
1. Tạo virtual environment
2. Cài đặt dependencies
3. Khởi động ứng dụng

## 📖 Hướng Dẫn Sử Dụng

### 📊 Tab Dashboard
- Xem thống kê tổng quan
- Top 10 biển báo phổ biến
- Lịch sử phát hiện gần đây
- Export dữ liệu ra CSV

### 📷 Tab Hình Ảnh
1. Click "Browse files" để chọn ảnh
2. Hệ thống sẽ tự động nhận diện
3. Xem kết quả và danh sách biển báo phát hiện

### 🎥 Tab Video
1. Upload video (.mp4, .avi, .mov)
2. Click "Bắt đầu xử lý"
3. Đợi quá trình hoàn tất
4. Xem video đã xử lý và thống kê chi tiết
5. Tải video về nếu cần

### 📹 Tab Webcam
1. Click "Bật Webcam"
2. Cho phép truy cập camera
3. Nhận diện realtime
4. **Chức năng**:
   - 📸 Chụp ảnh: Lưu frame hiện tại
   - 🔴 Ghi video: Ghi lại video
   - ⏹️ Dừng: Tắt webcam

## ⚙️ Cấu Hình

### Sidebar Settings

#### 🎯 Confidence Threshold
Điều chỉnh ngưỡng tin cậy (0.0 - 1.0). Giá trị cao hơn = chính xác hơn nhưng ít phát hiện hơn.

#### ⚡ Performance Settings
- **Skip Frames**: Số frames bỏ qua (tăng FPS)
- **Inference Size**: Kích thước ảnh inference (320/416/640/1280)

#### 🔊 Audio Settings
- Bật/tắt cảnh báo âm thanh
- Điều chỉnh thời gian giữa các cảnh báo

## 🏗️ Cấu Trúc Project

```
NhanDienBienBaoGiaoThong/
│
├── app.py                      # Streamlit entrypoint
├── requirements.txt            # Python dependencies
├── README.md                   # Project overview + quick start
│
├── utils/                      # Application modules
│   ├── audio_alert.py          # Audio system
│   ├── database.py             # SQLite helper
│   ├── font_utils.py           # Cross-platform font cache
│   ├── inference.py            # Image inference helpers
│   ├── model_cache.py          # Device/precision auto-detect
│   ├── path_utils.py           # Centralized path helpers
│   ├── video_processing.py     # Offline video pipeline
│   └── webcam_processing.py    # Realtime webcam pipeline
│
├── data/                       # Production dataset consumed by the app
│   ├── data.yaml               # Dataset config (57 classes)
│   ├── sign_labels_vi.json     # Label → name mapping
│   ├── traffic_signs.db        # SQLite detections database
│   ├── train/ | valid/ | test/ # YOLO-formatted splits
│
├── datasets/                   # Supporting datasets & backups (NEW grouping)
│   ├── backup/                 # 27-class legacy dataset & DB snapshot
│   ├── merged/                 # 57-class merged dataset (full history)
│   ├── new/                    # Latest raw annotations awaiting merge
│   └── raw/                    # Original Roboflow export (Vietnam Traffic Sign)
│
├── models/                     # Model weights & custom layers
│   ├── best.pt                 # Default production model
│   ├── best_27classes_backup.pt
│   ├── yolov8s_trained_best.pt
│   └── yolo_custom.py
│
├── runs/                       # Ultralytics training/validation artifacts
│   ├── detect/                 # Detection experiments (weights, charts, logs)
│   └── train/                  # Training runs (exp/, optimized_baseline/, …)
│
├── captured_images/            # Manual snapshots from webcam tab
├── recorded_videos/            # Videos recorded from webcam tab
├── input/                      # User-provided sample images/videos
├── output/                     # Exported CSV/video artifacts
├── temp/                       # Session temp files (auto-cleaned)
└── venv/                       # Local virtual environment (optional)
```

## 🎓 Training Model

### Chuẩn Bị Dataset

Dataset cần có cấu trúc:
```
data/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Train Model

```python
from ultralytics import YOLO

# Load model
model = YOLO('yolov8n.pt')

# Train
results = model.train(
    data='data/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='traffic_signs'
)
```

### Validate Model

```python
# Validate
metrics = model.val()
print(f"mAP50: {metrics.box.map50}")
print(f"mAP50-95: {metrics.box.map}")
```

## 🔧 Troubleshooting

### GPU không được nhận diện
```bash
# Kiểm tra CUDA
python -c "import torch; print(torch.cuda.is_available())"

# Cài đặt PyTorch với CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Lỗi font trên Linux
```bash
# Cài đặt fonts
sudo apt-get install fonts-dejavu fonts-liberation
```

### Webcam không hoạt động
- Kiểm tra quyền truy cập camera
- Thử camera_id khác (0, 1, 2...)
- Kiểm tra camera đang được sử dụng bởi app khác

### Lỗi module không tìm thấy
```bash
# Cài đặt lại dependencies
pip install -r requirements.txt --force-reinstall
```

## 📊 Performance Benchmarks

| Model    | Device | FPS (Webcam) | Inference Time | mAP50 |
|----------|--------|--------------|----------------|-------|
| YOLOv8n  | CPU    | 8-12 fps     | 80-120ms       | 0.85  |
| YOLOv8n  | GPU    | 25-30 fps    | 15-20ms        | 0.85  |
| YOLOv8s  | CPU    | 5-8 fps      | 120-180ms      | 0.88  |
| YOLOv8s  | GPU    | 20-25 fps    | 20-30ms        | 0.88  |

*Tested on: Intel i5-10400F, RTX 3060, 16GB RAM*

## 🗺️ Roadmap

### P1 - Nâng cao trải nghiệm
- [ ] Model selector trong sidebar
- [ ] Export video đã xử lý
- [ ] Biểu đồ theo ngày trong dashboard
- [ ] Auto fallback khi FPS thấp
- [ ] Object tracking (ByteTrack/StrongSORT)

### P2 - Chất lượng & Triển khai
- [ ] Validation script (mAP, PR curves)
- [ ] Data augmentation (rain, night, blur)
- [ ] Export ONNX/TensorRT/TFLite
- [ ] OCR cho biển tốc độ (P.127)
- [ ] Docker deployment
- [ ] Unit tests với pytest
- [ ] Pre-commit hooks (ruff/black)

## 📝 Changelog

### Version 2.0.0 (Current)
- ✅ Model caching + auto device + FP16
- ✅ Cross-platform font/audio
- ✅ Fixed confidence values
- ✅ Extracted labels to JSON
- ✅ Portable path handling
- ✅ Proper video/webcam cleanup

### Version 1.0.0
- Initial release
- Basic image/video/webcam processing
- Dashboard and statistics
- Database integration

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) - Object detection
- [Streamlit](https://streamlit.io/) - Web framework
- Vietnamese Traffic Sign Dataset - Training data
---

⭐ **Nếu project này hữu ích, hãy cho một Star!** ⭐

