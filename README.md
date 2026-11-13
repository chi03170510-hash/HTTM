# HTTM — Smart Classroom Camera Live Monitoring

Hệ thống giám sát lớp học trực tiếp từ camera sử dụng lõi AI **YOLOv8n** + **Behavior YOLO** (`Using Phone`, `Eating`, `Sleeping`), giữ nguyên giao thức kết nối camera (`DSHOW` -> `MSMF`) và giao diện OpenCV Live của `HTTM`.

## Tính năng chính
- **Phát hiện đối tượng & đếm người (`yolov8n.pt`)**: đếm sĩ số (`Occupancy`) và các vật dụng lớp học (`Cell Phone`, `Laptop`, `Book`, `Chair`, `Table`, ...).
- **Nhận diện 3 hành vi (`behavior_yolo.pt`)**: `Using Phone`, `Eating`, `Sleeping`.
- **Tự động chạy GPU CUDA (`CUDA:0`)**: tự động sử dụng GPU NVIDIA nếu có PyTorch CUDA, fallback an toàn sang CPU nếu máy không có GPU.
- **Giữ nguyên giao thức camera HTTM**: thử `DSHOW` rồi fallback `MSMF`, pipeline bất đồng bộ `frame_queue` / `result_queue`, chế độ xem live + bảng chẩn đoán 70/30.

---

## Cài đặt nhanh (Windows)

### Yêu cầu
- Windows 10/11 với webcam (`CAMERA_INDEX = 0` mặc định)
- **Python 3.11** (khuyến nghị, hỗ trợ `3.10+`)
- GPU NVIDIA + Driver mới (tuỳ chọn, để tăng tốc CUDA)

### 1. Clone repo và tạo môi trường tự động
```powershell
git clone https://github.com/chi03170510-hash/HTTM.git
cd HTTM
powershell -ExecutionPolicy Bypass -File .\setup_env.ps1
```
Script `setup_env.ps1` sẽ tự tạo `.venv`, kiểm tra `nvidia-smi` để cài PyTorch bản GPU (`cu128`) nếu có GPU NVIDIA, và cài đầy đủ các gói trong `requirements.txt`.

### 2. Hoặc cài đặt thủ công
```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip

# Nếu có GPU NVIDIA (bỏ qua dòng này nếu chỉ dùng CPU):
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

---

## Chạy Camera Live

### Cách 1: Chạy nhanh bằng `.bat` (hoặc nhấp đúp file)
```bat
run_camera_live.bat
```

### Cách 2: Chạy trực tiếp bằng Python trong `.venv`
```powershell
.\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py
```

### Phím điều khiển trên cửa sổ OpenCV
- `Q`: Thoát chương trình
- `I`: Bật/tắt bảng thông số chi tiết 70/30 bên phải (`3` hành vi + item inspector)
- `N` / `P`: Chuyển sang item trước/sau trong bảng chi tiết

---

## Kiểm tra trước khi chạy & Test tích hợp

Kiểm tra preflight (không mở camera thật):
```powershell
.\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_preflight.py
```

Chạy bộ test tích hợp:
```powershell
.\.venv\Scripts\python.exe -B -m unittest .\ai_training\integration\test_smart_classroom_integration.py -v
```

*(Nếu hệ thống có thêm `.venv-hybrid` tách riêng cho AI worker, `camera_coordinator_live.py` và `camera_live_preflight.py` sẽ tự động ưu tiên dùng `.venv-hybrid` làm worker; nếu không có thì chạy trực tiếp bằng `.venv`.)*

---

## Tuỳ chỉnh nhanh
- **Đổi chỉ số camera**: sửa `CAMERA_INDEX = 0` trong `ai_training/integration/camera_coordinator.py`.
- **Ép thiết bị suy luận**: đặt biến môi trường `HTTM_HYBRID_DEVICE=cpu` (hoặc `0` để bắt buộc GPU `CUDA:0`, mặc định `auto`).
