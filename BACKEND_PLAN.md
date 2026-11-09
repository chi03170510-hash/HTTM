# 📋 BACKEND DEVELOPMENT PLAN — HỆ THỐNG CẢNH BÁO NHẮM MẮT

> **Dự án**: HTTM (Hệ Thống Theo dõi Mắt)
> **Ngày tạo**: 15/09/2026
> **Mục tiêu**: Xây dựng Backend (FastAPI) kết nối AI Module (MediaPipe) với Frontend (React), lưu trữ vi phạm vào MySQL, đẩy cảnh báo realtime qua WebSocket.

---

## 📂 HIỆN TRẠNG DỰ ÁN

### Cấu trúc hiện tại
```
HTTM/
├── .gitignore
├── README.md
├── ai/
│   └── eye_detection/
│       ├── main.py              ← Script chạy độc lập, mở camera + MediaPipe + cv2.imshow
│       ├── config.py            ← EAR_THRESHOLD = 0.10
│       ├── ear_utils.py         ← Hàm tính EAR (euclidean_distance, get_eye_points, calculate_ear)
│       ├── analyze_ear.py       ← Script phân tích CSV (không liên quan đến backend)
│       ├── calibrate_ear.py     ← Script calibrate EAR (không liên quan đến backend)
│       ├── requirements.txt     ← mediapipe==1.0.1, opencv-python==5.0.0.93, numpy==2.5.3, ...
│       └── models/
│           └── face_landmarker.task   ← Model MediaPipe (~4MB)
```

### Những gì AI module đã làm được (file `main.py`)
- Mở camera/video bằng OpenCV
- Dùng MediaPipe `FaceLandmarker` (running_mode=VIDEO, num_faces=10) để detect landmarks
- Tính EAR (Eye Aspect Ratio) cho mắt trái/phải mỗi khuôn mặt
- So sánh `avg_ear < EAR_THRESHOLD (0.10)` → xác định OPEN/CLOSED
- Dùng dict `eye_closed_start = {}` để theo dõi timer cho từng `face_id`
- Khi `closed_duration >= WARNING_TIME (5.0s)` → vẽ cảnh báo lên frame (viền đỏ, text WARNING)
- Khi không detect khuôn mặt → `eye_closed_start.clear()` reset toàn bộ timer
- Landmark indices: LEFT_EYE = [33, 160, 158, 133, 153, 144], RIGHT_EYE = [362, 385, 387, 263, 373, 380]

### Những gì CHƯA có
- ❌ Backend (folder rỗng, chưa tạo)
- ❌ Database MySQL
- ❌ WebSocket server
- ❌ REST API cho lịch sử vi phạm
- ❌ Frontend React (đã khởi tạo nhưng chưa có nội dung, config trỏ tới `http://localhost:8000`)

---

## 🏗️ KIẾN TRÚC TỔNG THỂ

```
┌─────────────┐     WebSocket (realtime)     ┌──────────────────┐     Import trực tiếp    ┌─────────────────┐
│   Frontend  │ ◄──────────────────────────► │    Backend       │ ◄───── (Python) ──────  │   AI Module     │
│   (React)   │     REST API (lịch sử)       │   (FastAPI)      │                         │  (MediaPipe)    │
│  port 3000  │                              │   port 8000      │                         │  (ear_utils.py) │
└─────────────┘                              └────────┬─────────┘                         └─────────────────┘
                                                      │
                                                      │ SQLAlchemy + PyMySQL
                                                      ▼
                                              ┌───────────────┐
                                              │   MySQL DB    │
                                              │  (local)      │
                                              └───────────────┘
```

### Lý do chọn FastAPI
- Cùng Python với AI module → **import trực tiếp** `ear_utils.py`, `config.py`, không cần subprocess
- Hỗ trợ **WebSocket** native (starlette)
- Async/await → xử lý concurrent tốt
- Auto-generate Swagger docs tại `/docs`
- Port mặc định `8000` → khớp với config frontend

---

## 📁 CẤU TRÚC THƯ MỤC SAU KHI HOÀN THÀNH

```
HTTM/
├── ai/
│   └── eye_detection/           ← GIỮ NGUYÊN, KHÔNG SỬA
│       ├── main.py
│       ├── config.py
│       ├── ear_utils.py
│       ├── analyze_ear.py
│       ├── calibrate_ear.py
│       ├── requirements.txt
│       └── models/
│           └── face_landmarker.task
│
├── backend/
│   ├── main.py                  ← Entry point: uvicorn, mount routers, CORS, startup/shutdown
│   ├── config.py                ← Cấu hình DB, camera, paths (đọc từ .env)
│   ├── database.py              ← SQLAlchemy engine + SessionLocal + Base
│   ├── requirements.txt         ← Tất cả dependencies backend
│   ├── .env                     ← Biến môi trường (DB credentials, ...)
│   │
│   ├── models/                  ← SQLAlchemy ORM models
│   │   ├── __init__.py
│   │   └── violation.py         ← Model: Violation
│   │
│   ├── schemas/                 ← Pydantic schemas (request/response validation)
│   │   ├── __init__.py
│   │   └── violation.py         ← ViolationCreate, ViolationResponse, ViolationList, StatsResponse
│   │
│   ├── routers/                 ← API route handlers
│   │   ├── __init__.py
│   │   ├── violations.py        ← GET /api/violations, GET /api/violations/{id}, GET /api/violations/{id}/image, GET /api/stats
│   │   ├── camera.py            ← POST /api/camera/start, POST /api/camera/stop, GET /api/camera/status
│   │   └── websocket.py         ← WebSocket /ws/monitor
│   │
│   ├── services/                ← Business logic layer
│   │   ├── __init__.py
│   │   ├── eye_detection.py     ← Class EyeDetectionService (refactor logic từ ai/main.py)
│   │   └── camera_service.py    ← Class CameraService (quản lý camera loop, chụp ảnh, lưu DB)
│   │
│   └── uploads/
│       └── violations/          ← Thư mục lưu ảnh vi phạm (jpg)
│
├── .gitignore
└── README.md
```

---

# 🔷 PHASE 1: KHỞI TẠO PROJECT + DATABASE

**Mục tiêu**: Tạo cấu trúc backend, kết nối MySQL, tạo bảng.

## Bước 1.1: Tạo cấu trúc thư mục

Tạo tất cả folders và `__init__.py` như cấu trúc ở trên.

## Bước 1.2: Tạo `backend/requirements.txt`

```
fastapi==0.115.6
uvicorn[standard]==0.34.0
sqlalchemy==2.0.36
pymysql==1.1.1
python-dotenv==1.0.1
opencv-python==5.0.0.93
mediapipe==1.0.1
numpy==2.5.3
python-multipart==0.0.20
```

> **Lưu ý**: Phiên bản `mediapipe` và `opencv-python` phải **khớp** với `ai/eye_detection/requirements.txt` để tránh xung đột. Kiểm tra phiên bản chính xác trong file đó (mediapipe==1.0.1, opencv-python==5.0.0.93).

## Bước 1.3: Tạo `backend/.env`

```env
# Database
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password_here
DB_NAME=httm

# Camera
CAMERA_INDEX=0

# AI Model
MODEL_PATH=../ai/eye_detection/models/face_landmarker.task
```

## Bước 1.4: Tạo `backend/config.py`

Đọc biến môi trường từ `.env`:
- `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` → tạo `DATABASE_URL` dạng `mysql+pymysql://user:pass@host:port/dbname`
- `CAMERA_INDEX` (int, default 0)
- `MODEL_PATH` (string, resolve thành absolute path)
- `UPLOAD_DIR` = `os.path.join(os.path.dirname(__file__), "uploads", "violations")`

## Bước 1.5: Tạo `backend/database.py`

- Tạo `engine` từ `DATABASE_URL`
- Tạo `SessionLocal` (sessionmaker)
- Tạo `Base` (declarative_base)
- Tạo hàm `get_db()` (dependency injection cho FastAPI)
- Tạo hàm `init_db()` gọi `Base.metadata.create_all(bind=engine)` để auto-tạo bảng

## Bước 1.6: Tạo `backend/models/violation.py`

**Bảng `violations`:**

| Cột | Kiểu SQLAlchemy | Kiểu MySQL | Mô tả |
|-----|-----------------|------------|-------|
| `id` | `Integer, primary_key, autoincrement` | `INT AUTO_INCREMENT` | PK |
| `face_id` | `Integer, nullable=False` | `INT NOT NULL` | ID khuôn mặt từ AI (0, 1, 2...) |
| `started_at` | `DateTime, nullable=False` | `DATETIME NOT NULL` | Thời điểm bắt đầu nhắm mắt |
| `ended_at` | `DateTime, nullable=True` | `DATETIME NULL` | Thời điểm mở mắt (cập nhật sau) |
| `duration` | `Float, nullable=False` | `FLOAT NOT NULL` | Số giây nhắm mắt tại thời điểm lưu |
| `image_path` | `String(500), nullable=True` | `VARCHAR(500) NULL` | Đường dẫn relative tới ảnh (vd: `violations/20260915_103000_face0.jpg`) |
| `created_at` | `DateTime, default=datetime.utcnow` | `DATETIME` | Timestamp tạo record |

> **Tại sao 1 bảng, không cần bảng `violation_images` riêng?**
> Vì mỗi vi phạm chỉ chụp **1 ảnh** duy nhất (tại thời điểm phát hiện). Nếu sau này cần lưu nhiều ảnh/clip cho 1 vi phạm thì mới tách bảng.

## Bước 1.7: Tạo `backend/main.py` (skeleton)

```python
# Entry point - khởi tạo FastAPI app
# - Import routers (violations, camera, websocket) — nhưng Phase 1 chưa có, mount sau
# - CORS middleware: allow origins ["http://localhost:3000"] (React dev server)
# - Startup event: gọi init_db(), tạo thư mục uploads nếu chưa có
# - Root endpoint GET / trả về {"message": "HTTM Backend API"}
```

## Bước 1.8: Kiểm tra Phase 1

1. Tạo database MySQL tên `httm` thủ công: `CREATE DATABASE httm CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;`
2. Cập nhật password MySQL trong `.env`
3. Chạy: `cd backend && pip install -r requirements.txt && python main.py`
4. Kiểm tra: truy cập `http://localhost:8000` → thấy `{"message": "HTTM Backend API"}`
5. Kiểm tra: truy cập `http://localhost:8000/docs` → thấy Swagger UI
6. Kiểm tra MySQL: bảng `violations` đã được tạo

### ✅ Tiêu chí hoàn thành Phase 1
- [x] Server FastAPI chạy trên port 8000
- [x] Kết nối MySQL thành công
- [x] Bảng `violations` được tạo tự động
- [x] Swagger UI hoạt động
- [x] CORS cho phép localhost:3000

---

# 🔷 PHASE 2: REFACTOR AI MODULE THÀNH SERVICE

**Mục tiêu**: Tách logic phát hiện mắt từ `ai/eye_detection/main.py` thành class có thể import, KHÔNG sửa file gốc trong `ai/`.

## Bước 2.1: Tạo `backend/services/eye_detection.py` — Class `EyeDetectionService`

### Nguyên tắc refactor
- **KHÔNG sửa** bất kỳ file nào trong `ai/eye_detection/`. Giữ nguyên để team AI vẫn chạy độc lập.
- **Copy logic** từ `ai/eye_detection/main.py` vào class mới.
- **Import hàm tiện ích** từ `ai/eye_detection/ear_utils.py` bằng cách thêm `ai/eye_detection/` vào `sys.path`.
- **Import config** từ `ai/eye_detection/config.py` (lấy `EAR_THRESHOLD`).

### Class `EyeDetectionService`

```python
class EyeDetectionService:
    """
    Wrap logic phát hiện nhắm mắt từ MediaPipe.
    Stateful: giữ landmarker instance + timer cho mỗi face.
    """

    def __init__(self, model_path: str):
        """
        - Khởi tạo MediaPipe FaceLandmarker với model_path
        - running_mode = VIDEO, num_faces = 10
        - Khởi tạo dict eye_closed_start = {} (timer cho từng face_id)
        - Khởi tạo dict warning_saved = {} (đánh dấu face_id đã lưu ảnh chưa, tránh lưu lặp)
        - Lưu start_time = time.time() để tính timestamp_ms
        - frame_index = 0
        """

    def process_frame(self, frame: np.ndarray) -> dict:
        """
        Nhận 1 frame BGR từ OpenCV, trả về kết quả phân tích.

        Input:
            frame: numpy array BGR (từ cv2.VideoCapture.read())

        Output:
            {
                "faces_detected": True/False,
                "faces": [
                    {
                        "face_id": 0,
                        "eye_state": "OPEN" | "CLOSED",
                        "left_ear": 0.25,
                        "right_ear": 0.24,
                        "avg_ear": 0.245,
                        "closed_duration": 3.2,    # giây đã nhắm liên tục
                        "is_warning": False         # True nếu >= 5s
                    },
                    ...
                ]
            }

        Logic bên trong (copy từ main.py):
            1. Chuyển BGR → RGB
            2. Tạo mp.Image
            3. Tính timestamp_ms từ frame_index
            4. Gọi landmarker.detect_for_video()
            5. Với mỗi face:
               a. Lấy eye points (LEFT_EYE, RIGHT_EYE indices)
               b. Tính left_ear, right_ear, avg_ear
               c. So sánh avg_ear < EAR_THRESHOLD → CLOSED/OPEN
               d. Cập nhật eye_closed_start[face_id]
               e. Tính closed_duration
               f. Đặt is_warning = (closed_duration >= 5.0)
            6. Nếu không detect face → eye_closed_start.clear()
            7. Tăng frame_index
        """

    def reset(self):
        """Reset tất cả timer và trạng thái. Gọi khi stop camera."""
        # eye_closed_start.clear()
        # warning_saved.clear()
        # frame_index = 0
        # start_time = time.time()

    def mark_warning_saved(self, face_id: int):
        """Đánh dấu face_id đã lưu ảnh vi phạm, tránh lưu lặp lại."""
        # warning_saved[face_id] = True

    def is_warning_saved(self, face_id: int) -> bool:
        """Kiểm tra face_id đã lưu ảnh vi phạm chưa."""
        # return warning_saved.get(face_id, False)

    def clear_warning_saved(self, face_id: int):
        """Xoá đánh dấu khi face mở mắt lại (reset cho lần vi phạm tiếp theo)."""
        # warning_saved.pop(face_id, None)

    def close(self):
        """Giải phóng tài nguyên MediaPipe landmarker."""
```

### Chi tiết cách import từ AI module

```python
import sys
import os

# Thêm đường dẫn tới ai/eye_detection/ để import ear_utils và config
AI_MODULE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "ai", "eye_detection")
)
sys.path.insert(0, AI_MODULE_DIR)

from ear_utils import get_eye_points, calculate_ear
from config import EAR_THRESHOLD
```

### Constant cần giữ (copy từ `main.py`)
```python
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
WARNING_TIME = 5.0
```

## Bước 2.2: Test `EyeDetectionService` độc lập

Viết script test nhanh `backend/test_eye_service.py` (tạm, xoá sau):
```python
# Mở camera, tạo EyeDetectionService, gọi process_frame() cho mỗi frame
# In kết quả ra terminal
# Kiểm tra: đúng face_id, đúng eye_state, đúng is_warning khi nhắm mắt >= 5s
# KHÔNG dùng cv2.imshow (headless test) — chỉ print
```

### ✅ Tiêu chí hoàn thành Phase 2
- [x] `EyeDetectionService` khởi tạo thành công, load model MediaPipe
- [x] `process_frame()` trả về dict đúng format
- [x] Phát hiện đúng OPEN/CLOSED cho từng face
- [x] Timer hoạt động: closed_duration tăng dần khi nhắm mắt liên tục
- [x] `is_warning = True` khi nhắm >= 5s
- [x] Hỗ trợ multi-face (num_faces=10)
- [x] Không sửa bất kỳ file nào trong `ai/`

---

# 🔷 PHASE 3: CAMERA SERVICE + LƯU ẢNH + LƯU DATABASE

**Mục tiêu**: Xây dựng service quản lý camera loop chạy background, khi vi phạm → chụp ảnh + lưu DB.

## Bước 3.1: Tạo `backend/schemas/violation.py`

Pydantic schemas:

```python
class ViolationResponse(BaseModel):
    """Response cho 1 vi phạm"""
    id: int
    face_id: int
    started_at: datetime
    ended_at: Optional[datetime]
    duration: float
    image_path: Optional[str]
    created_at: datetime

    class Config:
        from_attributes = True    # để convert từ SQLAlchemy model


class ViolationListResponse(BaseModel):
    """Response cho danh sách vi phạm (có phân trang)"""
    items: List[ViolationResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class StatsResponse(BaseModel):
    """Response cho thống kê"""
    total_today: int
    total_this_week: int
    total_this_month: int
    total_all: int


class CameraStatusResponse(BaseModel):
    """Response cho trạng thái camera"""
    is_running: bool
    message: str
```

## Bước 3.2: Tạo `backend/services/camera_service.py` — Class `CameraService`

```python
class CameraService:
    """
    Singleton quản lý camera loop.
    Chạy trong background thread, gọi EyeDetectionService mỗi frame.
    """

    def __init__(self):
        """
        - self.cap = None                    # cv2.VideoCapture
        - self.is_running = False
        - self.eye_service = None            # EyeDetectionService instance
        - self.thread = None                 # Background thread
        - self.latest_result = None          # Kết quả mới nhất từ AI (để WebSocket đọc)
        - self.websocket_connections = set() # Danh sách WebSocket clients
        - self.db_session_factory = None     # SessionLocal để tạo DB session trong thread
        """

    def start(self, camera_index: int, model_path: str, db_session_factory):
        """
        Bắt đầu giám sát:
        1. Mở camera: cv2.VideoCapture(camera_index)
        2. Kiểm tra cap.isOpened() → nếu False, raise Exception
        3. Khởi tạo EyeDetectionService(model_path)
        4. Lưu db_session_factory
        5. Tạo và start background thread chạy self._camera_loop()
        6. self.is_running = True
        """

    def stop(self):
        """
        Dừng giám sát:
        1. self.is_running = False → thread loop sẽ break
        2. Chờ thread join (timeout 5s)
        3. Release camera: cap.release()
        4. Close eye_service
        5. Reset các biến
        """

    def _camera_loop(self):
        """
        Chạy trong background thread. QUAN TRỌNG NHẤT.

        while self.is_running:
            1. ret, frame = self.cap.read()
               - Nếu not ret → log lỗi, continue (hoặc break nếu camera mất)

            2. result = self.eye_service.process_frame(frame)
               - result = {"faces_detected": bool, "faces": [...]}

            3. self.latest_result = result

            4. Duyệt từng face trong result["faces"]:
               a. Nếu face["is_warning"] == True VÀ chưa lưu ảnh cho face này:
                  - Chụp ảnh: cv2.imencode('.jpg', frame) → lấy bytes
                  - Tạo tên file: f"violation_{timestamp}_{face_id}.jpg"
                    (timestamp format: %Y%m%d_%H%M%S)
                  - Lưu file ảnh ra: UPLOAD_DIR / tên_file
                  - Tạo record trong DB:
                    violation = Violation(
                        face_id = face["face_id"],
                        started_at = datetime khi bắt đầu nhắm (tính từ closed_duration),
                        duration = face["closed_duration"],
                        image_path = tên_file (relative path),
                        created_at = datetime.now()
                    )
                    db.add(violation) → db.commit()
                  - Đánh dấu: self.eye_service.mark_warning_saved(face_id)
                  - Gửi WebSocket message type "warning" cho tất cả clients
                    (bao gồm violation_id, image_url, ...)

               b. Nếu face["eye_state"] == "OPEN" VÀ face trước đó đang warning:
                  - Cập nhật ended_at cho violation cuối cùng của face_id này
                  - Gọi self.eye_service.clear_warning_saved(face_id)
                    → Cho phép lưu ảnh ở lần vi phạm tiếp theo

            5. Gửi WebSocket message type "status" cho tất cả clients
               (trạng thái realtime của tất cả faces)

            6. Điều khiển FPS: time.sleep(1/30) hoặc tùy chỉnh
               → Tránh CPU 100%. 30 FPS là đủ mượt.

        # Sau khi break loop:
        self.cap.release()
        """

    async def broadcast_websocket(self, message: dict):
        """
        Gửi message tới tất cả WebSocket clients đang kết nối.
        - Duyệt self.websocket_connections
        - Gọi ws.send_json(message)
        - Nếu ws bị disconnect → remove khỏi set
        """

    def register_websocket(self, websocket):
        """Thêm WebSocket client vào set."""

    def unregister_websocket(self, websocket):
        """Xoá WebSocket client khỏi set."""
```

### ⚠️ Lưu ý quan trọng về threading + async

- Camera loop chạy trong **thread riêng** (vì `cv2.VideoCapture.read()` là blocking I/O).
- FastAPI chạy trên **asyncio event loop**.
- Khi thread muốn gửi WebSocket (async) → dùng `asyncio.run_coroutine_threadsafe(coro, loop)` để schedule coroutine từ thread vào event loop.
- Cần lưu reference tới event loop: `self.loop = asyncio.get_event_loop()` trước khi start thread.

### ⚠️ Lưu ý về DB session trong thread

- **KHÔNG** dùng chung DB session giữa thread và async handlers.
- Trong `_camera_loop()` (thread), tạo DB session riêng bằng `self.db_session_factory()`, dùng xong `close()`.
- Mỗi lần cần INSERT, tạo session mới trong block `try/finally`.

### ⚠️ Lưu ý chỉ lưu ảnh 1 lần mỗi vi phạm

Flow cho 1 face:
```
Frame 1-100: OPEN                    → không làm gì
Frame 101:   CLOSED (bắt đầu nhắm)  → timer bắt đầu
Frame 102-250: CLOSED (2.5s)         → chưa đủ 5s, bỏ qua
Frame 251:   CLOSED (đúng 5s)        → ĐÂY LÀ LÚC: chụp ảnh, lưu DB, mark_warning_saved
Frame 252-400: CLOSED (7s)           → is_warning=True NHƯNG already_saved → bỏ qua, KHÔNG chụp lại
Frame 401:   OPEN (mở mắt lại)      → cập nhật ended_at, clear_warning_saved
Frame 500:   CLOSED (nhắm lần 2)    → timer reset, bắt đầu đếm lại từ 0
```

## Bước 3.3: Tạo singleton `camera_service` instance

Trong `backend/services/camera_service.py`, cuối file:
```python
# Singleton instance — toàn bộ app dùng chung 1 instance
camera_service = CameraService()
```

### ✅ Tiêu chí hoàn thành Phase 3
- [x] Camera mở/đóng đúng cách qua start/stop
- [x] Background thread chạy, không block API server
- [x] Mỗi vi phạm chỉ lưu 1 ảnh, không lặp
- [x] Ảnh lưu ra `backend/uploads/violations/` đúng format
- [x] Record INSERT vào MySQL thành công
- [x] `ended_at` được cập nhật khi mở mắt lại
- [x] WebSocket broadcast hoạt động

---

# 🔷 PHASE 4: REST API ENDPOINTS

**Mục tiêu**: Tạo các API endpoint cho frontend gọi.

## Bước 4.1: Tạo `backend/routers/camera.py`

```
POST /api/camera/start
    - Gọi camera_service.start(camera_index, model_path, SessionLocal)
    - Nếu đã running → trả 400 {"detail": "Camera đang chạy"}
    - Nếu lỗi mở camera → trả 500 {"detail": "Không mở được camera"}
    - Thành công → trả 200 {"is_running": true, "message": "Đã bắt đầu giám sát"}

POST /api/camera/stop
    - Gọi camera_service.stop()
    - Nếu chưa running → trả 400 {"detail": "Camera chưa chạy"}
    - Thành công → trả 200 {"is_running": false, "message": "Đã dừng giám sát"}

GET /api/camera/status
    - Trả về {"is_running": camera_service.is_running}
```

## Bước 4.2: Tạo `backend/routers/violations.py`

```
GET /api/violations
    Query params:
        - page: int = 1
        - page_size: int = 20
        - date_from: Optional[date] = None     # Filter từ ngày
        - date_to: Optional[date] = None       # Filter đến ngày
        - face_id: Optional[int] = None        # Filter theo face_id
    Response: ViolationListResponse
    Logic:
        - Query bảng violations, ORDER BY created_at DESC
        - Áp dụng filters nếu có
        - Phân trang: OFFSET = (page - 1) * page_size, LIMIT = page_size
        - Đếm total → tính total_pages

GET /api/violations/{id}
    - Trả về ViolationResponse
    - Nếu không tìm thấy → 404

GET /api/violations/{id}/image
    - Đọc image_path từ DB
    - Trả về file ảnh dùng FileResponse (content_type="image/jpeg")
    - Nếu không tìm thấy record hoặc ảnh → 404

GET /api/stats
    Response: StatsResponse
    Logic:
        - total_today: COUNT WHERE DATE(created_at) = today
        - total_this_week: COUNT WHERE created_at >= đầu tuần (Monday)
        - total_this_month: COUNT WHERE MONTH(created_at) = current month AND YEAR = current year
        - total_all: COUNT tất cả
```

## Bước 4.3: Mount routers trong `backend/main.py`

```python
app.include_router(camera_router, prefix="/api/camera", tags=["Camera"])
app.include_router(violations_router, prefix="/api", tags=["Violations"])
```

## Bước 4.4: Serve static files (ảnh vi phạm)

Ngoài endpoint `/api/violations/{id}/image`, có thể mount static folder cho truy cập trực tiếp:
```python
from fastapi.staticfiles import StaticFiles
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")
```
→ Ảnh có thể truy cập qua: `http://localhost:8000/uploads/violations/filename.jpg`

### ✅ Tiêu chí hoàn thành Phase 4
- [x] Tất cả endpoints hoạt động trên Swagger UI
- [x] Phân trang hoạt động đúng
- [x] Filter theo ngày hoạt động
- [x] Ảnh vi phạm có thể xem qua API
- [x] Stats trả về số liệu đúng
- [x] Camera start/stop hoạt động

---

# 🔷 PHASE 5: WEBSOCKET REALTIME

**Mục tiêu**: Frontend kết nối WebSocket để nhận trạng thái mắt và cảnh báo realtime.

## Bước 5.1: Tạo `backend/routers/websocket.py`

```python
@router.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """
    1. await websocket.accept()
    2. camera_service.register_websocket(websocket)
    3. try:
           while True:
               # Giữ connection sống
               # Có thể nhận message từ client (ping/pong, hoặc commands)
               data = await websocket.receive_text()
               # Xử lý nếu cần (hoặc bỏ qua)
       except WebSocketDisconnect:
           camera_service.unregister_websocket(websocket)
    """
```

## Bước 5.2: Format WebSocket messages

Backend gửi 2 loại message:

### Message loại `status` (gửi mỗi frame, ~30 lần/giây)

```json
{
    "type": "status",
    "timestamp": "2026-09-15T10:30:00.123",
    "camera_running": true,
    "faces_detected": true,
    "faces": [
        {
            "face_id": 0,
            "eye_state": "OPEN",
            "left_ear": 0.251,
            "right_ear": 0.243,
            "avg_ear": 0.247,
            "closed_duration": 0.0,
            "is_warning": false
        },
        {
            "face_id": 1,
            "eye_state": "CLOSED",
            "left_ear": 0.051,
            "right_ear": 0.048,
            "avg_ear": 0.050,
            "closed_duration": 3.2,
            "is_warning": false
        }
    ]
}
```

> **Tối ưu**: Không cần gửi 30 msg/s. Có thể giảm xuống 5-10 msg/s cho status (skip frames), chỉ gửi warning ngay lập tức. Dùng biến `last_ws_send_time` để throttle.

### Message loại `warning` (gửi 1 lần khi vi phạm được phát hiện)

```json
{
    "type": "warning",
    "timestamp": "2026-09-15T10:30:05.000",
    "face_id": 1,
    "duration": 5.1,
    "violation_id": 42,
    "image_url": "/api/violations/42/image",
    "message": "Phát hiện nhắm mắt >= 5 giây!"
}
```

### Message loại `warning_end` (khi mở mắt lại sau vi phạm)

```json
{
    "type": "warning_end",
    "timestamp": "2026-09-15T10:30:08.500",
    "face_id": 1,
    "total_duration": 8.5,
    "violation_id": 42
}
```

## Bước 5.3: Throttle WebSocket messages

Trong `_camera_loop()`, thêm logic throttle:
```python
WS_STATUS_INTERVAL = 0.1   # Gửi status tối đa 10 lần/giây
last_ws_status_time = 0

# Trong loop:
now = time.time()
if now - last_ws_status_time >= WS_STATUS_INTERVAL:
    # Gửi status message
    last_ws_status_time = now

# Warning message luôn gửi ngay, không throttle
```

### ✅ Tiêu chí hoàn thành Phase 5
- [x] WebSocket kết nối thành công từ browser/postman
- [x] Client nhận message status liên tục khi camera đang chạy
- [x] Client nhận message warning ngay khi phát hiện vi phạm
- [x] Client nhận message warning_end khi mở mắt
- [x] Nhiều client kết nối cùng lúc đều nhận message
- [x] Client disconnect không gây crash server

---

# 🔷 PHASE 6: TÍCH HỢP + KIỂM THỬ END-TO-END

**Mục tiêu**: Kiểm tra toàn bộ luồng từ đầu đến cuối.

## Bước 6.1: Kiểm thử end-to-end

### Test Case 1: Khởi động hệ thống
1. Chạy backend: `cd backend && uvicorn main:app --reload`
2. Truy cập `http://localhost:8000/docs` → Swagger UI hiển thị tất cả endpoints
3. Gọi `GET /api/camera/status` → `{"is_running": false}`

### Test Case 2: Bắt đầu giám sát
1. Gọi `POST /api/camera/start` → `{"is_running": true, ...}`
2. Kết nối WebSocket `ws://localhost:8000/ws/monitor`
3. Nhận message status liên tục → kiểm tra format đúng
4. Mở mắt bình thường → `eye_state: "OPEN"`, `is_warning: false`

### Test Case 3: Phát hiện vi phạm
1. Nhắm mắt trước camera >= 5 giây
2. WebSocket nhận message `type: "warning"` → kiểm tra có `violation_id`, `image_url`
3. Kiểm tra MySQL: có record mới trong bảng `violations`
4. Kiểm tra thư mục `uploads/violations/`: có file ảnh mới
5. Gọi `GET /api/violations` → thấy vi phạm vừa tạo
6. Gọi `GET /api/violations/{id}/image` → thấy ảnh

### Test Case 4: Mở mắt sau vi phạm
1. Mở mắt lại sau khi cảnh báo
2. WebSocket nhận message `type: "warning_end"`
3. Kiểm tra MySQL: `ended_at` được cập nhật

### Test Case 5: Nhiều vi phạm liên tiếp
1. Nhắm mắt → cảnh báo → mở mắt → nhắm mắt lại → cảnh báo lần 2
2. Kiểm tra: 2 records trong DB, 2 file ảnh khác nhau

### Test Case 6: Dừng giám sát
1. Gọi `POST /api/camera/stop` → `{"is_running": false, ...}`
2. Camera được release, thread dừng
3. WebSocket vẫn kết nối nhưng không nhận message mới

### Test Case 7: API lịch sử
1. Gọi `GET /api/violations?page=1&page_size=5` → phân trang đúng
2. Gọi `GET /api/violations?date_from=2026-09-15` → filter đúng
3. Gọi `GET /api/stats` → số liệu đúng

## Bước 6.2: Xử lý edge cases

- Camera không có / bị chiếm → trả lỗi rõ ràng, không crash server
- MediaPipe model file không tồn tại → trả lỗi khi start
- MySQL mất kết nối giữa chừng → try/except, log lỗi, tiếp tục camera loop
- WebSocket client disconnect đột ngột → remove khỏi set, không crash
- Gọi start khi đã running → trả 400
- Gọi stop khi chưa running → trả 400

## Bước 6.3: Logging

Thêm logging cơ bản:
```python
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("httm")

# Các log quan trọng:
logger.info("Camera started (index=%d)", camera_index)
logger.info("Camera stopped")
logger.warning("Violation detected: face_id=%d, duration=%.1fs", face_id, duration)
logger.error("Failed to save violation image: %s", str(e))
logger.info("WebSocket client connected (total: %d)", len(connections))
logger.info("WebSocket client disconnected (total: %d)", len(connections))
```

### ✅ Tiêu chí hoàn thành Phase 6
- [x] Tất cả 7 test cases pass
- [x] Edge cases không gây crash
- [x] Logs hiển thị đúng
- [x] Backend chạy ổn định >= 10 phút liên tục

---

# 📊 TỔNG KẾT PHASE

| Phase | Nội dung | Dependencies |
|-------|----------|--------------|
| **Phase 1** | Khởi tạo project + Database | Không |
| **Phase 2** | Refactor AI → EyeDetectionService | Phase 1 |
| **Phase 3** | CameraService + Lưu ảnh + Lưu DB | Phase 1, 2 |
| **Phase 4** | REST API Endpoints | Phase 1, 3 |
| **Phase 5** | WebSocket Realtime | Phase 3, 4 |
| **Phase 6** | Tích hợp + Kiểm thử E2E | Phase 1-5 |

---

# ⚠️ QUY TẮC BẮT BUỘC CHO AGENT THỰC HIỆN

1. **KHÔNG ĐƯỢC SỬA** bất kỳ file nào trong `ai/eye_detection/`. Chỉ **import** và **copy logic** từ đó.
2. **Mỗi phase phải test xong** trước khi qua phase tiếp theo.
3. **Ảnh vi phạm lưu dạng file** (JPG) trong `backend/uploads/violations/`, DB chỉ lưu đường dẫn relative.
4. **Mỗi vi phạm chỉ chụp 1 ảnh** (dùng cơ chế `mark_warning_saved`).
5. **Camera loop chạy trong thread riêng**, không block FastAPI event loop.
6. **CORS phải bật** cho `http://localhost:3000`.
7. **Port backend = 8000** (uvicorn mặc định).
8. **Dùng `python-dotenv`** để đọc `.env`, không hardcode credentials.
9. **Tất cả response dùng Pydantic schema**, không trả dict tay.
10. **File `.env` thêm vào `.gitignore`** (đã có sẵn rule `*.env` trong gitignore).

---

# 🔧 LỆNH CHẠY DỰ ÁN

```bash
# 1. Tạo database MySQL
mysql -u root -p -e "CREATE DATABASE httm CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"

# 2. Cài dependencies
cd backend
pip install -r requirements.txt

# 3. Cấu hình .env (sửa DB_PASSWORD)

# 4. Chạy backend
uvicorn main:app --reload --host 127.0.0.1 --port 8000

# 5. Truy cập Swagger
# http://localhost:8000/docs
```
