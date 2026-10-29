# HƯỚNG DẪN CÀI ĐẶT VÀ CHẠY DỰ ÁN HTTM

> Phiên bản hướng dẫn: **CH-20M** (giai đoạn kiểm thử). Hệ điều hành minh họa: **Windows 10/11**, PowerShell. Hướng dẫn này dành cho thành viên nhóm và người đánh giá dự án.

## 1. Giới thiệu

HTTM là dự án thử nghiệm sử dụng camera và AI để nhận diện khuôn mặt, theo dõi nhiều người và phát hiện các hành vi `sleeping`, `using_phone`. Chương trình sử dụng OpenCV, InsightFace, YOLO và SQLite. Khi hành vi được xác nhận đủ lâu (ngưỡng hiện tại khoảng 5 giây), hệ thống có thể lưu cảnh báo kèm ảnh.

**Lưu ý:** CH-20M chưa phải bản hoàn thiện. Trong thử nghiệm hai người, hệ thống đã nhận diện được sinh viên `001` nhưng YOLO chưa phát hiện ổn định hành vi `using_phone` của sinh viên này. Không dùng kết quả thử nghiệm để tự động kết luận vi phạm của người thật.

## 2. Yêu cầu chuẩn bị

- Máy tính Windows có camera hoặc điện thoại dùng làm camera; khuyến nghị dùng **Python 3.11 (64-bit)** vì đây là phiên bản môi trường đã sử dụng để thử nghiệm.
- Git, PowerShell, kết nối Internet để cài thư viện và tải model InsightFace lần đầu.
- Điện thoại có **DroidCam** và ứng dụng/driver tương ứng trên máy tính nếu dùng điện thoại làm camera.
- Quyền sử dụng hình ảnh của người tham gia thử nghiệm; chỉ dùng dữ liệu khuôn mặt khi có sự đồng ý.
- Các file mã nguồn của nhóm và checkpoint YOLO `best.pt`.

## 3. Tải mã nguồn

Mở PowerShell ở thư mục bạn muốn lưu dự án:

```powershell
git clone <URL_REPOSITORY_CUA_NHOM>
cd <THU_MUC_DU_AN>
```

Thay hai giá trị trong dấu `<...>` bằng URL và tên thư mục thực tế. Nếu đã có code, mở PowerShell tại thư mục gốc của dự án (nơi có thư mục `ai`).

**Lưu ý về nhánh Git:** Nhóm đang phát triển trên nhiều nhánh. Hãy xác nhận với nhóm nhánh đã chứa CH-20M; không mặc định rằng nhánh `main` đã có file này. Có thể kiểm tra:

```powershell
git branch -a
git status
```

## 4. Tạo môi trường Python

Tại thư mục gốc của dự án:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version
python -m pip install --upgrade pip
```

Nếu PowerShell chặn `Activate.ps1`, bạn **không bắt buộc phải kích hoạt**; có thể dùng trực tiếp `\.venv\Scripts\python.exe` trong tất cả lệnh bên dưới.

## 5. Cài thư viện

**Ưu tiên dùng file `requirements.txt` của nhóm nếu có và đã được kiểm thử:**

```powershell
python -m pip install -r requirements.txt
```

Nếu repository **chưa có** `requirements.txt`, có thể cài các thư viện chính được CH-20M sử dụng:

```powershell
python -m pip install ultralytics opencv-python numpy insightface onnxruntime
```

Các thư viện có thể yêu cầu công cụ biên dịch hoặc phiên bản tương thích trên từng máy. Nếu cài InsightFace thất bại, hãy gửi nguyên thông báo lỗi cho người phụ trách môi trường; không tự ý đổi sang Python 3.13.

Kiểm tra nhanh:

```powershell
python -c "import cv2, numpy, insightface, onnxruntime, ultralytics; print('Dependencies OK')"
```

## 6. Kiểm tra cấu trúc file

Tối thiểu, các file phục vụ CH-20M cần được đặt đúng vị trí:

```text
<THU_MUC_DU_AN>/
├── .venv/                              # Tạo trên từng máy, không commit
└── ai/
    └── behavior_detection/
        ├── test_behavior_tracking_ch20m.py
        ├── test_behavior_tracking_ch18.py
        ├── behavior_state_manager.py
        ├── behavior_timer.py
        ├── student_face_recognizer.py
        ├── warning_database.py
        ├── models/
        │   └── best.pt
        ├── data/
        │   └── students_faces.db         # Dữ liệu cá nhân: chuẩn bị riêng
        └── logs/                         # Được tạo khi chạy
```

CH-20M **import trực tiếp** các module CH-18, state manager, timer, recognizer và database; thiếu một trong các file này có thể gây `ModuleNotFoundError`.

Kiểm tra checkpoint:

```powershell
Test-Path ai\behavior_detection\models\best.pt
```

Kết quả phải là `True`. Checkpoint hiện dùng của nhóm là `best.pt`, có hai lớp: `sleeping` và `using_phone`. Nếu chưa có, lấy checkpoint chính thức từ người phụ trách mô hình/nhánh Git chứa model; không thay bằng model YOLO bất kỳ.

## 7. Chuẩn bị dữ liệu khuôn mặt và SQLite

- Dữ liệu sinh viên đã đăng ký được lưu trong `ai/behavior_detection/data/students_faces.db` ở môi trường kiểm thử hiện tại.
- Máy mới có thể chưa có cơ sở dữ liệu này. Cần thực hiện quy trình đăng ký khuôn mặt của nhóm hoặc dùng dữ liệu thử nghiệm được cấp quyền.
- File đăng ký từng được dùng trong dự án: `register_student_face.py` (kiểm tra file có tồn tại trong repository trước khi chạy).
- Không đưa database chứa khuôn mặt thật, ảnh cảnh báo hoặc log cá nhân lên GitHub công khai.

Nếu chỉ muốn kiểm tra kết nối camera và YOLO, hãy trao đổi với nhóm về bài kiểm thử không cần nhận diện danh tính. Không nên sao chép dữ liệu khuôn mặt của người khác để chạy thử.

## 8. Kết nối camera

1. Mở DroidCam trên điện thoại và phần mềm/driver tương ứng trên máy tính.
2. Kết nối theo hướng dẫn của DroidCam, kiểm tra hình ảnh đã hiển thị trên máy tính.
3. Đảm bảo không có Zoom/Teams hoặc chương trình khác chiếm camera.
4. CH-20M dùng hằng số `CAMERA_ID` được import từ `test_behavior_tracking_ch18.py`. Trên máy phát triển, giá trị đã sử dụng là **1**; trên máy khác có thể là **0**, **1** hoặc giá trị khác.

Kiểm tra camera bằng lệnh (thử `1`, nếu không mở được hãy thử `0`):

```powershell
python -c "import cv2; c=cv2.VideoCapture(1); print('Camera opened:', c.isOpened()); c.release()"
```

Nếu muốn thay camera, mở `ai/behavior_detection/test_behavior_tracking_ch18.py` và chỉnh `CAMERA_ID` theo camera đã kiểm tra. **Không sửa nhầm trong CH-20M**, vì file đó import giá trị từ CH-18.

## 9. Chạy CH-20M

Chạy tại **thư mục gốc dự án**:

```powershell
.\.venv\Scripts\python.exe -m py_compile ai\behavior_detection\test_behavior_tracking_ch20m.py
.\.venv\Scripts\python.exe ai\behavior_detection\test_behavior_tracking_ch20m.py
```

Lần chạy đầu, InsightFace có thể tải model `buffalo_l`; hãy chờ hoàn tất. Khi thành công, terminal sẽ hiển thị các dòng tương tự:

```text
InsightFace ready
YOLO classes: {0: 'sleeping', 1: 'using_phone'}
DroidCam connected | CH-20M started
```

Phím điều khiển (khi cửa sổ OpenCV đang được chọn):

| Phím | Tác dụng |
|---|---|
| `Q` | Thoát, giải phóng camera |
| `S` | Lưu ảnh chụp màn hình kiểm thử |

## 10. Xem log và kết quả

Log chẩn đoán:

```text
ai/behavior_detection/logs/behavior_tracking_ch20m.log
```

Xem 100 dòng cuối:

```powershell
Get-Content ai\behavior_detection\logs\behavior_tracking_ch20m.log -Tail 100
```

Ảnh cảnh báo được lưu trong:

```text
ai/behavior_detection/logs/warnings/
```

Các bản ghi cảnh báo được lưu qua module `warning_database.py` vào SQLite. **Đường dẫn database cảnh báo cần kiểm tra trong file `warning_database.py` của bản code hiện tại**, vì có thể khác database đăng ký khuôn mặt.

## 11. Lỗi thường gặp

| Hiện tượng | Cách kiểm tra |
|---|---|
| `No module named ...` | Đảm bảo đang dùng Python trong `.venv`, đã cài thư viện, đủ các module nội bộ |
| `Missing YOLO model` | Kiểm tra `models/best.pt` và `MODEL_PATH` trong CH-18 |
| `Cannot open DroidCam at index ...` | Mở DroidCam, kiểm tra kết nối, thử chỉ số camera khác trong CH-18 |
| InsightFace tải model lâu | Kiểm tra mạng và chờ lần tải đầu tiên |
| Không nhận ra Student ID | Kiểm tra dữ liệu đăng ký khuôn mặt, ánh sáng, góc nhìn; không tự hạ ngưỡng nhận diện |
| YOLO không phát hiện hành vi | Kiểm tra camera, vị trí trong khung hình, độ sáng, confidence và log chẩn đoán |
| Không có cảnh báo sau 5 giây | Xem `YOLO DETECTION`, `BEHAVIOR ASSOCIATION`, `BEHAVIOR TIMER` và trạng thái Track ID |
| Cửa sổ video giật/lag | Đóng ứng dụng nặng, giảm tải máy; báo người phụ trách trước khi thay tham số mô hình |

## 12. Kịch bản kiểm thử cho thành viên mới

1. Chạy chương trình và kiểm tra cửa sổ camera.
2. Cho **hai người đã đồng ý tham gia** ngồi trong khung hình; một người đã đăng ký, một người chưa đăng ký.
3. Kiểm tra Track ID và Student ID có hiển thị hợp lý.
4. Cho từng người thử hành vi `using_phone` riêng biệt trong 7–10 giây.
5. Quan sát vùng YOLO, cảnh báo, ảnh lưu và log.
6. Ghi lại trường hợp nhận diện sai hoặc không phát hiện, kèm thời điểm trong log (không chia sẻ ảnh khuôn mặt ngoài nhóm được phép).

**Kết quả mong đợi:** Chương trình hoạt động, có thể phát hiện và lưu cảnh báo khi điều kiện được đáp ứng. **Giới hạn hiện tại:** Việc phát hiện `using_phone` cho sinh viên `001` chưa ổn định trong thử nghiệm hai người; không coi việc không có cảnh báo là bằng chứng không có hành vi.

## 13. Quy tắc làm việc nhóm

- Mỗi người tạo `.venv` riêng, không commit `.venv/`, log, ảnh cảnh báo hoặc database cá nhân.
- Không sửa checkpoint `best.pt` tùy tiện; ghi rõ phiên bản model khi kiểm thử.
- Trước khi chạy hoặc merge code, kiểm tra nhánh Git và các thay đổi chưa commit (`git status`).
- Khi báo lỗi, gửi lệnh chạy, thông báo lỗi, thời điểm và phần log liên quan; che thông tin cá nhân nếu cần.
- CH-20M là **mốc dừng hiện tại** để báo cáo tiến độ. CH-20N chưa được triển khai.

---

**Checklist trước khi báo chạy thành công:** Python 3.11 ✓ · Thư viện ✓ · Module nội bộ ✓ · `best.pt` ✓ · Camera ✓ · Dữ liệu đăng ký hợp lệ (nếu cần nhận diện) ✓ · Chạy CH-20M ✓ · Xem log ✓.
