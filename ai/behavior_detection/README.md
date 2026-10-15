# Student Behavior Detection

Module AI phát hiện hành vi bất thường của sinh viên trong lớp học bằng YOLO11.

Module này thuộc hệ thống giám sát hành vi sinh viên trong lớp học và chịu trách nhiệm phát hiện hành vi trên từng frame camera.

## 1. Hành vi được phát hiện

Model hiện tại hỗ trợ 2 class:

| Class ID | Class name | Ý nghĩa |
|---|---|---|
| 0 | `sleeping` | Sinh viên ngủ/gục trong lớp |
| 1 | `using_phone` | Sinh viên sử dụng điện thoại |

> Hành vi `talking` không được xử lý trực tiếp bằng YOLO trong phiên bản hiện tại. Hành vi này dự kiến được xác định bằng person tracking, tương tác giữa các sinh viên và điều kiện thời gian.

---

## 2. Model

Model hiện tại:

```text
models/best_behavior_v1_4.pt
```

Kiến trúc:

```text
YOLO11n
```

Model V1.4 được fine-tune từ model V1.2 sau khi bổ sung hard-negative samples từ video thực tế nhằm giảm false positive, đặc biệt đối với class `using_phone`.

---

## 3. Dataset

Dataset huấn luyện được tổng hợp từ hai nguồn chính:

- Student Behaviors
- Ambient Intelligence Classroom

Hai class được chuẩn hóa thành:

```text
0: sleeping
1: using_phone
```

Dataset V1.3 dùng để fine-tune V1.4 gồm:

```text
Train: 5267 images
Valid: 532 images
Test : 532 images
```

Trong train có:

```text
1013 negative images
```

Trong đó có 13 hard-negative images được lấy từ các video test thực tế.

Dataset không được lưu trực tiếp trên GitHub do kích thước lớn.

---

## 4. Kết quả đánh giá V1.4

### Validation

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.819 | 0.708 | 0.797 | 0.501 |
| sleeping | 0.755 | 0.705 | 0.752 | 0.464 |
| using_phone | 0.883 | 0.711 | 0.842 | 0.539 |

### Test

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| All | 0.727 | 0.717 | 0.741 | 0.463 |
| sleeping | 0.668 | 0.733 | 0.698 | 0.417 |
| using_phone | 0.786 | 0.701 | 0.784 | 0.509 |

---

## 5. Confidence threshold

Threshold khuyến nghị hiện tại:

```python
SLEEPING_CONF = 0.40
PHONE_CONF = 0.40
```

Threshold được lựa chọn sau khi kiểm tra model trên ảnh và video thực tế nhằm cân bằng giữa false positive và false negative.

---

## 6. Cài đặt

Cài các thư viện cần thiết:

```bash
pip install ultralytics opencv-python
```

Python khuyến nghị:

```text
Python >= 3.10
```

---

## 7. Sử dụng BehaviorDetector

Import:

```python
from behavior_detector import BehaviorDetector
```

Khởi tạo model:

```python
detector = BehaviorDetector(
    model_path="models/best_behavior_v1_4.pt",
    sleeping_conf=0.40,
    phone_conf=0.40
)
```

Phát hiện hành vi trên một OpenCV frame:

```python
detections = detector.detect(frame)
```

---

## 8. Output

Output là một danh sách các detection.

Ví dụ:

```python
[
    {
        "class": "using_phone",
        "class_id": 1,
        "confidence": 0.8732,
        "bbox": [320, 150, 470, 520]
    }
]
```

Trong đó:

| Field | Ý nghĩa |
|---|---|
| `class` | Tên hành vi |
| `class_id` | ID của class |
| `confidence` | Độ tin cậy của model |
| `bbox` | Bounding box `[x1, y1, x2, y2]` |

Nếu không phát hiện hành vi:

```python
[]
```

---

## 9. Tích hợp với Person Tracking

Behavior Detection chỉ xác định:

```text
behavior
confidence
bbox
```

Module Person Detection / Tracking chịu trách nhiệm xác định:

```text
track_id
student_id
person_bbox
```

Luồng tích hợp:

```text
Camera Frame
      |
      +--------------------+
      |                    |
      v                    v
Person Detection      Behavior Detection
      |                    |
      v                    v
Person Tracking       sleeping / using_phone
      |                    |
      +---------+----------+
                |
                v
        BBox Association
                |
                v
       track_id + behavior
                |
                v
        Face Recognition
                |
                v
           student_id
                |
                v
             Backend
```

Ví dụ sau khi ghép behavior với student:

```json
{
    "track_id": 7,
    "student_id": "B21DCCN123",
    "behavior": "using_phone",
    "confidence": 0.87,
    "bbox": [300, 120, 500, 550]
}
```

Việc ghép behavior bbox với person bbox có thể thực hiện bằng IoU hoặc kiểm tra mức độ overlap giữa hai bounding box.

---

## 10. Talking Detection

`talking` không nằm trong model YOLO V1.4.

Hướng xử lý dự kiến:

```text
Person Detection
      ↓
Person Tracking
      ↓
Theo dõi tương tác / hướng đầu / vị trí
      ↓
Duy trì trạng thái theo thời gian
      ↓
Talking > 5 seconds
      ↓
Behavior Event
```

Không nên chỉ sử dụng một frame hoặc chỉ dựa vào việc quay đầu để kết luận sinh viên đang nói chuyện.

---

## 11. Cấu trúc thư mục

```text
behavior_detection/
│
├── models/
│   └── best_behavior_v1_4.pt
│
├── scripts/
│   ├── build_behavior_dataset.py
│   ├── build_behavior_v1_1.py
│   ├── build_behavior_v1_2.py
│   ├── check_dataset_quality.py
│   ├── check_labels.py
│   ├── check_leakage_v1_2.py
│   ├── test_image.py
│   └── test_video.py
│
├── behavior_detector.py
└── README.md
```

Dataset, training outputs, test images và test videos được loại khỏi Git repository thông qua `.gitignore`.

---

## 12. Known Limitations

Model hiện tại vẫn còn một số hạn chế:

- Có thể bỏ sót hành vi `using_phone` khi điện thoại nhỏ hoặc bị che khuất.
- Khả năng phát hiện giảm trong lớp học đông người.
- `sleeping` có thể bị nhầm với một số tư thế cúi đầu.
- Kết quả phụ thuộc vào góc camera, ánh sáng và độ phân giải.
- Model phát hiện hành vi theo frame, chưa tự xác định danh tính sinh viên.
- `talking` cần xử lý bằng tracking và logic thời gian riêng.

Các trường hợp lỗi thực tế nên được thu thập làm hard-negative hoặc additional training samples cho các phiên bản tiếp theo.

---

## 13. Current Version

```text
Model: Behavior Detection V1.4
Architecture: YOLO11n

Classes:
0 - sleeping
1 - using_phone

Recommended confidence:
sleeping    = 0.40
using_phone = 0.40
```