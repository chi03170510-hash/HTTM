# CH-20 — Kiểm thử tích hợp duration bằng dữ liệu giả lập

Ngày: 2026-11-10.

**Kết quả cuối:** 16/16 test harness mới và 105/105 test hồi quy đạt;
**121 test đạt, 0 failure, 0 error**. Association trên predictions đã lưu đạt 2/2 ảnh.
Đây là kiểm thử phần mềm bằng dữ liệu tổng hợp, không phải bằng chứng model/camera mới.

## 1. Files changed

Chỉ thêm hai file trong `ai_training/integration`:

- `test_ch20_duration_pipeline.py`: harness độc lập và 16 test tích hợp.
- `CH20_DURATION_PIPELINE_REPORT.md`: báo cáo này.

Không sửa file production, test cũ, weights, threshold, camera index, CH-18/CH-19
hoặc dependency. Không thêm chế độ giả lập/flag/import vào Camera Live production.

## 2. Cách tạo dữ liệu giả lập

`observation()` tạo phản hồi ở ranh giới hai AI worker:

- Face ID reply có bbox tổng hợp, `student_id="SV001"` và score giả lập.
- Hybrid reply có person reference, bbox và bốn behavior theo schema IMAGE_ONLY.
- `using_phone` được điều khiển bằng state của từng test. Các behavior khác mặc
  định là sleeping/suspected, turning_back/unknown và eating/not_detected.
- Các ảnh nguồn chỉ là mảng BGR 480×640 tổng hợp, không có ảnh sinh viên hay dữ
  liệu mặt thật. JPEG tạm được ghi/đọc và dọn dẹp bởi AI loop hiện có.
- Fixture không cung cấp Person Track ID hoặc ép Student ID vào tracker. P1/P2
  được PersonTracker thật sinh ra từ bbox và thời gian cập nhật.
- Đồng hồ completion giả lập cách 1.5 giây, tách biệt với `captured_at`. Điều này
  cho phép kiểm tra khoảng capture >5 giây mà P1 chưa hết hạn do completion gap.

Đây là điều khiển phản hồi model trong test, không khẳng định model đã nhận dạng
một hành vi thực tế hoặc thay confidence threshold để tạo detected.

## 3. Đường đi qua pipeline thật

Harness thực hiện hai bước tuần tự, không chạy model thật:

```text
Ảnh tổng hợp + captured_at → frame_queue
  → ai_loop() thật
  → hai worker reply giả lập
  → FaceTracker thật → Association V4 thật → PersonTracker thật
  → BehaviorDuration thật → snapshot đúng frame → result_queue

Các result vừa tạo → result_queue của main() thật, từng result một
  → main() nhận result và cập nhật latest_result
  → draw_ai_panel() thật → OpenCV putText thật trên ảnh tổng hợp
```

Chỉ thay Worker, camera/GUI/thread giao nhận, đồng hồ và bộ đếm frame cho các ca
fault injection. Không thay Association V4, FaceTracker, PersonTracker,
BehaviorDuration hoặc logic dựng Panel.

`PipelineHarness._run_main()` phát mỗi AI result qua result queue mà `main()`
đang dùng, rồi cho thêm một lần render không có AI result mới. Các ca nhiều người
dùng chính phím N/P của main để đọc cả hai trang; Q kết thúc.

Nhãn được ghi nhận từ lời gọi `cv2.putText` có `wraps` nên OpenCV vẫn vẽ pixel
thật. Harness kiểm tra Panel có pixel chữ. Không gọi trực tiếp BehaviorDuration
trong test để tạo kết quả đưa lên UI, và không tự xây result có sẵn duration.

Worker reply giữ đúng Frame ID/path của request. Snapshot được so pixel với
JPEG cả hai worker giả lập đã đọc; file tạm không còn khi UI bắt đầu replay.

## 4. Kết quả chuỗi detected theo từng frame

Test `test_exact_detected_sequence_passes_real_pipeline_and_main_panel` đạt:

| Frame | captured_at | Student | Person Track | Association | Nhãn thực tế gửi tới OpenCV Panel |
| --- | --- | --- | --- | --- | --- |
| 1 | 100.0 | SV001 | P1 | matched | `Using Phone: detected ~0.0s` |
| 2 | 101.5 | SV001 | P1 | matched | `Using Phone: detected ~1.5s` |
| 3 | 103.0 | SV001 | P1 | matched | `Using Phone: detected ~3.0s` |
| 4 | 104.5 | SV001 | P1 | matched | `Using Phone: detected ~4.5s` |

Association V4 được gọi đủ 4 lần. Có 8 lần vẽ qua main: mỗi result một lần mới
và một lần render lại. Tuổi kết quả thay đổi giữa hai lần vẽ nhưng nhãn duration
giữ nguyên. Sleeping, Turning Back và Eating đều hiển thị `--`.

## 5. Kết quả reset và các ranh giới

Ký hiệu `--` là không có duration, không phải zero; sau khi đủ điều kiện trở lại
thì episode mới bắt đầu từ `~0.0s`.

| Tình huống | Kết quả qua pipeline và Panel |
| --- | --- |
| detected → suspected → detected | `~0.0, ~1.5, --, ~0.0`; P1/SV001 giữ nguyên. |
| detected → unknown → detected | `~0.0, ~1.5, --, ~0.0`. |
| detected → not_detected → detected | `~0.0, ~1.5, --, ~0.0`. |
| Đổi Person Track | Di chuyển bbox để IoU không khớp: tracker thật sinh P2; duration `~0.0, ~1.5, ~0.0, ~1.5`. |
| Đổi Student ID | SV001 → SV002 trên cùng P1; episode mới từ `~0.0s`. |
| Mất association | Đặt face ra ngoài person bbox; V4 trả không có match, Student Unknown/duration `--`; trở lại từ zero. |
| Student Unknown | Dù stable face label còn SV001, nhận dạng hiện tại Unknown vẫn ngắt duration; biết lại thì từ zero. |
| Hai người cùng Student ID | V4 vẫn trả hai match hình học; lớp duration chặn cả hai identity. Hai trang P1/P2 đều Student Unknown và `detected --`. Hết trùng thì bắt đầu lại từ zero. |
| Capture gap >5 giây | Capture 101.5 →107.0: duration reset về zero trong khi P1 vẫn giữ nguyên. |
| Frame ID trùng, cùng capture | Trả lại cùng BehaviorFrame; chuỗi `~0.0, ~1.5, ~1.5, ~3.0`. |
| Frame ID trùng nhưng capture khác | Snapshot validation từ chối metadata không nhất quán; result lỗi được AI loop trả về, không đưa result sai lên Panel. Đây là ca lỗi đầu vào có chủ đích. |
| Timestamp không tăng | Cùng timestamp hoặc lùi timestamp: duration `--`, timestamp_valid=False; lần hợp lệ tiếp theo bắt đầu zero. |
| Người biến mất rồi quay lại | Frame trống xóa episode; tracker sinh P2 khi quay lại, duration zero. |
| Gap đúng 5 giây | Được tiếp tục; tổng duration lên `~6.5s` vẫn không WARNING/DB/API. |
| Hai Student khác nhau | SV001/P1 và SV002/P2 có duration độc lập; frame cuối tương ứng `~3.0s` và `~1.5s` trên hai trang thực tế. |

Frame ID duplicate được tiêm bằng bộ đếm vòng lặp chỉ trong harness; production
bình thường tự tăng Frame ID. Kiểm tra duplicate có capture khác xác nhận pipeline
từ chối dữ liệu sai, không phải có lỗi test hoặc tự đổi semantics duration.

## 6. Hiển thị và side effects

- Panel thực tế nhận đúng `Student: SV001`, `Person Track: P1`, Frame ID và các
  nhãn duration ở bảng trên; output có kích thước 1280×720.
- Render lặp chỉ đổi result age, không đổi giá trị duration bất biến.
- Các state không detected có `--`; association/identity không hợp lệ cũng có `--`
  ngay cả khi behavior đầu vào vẫn detected.
- Không có WARNING trong nhãn UI hoặc stdout/stderr được thu từ pipeline.
- Harness chặn `subprocess.Popen`, SQLite connect, socket connect/create_connection
  và URL open; gọi nhầm các side effect này sẽ làm test thất bại. Không chạy model,
  mở camera/GUI thật, gửi API hay ghi SQLite trong kiểm thử.

## 7. Regression tests đã chạy

Kết quả cuối:

| Bộ test | Số case | Đạt | Failure | Error |
| --- | --- | --- | --- | --- |
| Harness duration pipeline mới | 16 | 16 | 0 | 0 |
| BehaviorDuration | 17 | 17 | 0 | 0 |
| Snapshot/Overlay | 10 | 10 | 0 | 0 |
| Camera Live/Panel/Association V4 | 14 | 14 | 0 | 0 |
| Bố cục Live 70%/Panel 30% | 8 | 8 | 0 | 0 |
| PersonTracker | 17 | 17 | 0 | 0 |
| Association độc lập | 5 | 5 | 0 | 0 |
| Hybrid V7 | 34 | 34 | 0 | 0 |
| **Tổng** | **121** | **121** | **0** | **0** |

Thêm test association từ predictions cục bộ đã lưu: **2/2 ảnh đạt**, dùng vị trí
face tổng hợp. Không tính hai ảnh thành lần kiểm thử camera hoặc inference mới.

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_ch20_duration_pipeline.py -v
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_behavior_*.py -v
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_person_tracking.py -v
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_live_layout.py -v
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_person_tracker.py -v
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
.\.venv\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -v
```

Lần chạy đầu trong quá trình tạo harness có 16 case với 17 failure (một case có
hai subtest). Nguyên nhân là NumPy lazy-load `numpy.testing` gọi truy vấn phiên
bản Windows qua subprocess và đụng guard chặn tiến trình. Đã sửa riêng harness
dùng phép so sánh pixel `numpy.array_equal` tương đương, không nới guard và không
sửa production. Sau sửa, harness chạy 16/16 đạt như bảng kết quả cuối.

## 8. Production code

Không thay đổi production code. Không thêm dependency injection hook hoặc chế độ
fake vào production vì patch tại ranh giới worker và camera/GUI đã đủ cho test.
Dữ liệu giả chỉ được tạo khi gọi harness/test; production không import file test.

Đối chiếu hash các file có sẵn với baseline để xác nhận không đổi source/model/
config/test cũ. Không commit hoặc push; dừng ở CH-20.

## 9. Giới hạn camera thực tế

Theo thông tin người dùng: camera đã xử lý 10/10 frame, nhận SV001, giữ P1 và
association matched 10/10; chưa có chuỗi detected liên tiếp. Đây là thông tin
người dùng cung cấp, không phải camera được chạy lại trong lượt kiểm thử này.

Harness chưa xác nhận độ chính xác model khi phát hiện using_phone, duration tăng
trên camera từ model thật, tracking trong lớp đông/che khuất/giao nhau, FPS hoặc
độ trễ/thread scheduling trên thiết bị. Hai bước replay là tuần tự với thời gian
ảo, không thay thế thử nghiệm đồng thời của preview và inference thật.

## 10. Đánh giá nghiệm thu

**Đủ bằng chứng nghiệm thu chức năng phần mềm duration → Panel trong phạm vi
quy tắc CH-20 được kiểm thử:** chuỗi detected, reset, liên kết danh tính, chống cập
nhật trùng, hiển thị và không phát sinh cảnh báo/DB/API đều đạt qua pipeline tích hợp.

**Chưa đủ để nghiệm thu toàn bộ CH-20 trên camera/model thật.** Điều kiện còn lại
là người dùng quan sát và xác nhận chuỗi detected từ model thật cùng duration trên
UI. Không diễn giải kết quả giả lập thành kết quả camera hoặc tự chuyển CH-21.
