# CH-20 — Behavior Duration + AI Snapshot

Ngày: 2026-11-10.

**Trạng thái: hoàn thành phần triển khai và kiểm thử tự động; chờ kiểm thử camera.**
Chưa nghiệm thu CH-20 và chưa chuyển CH-21. Lượt này không chạy iPhone/iVCam,
InsightFace hoặc inference model thật; không lấy kết quả tổng hợp làm bằng chứng camera.

## Files changed

Tất cả trong `ai_training/integration`:

| File | Thay đổi |
| --- | --- |
| `behavior_duration.py` — mới | Timer ước lượng độc lập, kết quả bất biến, kiểm tra liên kết danh tính và frame. |
| `behavior_overlay.py` — mới | Snapshot giữ pixel riêng, kiểm tra metadata, resize/clipping bbox và vẽ nhãn. |
| `camera_coordinator_live.py` | Nối duration một lần/frame AI; giữ snapshot; hiển thị ba vùng live/snapshot/panel; phân trang. |
| `test_behavior_duration.py` — mới | 17 unit test duration. |
| `test_behavior_overlay.py` — mới | 10 unit test snapshot/overlay. |
| `test_camera_person_tracking.py` | Mở rộng từ 8 lên 14 test tích hợp, giữ các regression tracker có sẵn. |
| `CH20_IMPLEMENTATION_REPORT.md` — mới | Cơ chế, kết quả kiểm thử và hướng dẫn nghiệm thu. |

Đã kiểm tra Git status/diff trước sửa. `.gitignore` có thay đổi sẵn và nhiều file
integration chưa được Git theo dõi. Bản sửa không ghi đè chúng bằng phiên bản từ Git.
FaceTracker, PersonTracker, Association V4, Hybrid V7, worker, config, weights,
camera index 2 và CH-18/CH-19 cũ được giữ nguyên.

## A — Cơ chế duration và AI Panel

`BehaviorDurationTracker` chỉ được gọi trong `ai_loop()` sau khi Association V4
đã được gắn Person Track ID. Một instance thuộc một phiên chạy, với episode
riêng cho `(Person Track ID, behavior)` và Student ID đi kèm episode đó.

- Timestamp là `captured_at` từ queue camera, không phải thời điểm inference xong.
- Chỉ `state="detected"` với Student ID hợp lệ và association duy nhất mới tạo episode.
- Quan sát đầu có `0.0s`; các quan sát tiếp theo trả `captured_at - started_at`.
- `suspected`, `unknown`, `not_detected`, thiếu behavior, mất association, Student ID
  Unknown, mất người hoặc identity mơ hồ đều không có duration: hiển thị `--`.
- Đổi Student ID/Person Track ID tạo episode mới. Không nối lại timer cũ theo Student ID.
- Face Track ID hoặc reference đổi vẫn có thể giữ episode nếu P và Student ID
  tiếp tục được ghép duy nhất trong frame đó.
- Cùng Student ID xuất hiện ở nhiều P, cùng face bị gán nhiều P, reference/match
  trùng hoặc ghép sai reference đều bị chặn gán duration cho sinh viên.
- Khoảng capture vượt 5 giây ngắt episode; đúng 5 giây vẫn được tiếp tục.
- Frame ID trùng trả nguyên kết quả đã tính, không xử lý payload mới. Frame ID cũ
  bị từ chối. Timestamp không tăng xóa episode, trả duration `--`, không kéo mốc
  thời gian lùi; timestamp hợp lệ tiếp theo bắt đầu lại từ zero.
- Kết quả `BehaviorFrame`, `PersonBehaviorView`, `BehaviorDuration` là dataclass
  bất biến. Không cộng thời gian trong render và không tạo warning/violation state.

Panel hiển thị Student ID, Person Track ID, trạng thái liên kết và cả bốn behavior:
`sleeping`, `using_phone`, `turning_back`, `eating`. Duration luôn có dấu `~`, ví dụ
`Using Phone: detected ~3.8s`. Đây là khoảng quan sát ước lượng, không chứng minh
hành vi liên tục giữa các frame hoặc xác nhận vi phạm.

Panel có hai người mỗi trang; N/P đổi trang. Snapshot vẫn vẽ tất cả person boxes
hợp lệ trong kết quả AI đó. Thông tin debug Face Track/Current Face ID vẫn là
thông tin nhận dạng nguồn, không được dùng để ép giữ duration.

## B — Đồng bộ snapshot và bbox

1. Camera Live vẫn ghi `frame.jpg` như trước.
2. Ngay trong AI loop, giải mã JPEG đó thành `snapshot_image`. Cả hai worker
   đọc cùng file không bị ghi đè trong thời gian xử lý frame hiện tại.
3. Sau khi có association, person tracks và duration, tạo `AISnapshot` chứa
   Frame ID, `captured_at`, `BehaviorFrame` và bản sao pixel BGR chỉ đọc.
4. Constructor kiểm tra metadata behavior và snapshot cùng frame. Không giữ
   đường dẫn ảnh làm nguồn render; snapshot vẫn dùng được sau khi file tạm bị xóa.
5. `render_snapshot()` chỉ dùng bbox người từ person tracking, không dùng bbox
   điện thoại/thức ăn của behavior. Bbox được clip theo ảnh nguồn rồi biến đổi
   với đúng kích thước resize và offset letterbox của ảnh.
6. `RenderedSnapshot` giữ metadata. UI kiểm tra lại metadata với result trước
   khi ghép vào bố cục, từ chối cache của frame khác.
7. UI cache snapshot đã vẽ cho result mới nhất. Vòng lặp preview chỉ ghép cache
   và cập nhật tuổi ảnh; không vẽ bbox lên video live, không đọc lại `frame.jpg`.

Ba vùng hiển thị: **LIVE | AI SNAPSHOT - HISTORICAL | AI Panel**. Snapshot có
Frame ID, tuổi ảnh và nhãn ước lượng. Sau hơn 5 giây không có kết quả AI mới kể
từ `completed_at`, Panel có dòng `STALE`; các giá trị duration vẫn đóng băng.
Tuổi ảnh tính từ `captured_at`, bao gồm cả thời gian chờ/xử lý.

Cửa sổ có thể resize, khởi tạo 1280×720. Nội dung snapshot mặc định fit trong
640×480, giữ tỉ lệ và có phần header riêng. Việc resize cửa sổ không thay đổi ảnh
đầu vào model hay camera index. N/P đổi trang người, Q thoát. Giới hạn 10 frame AI
và cơ chế đóng worker được giữ nguyên.

## Tests executed — kết quả thực tế

Giai đoạn A đã chạy đạt 15 unit test duration và 9 test tích hợp trước khi thêm
snapshot. Sau B và rà soát, các bộ kiểm thử cuối cùng bên dưới đều đạt:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_behavior_duration.py -v
# Ran 17 tests; OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_behavior_overlay.py -v
# Ran 10 tests; OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_person_tracking.py -v
# Ran 14 tests; OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_person_tracker.py -v
# Ran 17 tests; OK

.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS: 5/5 association tests

.\.venv\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# TOTAL MATCHED IMAGES: 2/2

.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -v
# Ran 34 tests; OK
```

Tổng: 97 test case đạt và association trên 2/2 ảnh từ predictions đã lưu.
Test artifact dùng vị trí face tổng hợp; không phải inference hoặc camera mới.
Association V4 thật được thực thi trong các test Camera Live với worker giả lập.

Các trường hợp bắt buộc đã bao phủ:

| Yêu cầu | Bằng chứng kiểm thử |
| --- | --- |
| Bốn states, nhiều behavior, nhiều P | Duration unit test; mọi behavior có episode riêng, chỉ detected có số giây. |
| Đổi Student ID/P, Unknown, mất association/người | Episode reset; quay lại không nhận lại duration cũ. |
| Student ID hoặc face bị gán nhiều người | Chặn attribution, reset episode. Có test qua Association V4 thật. |
| Frame trùng/cũ, timestamp không tăng | Không tăng duration; kết quả bất biến hoặc từ chối input cũ. |
| Khoảng capture >5 giây | Reset; kiểm tra cả ranh giới 5.0/5.001 giây. |
| Capture time khác processing time | Test integration dùng capture 10/12/15 và completion 101/104/107; duration cuối là 5 giây. |
| Render lại | Render ở các thời điểm khác nhau vẫn cùng `~7.6s`. |
| Bbox sát biên, resize, letterbox | Kiểm tra tọa độ rectangle sau clip/scale/offset, kể cả bbox ngoài ảnh/không hợp lệ. |
| Snapshot/frame/bbox cùng metadata | Từ chối Frame ID/timestamp sai ở constructor, renderer và cache UI. |
| File tạm ghi đè/xóa | So sánh pixel snapshot với ảnh JPEG cả hai worker giả đọc; render sau khi file đã mất. |
| Live không có bbox lịch sử | Kiểm tra vùng pixel live giữ nguyên và composition không gọi rectangle. |
| Cache hiệu quả | Main loop giả lập ba frame preview chỉ gọi snapshot renderer một lần. |
| Không WARNING/DB/API | Patch chặn SQLite và kết nối socket trong test; duration >5 giây vẫn chỉ là estimate, nhãn không có WARNING. |
| Nhiều người trong Panel | Người thứ ba truy cập được qua trang thứ hai. |

Đã xem bố cục tổng hợp 1640×850 với hai người giả lập, preview và snapshot
khác nhau. Đây chỉ là kiểm tra bố cục, không phải bằng chứng nhận dạng camera.

## Ảnh hưởng hiệu năng

**Dự kiến:** không tăng số lần inference. AI loop thêm một lần giải mã JPEG và
sao chép ảnh mỗi kết quả; UI vẽ/resize snapshot một lần mỗi kết quả mới được hiển
thị, rồi ghép vùng cache vào preview. Panel và tuổi ảnh vẫn được vẽ theo UI.

Ảnh BGR 640×480 chiếm khoảng 0.92 MB cho mỗi bản ảnh; một số bản tạm/cache cùng
tồn tại. Bố cục rộng hơn cũng tăng chi phí ghép ảnh mỗi frame preview. Thời gian
`elapsed` trong log nay bao gồm giải mã snapshot và tạo dữ liệu duration/snapshot;
`face_time` và `hybrid_time` tiếp tục đo riêng worker như trước.

**Số đo thực tế:** chưa đo FPS preview hoặc latency AI trên iPhone sau sửa.
Thời gian chạy unit test và ảnh tổng hợp không được coi là số đo camera. Cần
người dùng đánh giá độ mượt và log thời gian trên cùng máy/độ phân giải.

## Rủi ro còn lại

- Các quan sát cách nhau vài giây có thể bỏ qua lúc hành vi dừng rồi tiếp tục;
  duration vì vậy là ước lượng theo mẫu, không phải thời lượng vi phạm.
- PersonTracker IoU/Association V4 vẫn có giới hạn khi giao nhau, che khuất hoặc
  người thay nhau đứng cùng vị trí giữa hai lần AI.
- Mất detection/association hoặc Unknown làm episode bị chia nhỏ có chủ đích.
- Nhãn cố tìm vị trí trống và có connector về bbox; lớp đông vẫn có thể chồng
  nhãn. Panel phân trang cung cấp cách đọc bổ sung.
- Giới hạn 10 frame AI phù hợp từng lượt kiểm tra ngắn, chưa chứng minh ổn định
  dài hạn. Không thay đổi giới hạn này trong task.

## Manual testing và nghiệm thu CH-20

Kết nối iPhone/iVCam và camera index 2 như các lượt PersonTracker trước. Chạy:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$ch20Log = Join-Path $env:TEMP ('httm-ch20-{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py 2>&1 |
    Tee-Object -FilePath $ch20Log
```

Mỗi tình huống có thể chạy lại một lượt 10 frame AI:

1. Một người: đủ `Completed AI results: 10`; Frame ID trên snapshot và Panel
   khớp nhau. Preview tiếp tục chuyển động; bbox chỉ xuất hiện ở snapshot.
2. Với behavior mà Hybrid thực tế trả `detected`: lần đầu `~0.0s`, những lần
   detected liên tiếp có duration theo chênh lệch capture. Không cộng theo FPS UI.
3. Khi behavior thành suspected/unknown/not_detected, duration phải là `--`.
   Không diễn giải việc model chưa detected thành lỗi timer hoặc ép thay threshold.
4. Nếu chuỗi detected vượt 5 giây, vẫn chỉ hiển thị estimate; không có cảnh báo
   vi phạm, ảnh vi phạm, SQLite hoặc API.
5. Rời camera đến khi có kết quả AI không thấy người rồi quay lại: ID/episode
   mới. Mất nhận dạng hoặc association: Student Unknown, duration `--`.
6. Nhiều người: kiểm tra behavior thuộc đúng bbox/P/Student. Nếu có nhiều ID đã
   đăng ký, kiểm tra các duration độc lập; người chưa nhận dạng vẫn là Unknown.
7. Khi có hơn hai người, dùng N/P để kiểm tra tất cả các trang. Resize cửa sổ,
   kiểm tra bbox vẫn đúng ảnh snapshot. Q thoát sớm nếu cần.
8. Ghi nhận độ mượt preview, Processing/Result age và mọi lần ID/duration bị ngắt.
   Chỉ cần báo Frame ID, P ID, trạng thái/duration và hiện tượng; không cần gửi ảnh mặt.

Chỉ nghiệm thu CH-20 sau khi người dùng xác nhận các điều kiện camera phù hợp,
đúng liên kết sinh viên, duration/reset, bbox/snapshot cùng frame và độ mượt chấp
nhận được. Dừng chờ kết quả camera, không tự chuyển CH-21.
