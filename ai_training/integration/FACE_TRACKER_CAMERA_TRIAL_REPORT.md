# FaceTracker Camera Trial — max_age 4.0s, opt-in only

Ngày: 2026-11-10. Đã triển khai cơ chế opt-in và kiểm thử tự động.
**Chưa chạy camera trong phiên này; chờ người dùng kiểm thử trước khi kết luận trial.**

## Files changed

| File trong `ai_training/integration` | Thay đổi |
|---|---|
| `camera_coordinator_live.py` | Đọc/kiểm tra biến trial trước khi mở camera; dùng tham số đã kiểm tra khi tạo FaceTracker trong AI worker; thông báo console khi trial bật |
| `test_face_tracker_camera_trial.py` | 18 test mới, tái sử dụng harness CH-20/CH-21 và dữ liệu Diagnostic A/B |
| `FACE_TRACKER_CAMERA_TRIAL_REPORT.md` | Báo cáo và hướng dẫn chạy camera |

Đã đọc source và Git status/diff trước khi sửa. Workspace có nhiều file chưa commit;
không ghi đè các thay đổi đó. Đối chiếu SHA-256 của 104 file có sẵn thuộc source Python,
JSON/JSONL và một số file workspace: chỉ `camera_coordinator_live.py` thay đổi.
JSONL camera gốc và `face_tracker_debug_summary.json` giữ nguyên.

Không sửa FaceTracker/PersonTracker, greedy matching, IoU threshold, Association V4,
BehaviorDuration, WarningObserver, model, weights, confidence threshold, camera index,
CH-18/19 hoặc cấu hình WARNING. Không thêm dependency, commit/push hay triển khai CH-22.

## Cấu hình và phạm vi

| `HTTM_FACE_TRACKER_TRIAL_MAX_AGE` | Camera Live FaceTracker |
|---|---|
| Không tồn tại | Baseline `max_age=2.0s` |
| Đúng chuỗi `4.0` | Trial `max_age=4.0s` |
| Giá trị khác | `ValueError` có tên biến, giá trị sai và hướng dẫn bỏ biến để về baseline |

Chỉ chấp nhận literal `4.0`; `4`, `4.00`, `2.0`, `5.0`, chuỗi rỗng, khoảng trắng,
NaN/Infinity và các giá trị khác đều bị từ chối. Muốn chạy baseline phải bỏ biến.
`main()` kiểm tra trước khi mở camera/GUI hoặc tạo thread; AI worker kiểm tra lại
trước khi tạo FaceTracker/workers và trả lỗi/finished sạch nếu được gọi độc lập.

Trial bật in ra:

```text
[FACE TRACKER TRIAL] max_age=4.0s enabled for this Camera Live session; production default remains 2.0s.
```

Biến này chỉ được đọc trong Camera Live. Constructor `FaceTracker()` ở nơi khác vẫn
mặc định 2.0s; Camera Live PersonTracker vẫn 5.0s, IoU 0.3, keep_unmatched=False.
Sleeping, using_phone, turning_back và eating WARNING vẫn mặc định disabled.

## Debug và chi phí

Không đổi schema AI result hoặc JSONL. Diagnostic A đã đọc tham số từ instance:

- `face_tracker.parameters.max_age`: 2.0 hoặc 4.0 theo phiên.
- `parameters.max_age` ở root JSONL là PersonTracker: vẫn 5.0.
- Cùng session giữ frame ID, capture/update timestamps, detections, score, IoU,
  expiry/matched/new và mapping association như trước.

Không tự bật debug hay snapshot. Face debug vẫn cần cả `HTTM_PERSON_DEBUG=1` và
`HTTM_FACE_DEBUG=1`; snapshot chỉ lưu khi có flag riêng. Quota mặc định giữ
10 frame và 20 MiB/session. Test xác nhận session không có ảnh khi snapshot tắt.
Không thêm embedding hoặc dữ liệu cá nhân mới vào log, không SQLite/API/network.

Phần mới đọc environment hai lần lúc khởi động (main và AI worker), không thêm việc
vào mỗi frame hoặc vòng render. Khi debug tắt, test xác nhận không gọi observer,
candidate-IoU diagnostics hoặc writer. Giữ track lâu hơn có thể làm tăng số candidate
IoU của thuật toán hiện có. Chưa đo FPS, latency hoặc chi phí debug trên camera thật.

## Kết quả kiểm thử thực tế

**262/262 test đạt, 0 failure, 0 error, 0 skip** trong lượt chạy hồi quy cuối.
18 test mới nằm trong tổng này; không cộng lại các lượt chạy lặp.

| Nhóm | Đạt |
|---|---:|
| Camera trial mới | 18/18 |
| FaceTracker Diagnostic A và pipeline | 25/25 |
| FaceTracker Diagnostic B và pipeline | 25/25 |
| PersonTracker và Diagnostic A/replay | 39/39 |
| BehaviorDuration, CH-20 pipeline, Snapshot/Overlay | 43/43 |
| WarningObserver, CH-21 pipeline | 47/47 |
| Camera Live/Person integration, Panel/Layout | 22/22 |
| Hybrid person diagnostics | 4/4 |
| Hybrid V7 | 34/34 |
| Association script | 5/5 |
| **Tổng** | **262/262** |

Ngoài tổng unittest/script trên, kiểm tra Hybrid V7 output đã lưu đạt **2/2 ảnh**.
Association V4 thực thi trong các harness tích hợp; script association riêng kiểm tra
association nền. Không chạy script camera thủ công trong bộ tự động.

Các lệnh đã chạy, từ repository root trừ block có `Push-Location`:

```powershell
Push-Location .\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_face_tracker_camera_trial test_face_tracker_parameter_evaluation test_face_tracker_parameter_pipeline test_face_tracker_diagnostics test_face_tracker_diagnostic_pipeline test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# 219 tests OK, 22.507s trong lượt chạy này
..\..\.venv-hybrid\Scripts\python.exe -m unittest test_hybrid_person_diagnostics -q
# 4 tests OK
Pop-Location
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -q
# 34 tests OK
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS 5/5
.\.venv-hybrid\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# 2/2 images
```

### Replay đúng 10 frame camera qua Camera Live

Nguồn chỉ đọc:
`ai_training/runs/person_tracker_debug/session_9cbaa96797d7444a8b0fcc77f849eafd/frames.jsonl`.
SHA-256 giữ nguyên:
`f404c29b9dd749c6351001a81fc761f390c63c40e26eccaede1245fd82688b53`.

Harness dùng FaceTracker thực, cấu hình qua environment thật, AI loop, PersonTracker,
Association V4, duration, warning observer và main/panel thực. Mock chỉ camera/GUI,
workers và clock; chặn model process, SQLite và các đường network của harness.
Giữ nguyên bbox, detection order, raw Student ID, score, captured_at và chính xác
face_tracker_updated_at. Kiểm tra cả Face/Person JSONL replay đều consistent.

| Kết quả | Baseline 2.0s | Trial 4.0s |
|---|---|---|
| Face Track ID theo frame | 1,1,2,3,4,5,6,6,7,7 | 1,1,1,1,1,1,1,1,1,1 |
| ID được tạo / ID switch / expiry | 7 / 6 / 6 | 1 / 0 / 0 |
| Matched với track cũ | 3 | 9 |
| Person P1 / SV001 / matched | 10/10 | 10/10 |

Panel nhận đủ 10 frame và 10 lần render lại. Trang P1 hiển thị T1/SV001 khi trial bật;
các trang person phụ không được association vẫn hiển thị Unknown, không mượn T1.
Đây là replay geometry/recognition/timestamp, không phải chạy lại model: pixel là ảnh
giả lập và behavior camera không có trong JSONL nên để unknown, không tạo evidence.
Không coi processing time giả lập là số đo latency thực tế.

### Các kiểm tra bảo vệ và rủi ro

- 14 giá trị environment sai bị từ chối trước VideoCapture, GUI, thread hoặc Worker.
- Bỏ biến sau trial khôi phục baseline ngay trong cùng process, không cache cấu hình.
- Trial giữ ID ở gap đúng 4.0s; gap 4.001s tạo Face ID mới. Person vẫn P1 ở hai gap này.
- CH-20 vẫn tăng theo captured_at: 0, 1.5, 3.0, 4.5s; render lại không tăng duration.
- Chuỗi 3s/quan sát cho duration 0,3,6,9,12s. Bật warning chỉ trong mock test tạo một
  experimental event ở frame 3; cấu hình production được phục hồi và vẫn all-disabled.
- Unknown, duplicate Student ID, mất association hoặc thiếu face ngắt episode; các
  quan sát hợp lệ sau đó bắt đầu duration 0. Không kế thừa warning episode cũ.
- Đổi SV001 sang SV002 trên cùng Face/Person ID reset duration về 0. Không phát warning
  ngay lúc đổi; event thử nghiệm mới chỉ xuất hiện khi SV002 đủ chuỗi riêng.
- Hai face phân biệt rõ vẫn giữ hai ID khi detection list đảo thứ tự.
- Synthetic replacement cùng vị trí sau 3s: baseline tạo ID mới; trial giữ ID cũ cho
  actor khác. Test chủ động xác nhận **một wrong-actor continuation**, không coi đó là pass
  về nhận dạng. Synthetic crossing vẫn có hai ID switch/hai wrong-actor continuations.
- Harness chặn SQLite/API/network và kiểm tra WARNING production không xuất hiện,
  kể cả duration trên 5s. Không tạo sự kiện vi phạm chính thức.

Không sửa expected results của bất kỳ test cũ nào.

## PowerShell: chạy camera iPhone/iVCam

### Bật trial 4.0s và Face/Person debug, không lưu ảnh

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$env:HTTM_FACE_TRACKER_TRIAL_MAX_AGE = '4.0'
$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_FACE_DEBUG = '1'
Remove-Item Env:HTTM_PERSON_DEBUG_SNAPSHOTS -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Camera index vẫn 2. JSONL mới nằm trong `ai_training/runs/person_tracker_debug/session_*/frames.jsonl`;
không sửa session cũ. Để so sánh, lưu tên session baseline và trial riêng.

### Chạy baseline 2.0s / tắt trial

```powershell
Remove-Item Env:HTTM_FACE_TRACKER_TRIAL_MAX_AGE -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Hai debug flags ở trên vẫn bật để ghi baseline. Muốn chạy không debug:

```powershell
Remove-Item Env:HTTM_PERSON_DEBUG,Env:HTTM_FACE_DEBUG,Env:HTTM_PERSON_DEBUG_SNAPSHOTS -ErrorAction SilentlyContinue
```

Các biến chỉ áp dụng cho process hiện tại và process con; không dùng `setx` hay đổi
cấu hình hệ thống. Trial không tự bật WARNING. Không cần bật snapshot để chạy trial.

## Tiêu chí nghiệm thu camera và giới hạn

1. Trial in đúng thông báo; debug ghi face max_age=4.0 và person max_age=5.0.
   Baseline không có thông báo trial, face max_age=2.0.
2. Hoàn thành 10/10 AI frame. Với chuỗi một face, IoU >=0.3 và age kể từ lần matched
   gần nhất <=4.0s thì giữ Face ID; nếu age >4.0s, expiry và ID mới là đúng thiết kế.
3. Kiểm tra Student ID/association và người thực tế độc lập với độ ổn định Face ID.
   Quan sát riêng quay đầu, hai người gần nhau/crossing và người thay thế cùng vị trí.
   Nếu ID nối sai thì ghi là lỗi identity, không chấp nhận chỉ vì ít ID switch.
4. Unknown, duplicate Student ID, mất association hoặc đổi Student ID phải ngắt chuỗi
   duration/experimental episode theo CH-20/21. Render lại không tăng duration.
5. Layout 70/30, N/P/Q, video live độc lập vẫn hoạt động; không bbox lịch sử trên live,
   không WARNING production, API hoặc SQLite side effects.
6. Bỏ trial env và chạy lại để xác nhận baseline phục hồi.

Rủi ro còn tồn tại: IoU-only không phân biệt được người thay thế cùng bbox trong TTL,
có thể tráo ID khi crossing, hoặc mất track khi bbox nhảy mạnh. Kéo TTL từ 2 lên 4s làm
dài cửa sổ nối nhầm; identity reset chỉ bảo vệ khi recognition/association phản ánh
được thay đổi. Recognition gán sai nhưng vẫn hợp lệ có thể không bị phát hiện.

Chưa xác minh camera scheduling, FPS/latency, chuỗi detected thật, nhiều người thật,
cảnh replacement/crossing thật hoặc chất lượng recognition/model trong phiên này.
**Chỉ nghiệm thu được cơ chế opt-in và automated tests. Chờ kết quả camera trước khi
kết luận trial hoặc đề nghị đổi default. Dừng tại đây, không chuyển CH-22.**
