# AI3 — Camera Live stability, offline

Ngày kiểm tra: 2026-11-11. Chỉ AI; không mở camera, không inference/train,
không đổi model/config, không cài hoặc gỡ package. WARNING vẫn disabled;
Identity Shadow không cấp SID production, `T_hold=None`.

Bằng chứng trong thư mục mới:
`ai_training/runs/camera_live_ai3_validation_ce248929b1014353a1b3723109ffb5a3/`.
Không ghi đè kết quả cũ. `mock_stress.json` là phép đo độc lập bằng `.venv`;
`final_regression/full_offline_regression.json` là regression trên source cuối
bằng `.venv-hybrid`; bản 701 test trước đó được giữ riêng ở thư mục gốc artifact.

## 1. Audit vòng đời

| Tài nguyên | Contract thực tế / vị trí | Kết luận offline |
| --- | --- | --- |
| VideoCapture | `camera_coordinator_live.main`, dòng 446–452, 567–591: DSHOW → MSMF; release lần mở thất bại và trong finally | Backend, Q, read failure được mock; chưa xác minh driver iVCam |
| Face/Hybrid worker | `camera_coordinator_live.ai_loop`, dòng 91–103, 295–306; `camera_live_runtime.LiveWorker`, dòng 79–157 | Hai worker tuần tự; cleanup độc lập, terminate/wait rồi kill/wait; đã sửa hai lỗi dưới đây |
| Queue camera/kết quả | `camera_coordinator_live.main`, dòng 454–455, 492–505; `camera_live_runtime.publish_latest`, dòng 27–42 | 1 frame đầu vào / 3 kết quả; UI bỏ backlog, không bỏ lần cập nhật DurationTracker |
| IPC stdout | `LiveWorker._read_output`, dòng 101–115; inherited protocol `camera_coordinator.Worker.process_frame`, dòng 50–94 | Sau sửa: tối đa 3 item gồm EOF; stdout vượt protocol gây lỗi rõ ràng và dừng phiên |
| JPEG/sidecar tạm | `ai_loop`, dòng 105–135; `bbox_visual_metadata.write_source` | Một TemporaryDirectory; dùng lại `frame.jpg` và tên sidecar cố định; đã đo 1 hoặc 3 file đang tồn tại khi Person Debug OFF |
| AISnapshot | `behavior_overlay.AISnapshot.__post_init__`, dòng 27–38; `ai_loop`, dòng 247–265 | Copy pixels readonly theo observation; queue/cache không giữ lịch sử không giới hạn |
| Overlay/cache | `person_snapshot_overlay.PersonSnapshotOverlay`, dòng 39–44, 72–124 | Một render cache và một source snapshot; key bao gồm observation/metadata; repaint chỉ đổi nhãn tuổi, không gọi DurationTracker |
| Identity Shadow | `ai_loop`, dòng 82–83; `identity_continuity_diagnostics`, dòng 146–160, 191–248 | Continuous budget 10 frame; tối đa 100 frame/1 MiB; JSONL mở trong context manager; giữ một previous behavior frame; hết budget không dừng AI |
| Person Debug | `person_tracker_diagnostics.DiagnosticConfig`, dòng 25–52; `record`, dòng 279–296 | Mặc định 10 frame/20 MiB; giới hạn cấu hình 100 frame/100 MiB; snapshot diagnostic là dữ liệu lưu có chủ ý, không phải file tạm |
| Tracker | `person_tracker.update`, `recognition/face_tracker.py` | Xóa track expired; Face labels deque(maxlen=5); không sửa tracker trong AI3 |
| OpenCV windows | `main`, dòng 479, 543, 567–578; `PersonSnapshotOverlay.show` | GUI/overlay lỗi được mock; destroyAllWindows và join AI thread; không tạo GUI thật trong test |

Face worker load model/database một lần (`face_id_stream_worker.main`, dòng 24–28).
Hybrid worker load model một lần (`hybrid_v7_stream_worker.HybridV7Stream.__init__`,
dòng 49–80); close MediaPipe face/object resources ở dòng 196–203. Parent hiện
terminate subprocess khi dừng, nên không bảo đảm `finally` Python của child được
chạy trên Windows; OS giải phóng tài nguyên khi process kết thúc. Chưa đo native
ONNX/Torch/MediaPipe/CUDA allocator hoặc bộ nhớ driver.

## 2. Hai lỗi đã tái hiện và sửa tối thiểu

1. **Reader thread không start được sau Popen:** constructor cũ ném exception
   trước khi `ai_loop` nhận ownership. Process và hai pipe không được cleanup.
   Mock trước sửa xác nhận `terminate=False`, `stdin_closed=False`,
   `stdout_closed=False` (`audit_reproduction_before.json`).
   `LiveWorker.__init__` nay tự close resource nếu constructor thất bại; không
   join một thread chưa start. Không sửa legacy `camera_coordinator.Worker`.
2. **Queue IPC không giới hạn:** reader cũ đưa 5.000 dòng stdout bất thường và
   EOF vào queue, thành 5.001 item, maxsize=0. LiveWorker nay dùng queue3 và từ
   chối backlog ngoài protocol. Đây là giới hạn IPC, không phải threshold AI.
   Không âm thầm bỏ response hợp lệ; frame/image validation và worker timeout
   90 giây của protocol giữ nguyên.

Test bổ sung cho phần IPC mới tái hiện lịch reader bắt đầu sau khi
`process_frame` đã bọc wait queue. Reader nay luôn dùng `_response_queue` gốc;
không phụ thuộc thời điểm thread được schedule. Reproduction và test sau sửa
được lưu; đây là bảo vệ tương tác của thay đổi AI3, không phải lỗi thứ ba đã có
trước AI3.

Chỉ file production `camera_live_runtime.py` được sửa. Không sửa `main`, worker
AI, live preview, panel, overlay, tracker, association, duration hoặc warning.

## 3. Mock stress thực chạy

Chạy real `ai_loop`, tracker/V4/DurationTracker/WarningObserver, JPEG encode/decode,
AISnapshot, sidecar và renderer trên ảnh synthetic 320×240, hai người. Camera và
workers là fake, không load model. UI consumer được giữ chậm để mailbox đầy.
Capture timestamp giả lập tăng 1,5s/observation; đây không phải phiên camera
chạy 1.000 frame theo wall clock hoặc phép đo inference latency.
Không lưu call history từng frame trong fake camera/worker. RSS lấy Windows
GetProcessMemoryInfo/WorkingSetSize, không thêm dependency.

| Observations mỗi lượt | Tổ hợp Overlay × Shadow | Queue input/result tối đa | Snapshot tối đa | Cache | File tạm còn sau cleanup |
| --- | --- | --- | --- | --- | --- |
| 100 | OFF/OFF, ON/OFF, OFF/ON, ON/ON | 1 / 3 | 4 OFF, 5 ON | 0 OFF, 1 ON | 0 |
| 500 | OFF/OFF, ON/OFF, OFF/ON, ON/ON | 1 / 3 | 4 OFF, 5 ON | 0 OFF, 1 ON | 0 |
| 1.000 | OFF/OFF, ON/OFF, OFF/ON, ON/ON | 1 / 3 | 4 OFF, 5 ON | 0 OFF, 1 ON | 0 |

12 lượt = 6.400 observations. Sáu profile bổ sung: slow worker (100 obs), timeout,
worker exit, exception giữa phiên, capture read failure (mỗi lỗi hoàn tất 39 obs),
stop event sau 40 obs. Tổng cả 18 profile: 6.696 observations.

Tất cả profile cleanup mock workers/capture, xóa temp directory; snapshot còn
trong mailbox sau vòng AI là 2, sau drain/release là 0; không duplicate frame ID,
không warning event. Shadow ON ghi 10 dòng/36.823 bytes rồi ngừng ghi đúng budget.
SHA256 toàn bộ chuỗi input DurationTracker giống nhau giữa bốn tổ hợp của từng
độ dài. Q thực tế được kiểm tra ở real `main` + AI thread với camera/workers/GUI
mock trong `test_camera_live_continuous` (30 test), không phải coi stop event ở
stress adapter là kiểm tra phím Q.

RSS samples trong 12 lượt: **43,305–47,898 MiB**; RSS cuối lượt **42,699–44,676
MiB**. Ở bốn lượt 1.000 obs, RSS cuối **44,316–44,676 MiB**. Đây là một process
test dùng lại allocator, có warm-up; không đặt ngưỡng RSS mới và không dùng số
này để kết luận inference không leak, FPS, hoặc độ ổn định camera thật.

Stress được chạy lại trong full regression cuối bằng `.venv-hybrid`: **8/8 test
PASS**, cùng 18 profile/6.696 observations, cùng queue/cache/snapshot limits và
0 file/snapshot còn lại sau release. RSS samples ở suite này **270,414–273,645
MiB**. Suite đã import các thư viện Hybrid và chạy các test trước đó, khác process
đo độc lập; không gộp hai phép đo thành một chuỗi RSS hoặc quy tăng bộ nhớ cho
inference. Chi tiết trong `final_regression/mock_stress.json`.

Môi trường stress độc lập: Windows 10 build 19045 x64, Python 3.11.9 `.venv`,
NumPy 2.4.6 / OpenCV 4.11.0. Thông tin interpreter/package/build riêng lưu trong preflight và
`opencv_runtime_audit.json`.

## 4. Hai WARN preflight

**OpenCV:** `.venv-hybrid` có `opencv-python`, `opencv-python-headless` và
`opencv-contrib-python`, đều **5.0.0.93**, cùng site-packages. Import cv2 thực
thành công, `cv2.__version__=5.0.0`, build `GUI: WIN32UI`, `Win32 UI: YES`.
Các distribution chia sẻ namespace/files, nên cập nhật/gỡ một gói có thể thay
binary đang được dùng. Chưa kiểm tra window/driver thật; không tự gỡ/cài lại.
Coordinator dùng `.venv` riêng; Hybrid worker không trực tiếp mở camera/window.

**Duplicate JSON:** `config/hybrid_v7.json` đọc bằng `json.loads`, last key wins:

| Key | Giá trị đầu (dòng) | Giá trị cuối / hiệu lực (dòng) |
| --- | --- | --- |
| food_mouth_distance_person_height | 0.20 (9) | 0.16 (70) |
| draw_person_boxes_debug | false (26) | true (58) |
| draw_landmarks_debug | false (28) | true (59) |

Selector checkpoint độc lập trong `src/behavior_checkpoint.py`; không có key
confidence inference/Sleeping/Using Phone bị lặp. Food distance được đọc bởi
`hybrid_v7.assess_food_head` dòng 649 trong bước V6, sau đó V6.1/V7 tính lại eating
bằng `assess_food_head_v6_1` (dòng 458 và 224). Hai draw flags dùng ở image runner
(dòng 146/190), không điều khiển live overlay hiện tại. Không thể xóa bản trùng
mà giữ giá trị đầu vì sẽ đổi parsed config. Không thấy bằng chứng cần đổi config
trong AI3; SHA256 giữ nguyên:
`bd0de5edea222a1161ac577697eb64a87504413216a9487429770fc5be967f4a`.

## 5. Kiểm thử và cách chạy lại offline

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'
$env:PYTHONPATH = 'ai_training/integration;ai_training'
& .\.venv\Scripts\python.exe -B -m unittest test_camera_live_stability test_camera_live_continuous -q
& .\.venv-hybrid\Scripts\python.exe -B .\ai_training\runs\camera_live_ai3_validation_ce248929b1014353a1b3723109ffb5a3\final_regression\run_offline_regression.py
```

Stress ban đầu: 7/7 PASS, gồm 18 profile và 4 test IPC/lifecycle. Sau đó bổ sung
test lịch reader/proxy; Camera Live continuous + resource failure: **35/35 PASS**.
Regression toàn bộ trên source cuối được ghi
trong `final_regression/full_offline_regression.json`, gồm tất cả integration unit tests, 5
association function tests và 34 Hybrid V7 unit tests. Runner chặn VideoCapture
thật và YOLO construction thật. `.venv` thiếu ultralytics nên không dùng để chạy
full suite; lần gọi sai môi trường dừng ở import, chưa chạy test. Không cài thêm.

**Regression cuối: 702/702 PASS, 0 FAIL, 0 ERROR, 0 SKIP.** Hai module lịch sử
`test_face_tracker_camera` và `test_hybrid_v7_real_output` có 0 unittest được
discover; không tính chúng là bằng chứng đã chạy camera/inference. 1.176 file
có sẵn đã được so SHA256: chỉ `camera_live_runtime.py` thay đổi, không có file
bị mất; toàn bộ model/config/recognition/enrollment/src giữ nguyên. Bằng chứng
trong `final_verification.json`; input Duration được kiểm tra qua hash trong
stress, cùng regression CH20/CH21/Using Phone/Shadow/overlay.

Preflight hiện tại: 56 PASS / 2 WARN / 0 FAIL. Preflight không xác nhận model
accuracy, class mapping bằng inference, camera backend hoặc native runtime lâu.

## 6. Giới hạn và bước AI4 khi có camera

- Chưa nghiệm thu iVCam, DSHOW/MSMF, sleep/wake, rút kết nối hoặc driver giữ
  `cap.read()` không trả về. Source gọi `cap.read()` đồng bộ; nếu native call treo,
  phím Q chưa thể được xử lý cho đến khi call trả về. Mock read failure có trả về
  không chứng minh trường hợp driver treo đã được xử lý.
- IPC được bound theo số item, không phải tổng bytes. Dữ liệu inference thật,
  độ phân giải lớn, nhiều người và native allocator cần đo trong phiên thật.
- Đóng cưỡng bức cả ứng dụng/OS có thể bỏ lại thư mục tạm; cleanup test áp dụng
  cho đường thoát/error được Python xử lý. Không xóa diagnostic có chủ ý.
- Shadow continuous chỉ ghi 10 observations, không cung cấp telemetry cả phiên
  dài. Console ghi mỗi AI result; tốc độ terminal/disk thực chưa được stress.
- Mỗi worker đợi response tối đa 90s; Q có thể hủy wait theo poll 0.1s; hai worker
  close có wait hữu hạn và main join20s. Native/OS error có thể được báo là cleanup
  failure; không khẳng định child process thật đã được đo trong test mock.

Dùng runbook/checklist AI2 hiện có cho AI4: preflight → phiên giới hạn10 → phiên
continuous; thử Overlay/Shadow, một/hai sinh viên, Using Phone/Sleeping và SID
Unknown/Ambiguous; quan sát Frame ID/STALE, duration chỉ tăng theo observation;
Q và mất kết nối phải release camera/window/process. Trong phiên dài ghi lại
RSS parent + hai child PID, sự tăng ổn định sau warm-up và phần dư sau khi dừng.
Chỉ người kiểm thử mới xác nhận behavior/actor truth. Chưa có case camera AI4
nào được đánh dấu PASS. Không thêm công cụ hoặc thuật toán AI4 trong phase này.

## 7. Files và rollback

- Sửa `camera_live_runtime.py`: hai lỗi IPC/lifecycle có reproduction.
- Tạo `test_camera_live_stability.py`: mock stress + unit tests.
- Tạo tài liệu này; kết quả/backup nằm trong artifact mới nêu trên.

Rollback riêng file đã sửa, sau khi kiểm tra không có thay đổi mới hơn:

```powershell
Copy-Item -LiteralPath .\ai_training\runs\camera_live_ai3_validation_ce248929b1014353a1b3723109ffb5a3\rollback_before\camera_live_runtime.py -Destination .\ai_training\integration\camera_live_runtime.py
```

Không dùng git reset/clean vì integration và nhiều thay đổi của người dùng hiện
chưa được commit. Backup là source ngay trước AI3; rollback phục hồi cả hai rủi
ro cũ. Test/tài liệu mới có thể để lại để đối chiếu, không cần xóa dữ liệu.
