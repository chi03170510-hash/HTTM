# CH-21 — Experimental WARNING, mặc định DISABLED

Ngày kiểm thử: 2026-11-10.

**Kết quả:** 27/27 unit test CH-21, 20/20 integration test CH-21 và 121/121
regression test đạt. Tổng cộng **168 test đạt, 0 failure, 0 error, 0 skip** ở
phiên bản cuối. Kiểm tra association bổ sung trên predictions đã lưu đạt 2/2 ảnh;
hai ảnh này được báo cáo riêng, không cộng vào 168 test.

Đủ cơ sở nghiệm thu chức năng phần mềm experimental bằng dữ liệu kiểm soát.
Chưa kiểm chứng cảnh báo với chuỗi inference/camera thật; không phải nghiệm thu
khả năng xác nhận vi phạm. Camera Live production tiếp tục tắt toàn bộ cảnh báo.

## 1. Files changed

| File trong `ai_training/integration` | Thay đổi |
|---|---|
| `behavior_warning.py` | Module mới: cấu hình bất biến, kiểm tra evidence, observer, episode, event và view bất biến |
| `camera_coordinator_live.py` | Khởi tạo observer trong AI worker, thêm `warning_frame` vào result và đọc nó trên Panel |
| `test_behavior_warning.py` | 27 unit test, fixture evidence độc lập với model |
| `test_ch21_warning_pipeline.py` | 20 integration test, tái sử dụng harness CH-20 |
| `CH21_IMPLEMENTATION_REPORT.md` | Báo cáo này và lệnh kiểm thử lại |

Đã đọc Git status/diff và source trước khi sửa. Thư mục integration vốn chưa
được Git theo dõi, vì vậy còn lưu bản đối chiếu trước khi sửa và so SHA-256 của
101 file có sẵn: chỉ `camera_coordinator_live.py` thay đổi trong nhóm này.
`.gitignore` có chỉnh sửa từ trước và được giữ nguyên.

Giữ nguyên BehaviorDuration, snapshot/overlay, FaceTracker, PersonTracker,
Association V4, Face ID/InsightFace, Hybrid V7, cấu hình inference, weights,
camera index 2 và code CH-18/CH-19. Không thêm dependency, commit hoặc push.

## 2. Cấu hình an toàn

`WarningConfig` là frozen dataclass; chỉ nhận giá trị boolean rõ ràng.
`DEFAULT_WARNING_CONFIG` và `camera_coordinator_live.WARNING_CONFIG` mặc định:

| Behavior | Enabled |
|---|---|
| sleeping | False |
| using_phone | False |
| turning_back | False |
| eating | False |

Không có CLI/environment flag tự bật cảnh báo. Chỉ test process dùng
`patch.object(live, "WARNING_CONFIG", WarningConfig(...=True))`, sau đó khôi phục.
Không có dữ liệu/harness giả lập được import vào production.

Khi tất cả tắt, behavior/duration CH-20 vẫn hiển thị bình thường; Panel thêm
`Experimental alerts: disabled`. Test đưa cả bốn behavior detected kéo dài qua
observer mặc định và xác nhận không có event.

## 3. Observer, episode và thời gian

Mỗi camera session có một `BehaviorWarningObserver`. `ai_loop()` gọi observer
một lần sau BehaviorDuration và xác thực metadata snapshot, trước khi đưa result
vào queue. Đầu vào là `BehaviorFrame` bất biến, matches kèm evidence hiện tại và
`observed_at=time.perf_counter()`; cùng hệ thời gian monotonic với `captured_at`.

Observer quản lý episode riêng cho Person Track, Student ID và behavior; không
gọi BehaviorTimer hoặc BehaviorStateManager cũ. Vòng đời:

`CLEAR → CANDIDATE → WARNING_EXPERIMENTAL`.

Một quan sát đủ điều kiện cần đồng thời có:

- Behavior được bật và state `detected` nhất quán với decision/status IMAGE_ONLY.
- Person Track ID hợp lệ, identity hiện tại matched và Student ID hợp lệ, duy nhất.
- Một association chính xác `matched`, cùng person reference, Student ID và Face
  Track ID với BehaviorFrame. `matched_v4_heuristic` bị chặn riêng ở CH-21.
- Evidence đúng hợp đồng behavior bên dưới, không thiếu hay mâu thuẫn.
- Frame ID mới, timestamp hữu hạn và tăng; capture không ở tương lai.
- Tuổi ảnh tại lúc xét từ 0 đến 5 giây, gap giữa các capture tối đa 5 giây.

Quan sát đủ điều kiện đầu tiên bắt đầu qualified span ở 0.0 giây. Cảnh báo chỉ
phát khi **cả duration gốc CH-20 và qualified span đều >5.0**, không so chuỗi đã
làm tròn. Không đặt ngưỡng confidence mới hoặc số mẫu tùy ý. Với gap tối đa 5
giây và span >5 giây, cần ít nhất ba quan sát đủ điều kiện.

Evidence/association/freshness mất điều kiện thì qualified span về 0, kể cả khi
duration CH-20 vẫn tiếp tục. Cờ đã phát event được giữ trong cùng episode, tránh
phát lại sau khi evidence/freshness phục hồi. Negative state, mất identity,
đổi Person/Student, absence, CH-20 reset hoặc capture gap >5 kết thúc episode.
Người quay lại bắt đầu episode mới, không kế thừa warning cũ.

Duplicate frame hợp lệ trả trạng thái đã lưu nhưng `events=()`, không cộng mẫu
hay thời gian. Timestamp không tăng, frame cũ hoặc metadata không hợp lệ ngắt
chuỗi và không phát event; thời gian tham chiếu không bị lùi. Trong Camera Live,
duplicate ID có capture khác bị AISnapshot chặn trước observer, theo cơ chế CH-20.

## 4. Evidence thực tế theo từng behavior

Điều kiện chung: `state/decision=detected`, `status=IMAGE_ONLY`, có danh sách
`supporting_evidence`, `contradicting_evidence` rỗng, evidence dạng dict và không
có `missing_evidence` đang được báo. IMAGE_ONLY không xác nhận hành vi liên tục.

| Behavior | Evidence được kiểm tra |
|---|---|
| sleeping | Support `behavior_yolo_sleeping_candidate`; bbox ngủ hợp lệ; model confidence và `sleeping_yolo_confidence` hữu hạn, đạt `sleeping_yolo_threshold` do V7 cung cấp. Cấu hình hiện tại là 0.75, không sửa ngưỡng này. |
| using_phone | Hai support YOLO candidate và unique same-person COCO phone; `phone_association_status=ASSOCIATED_GEOMETRICALLY`; model score hợp lệ; detection IDs không rỗng/trùng, số ID khớp số bbox hợp lệ; các tỷ lệ area/height/width không vượt limits hiện có trong `phone_object_geometry`. Không tự áp ngưỡng 0.75 của sleeping cho phone. |
| eating | Support unique same-person food và food gần miệng; `same_person_food_association=True`; `mouth_source=face_landmark`; food/mouth matches có ID, class, object confidence và khoảng cách hữu hạn nằm trong threshold V7 cung cấp. Không yêu cầu model confidence hành vi vì V7 có thể trả None cho eating. |
| turning_back | Support hướng đầu phù hợp hướng sau đã hiệu chuẩn; `rear_direction_calibrated=True`; yaw, target và tolerance hữu hạn, yaw nằm trong tolerance đã cung cấp. Vẫn phải detected ở đầu vào. |

Căn cứ source: `src/hybrid_v7.py` (`spatial_v7`, `spatial_v6`, `_set_decision`,
`phone_object_geometry_plausible`) và evidence hướng đầu được giữ từ
`src/hybrid_v3.py`. Observer chỉ đọc các trường đó, không chạy lại inference.

**Giới hạn turning_back:** pipeline IMAGE_ONLY hiện giữ suspected/unknown; cấu
hình hướng sau đang null. Bật flag thử nghiệm cũng không nâng các state này lên
detected. Ca positive turning_back dùng fixture detected có calibration hợp lệ;
không chứng minh V7 hiện có thể phát cảnh báo turning_back với camera thật.

## 5. Unit tests

`test_behavior_warning.py`: **27/27 đạt**. Bao phủ các mốc 4.9/5.0/5.01, mỗi
episode một event, duplicate/old frame, timestamp NaN/Inf/không tăng/tương lai,
capture age 5 và 5.01, gap đúng 5 và >5, disabled, identity/P ID không hợp lệ,
Student Unknown/trùng, đổi Person/Student, mất association, evidence thiếu/mâu
thuẫn, geometry/ID/mouth/calibration sai, nhiều behavior và nhiều sinh viên,
negative state/absence rồi episode mới. Có ca duration CH-20 tiếp tục nhưng
qualified span phải bắt đầu lại, và ca latch vẫn giữ sau evidence/stale break.

## 6. Integration tests và kết quả Panel

`test_ch21_warning_pipeline.py`: **20/20 đạt**. Harness CH-20 được tái sử dụng,
chỉ thay ranh giới Worker, thời gian, frame numbering, camera/GUI/thread giả lập.
Ảnh là mảng BGR tổng hợp; Person ID được tracker thật tạo từ bbox.

Luồng thực thi:

```text
Worker replies giả lập với evidence có kiểm soát
  → ai_loop thật → FaceTracker → Association V4 → PersonTracker
  → BehaviorDuration → AISnapshot đúng frame → BehaviorWarningObserver
  → result_queue → main thật → draw_ai_panel → OpenCV putText thật
```

Không chỉ gọi trực tiếp observer/duration rồi dựng sẵn UI result. Cấu hình mock
chỉ tồn tại trong test. Completion thông thường trễ capture 1.5 giây; ca gap dài
cũng đẩy completion về sau capture, không vô tình kiểm thử bằng timestamp tương
lai. Các fixture CH-20 cũ với capture 100/completion 200 được dùng riêng để
xác nhận stale bị chặn, không dùng cho positive warning.

Ca biên đi qua toàn bộ pipeline và Panel:

| Frame | captured_at | Duration gốc / qualified | Event mới | Nhãn behavior |
|---|---:|---:|---|---|
| 1 | 198.50 | 0.00 | Không | ~0.0s |
| 2 | 200.00 | 1.50 | Không | ~1.5s |
| 3 | 201.50 | 3.00 | Không | ~3.0s |
| 4 | 203.40 | 4.90 | Không | ~4.9s |
| 5 | 203.50 | 5.00 | Không | ~5.0s |
| 6 | 203.51 | 5.01 | Một WARNING_EXPERIMENTAL | ~5.0s |

Frame 6 vẫn hiển thị `~5.0s` do làm tròn nhưng phát event từ giá trị gốc 5.01.
Panel có Student SV001, Person P1, behavior, Frame 6,
`WARNING (EXPERIMENTAL)` và `Not an official violation`.

Ca evidence ngắt ở frame 3: frame 5 có CH-20 duration 6.0 nhưng qualified chỉ
1.5, không warning; frame 8 qualified 6.0/CH-20 10.5 mới phát event. Ca V4 thật
với hai bbox chồng lấn cho `matched_v4_heuristic` giữ duration 6.0 nhưng không
cảnh báo. Hai sinh viên, hai behavior tạo đúng bốn event độc lập và hai trang N/P.

Một delivery và một cached repaint được chạy qua main cho từng AI observation.
Observer được gọi đúng số frame AI, không gọi từ render. Repaint thêm tới tuổi
ảnh 20 giây không đổi duration/event; sau 5 giây nhãn active bị thay bằng
`STALE experimental alert (historical)`, kể cả khi completion còn mới.

## 7. Regression tests và lệnh tái lập

| Bộ test | Đạt | Lỗi |
|---|---:|---:|
| CH-21 observer unit | 27 | 0 |
| CH-21 warning pipeline | 20 | 0 |
| CH-20 duration pipeline | 16 | 0 |
| BehaviorDuration | 17 | 0 |
| Snapshot/Overlay | 10 | 0 |
| Camera/Panel/Person integration | 14 | 0 |
| Camera Live layout | 8 | 0 |
| PersonTracker | 17 | 0 |
| Association script hiện có | 5 | 0 |
| Hybrid V7 | 34 | 0 |
| **Tổng** | **168** | **0** |

Script association 5/5 hiện có kiểm tra hàm association nền. Association V4 thật
được thực thi trong các harness CH-20/CH-21 và Camera integration, gồm ca
`matched_v4_heuristic` riêng bên trên. Predictions đã lưu: thêm 2/2 ảnh đạt
`test_hybrid_v7_real_output.py`, không chạy model mới.

Lệnh đã chạy ở bản cuối, PowerShell:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# Ran 129 tests, OK (lần chạy gộp: 7.883s)

Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS: 5/5
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -v
# Ran 34 tests, OK
.\.venv-hybrid\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# TOTAL MATCHED IMAGES: 2/2
```

Hai file test CH-21 cũng đã chạy riêng bằng `unittest discover`, lần cuối đạt
27 và 20 case. Subtest không được cộng thành case độc lập. Trong quá trình bổ
sung ca heuristic, fixture bbox đầu tiên còn được V2 nhận là matched; đã sửa
bbox để tạo đúng ambiguity và chạy lại ca V4 cùng toàn bộ suite gộp thành công.

## 8. SQLite/API và side effects

Module observer chỉ dùng thư viện chuẩn tính toán/dataclass và BehaviorFrame;
không có persistence, network hoặc legacy timer dependencies. Event chỉ nằm
trong tuple bất biến của AI result, không dispatch sang hệ thống khác.

Harness chặn `sqlite3.connect`, `socket.socket.connect`,
`socket.create_connection`, `urllib.request.urlopen`, `subprocess.Popen`; ca
positive còn chặn `HTTPConnection.request`. Chạm các đường này sẽ làm test lỗi.
Positive warning vẫn đi qua tới UI, không có side effect bị gọi. Test bổ sung
kiểm tra module không import thư viện DB/API/legacy timer.

JPEG tạm và snapshot trong RAM là cơ chế CH-20 có sẵn; không lưu bản ghi vi
phạm, không gửi event hoặc dữ liệu nhận dạng ra ngoài.

## 9. Ảnh hưởng Camera Live và hiệu năng

Giữ LIVE CAMERA 896/1280 chiều rộng và Panel 384/1280, resize giữ tỷ lệ, N/P/Q,
preview độc lập worker, snapshot ẩn nhưng còn trong RAM. Panel chỉ đọc frozen
view, không tăng duration/phát event, không vẽ bbox lịch sử lên live.

Đã kiểm tra pixel live, tỷ lệ ảnh và tọa độ chữ không bị cắt; ảnh Panel tổng hợp
1280×720 với warning cũng đã được xem trực quan. Đây không phải ảnh camera thật.
Khi test bật nhiều cảnh báo, canvas có thể cao thêm để không cắt chữ; tỷ lệ hai
cột giữ nguyên. Cấu hình production disabled giữ kích thước layout cũ trong test.

Chi phí dự kiến tuyến tính theo số người × 4 behavior, cộng kiểm tra danh sách
evidence; trạng thái chỉ giữ episode hiện tại và frame gần nhất. Không thêm
inference, copy ảnh, thread hoặc I/O vào observer. 7.883s là thời gian chạy 129
test với dữ liệu tổng hợp, **không phải** số đo overhead/latency/FPS camera thật.
Chưa đo hiệu năng Camera Live sau thay đổi.

## 10. Giới hạn và rủi ro

- Các khoảng giữa ảnh có thể bỏ lỡ lúc hành vi dừng; qualified span vẫn là ước
  lượng, không chứng minh hành vi liên tục hoặc vi phạm chính thức.
- False positive/negative của model, nhận dạng nhầm hoặc hoán đổi IoU ID chưa
  được loại bỏ bởi observer; cổng metadata/evidence chỉ giảm phạm vi phát cảnh báo.
- Sleeping dựa vào YOLO; phone dựa vào candidate + object cùng người; eating dựa
  vào food/miệng tĩnh. Score không phải xác suất vi phạm đã hiệu chuẩn.
- Image age hoặc gap >5, association heuristic, mất khuôn mặt/identity/evidence
  sẽ chặn cảnh báo và có thể bỏ sót hành vi thật. Đây là lựa chọn thận trọng.
- Turning_back IMAGE_ONLY hiện không cung cấp detected đã đủ điều kiện; positive
  fixture không phải kết quả model thật.
- Harness chạy tuần tự, không kiểm chứng scheduling đồng thời của thread, độ
  mượt iVCam hoặc chuỗi detected từ inference thực tế.

## 11. Chạy lại camera iPhone/iVCam

Không chạy camera thật trong lượt kiểm thử này. Kết nối iPhone/iVCam index 2 và
chạy cấu hình production mặc định, **không bật bất kỳ behavior warning nào**:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Kiểm tra `Completed AI results: 10`, LIVE/Panel 70/30, không bbox lịch sử, các
trường identity/association/behavior/duration còn đúng, dòng
`Experimental alerts: disabled`, N/P và Q. Production sẽ không hiện warning dù
duration vượt 5; muốn kiểm tra nhánh positive, chạy test harness riêng bên trên.

## 12. Đánh giá nghiệm thu và điểm dừng

**Đủ điều kiện nghiệm thu chức năng phần mềm CH-21 experimental bằng automated
tests**, với cấu hình production mặc định disabled, event một lần/episode,
qualified evidence span độc lập và UI phân biệt historical/stale.

Chưa xác nhận hiệu quả phát hiện/cảnh báo với model và camera thật, chưa đo FPS
thực tế, chưa phải hệ thống xác nhận vi phạm. Chờ người dùng kiểm thử camera và
quyết định nghiệm thu thực tế. Dừng ở CH-21; không triển khai CH-22.
