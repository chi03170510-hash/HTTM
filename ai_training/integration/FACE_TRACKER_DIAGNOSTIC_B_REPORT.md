# FaceTracker Diagnostic B — Offline Parameter Evaluation

Ngày: 2026-11-10. **Chỉ đánh giá offline. Camera Live production vẫn max_age=2.0s.**

Kết quả camera replay: 4.0s và 5.0s đều giảm số lần đổi Face ID từ 6 xuống 0.
Các counterexample có ground truth lại chứng minh TTL dài hơn có thể nối người
thay thế vào track cũ. Ưu tiên 4.0s cho bước thử nghiệm được phê duyệt tiếp theo;
chưa đủ căn cứ chấp nhận đổi production chỉ vì ID ổn định.

**244/244 test đạt, 0 failure, 0 error, 0 skip**: 25 test mới + 219 regression.
Ngoài ra có 22 tình huống rủi ro × 3 cấu hình = 66 trial, không cộng chúng như
66 unittest độc lập. Predictions đã lưu đạt thêm 2/2 ảnh, báo cáo riêng.

## 1. Files changed và bảo toàn production

Chỉ tạo năm file mới trong `ai_training/integration`:

| File | Nội dung |
|---|---|
| `evaluate_face_tracker_parameters.py` | CLI/engine offline, immutable inputs, camera replay, 22 risk scenarios và benchmark |
| `test_face_tracker_parameter_evaluation.py` | 21 test camera/metric/risk/immutability |
| `test_face_tracker_parameter_pipeline.py` | 4 test qua pipeline cho cả ba tuổi thọ bằng test-only factory injection |
| `FACE_TRACKER_DIAGNOSTIC_B_RESULTS.json` | Kết quả máy đọc được, trace từng frame, metric/risk/benchmark |
| `FACE_TRACKER_DIAGNOSTIC_B_REPORT.md` | Báo cáo này |

Đã đọc source/Git status trước khi làm. So SHA-256 của 88 file có sẵn cho kết quả
**không file nào thay đổi**, gồm source, config, weights, JSONL nguồn và
`face_tracker_debug_summary.json`. Không sửa expected result của test cũ.

Không sửa Camera Live, FaceTracker, PersonTracker, Association V4, BehaviorDuration,
WarningObserver, model/weights/thresholds/camera index, diagnostics production,
CH-18/19 hoặc WARNING defaults. Không thêm dependency, không chạy camera/model
mới, không commit/push hoặc triển khai CH-22.

## 2. Phương pháp replay và định nghĩa metric

Nguồn duy nhất cho camera replay:

`ai_training/runs/person_tracker_debug/session_9cbaa96797d7444a8b0fcc77f849eafd/frames.jsonl`

SHA-256 nguồn trước/sau:

`f404c29b9dd749c6351001a81fc761f390c63c40e26eccaede1245fd82688b53`

Đọc trực tiếp face_tracker_updated_at, bbox, thứ tự detection, current Student ID
và recognition score. Không dùng capture gap, Person update gap hoặc processing
time để thay Face update timestamp. Input được chuyển thành frozen dataclasses
và tuples, rồi đưa nguyên chuỗi vào ba FaceTracker độc lập:

```python
FaceTracker(iou_threshold=0.3, max_age=2.0)
FaceTracker(iou_threshold=0.3, max_age=4.0)
FaceTracker(iou_threshold=0.3, max_age=5.0)
```

Đây là class production thật, không có bản thuật toán thay thế. Ba input digest
đều bằng `91aa2b416226d5e53073429f97b5fb71710833c67cbedec3d5ac21ab58b2e7df`.
Current Student ID/score không bị thay để ép matching. Stable label đầu ra có
thể khác do track giữ lịch sử lâu hơn, đó là hệ quả của tracker hiện tại.

Baseline 2.0 được đối chiếu trước: state trước/sau, last_seen, stable label,
next_id và output khớp log gốc **10/10 frame**.

Metric được tách riêng:

- `reference_id_changes`: Face ID đổi giữa hai lần quan sát liên tiếp của cùng
  reference, kể cả khi có frame thiếu detection ở giữa. Camera dùng P1 đã được
  association như một **reference quan sát**, không phải ground truth độc lập.
- `retained_reference_observations`: reference giữ ID so với lần quan sát trước;
  frame đầu không có transition để tính.
- `wrong_actor_continuations`: một track được nối sang ground-truth actor khác.
  Actor labels chỉ có trong fixture synthetic và không được đưa vào thuật toán.
- Input không thể phân biệt từ post-state được đánh dấu ambiguous. Không chọn
  tùy tiện detection index, không coi thiếu ground truth là zero identity error.

Camera kết quả `wrong_actor_continuations=null`, `identity_metrics_complete=false`.
Không suy ra identity accuracy, IDF1/MOTA hoặc false-positive rate từ 10 frame này.

## 3. Kết quả 2.0 / 4.0 / 5.0 trên session thật

Face update gaps: 9 interval, trung bình **2.130773s**, nhỏ nhất **1.727144s**,
lớn nhất **2.414767s**. Không tính thời gian khởi động frame đầu thành update gap.

| Metric | 2.0s | 4.0s | 5.0s |
|---|---:|---:|---:|
| Số Face ID tạo ra | 7 | 1 | 1 |
| Đổi Face ID trên reference P1 | 6 | 0 | 0 |
| Expiry | 6 | 0 | 0 |
| Giữ ID từ lần quan sát trước | 3/9 | 9/9 | 9/9 |
| Matching thành công vào track cũ | 3 | 9 | 9 |
| IoU nhỏ nhất trong các match | 0.542515 | 0.542515 | 0.542515 |
| IoU trung bình trong các match | 0.756136 | 0.840573 | 0.840573 |
| Cặp IoU được tracker xét | 3 | 9 | 9 |
| Peak track được giữ | 1 | 1 | 1 |
| Binding association bằng baseline | 10/10 | 10/10 | 10/10 |

Chuỗi ID theo frame:

```text
2.0s: 1, 1, 2, 3, 4, 5, 6, 6, 7, 7
4.0s: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
5.0s: 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
```

Association được tái tạo bằng V4 thật từ recorded person bbox và recorded Person
Track map, giữ nguyên current Face ID recognition logic của coordinator.
Baseline identity/geometry/status projection khớp log **10/10**. Với candidate,
Person P1, Student SV001, status matched và detection binding vẫn giữ **10/10**;
Face Track ID được phép khác vì chính là biến đang đánh giá.

Không tái tạo đầy đủ behavior payload của association: JSONL không lưu behavior
decisions/evidence gốc. Không tuyên bố replay inference hoặc episode camera thật.

## 4. Số ID switch giảm được

Cả hai candidate loại bỏ **6/6 lần đổi ID quan sát được (100%) trong riêng
session này**; số ID tạo ra giảm 7→1. Không có lợi ích thêm của 5.0 so với 4.0
trên chuỗi 10 frame này.

Đây là giảm fragmentation theo reference P1, không phải chứng minh nhận đúng
người. Đặc biệt, người mới nhận cùng track có thể làm ID-change count bằng 0
trong khi identity continuation sai; metric rủi ro dưới đây phát hiện việc đó.

## 5. Kết quả các tình huống offline

Các con số dưới đây là kết quả fixture có kiểm soát, không phải thống kê xác
suất lỗi thực tế. Bbox jump mô phỏng tình huống quay đầu làm detector đổi bbox;
không kiểm thử head pose/model thật. Ground-truth actor tách khỏi Student ID:
đổi recognition label có thể vẫn là cùng actor, và người thay thế là actor khác.

| Tình huống | 2.0s | 4.0s | 5.0s |
|---|---|---|---|
| Một người đứng yên, 10 frame, gap 1.5 | 1 ID, 0 đổi | Như baseline | Như baseline |
| Quay đầu/bbox jump dưới ngưỡng IoU | 2 lần đổi ID | Vẫn 2 lần đổi | Vẫn 2 lần đổi |
| Hai người gần nhau đứng yên, gap 3s | 12 ID, 10 lần đổi | 2 ID, 0 đổi, 0 nối sai | Như 4.0 |
| Hai người đổi vị trí sau 3s | 0 nối track sang actor khác | **2 nối sai** | **2 nối sai** |
| Hai người đi ngang, gap 1.5s | **2 nối sai** | **2 nối sai** | **2 nối sai** |
| Người thay thế cùng vị trí sau 1.5s | **1 nối sai** | **1 nối sai** | **1 nối sai** |
| Người thay thế cùng vị trí sau 3s | Track mới, 0 nối sai | **1 nối sai** | **1 nối sai** |
| Người thay thế cùng vị trí sau 4.5s | Track mới, 0 nối sai | Track mới, 0 nối sai | **1 nối sai** |
| Thiếu một frame, cùng người trở lại ở 3s | Đổi ID | Giữ ID | Giữ ID |
| Thiếu nhiều frame, cùng người trở lại ở 4.5s | Đổi ID | Đổi ID | Giữ ID |
| Student ID tạm Unknown | Face ID vẫn giữ | Như baseline | Như baseline |
| Đảo thứ tự hai detection, bbox phân biệt | Không đổi ID | Không đổi ID | Không đổi ID |
| Hai bbox trùng, đảo thứ tự detection | **2 nối sai** | **2 nối sai** | **2 nối sai** |
| Person P1 giữ nguyên, Face gap 3s | Face đổi 3 lần | Face không đổi | Face không đổi |
| Cùng actor/bbox, recognition label đổi | Face ID giữ | Như baseline | Như baseline |
| Hai recognition score gần nhau/đảo thứ hạng score | Matching theo geometry không đổi | Như baseline | Như baseline |

Các fixture known-label overlap cho phép audit output bằng raw label/score đã
ghi vào track sau update; đây không phải dùng identity để sửa matching. Nếu cả
bbox/label/score không phân biệt được, test riêng xác nhận metrics incomplete,
không tự gán ground truth cho output.

Biên expiry: số ID tạo qua hai observation cùng bbox/SID:

| Update gap | max_age 2.0 | max_age 4.0 | max_age 5.0 |
|---:|---:|---:|---:|
| 2.000 | 1 | 1 | 1 |
| 2.001 | 2 | 1 | 1 |
| 4.000 | 2 | 1 | 1 |
| 4.001 | 2 | 2 | 1 |
| 5.000 | 2 | 2 | 1 |
| 5.001 | 2 | 2 | 2 |

Không đổi điều kiện strict `age > max_age`. Không sửa epsilon hoặc threshold.

## 6. Nguy cơ nối nhầm identity

IoU-only không thể phân biệt cùng người quay lại với người khác xuất hiện cùng
vị trí trong cửa sổ retention. Recognition score/SID không tham gia matching.
Do đó tăng tuổi thọ mở rộng cửa sổ có thể nối nhầm, dù Student ID mới được model
nhận dạng chính xác. Baseline 2.0 cũng có giới hạn này ở gap 1.5s.

Hai fixture miss-many/replacement có thể có cùng chuỗi hình học nhưng actor
thực khác nhau: 5.0 giúp khôi phục đúng ở fixture cùng người, đồng thời nối sai
ở fixture người thay thế. Không thể kết luận candidate an toàn chỉ bằng số ID.

Chưa có tiêu chí định lượng chấp nhận rủi ro do người dùng đặt. Counterexample
hiện có đủ để **không tuyên bố 4.0 hoặc 5.0 đã đạt an toàn identity production**.
Không suy ra tỷ lệ lỗi từ 22 fixture tổng hợp.

## 7. CH-20 và CH-21

Test-only factory tạo tracker độc lập cho 2/4/5, xác nhận coordinator vẫn yêu cầu
production max_age=2.0/IoU=.3 trước khi inject; patch được khôi phục sau mỗi test.
Chạy real AI loop/V4/PersonTracker/Duration/Warning/Panel bằng model replies mock,
timestamps capture/completion phù hợp và cùng input cho cả ba cấu hình.

- Face ID đổi ở 2.0 nhưng giữ ở 4/5; duration cùng là **0,3,6,9,12s**, cùng một
  warning episode và một event experimental ở frame 3 trong mock config bật.
- Unknown, duplicate Student ID hoặc association mất: **0,3,6,--,0,3,6s**;
  episode cũ kết thúc, frame 3 và frame 7 là hai event thuộc episode khác nhau.
- Đổi SV001→SV002 trên cùng P1/bbox: duration về **0** ở frame 4 dù candidate
  giữ Face ID 1. Không event ở frame đổi identity; SV002 chỉ có event mới ở
  frame 6 sau qualified span 6s. Stable label cũ không ép giữ identity hiện tại.
- Mock dùng production WARNING config disabled: duration vẫn tăng nhưng **0
  warning event**. Sau patch, defaults vẫn disabled, FaceTracker mặc định vẫn 2.0.
- Harness chặn SQLite/socket/API/subprocess model thật; không tạo vi phạm chính
  thức hoặc gửi event ra backend. Positive warnings chỉ tồn tại trong test mock.

Không có behavior/evidence gốc trong log camera để xác nhận actual duration hay
warning trên ảnh thật. Capture gap frame 1→2 thực tế là **21.690111s** do cold
start, nên CH-20 sẽ ngắt chuỗi nếu đang detected, độc lập việc Face ID được giữ.
Tăng max_age không thay đổi capture-gap/freshness rules 5 giây của CH-20/21.

## 8. Regression tests

| Nhóm | Đạt |
|---|---:|
| Offline evaluation mới | 21 |
| Pipeline parameter evaluation mới | 4 |
| Face Diagnostic A | 25 |
| Person Diagnostic A, gồm worker diagnostics | 26 |
| PersonTracker | 17 |
| Association script nền | 5 |
| BehaviorDuration | 17 |
| CH-20 pipeline | 16 |
| CH-21 observer/pipeline | 47 |
| Camera/Panel integration + Layout | 22 |
| Snapshot/Overlay | 10 |
| Hybrid V7 | 34 |
| **Tổng** | **244** |

0 failure/error/skip ở bản cuối. Association V4 thật còn chạy trong các pipeline
test; script nền 5/5 không bị gọi nhầm là suite V4 độc lập. Stored predictions
2/2 ảnh đạt, ngoài số test trên. 66 risk trials không cộng thành unittest cases.

Lệnh đã chạy:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_face_tracker_parameter_evaluation test_face_tracker_parameter_pipeline test_face_tracker_diagnostics test_face_tracker_diagnostic_pipeline test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# Ran 201 tests, OK (11.410s)
..\..\.venv-hybrid\Scripts\python.exe -m unittest test_hybrid_person_diagnostics -q
# Ran 4 tests, OK

Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -q
# Ran 34 tests, OK
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# 5/5 PASS
.\.venv-hybrid\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# 2/2 images
```

Chạy lại evaluator không ghi file:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe .\ai_training\integration\evaluate_face_tracker_parameters.py --benchmark-repeats 200
```

Để xuất JSON khác, thêm `--output <đường-dẫn-file-mới>`. CLI dùng exclusive create,
từ chối overwrite source JSONL hoặc file output có sẵn. Kết quả đã xuất nằm tại
`FACE_TRACKER_DIAGNOSTIC_B_RESULTS.json`; không cần chạy camera để tái lập.

## 9. Chi phí hiệu năng

Benchmark đo riêng vòng FaceTracker.update trên đúng 10 input camera, lặp 200
lần, tạo tracker mới mỗi lượt; không gồm model, ảnh, association, audit hoặc
serialize. Median thời gian bình quân mỗi update ở lần xuất artifact:

| max_age | Median µs/update | Minimum µs/update |
|---:|---:|---:|
| 2.0 | 5.500 | 5.410 |
| 4.0 | 6.335 | 6.030 |
| 5.0 | 8.155 | 6.120 |

Đây là microbenchmark nhỏ chịu nhiễu scheduling, không chứng minh 5.0 luôn chậm
hơn 4.0 và không đại diện FPS camera. Chi phí trên input một face nhỏ hơn nhiều
so với inference vài giây; cảnh đông/người mất quan sát có thể giữ nhiều track
hơn, tăng số cặp IoU. Chưa benchmark classroom nhiều người hoặc camera thật.

## 10. Khuyến nghị max_age

Trong ba cấu hình được phê duyệt đánh giá, **ưu tiên 4.0s làm ứng viên cho thử
nghiệm camera có kiểm soát tiếp theo**, nếu người dùng phê duyệt. Đây là candidate
nhỏ nhất đã so sánh mà loại bỏ fragmentation trong session hiện tại; không phải
ngưỡng tối ưu toàn cục. 5.0 không thêm lợi ích camera này và thêm false-continuation
ở gap 4.5s, dù có thể giữ cùng actor qua occlusion dài hơn trong fixture khác.

Không khuyến nghị đổi production ngay chỉ từ kết quả ID ổn định. Cả 4/5 đều có
counterexample nối sai ở gap 3s; cần chấp thuận tradeoff và thử departure/
replacement/crossing với nhiều người. Production vẫn **2.0s**.

## 11. Giới hạn chưa kiểm chứng

- Chưa chạy camera với 4.0/5.0; không có dữ liệu model/face detection mới.
- Một session 10 frame, một face, không có ground truth identity độc lập; không
  đo được xác suất nối sai trong lớp hoặc generalization.
- Các risk fixtures mô phỏng bbox/identity/timing, không mô phỏng đầy đủ ánh sáng,
  occlusion, motion blur, recognition error hoặc hình dáng khuôn mặt thật.
- Association chỉ tái tạo projection identity/geometry/status; thiếu behavior
  payload nên CH-20/21 camera thật chưa replay toàn bộ từ log này.
- Known identity đổi được gates CH-20/21 reset; misrecognition/association sai
  kéo dài vẫn có thể gắn evidence sai. Pure IoU và TTL không giải quyết hết.

## 12. Những thay đổi production cần phê duyệt và điểm dừng

Nếu tiếp tục, cần phê duyệt riêng một pilot thay **chỉ max_age của FaceTracker
trong Camera Live từ 2.0→4.0**, giữ threshold/greedy/model/current-identity và
CH-20/21 safeguards. Không tự đổi default FaceTracker toàn cục cho các app cũ.

Cần thống nhất mức rủi ro nối người thay thế có thể chấp nhận và tiêu chí nghiệm
thu nhiều người. Bất kỳ thay đổi vòng đời unmatched, identity gating hoặc
embedding matching nào là phạm vi riêng, không được thực hiện trong Diagnostic B.

**Dừng sau đánh giá offline và báo cáo. Không thay production, không commit/push,
không triển khai CH-22. Chờ phê duyệt của người dùng.**
