# FaceTracker Diagnostic C — Offline Matching Evaluation

Ngày: 2026-11-10. **Chỉ thêm công cụ/kiểm thử/kết quả Diagnostic C; không sửa production.**

25/25 test Diagnostic C đạt. Toàn bộ kiểm thử liên quan: **286/290 đạt, 4 failure,
0 error, 0 skip**. Bốn failure là test cũ kỳ vọng camera index 2, trong khi source
workspace hiện tại đặt index 0; chạy riêng không nạp Diagnostic C vẫn lỗi như vậy.
Không sửa production hoặc expected results để làm chúng đạt.

## 1. File tạo mới và bảo toàn dữ liệu

Tạo bốn file trong `ai_training/integration`:

| File | Nội dung |
|---|---|
| `evaluate_face_tracker_matching_c.py` | CLI, loader session 4.0s, baseline tracker thật, hai matcher hình học offline và metric |
| `test_face_tracker_matching_c.py` | 25 unit/integration test offline, kiểm tra ground truth và bảo toàn source |
| `FACE_TRACKER_DIAGNOSTIC_C_RESULTS.json` | Kết quả/trace 10 frame camera và 25 tình huống × 4 phương án |
| `FACE_TRACKER_DIAGNOSTIC_C_REPORT.md` | Báo cáo này |

Đã đọc source/Git status trước khi code và trình bày kế hoạch. Đối chiếu SHA-256 của
120 file có sẵn (source/config/JSONL/báo cáo và các file workspace được ghi nhận):
không file cũ nào thay đổi. JSONL nguồn, báo cáo A/B, hai file phân tích của người
dùng, Camera Live/trial/fallback, FaceTracker, PersonTracker, Association, CH-20/21
giữ nguyên. Không cài dependency, chạy camera/model mới, SQLite/API, commit/push
hoặc triển khai CH-22. Không thay model/weights/confidence threshold production.

## 2. Dữ liệu và phương pháp

Nguồn:
`ai_training/runs/person_tracker_debug/session_37712fb9ffff45f5a9219897b4785309/frames.jsonl`.

SHA-256:
`6818f36ad6d3c8fd1c84708fa39072e7f43bf996da67c0cd1b1ed91594c592bd`.

Session có 10 frame, 13 face detections, IoU production 0.3, max_age thử nghiệm 4.0s.
Loader trước hết chạy replay bằng FaceTracker thật và xác nhận **10/10 frame** có
state/counter đúng log. Mapping detection → Face Track cũng khớp **10/10 frame**.
Evaluator B hiện chỉ nhận session baseline 2.0s; Diagnostic C dùng loader mới,
không sửa evaluator B hoặc replay cũ.

Cả bốn phương án dùng cùng detections, detection order, frame ID, captured_at và
face_tracker_updated_at. Raw Student ID/score được giữ nguyên cho baseline, nhưng
không tham gia quyết định matching hình học hoặc tạo actor ground truth. Không
đọc Person ID/association để nối track hay gán nhãn đúng/sai. Timestamps không được
suy ra từ processing time; giữ nguyên max_age=4.0s và expiry `age > max_age` trước
matching cho mọi phương án. Mất một detection không cập nhật last_seen của track.

Baseline/control dùng FaceTracker thực. Để audit detection index ngay cả khi hai
bbox/score/Student ID giống hệt nhau, mỗi bbox đầu vào có tuple object riêng; tracker
hiện tại lưu đúng object đó. Audit đọc object sau update, không tác động decision.
Có test xác nhận và fail rõ nếu hợp đồng lưu bbox của tracker thay đổi.

Hai matcher hình học là instance độc lập chỉ lưu bbox/last_seen/counter; không được
import vào Camera Live. Chúng giữ greedy one-to-one, thứ tự cấp ID mới theo detection
input và giữ unmatched track tới expiry. Không motion prediction/embedding/ReID.

## 3. Bốn phương án và tham số thử nghiệm

1. **IoU-only:** tracker thật, IoU >= 0.3, max_age 4.0s.
2. **IoU relaxed control:** tracker thật, IoU >= 0.05, max_age 4.0s, chỉ offline.
   Đối chứng này tách lợi ích của nới IoU khỏi lợi ích của center/size.
3. **Geometry:** khoảng cách tâm chuẩn hóa `d = distance / max(diagonal_old, diagonal_new)`;
   size similarity `s = min(min(w_old,w_new)/max(w_old,w_new), min(h_old,h_new)/max(h_old,h_new))`.
   Điểm `0.5*IoU + 0.3*max(0,1-d/0.6) + 0.2*s`.
   Chỉ tạo candidate khi đồng thời IoU >=0.05, d <=0.6, s >=0.5 và score >=0.35.
   Greedy chọn score lớn nhất, tie-break theo track ID/detection index như baseline.
4. **Geometry conservative:** cùng score/gates, thêm mutual best và khoảng cách điểm
   top-1/top-2 >=0.15 ở cả phía track lẫn detection. Nếu nhỏ hơn margin thì từ chối;
   không có đối thủ thứ hai thì không áp dụng margin. So với ma trận candidate gốc,
   không chuyển sang lựa chọn thứ hai sau khi cặp mơ hồ bị từ chối. Detection chưa
   match nhận ID mới; track cũ unmatched vẫn theo TTL.

Đây là bộ tham số thăm dò được ghi rõ, chưa hiệu chỉnh trên dữ liệu đại diện hoặc
đánh giá held-out. Gate 0.05/score 0.35/margin 0.15 không phải confidence threshold
model và không được đưa vào production. Không chọn bộ tham số bằng cách tối ưu
Student ID/Person ID. Không thể suy ra nhận dạng đúng chỉ từ việc ít track hơn.

## 4. Định nghĩa metric và ground truth

- `ids_created`: tổng số ID cấp; `new_ids_after_first_frame`: số ID cấp sau frame
  đầu, gồm cả người mới đến. Hai số này **không phải ID switches**.
- `id_changes`: một actor có nhãn độc lập nhận track khác lần quan sát có nhãn trước,
  kể cả sau một lần mất detection. Đây là định nghĩa theo chuỗi quan sát, không phải
  bộ metric MOT chuẩn đầy đủ.
- `false_continuations`: track được dùng tiếp cho actor có nhãn khác actor trước đó.
  Không cộng actor mới tới nhưng nhận một track hoàn toàn mới.
- `expiry_count`: số track bị xóa vì age >4.0s ở các update có trong dữ liệu.
  Không thêm frame giả cuối cùng để flush các track còn giữ.
- `ambiguity_rejections`: số detection không match do cạnh tranh điểm mơ hồ.

**Camera chưa có actor ground truth:** `id_changes` và `false_continuations` đều
`null`, coverage 0, `identity_metrics_complete=false`. Không dùng SV001/P1 hoặc
Face Track ID của thuật toán cũ làm nhãn actor. Không được đọc `null` thành 0.

Synthetic scenarios có actor A/B độc lập với Student ID. Test thay toàn bộ Student
ID thành cùng một chuỗi và score thành cùng một giá trị: trace/metric không đổi.
Test đổi Person mapping trong log cũng không làm thay đổi input Diagnostic C.

CLI có thể nhận annotation thủ công theo `(frame_id, detection_index, actor)` từ
người xem video/ảnh và nhận biết actor thực. Annotation chỉ phục vụ đánh giá sau
matching, không được truyền vào score/gating. Annotation một phần cho coverage
không đầy đủ và chỉ đếm được các lỗi quan sát được, không xác nhận zero-error.

## 5. Kết quả camera replay

| Phương án | ID tạo | ID mới sau F1 | Matched links | Expiry | ID changes / false continuations |
|---|---:|---:|---:|---:|---|
| IoU-only 0.3 | 5 | 4 | 8 | 4 | null / null |
| IoU-only 0.05 control | 4 | 3 | 9 | 3 | null / null |
| Geometry | 4 | 3 | 9 | 3 | null / null |
| Geometry conservative | 4 | 3 | 9 | 3 | null / null |

Không phương án nào gặp cạnh tranh mơ hồ trong 10 frame này. Dưới đây là track theo
**detection index của từng frame**, không phải track theo actor đã được xác minh:

| Frame | IoU-only 0.3 | Ba phương án còn lại |
|---|---|---|
| 1 | [1] | [1] |
| 2 | [1,2] | [1,2] |
| 3–4 | [3,2] | [1,2] |
| 5 | [3,4] | [1,3] |
| 6–8 | [3] | [1] |
| 9 | [] | [] |
| 10 | [5] | [4] |

**Frame 3:** track 1 → detection 0 có update gap 3.326s, IoU 0.172204, normalized
center distance 0.374843, size similarity 0.834483. Baseline không đạt IoU 0.3;
geometry score 0.365577 vượt gate 0.35 và giữ ID. IoU relaxed control cũng giữ ID.
Vì thế camera replay này chưa chứng minh center/size tốt hơn việc chỉ hạ IoU.

**Frame 10:** track được thấy gần nhất ở frame 8; frame 9 không có face. Age thực
là 4.034463s, khác consecutive update gap F9→F10 là 2.346807s. Mọi phương án đều
expire trước matching và cấp ID mới. IoU với bbox cũ 0.287625 còn thấp hơn baseline
0.3, nên chỉ tăng TTL cũng chưa đủ để baseline match cặp này.

## 6. Tình huống có actor ground truth

25 scenarios gồm 22 scenario tái sử dụng từ Diagnostic B (bỏ toàn bộ Person/reference
fields khi tạo input) và 3 trường hợp bổ sung: replacement có frame vắng người,
size jump, center jump. Mỗi scenario chạy bốn phương án: **100 trial**, không cộng
chúng thành 100 unittest độc lập.

Mỗi ô dưới là **ID changes / expiry / false continuations**:

| Scenario | IoU 0.3 | IoU 0.05 | Geometry | Conservative |
|---|---|---|---|---|
| Một người đứng yên | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Hai người gần nhau | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| Crossing | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 2 |
| Swap positions | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 2 |
| Replacement sau 3s | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| Replacement có frame trống | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| Overlapping reordered | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 2 | 2 / 0 / 0 |
| Size jump cùng actor | 0 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |
| Center jump cùng actor | 1 / 0 / 0 | 0 / 0 / 0 | 1 / 0 / 0 | 1 / 0 / 0 |

Ở overlapping_reordered, conservative từ chối hai detection mơ hồ và tạo **4 ID**
thay vì 2; giảm false continuation bằng cách ngắt liên tục. Không giải quyết được
crossing/swap có hình học trông vẫn tự tin hoặc replacement cùng bbox. Đổi thứ tự
detection khi bbox phân biệt rõ vẫn giữ hai track. Các boundary 2.0/2.001/4.0 giữ ID;
4.001/5.0/5.001 expire ở mọi phương án. Nhãn Unknown/score gần nhau không quyết định
matching. Size/center gate có thể ngắt cả một actor đúng khi bbox thay đổi mạnh.

## 7. Tests đã chạy và kết quả thực tế

| Nhóm | Đạt / chạy | Failure |
|---|---:|---:|
| Diagnostic C mới | 25/25 | 0 |
| Camera backend | 0/3 | 3 |
| FaceTracker Camera Trial | 17/18 | 1 |
| FaceTracker Diagnostic A + pipeline | 25/25 | 0 |
| FaceTracker Diagnostic B + pipeline | 25/25 | 0 |
| PersonTracker + Diagnostic A/replay | 39/39 | 0 |
| BehaviorDuration + CH-20 + Snapshot/Overlay | 43/43 | 0 |
| WarningObserver + CH-21 | 47/47 | 0 |
| Camera Live/Person integration + Panel/Layout | 22/22 | 0 |
| Hybrid person diagnostics | 4/4 | 0 |
| Hybrid V7 | 34/34 | 0 |
| Association script | 5/5 | 0 |
| **Tổng** | **286/290** | **4** |

Kiểm tra predictions đã lưu riêng đạt **2/2 ảnh**. Không tính lại các lần rerun.

Bốn test lỗi:

- `test_camera_live_backend`: cả ba test hard-code index 2 trong message/timeline.
- `test_face_tracker_camera_trial.test_trial_is_scoped_to_camera_live_not_face_tracker_class_defaults`:
  assertion `live.CAMERA_INDEX == 2` thất bại.

Source hiện tại: `camera_coordinator.py:18` là `CAMERA_INDEX = 0`; Camera Live import
giá trị này. SHA-256 khớp bản ghi trước khi code Diagnostic C. Chạy riêng bốn test
cũ, không import module Diagnostic C, vẫn **4 failure/4 tests** với cùng lỗi 0 !=2.
Đây là lệch cấu hình/test có sẵn trong workspace. Không sửa source, mock global
camera index hay thay expected để che các failure.

Lệnh đầy đủ, từ repository root:

```powershell
Push-Location .\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_face_tracker_matching_c test_camera_live_backend test_face_tracker_camera_trial test_face_tracker_parameter_evaluation test_face_tracker_parameter_pipeline test_face_tracker_diagnostics test_face_tracker_diagnostic_pipeline test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# 247 tests, 243 pass, 4 failures, 15.738s
..\..\.venv-hybrid\Scripts\python.exe -m unittest test_hybrid_person_diagnostics -q
# 4 OK
Pop-Location
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -q
# 34 OK
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS 5/5
.\.venv-hybrid\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# 2/2 images
```

25 test mới bao phủ nguồn/metadata/order/clock, equivalence baseline, input bất biến,
Student/Person invariance, ambiguity/no fallback, center/size, missing frames/expiry,
ground truth coverage/manual annotations, log lỗi, output exclusive và không có
camera/model/SQLite/network side effects. CH-20/21 regression giữ nguyên episode
guards; production WARNING không được bật. Không chạy script camera thủ công.

## 8. Cách chạy lại và bảo vệ output

Chạy evaluation không ghi file:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe .\ai_training\integration\evaluate_face_tracker_matching_c.py
```

Ghi lần đánh giá mới, chọn tên chưa tồn tại:

```powershell
.\.venv\Scripts\python.exe .\ai_training\integration\evaluate_face_tracker_matching_c.py --output .\ai_training\integration\FACE_TRACKER_DIAGNOSTIC_C_RESULTS_rerun.json
```

CLI dùng exclusive create `open('x')`, từ chối ghi đè file có sẵn, JSONL nguồn hoặc
annotation. Kiểm tra hash nguồn trước/sau evaluation. JSON hiện tại khoảng 526 KiB,
chứa input bbox theo index, timestamps, actor labels (null ở camera), pair features,
gates/reasons, assignments, unmatched/expired/new và metric. Không lưu ảnh/embedding.

Nếu có annotation được người xem video xác nhận, format là một JSON list:

```json
[
  {"frame_id": 1, "detection_index": 0, "actor": "reviewed_actor_A"},
  {"frame_id": 2, "detection_index": 0, "actor": "reviewed_actor_A"}
]
```

Đây chỉ là ví dụ format, **không phải nhãn được xác nhận của session camera**.
Truyền `--annotations <file>` và `--output <ten-moi.json>` để đo theo nhãn thủ công;
hai annotation ví dụ chỉ cho coverage một phần. Không tạo nhãn bằng Student/Person ID.

## 9. Kết luận và rủi ro

Diagnostic C đã tái hiện baseline và tạo được phương án từ chối mơ hồ offline.
Camera replay giảm một lần cấp ID/một expiry, nhưng đối chứng chỉ nới IoU cũng có
cùng kết quả; chưa có bằng chứng geometry tốt hơn trên identity thực.

Rủi ro: geometry-only không thấy được người thay thế cùng vị trí; crossing/swap có
thể tự tin nối sai; gate/margin có thể phân mảnh actor đúng. Unmatched tracks được
giữ tới TTL nên có thể cạnh tranh hoặc được nối lại ở frame sau. Không có appearance,
motion prediction hay dữ liệu ground truth camera để lượng hóa đầy đủ tradeoff.
Các weights/gates chưa được hiệu chỉnh trên nhiều session hoặc held-out set.

Không có chi phí Camera Live mới vì production không import Diagnostic C. Chi phí
offline tăng theo số cặp track×detection và sort candidates; chưa đo FPS/model
latency camera, không suy ra camera performance từ thời gian unittest.

Bước tiếp theo phù hợp là thu/đánh nhãn actor độc lập cho nhiều cảnh camera rồi
đánh giá lại fragmentation và false continuations trên tập riêng. Chưa có cơ sở
đưa matcher này vào production. Bốn regression failure camera-index cần xử lý như
một công việc cấu hình/test riêng. Dừng sau Diagnostic C, không chuyển CH-22.
