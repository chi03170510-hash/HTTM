# HTTM Camera Live Test Toolkit — offline

Toolkit chỉ đọc log, dùng Python standard library. Không import OpenCV/Ultralytics, không mở webcam, chạy inference, chỉnh model hoặc thay đổi pipeline production. Không cần cài dependency.

Để đối chiếu identity với actor truth thủ công theo frame/đoạn thời gian, dùng công cụ độc lập trong [CAMERA_IDENTITY_TRUTH_GUIDE.md](CAMERA_IDENTITY_TRUTH_GUIDE.md). Công cụ bind SHA256/session của hai logger, yêu cầu timestamp khớp chính xác và không tự lấy SID production làm ground truth. Không cấp SID shadow cho pipeline hoặc chọn T_hold.

## Schema audit từ source hiện tại

| Dữ liệu | PersonTracker Debug | Shadow Phase 1 cũ | Shadow behavior logging hiện tại |
|---|---|---|---|
| Schema | person_tracker_diagnostic_v1 | identity_continuity_shadow_v1 | identity_continuity_shadow_v1 |
| Frame/capture timestamp | Có | Có | Có |
| Person Track | tracks_after; chỉ track last_seen==tracker_updated_at là detection hiện tại | observations; bỏ track missing/expired khi đếm current | sleeping[] / using_phone[] production rows; observations shadow riêng |
| SID/association | Raw associations; không chứng minh production attribution | face SID và shadow proposal, không thay thế SID production | SID/association_status/identity_status từ BehaviorFrame trong cả hai mảng |
| Face/continuity | FaceTracker diagnostic tùy opt-in | Có; face_present=null khác false | Có |
| Sleeping state/confidence/duration | Không | Không | Có trong sleeping[]; confidence có thể null |
| Lý do duration | Không | Không | duration_event/blockers/reset_reasons; phân loại diagnostic |
| Using Phone state/confidence/duration | Không | Không | Có trong using_phone[]; confidence có thể null |

Nguồn audit: `person_tracker_diagnostics._record`, `identity_continuity_diagnostics.record/behavior_diagnostic`, `BehaviorDurationTracker.update`, `camera_coordinator_live.ai_loop`.

Hiện có log PersonTracker camera cũ và mẫu Shadow tổng hợp từ test; chưa có session Shadow camera thật trong thư mục mặc định tại thời điểm chuẩn bị toolkit. Không coi các mẫu synthetic là camera thật.

Log mới khi bật Shadow chứa cả Sleeping và Using Phone. Log cũ chỉ có Sleeping vẫn đọc được; Using Phone thiếu được báo MISSING DATA / INCONCLUSIVE, không tự bổ sung từ snapshot hoặc inference. Kết quả Using Phone camera PASS trước đây không bị phủ định khi log cũ thiếu trường.

### Schema behavior bổ sung, tương thích ngược

Giữ `schema=identity_continuity_shadow_v1`; thêm `behavior_diagnostic_version=2`. Không thay schema kết quả AI, AISnapshot hoặc AI Panel. Một record gồm hai mảng `sleeping[]`, `using_phone[]`, mỗi mảng có một row cho mỗi person trong BehaviorFrame, kể cả khi identity không hợp lệ. Không có person thì cả hai mảng rỗng; không có kết quả production thì `production_observation_status=not_provided`, không suy ra state.

Các trường chung mỗi row:

- `person_track_id`, `student_id`, `association_status`, `identity_status`: giá trị production đã xác định; SID có thể null.
- `identity_continuity_status/reason`, `current_face_student_id`, `current_face_present`, `shadow_proposed_student_id`: thông tin shadow riêng, không cấp identity cho production.
- `<behavior>_state`, `<behavior>_duration_seconds`: sao chép từ BehaviorFrame. Duration null khác 0.
- `<behavior>_model_confidence`, `<behavior>_decision_reason`: đọc Hybrid output đúng person_ref duy nhất; thiếu hoặc mapping không duy nhất thì null, không mượn score Face Recognition.
- `duration_event`, `duration_blockers`, `duration_reset_reasons`: dùng chung một hàm phân loại diagnostic cho cả hai hành vi. Các lý do là diễn giải từ output hiện tại/trước đó, không đọc/sửa timer nội bộ và không chứng minh identity vật lý.

Trường cấp frame giữ `duration_capture_gap_seconds`, `duration_max_gap` lấy từ tracker hiện tại, `production_timestamp_valid`, `previous_logged_frame_id`, `duration_missing_person_track_ids`, `duration_reason_source=diagnostic_classification_of_production_outputs`.

Analyzer ghép hai mảng bằng Person Track ID, không dùng thứ tự row hoặc SID để ghép. PID trùng, tập PID khác nhau hoặc identity mâu thuẫn được báo rõ; không âm thầm chọn identity. Hai mảng không làm tăng gấp đôi số person/SID. Hàm `sleeping_diagnostic` vẫn giữ entry point tương thích cho caller cũ.

## Phân tích một log/session

Từ repository root:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_test_toolkit.py analyze `
  .\ai_training\runs\person_tracker_debug\session_808dd75fae434122b1d0a03c44fa819c `
  --provenance recorded
```

Có thể đưa trực tiếp đường dẫn file JSONL. Một thư mục phải có `frames.jsonl` hoặc đúng một file JSONL; toolkit không tự chọn session mới nhất hoặc ghép log khác session.

`--provenance recorded|synthetic|unverified` là khai báo của người dùng, không phải chứng minh độc lập. Mặc định unverified. Schema fixture luôn bị đánh dấu synthetic dù khai báo recorded.

Mặc định tạo thư mục báo cáo UUID mới tại `ai_training/runs/camera_test_toolkit/analyze_<uuid>/`. Có thể dùng `--output-dir <THU_MUC_MOI>`. Thư mục đã tồn tại, kể cả rỗng, bị từ chối để không ghi đè.

Output:

- `analysis.md`: facts từ log, dữ liệu thiếu, timeline có nhãn observed/logged_diagnostic/inferred và technical checks.
- `analysis.json`: dữ liệu chuẩn hóa, source SHA256, coverage, issues và events để kiểm toán.
- `frames.csv`: Frame ID/capture, current tracks và độ đầy đủ dữ liệu, kể cả frame không có person.
- `observations.csv`: mỗi person × behavior; gồm SID/PID/association/identity_status, state, confidence, decision_reason, duration, event, blockers/reset reasons. state_available/duration_available phân biệt thiếu key với duration null đã được ghi.
- `events.csv`: mất association/SID, đổi tập track, duplicate SID, expiry count trong summary, duration tăng/đứng/reset candidate và timestamp anomalies.

State không có dữ liệu được biểu diễn `states=null` / MISSING DATA, không phải not_detected hoặc “0 lần detected”. Duration null khác duration 0. Exact duplicate record không cộng lặp frame; cùng Frame ID nhưng payload khác là lỗi dữ liệu.

`analysis.md` có bảng state counts và bảng observation chi tiết cho cả hai behavior; `analysis.json` giữ metadata/evidence tương ứng. `test_results.md/.csv/.json` dùng dữ liệu này kiểm tra các case S/P, vẫn cần human review để nghiệm thu. Lệnh analyze mặc định đọc cả hai hành vi, không cần cờ chọn behavior.

Các cờ `*_available` mô tả có trường hợp lệ trong log, không có nghĩa giá trị đã nhận diện thành công. SID null/Unknown được chuẩn hóa thành chưa có SID hợp lệ; không dùng shadow proposal điền vào chỗ trống này. Tập current tracks đổi hoặc số association giảm không chứng minh actor nào đã rời/được thay thế.

## Manifest và báo cáo 16 case

Checklist đầy đủ: `CAMERA_LIVE_TEST_CHECKLIST.md`.

Tạo manifest mới (hoặc sao chép file example trước khi chỉnh):

```powershell
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_test_toolkit.py init-manifest `
  --output .\camera_test_manifest.local.json
```

Mặc định cả 16 case đều `run=false` → NOT RUN. Sau một test thật, sửa entry tương ứng:

```json
{
  "case_id": "S1",
  "run": true,
  "log": "ai_training/runs/identity_continuity_shadow/session_REPLACE/frames.jsonl",
  "provenance": "recorded",
  "frame_ids": [],
  "ground_truth": {
    "confirmed": false,
    "verdict": "INCONCLUSIVE",
    "reviewer": "",
    "evidence": ""
  }
}
```

Đường dẫn log tương đối theo thư mục chứa manifest; đường dẫn tuyệt đối cũng được hỗ trợ. `frame_ids=[]` nghĩa là toàn log. Nếu chọn frame nhưng frame không có, không được PASS bằng cách bỏ qua phần thiếu.

Chỉ người kiểm thử mới đổi `confirmed=true`, ghi reviewer, evidence gắn với Frame ID và verdict PASS/FAIL/INCONCLUSIVE. Không lấy model confidence hoặc detected để tự điền phần này. Không điền PASS cho thử nghiệm chưa thực hiện.

```powershell
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_test_toolkit.py report `
  .\camera_test_manifest.local.json
```

Output trong thư mục mới: `test_results.md`, `test_results.csv`, `test_results.json`, gồm đủ bốn tổng PASS/FAIL/INCONCLUSIVE/NOT RUN. Báo cáo tách từng automatic check khỏi manual review.

Quy tắc:

- NOT RUN: chưa khai báo đã chạy.
- INCONCLUSIVE: log/path/field/ground truth thiếu; provenance synthetic/unverified không đủ cho case camera thật.
- FAIL: điều kiện kỹ thuật kiểm chứng được từ log ghi nhận lỗi, hoặc có verdict FAIL do người quan sát xác nhận. Technical FAIL không tự là kết luận model nhận diện sai.
- PASS: mọi technical check bắt buộc PASS **và** người kiểm thử xác nhận ground truth PASS, có reviewer/evidence, provenance được khai báo recorded.
- `duration_progress`: chỉ đánh giá cặp observation liền nhau có cùng reported PID/SID, identity/state/timestamp đủ điều kiện và max gap đã ghi trong log. Bất kỳ cặp đủ điều kiện nào không tăng sẽ không PASS. Không tự đặt max gap khi log thiếu.
- Không chuyển shadow SID thành production SID; raw duplicate association claim khác duplicate SID thực sự được attribution trong production.

Exit code: 0 = công cụ hoàn tất, có thể vẫn có INCONCLUSIVE/NOT RUN; 1 = có technical FAIL hoặc case FAIL đã được báo cáo; 2 = lỗi invocation/input/manifest hoặc output đã tồn tại. Exit code 0 không có nghĩa camera/model PASS.

## Fixture format cho kiểm thử offline

Schema `camera_test_fixture_v1` do toolkit định nghĩa chỉ cho dữ liệu tổng hợp, không phải schema Camera Live hiện tại. Bắt buộc `synthetic=true`. Mỗi frame có frame_id/captured_at/people; mỗi person có person_track_id, reported student_id/association/identity_status và behaviors.sleeping/using_phone chứa state, estimated_seconds, model_confidence nếu có.

Format này cho phép kiểm thử analyzer độc lập. Các test integration còn ghi JSONL thực bằng Shadow writer từ worker replies tổng hợp, rồi đọc lại bằng analyzer. Đó là kiểm thử offline, không phải session camera thật hoặc model inference. Không giả lập replay camera rồi tuyên bố PASS.

Chạy test:

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s .\ai_training\integration `
  -p test_camera_live_test_toolkit.py -v
```

## Khi có webcam

1. Chọn một case và xác định nhãn/actor cần người quan sát xác nhận trước khi chạy. Không cố ghép tất cả case vào một phiên 10 AI frames.
2. Bật Shadow diagnostic trong đúng PowerShell dùng chạy Camera Live; không bật snapshot, thay model hoặc threshold qua toolkit:

```powershell
$env:HTTM_IDENTITY_SHADOW = '1'
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

3. Chờ thông báo first frame written và ghi lại đường dẫn session thực tế. Dùng observer/ghi chú độc lập nếu không nhìn được Panel lúc cúi đầu; không mặc định face absence chứng minh đang Sleeping.
4. Thực hiện checklist, dừng bằng Q khi cần; kiểm tra đủ các giai đoạn qua Frame ID. Không có record thì chưa thể phân tích kết quả.
5. Chạy analyze trên đúng file vừa ghi; xem cả Sleeping và Using Phone, thiếu trường và timeline trước khi kết luận. Log cũ thiếu Using Phone vẫn giữ các điều kiện tương ứng INCONCLUSIVE. Không có identity hợp lệ thì duration null có thể đúng chính sách hiện tại.
6. Cập nhật manifest với evidence/ground truth thật, chạy report; chỉ chốt những case có đủ bằng chứng.

## Giới hạn và rollback

- PID/SID ổn định không chứng minh cùng actor. Bbox, confidence và detected không thay thế ground truth.
- Không đánh giá precision/recall/mAP khi chưa có nhãn frame/actor tương ứng. Không chẩn đoán ngủ sinh lý.
- Không cấp identity từ Shadow cho production; Sleeping Duration có thể null đúng thiết kế khi mất SID.
- Shadow diagnostic mặc định tắt; chỉ writer và analyzer được mở rộng, không sửa detection, duration, Camera Coordinator hoặc WarningObserver. Tắt ghi diagnostic bằng `Remove-Item Env:HTTM_IDENTITY_SHADOW -ErrorAction SilentlyContinue` trong PowerShell dùng khởi động phiên kế tiếp. Rollback instrumentation hoàn toàn: phục hồi các file writer/analyzer/test/tài liệu theo bản sao trước thay đổi; không reset Git toàn repo và không xóa log/báo cáo cũ.
- WARNING giữ nguyên disabled; không commit/push, training, inference hoặc webcam access trong tác vụ chuẩn bị toolkit.
