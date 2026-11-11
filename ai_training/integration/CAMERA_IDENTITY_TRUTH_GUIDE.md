# Real-camera identity safety — chuẩn bị và đánh giá offline

Mục đích là so sánh identity production/shadow với **nhãn actor do người kiểm thử xác nhận**.
Công cụ không gọi model, camera, resolver hoặc DurationTracker; không điều chỉnh retention.
`T_hold=None`, Shadow không cấp SID cho production và WARNING vẫn disabled.
PASS chỉ có nghĩa các identity được log khớp với nhãn đã cung cấp trong phạm vi đánh giá;
không nghiệm thu camera/model hoặc phê duyệt một T_hold.

## Audit schema thực tế

| Nguồn | Dữ liệu dùng | Giới hạn |
|---|---|---|
| `person_tracker_diagnostic_v1` | frame_id, captured_at, tracker_updated_at, clock, tracks_after PID/bbox/last_seen, raw associations, details_truncated, snapshot metadata nếu có | SID association không chứng minh SID đã được production attribution chấp nhận. Track last_seen khác update time không phải detection hiện tại. |
| `identity_continuity_shadow_v1` | observations PID, proposed_student_id, status, face_present, track_continuity; V2 sleeping/using_phone rows có student_id/identity_status/association_status từ BehaviorFrame | Không có person bbox; không có ground truth. Legacy không có production rows thì production SID MISSING, không mượn proposal. |
| Camera Live Test Toolkit | strict JSON, resolve session/file và logic đọc production rows Sleeping/Using Phone dùng chung | Analyzer cũ tách từng session, không tự gán actor truth hoặc ghép hai logger. Công cụ mới dùng lại các helper đọc dữ liệu, không sửa pipeline hoặc schema cũ. |

Hai logger sinh UUID session độc lập và JSONL **không có capture-session ID chung**.
Tên thư mục source được dùng làm `sources.<role>.session_id`; `capture_session_id`
là tên nhóm phiên do người kiểm thử khai báo, không phải bằng chứng tự động.

## Quy tắc đối chiếu

1. Bind đường dẫn file, session thư mục và SHA256 chính xác cho từng nguồn. Đổi file,
   session hoặc nội dung sau gán nhãn sẽ bị từ chối; không tự chuyển nhãn sang file mới.
2. Khi có hai nguồn, người kiểm thử xác nhận chúng từ cùng một lần chạy bằng
   `pairing.confirmed/reviewer/evidence`. Chỉ cùng Frame ID chưa đủ để join.
3. Join mặc định bằng Frame ID và bắt buộc **captured_at, tracker_updated_at, clock khớp chính xác**.
   Clock phải `perf_counter`; không đoán offset/tolerance, làm tròn, nearest-time hoặc thứ tự dòng.
4. Frame ID khác nhau: cung cấp `frame_map` one-to-one, có người xác nhận từng cặp.
   Hai timestamp vẫn phải khớp chính xác. Timestamp khác nhau hoặc khác hệ clock
   chưa có công cụ calibration được chứng minh: INCONCLUSIVE, không override bằng lời xác nhận.
5. Trong một cặp frame, ghép bằng Person Track ID duy nhất, không dùng SID hoặc thứ tự detection.
   Tập PID khác nhau, metadata Sleeping/Phone mâu thuẫn hoặc frame lỗi sẽ chặn join.
6. Actor label chỉ có hiệu lực trên observation được liệt kê cụ thể với đủ PID,
   Frame ID, capture và update time. Không truyền label theo lịch sử PID.

SHA256 xác nhận dữ liệu không đổi, không xác nhận nhãn người quan sát đúng.
Những xác nhận manual vẫn cần bằng chứng độc lập để reviewer kiểm tra.

## Schema actor truth `httm_actor_truth_v1`

CLI `init` tạo `sources`, `pairing`, `observation_catalog`; **actors/annotations để trống**.
Catalog chỉ có key observation, bbox và snapshot metadata; không tự copy SID AI làm roster.
Ví dụ dưới đây minh họa cấu trúc, không phải dữ liệu camera đã được xác nhận:

```json
{
  "schema": "httm_actor_truth_v1",
  "capture_session_id": "run_sleeping_two_students_01",
  "provenance": "recorded",
  "sources": {
    "person": {"path": "ABSOLUTE_PERSON_FRAMES_JSONL", "session_id": "session_PERSON", "sha256": "FROM_INIT"},
    "shadow": {"path": "ABSOLUTE_SHADOW_FRAMES_JSONL", "session_id": "session_SHADOW", "sha256": "FROM_INIT"}
  },
  "pairing": {
    "confirmed": true, "reviewer": "nguoi_kiem_thu",
    "evidence": "Hai duong dan session tu cung mot lan chay, doi chieu metadata frame", "frame_map": []
  },
  "actors": [
    {"actor_id": "actor_a", "expected_student_id": "SV001", "basis": "manual_roster",
     "independent_of_ai": true, "confirmed": true, "reviewer": "nguoi_kiem_thu",
     "evidence": "Nguoi tham gia A duoc doi chieu danh sach lop doc lap voi ket qua AI"}
  ],
  "annotations": [
    {"kind": "frame", "actor_id": "actor_a", "basis": "manual_visual",
     "independent_of_ai": true, "confirmed": true, "reviewer": "nguoi_kiem_thu",
     "evidence": "Actor A trong bang chung dung observation frame 1; khong lay nhan tu SID AI",
     "members": [
       {"person_frame_id": 1, "person_track_id": "P1", "captured_at": 101.0, "tracker_updated_at": 201.0}
     ]}
  ]
}
```

Thêm actor_b/actor_c tương tự từ roster độc lập. Actor ID là nhãn người thật trong
phiên, không phải tên khác của PID hoặc SID AI. Có actor nhưng chưa biết SID chuẩn
thì để expected_student_id=null: actor truth có thể hợp lệ nhưng chưa chấm đúng/sai SID.

Để ghi nhãn đoạn thời gian, dùng `kind=segment`, thêm `capture_start/capture_end`,
liệt kê **từng member đã quan sát và xác nhận**. Các member có thể có PID khác nhau
nếu người kiểm thử xác minh cùng actor. Không tự mở rộng label cho mọi PID/frame
trong khoảng. Người đổi vị trí/thay thế phải được tách theo actor thật, kể cả PID giữ nguyên.

Mapping Frame ID khác nhau, chỉ khi có bằng chứng:

```json
{"person_frame_id": 1, "shadow_frame_id": 101, "confirmed": true,
 "reviewer": "nguoi_kiem_thu", "evidence": "capture/update timestamps bang nhau va dung cung observation"}
```

Không cho một actor chiếm hai PID trong cùng frame; label trùng/xung đột bị loại.
Không dùng `detected`, confidence, production SID, shadow SID hoặc PID ổn định để
khẳng định actor. Tool kiểm tra khai báo độc lập, không thể xác thực người nhập có nhìn đúng hay không.

## Workflow khi có webcam — chưa thực hiện trong tác vụ này

1. Chọn người quan sát và actor_a/actor_b; ghi roster thủ công trước khi xem prediction.
   Dùng đặc điểm nhận biết do người quan sát xác nhận và lưu bằng chứng theo frame.
2. Trong cùng PowerShell, bật hai diagnostic hiện có rồi chạy Camera Live:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_IDENTITY_SHADOW = '1'
# Nếu người kiểm thử chủ động cần ảnh observation để gán nhãn, cơ chế có sẵn:
# $env:HTTM_PERSON_DEBUG_SNAPSHOTS = '1'
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Đây là hướng dẫn cho tương lai; công cụ không tự chạy lệnh camera hoặc bật snapshot.
Snapshot là lựa chọn thu thập dữ liệu hiện có, không thêm recorder/ảnh/embedding mới.
Không đặt env hold period, không sửa camera index, model hoặc threshold.

3. Lưu **cả hai đường dẫn session thực tế trong console** và case ID. Mỗi phiên hiện
   tối đa 10 AI observations; không cố ghép nhiều lần khởi động thành một phiên.
4. Với từng AI Frame ID, xác minh actor trong đúng observation. AI Panel hiển thị
   kết quả lịch sử; ảnh live bên trái có thể khác thời điểm inference. Không dùng
   ảnh live hiện tại để gán nhãn frame AI cũ nếu chưa có bằng chứng tương ứng.
   Khi có snapshot đúng frame, đối chiếu Frame ID/capture và metadata ảnh đã lưu.
   Công cụ mới không mở ảnh hoặc xác thực bằng chứng hình ảnh thay người kiểm thử.
   Nếu không nhìn rõ actor hoặc không có evidence đúng frame, để observation chưa gán nhãn.
5. Sau khi dừng camera, tạo template từ hai session vừa ghi:

```powershell
$personSession = '.\ai_training\runs\person_tracker_debug\session_PERSON_REPLACE'
$shadowSession = '.\ai_training\runs\identity_continuity_shadow\session_SHADOW_REPLACE'
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_identity_truth.py init `
  --person $personSession --shadow $shadowSession `
  --capture-session-id 'run_two_students_01' --provenance recorded `
  --output .\actor_truth_run01.json
```

Sửa file JSON theo schema: xác nhận pairing, khai báo roster độc lập, thêm annotations.
Copy key frame/PID/timestamps từ catalog sau khi đã xác định actor bằng chứng thủ công;
catalog không tự gán actor hoặc expected SID. Nguồn log tương đối được resolve theo
thư mục chứa truth JSON; init mặc định ghi đường dẫn tuyệt đối.

6. Chạy đánh giá và xem sai identity trước khi xem retention đúng:

```powershell
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_identity_truth.py evaluate `
  .\actor_truth_run01.json --output-dir .\ai_training\runs\actor_truth_run01_NEW
```

Output directory phải chưa tồn tại. `init` cũng không ghi đè file truth cũ.
Exit 0 = hoàn tất PASS hoặc INCONCLUSIVE (đọc status); exit 1 = có wrong SID hoặc
duplicate SID production/shadow đã báo cáo; exit 2 = input/schema/hash/mapping không hợp lệ.
Không coi exit 0 là Camera Live PASS.

## Báo cáo và cách diễn giải

- `identity.md`: counts, bảng actor/expected/production/shadow/verdict, event timeline,
  duplicate claims, mismatch/thiếu bằng chứng và giới hạn.
- `identity.csv`: mỗi current PersonTracker detection; có hai session ID, Frame ID,
  timestamps/PID/bbox, actor truth, expected SID, hai identity channels, reviewer/evidence,
  lý do không chấm được, face presence và shadow reason.
- `identity.json`: đầy đủ metadata/source hashes, metrics, events, issues và rows.
- `events.csv`: mất face/nhận diện lại, PID đổi với cùng actor thủ công, actor khác
  trên cùng PID, bbox overlap và SID sai. Đây là thay đổi giữa các observation đã
  gán nhãn, không chứng minh liên tục trong khoảng trống.
- `unmatched_shadow.csv`: dữ liệu shadow chưa ghép, không bị âm thầm bỏ qua.

Tổng observation = current PersonTracker detections; retained tracks không được
quan sát không tính là người hiện tại. Shadow không ghép được được đếm riêng.
Mỗi channel có CORRECT/WRONG/UNKNOWN/MISSING/UNSCORABLE. `ambiguous_status` là trạng
thái log, không tự đồng nghĩa model nhận diện sai. False retention chỉ tính khi
status `track_retained` và proposal sai theo actor truth hợp lệ.

Duplicate SID được tính theo nhiều PID khác nhau trong cùng **frame nguồn** của
Shadow log, riêng production và proposal; có thể phát hiện khi chưa có actor truth
hoặc chưa pair nguồn. Raw association duplicate được giữ riêng, không tự coi là
duplicate SID attribution production. Một PID xuất hiện trong hai behavior rows
không bị đếm thành hai người.

PASS chỉ khi mọi observation chấm được, identity hai channel đều đúng, không
unknown/ambiguous/missing/unmatched/issue. FAIL khi có wrong identity có nhãn hoặc
duplicate SID nguồn production/shadow. Còn lại INCONCLUSIVE. Duplicate là lỗi
claim log; không phải kết luận actor ground truth từ SID.

## Những trường hợp INCONCLUSIVE

- Thiếu/không xác nhận roster, actor label hoặc bằng chứng độc lập; không thấy actor rõ.
- Hai annotation xung đột hoặc một actor được gán cho hai PID cùng frame.
- Nguồn chưa được xác nhận cùng phiên, timestamp/clock khác, frame mapping không chứng minh được.
- Source bị truncate, timestamp không hợp lệ, PID sets/production metadata mâu thuẫn.
- Log cũ chỉ có raw associations hoặc shadow proposal, chưa có production attribution.
- Actor có nhãn nhưng SID chuẩn chưa biết; identity Unknown/ambiguous hoặc source/frame bị thiếu.
- Có shadow observations chưa đối chiếu, hoặc session không có observation.

Hash/session mismatch, JSON lỗi, duplicate Frame ID hoặc mapping không hợp lệ
được từ chối rõ ràng bằng exit 2; không cố chấm một tập dữ liệu bị thay đổi.
Không tự biến log cũ thiếu nhãn thành camera ground truth.

## Kịch bản camera cần ưu tiên

Thực hiện các case IC trong checklist hiện có: mất face rồi nhận diện lại;
PID đổi; hai người đổi chỗ/crossing/overlap; một người rời và người khác thay thế
cùng vị trí, gồm thay thế giữa AI updates; recognition sai/trùng SID. Ghi actor
truth trước khi đối chiếu prediction. Retention sai phải được xem trước continuity đúng.
Using Phone/Sleeping chỉ là dữ liệu behavior phụ trợ, không dùng làm evidence identity.

## Kiểm thử offline và rollback

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s .\ai_training\integration -p test_camera_identity_truth.py -v
```

Fixture module tạo log/roster/nhãn synthetic độc lập. Các output retention cố ý
adversarial, gồm vi phạm uniqueness, không phải mô phỏng đã bật retention trên
Camera Live. Không lấy tỷ lệ test fixture làm precision/recall hoặc risk trên camera.
Fixture source có `synthetic=true` nên không thể nâng thành recorded chỉ bằng đổi manifest.

Rollback: ngừng sử dụng ba module mới và hướng dẫn này; production không có gì
cần phục hồi. Link hướng dẫn trong CAMERA_LIVE_TEST_TOOLKIT.md có thể bỏ riêng nếu
cần, giữ mọi sửa đổi khác của người dùng. Không xóa log/truth/report cũ hoặc reset Git.
