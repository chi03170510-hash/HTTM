# PersonTracker Diagnostic A — Instrumentation + Offline Replay

Ngày: 2026-11-10. Phạm vi: công cụ chẩn đoán opt-in; không sửa thuật toán hay
tham số production. Không chạy camera thật trong lượt này.

**Kết quả cuối: 194/194 test đạt, 0 failure, 0 error, 0 skip**, gồm 26 test mới
và 168 regression test đã có. Hai lần chạy CLI replay A/B trả exit code 0,
consistent cả ba frame. Association trên predictions đã lưu đạt thêm 2/2 ảnh,
báo cáo riêng ngoài 194 test.

## 1. Files changed và kiểm tra trước khi sửa

| File trong `ai_training/integration` | Thay đổi |
|---|---|
| `person_tracker_diagnostics.py` | Mới: cấu hình, metadata projection, sidecar tạm, log JSONL, snapshot và quota |
| `replay_person_tracker_diagnostics.py` | Mới: CLI đọc log và replay tracker, không chạy AI hoặc camera |
| `camera_coordinator_live.py` | Copy state trước update khi opt-in; ghi diagnostic sau khi đưa result vào queue |
| `hybrid_v7_stream_worker.py` | Khi opt-in, xuất provenance person thực có qua sidecar tạm |
| `test_person_tracker_diagnostics.py` | 15 test instrumentation/integration |
| `test_person_tracker_diagnostic_replay.py` | 7 test bbox/timing/replay |
| `test_hybrid_person_diagnostics.py` | 4 test method worker thật với model boundary được mock |
| `PERSON_TRACKER_DIAGNOSTIC_A_REPORT.md` | Báo cáo và hướng dẫn sử dụng |

Đã đọc Git status/source/test trước khi sửa và đối chiếu bản trước sửa.
SHA-256 của 77 file source/config/weights có sẵn xác nhận chỉ hai file production
trong bảng thay đổi. `.gitignore` có thay đổi từ trước được giữ nguyên; thư mục
integration vốn đang untracked nên không dùng riêng `git diff` để xác định scope.

Giữ nguyên `person_tracker.py`, greedy matching, `iou_threshold=0.3`, Live
`max_age=5.0`, `keep_unmatched=False`; không đổi FaceTracker, Association V4,
BehaviorDuration, WarningObserver, WARNING mặc định disabled, camera index,
confidence thresholds, model/weights hoặc CH-18/CH-19. Không thêm dependency,
commit/push hoặc triển khai CH-22. Không sửa expected result của test cũ.

## 2. Bật/tắt debug

| Biến môi trường | Mặc định | Ý nghĩa |
|---|---|---|
| `HTTM_PERSON_DEBUG` | Tắt | Chỉ chuỗi `1` bật diagnostic |
| `HTTM_PERSON_DEBUG_SNAPSHOTS` | Tắt | Chỉ `1` và master debug đang bật mới lưu PNG |
| `HTTM_PERSON_DEBUG_DIR` | `ai_training/runs/person_tracker_debug` dưới repository | Thư mục chứa session mới |
| `HTTM_PERSON_DEBUG_MAX_FRAMES` | 10 | Giới hạn session; chặn tối đa 100 |
| `HTTM_PERSON_DEBUG_MAX_BYTES` | 20971520 (20 MiB) | JSONL + PNG/session; chặn tối đa 100 MiB |

Giá trị quota âm được đưa về 0, vượt trần bị giới hạn; giá trị không phải số
dùng mặc định. Quota 0 không ghi dữ liệu. Camera Live vẫn dừng AI ở giới hạn
10 frame cũ; không tăng MAX_AI_FRAMES vì debug.

Khi tắt: không tạo session, sidecar, ảnh hoặc log file mới. Log console
`[PERSON TRACKING]` có sẵn được giữ nguyên. Khi bật: tạo một thư mục
`session_<uuid>` riêng, không ghi đè session/file có sẵn.

Quota áp dụng theo session; không tự xóa session cũ. Chạy debug nhiều lần tạo
nhiều session để người dùng chủ động quản lý. Vượt quota thì dừng diagnostic,
AI vẫn tiếp tục. Nếu chỉ ảnh vượt quota, bỏ ảnh với reason `byte_limit` và vẫn
ghi metadata nếu còn chỗ.

## 3. Metadata thực sự thu được và schema

Coordinator đã có: frame/capture/completion timestamps, processing times,
snapshot đúng frame, tracked people, matches và cấu hình tracker. Track object
đã có bbox, person reference và last_seen. Các trường này không cần thay schema
production. Diagnostic ghi:

- `frame_id`, `captured_at`, `tracker_updated_at`, `completed_at`, image dimensions.
- `tracker_update_gap`, `captured_at_gap` tách riêng `processing.elapsed/face/hybrid`.
- Tham số tracker thực tế; trạng thái trước/sau gồm Track ID, person_ref, bbox,
  last_seen và age_seconds tính tại tracker_updated_at.
- Các cặp candidate IoU, có đạt threshold hay không, có được tracker xét hay
  không, và selection suy ra từ kết quả update thực tế.
- Decision `expired`, `unmatched_removed`, `matched`, `created`; reason phân biệt
  vượt max_age, không có IoU đạt ngưỡng, hoặc thua ghép greedy.
- Association status, Person Track ID, Face Track ID và Student ID hiện tại.

Không chạy lại matching để sửa kết quả. Candidate IoU được tính sau update trên
bản copy state trước update. Với track đã hết hạn, IoU vẫn được ghi để phân tích
phản thực tế nhưng `considered_by_tracker=False`: không diễn giải thành tracker
đã thực hiện IoU matching cho track đó.

Hai detection hoàn toàn trùng reference/bbox không cho phép xác định detection
index được chọn chỉ từ output; trường `selected=null` thay vì đoán. Tối đa 100
track/detection/match được ghi chi tiết để chặn chi phí/dung lượng; record vượt
phạm vi được đánh dấu `details_truncated`, CLI từ chối replay không đầy đủ.

Confidence/source person có thật trong `obj.person_detections` của Hybrid nhưng
worker cũ bỏ các trường này khi trả lời. Đã thêm kênh debug riêng:

```text
Hybrid obj.person_detections
  → projection: person_ref/bbox + confidence/class/sources/quality thực có
  → một file frame.person_debug.json bên cạnh JPEG tạm
  → coordinator kiểm tra frame ID, image path, dimensions và bbox/reference
  → frames.jsonl trong session
```

Sidecar tối đa 64 KiB, đồng thời không vượt quota bytes đã cấu hình, chỉ giữ một
file được ghi lại, tối đa số frame debug. Nó nằm trong TemporaryDirectory của
AI loop nên được dọn cùng JPEG tạm. Worker sidecar không lưu ảnh.

**Schema JSON trả về của worker và result đưa vào Panel giữ nguyên cả khi debug
bật và tắt.** Nếu không có metadata, lookup mơ hồ, sidecar sai frame hoặc không
ghi được, log ghi trạng thái không có dữ liệu và không tạo confidence/source.
Không lấy confidence của behavior để điền confidence person.

Whitelist không lưu embedding, tên sinh viên, face crops hoặc toàn bộ payload
model. Giá trị confidence/source của P2/P4 trong camera cũ vẫn chưa thu được;
các giá trị trong test là fixture, không phải số đo camera.

## 4. Snapshot debug

Mặc định không lưu. Chỉ bật qua `HTTM_PERSON_DEBUG_SNAPSHOTS=1` cùng master flag.
Không cần đổi cơ chế snapshot CH-20: dùng trực tiếp `AISnapshot.image` đã giữ
pixel của JPEG mà cả hai worker đọc. PNG lưu lossless các pixel đã decode đó;
không đọc lại frame.jpg và không lấy frame preview live mới.

Mỗi PNG có file name theo observation/frame ID; record JSONL chứa file name,
frame ID, captured_at và SHA-256 của PNG. Kiểm tra metadata trước khi lưu. Test
đã ghi đè JPEG input rồi xác nhận PNG vẫn bằng đúng pixel inference đã giữ.

JSONL + PNG không vượt quota bytes/session. Lỗi storage hoặc encode chỉ dừng/bỏ
diagnostic tương ứng, không dừng inference/tracker. Không SQLite, API hoặc network.

## 5. Offline replay chính xác bbox frame 4–6

Test dùng đúng năm bbox đã cung cấp, PersonTracker thật và policy Live hiện tại.
Frame 4 là seed đầu tiên của replay tổng hợp để tracker tự cấp P1/P2; không phục
hồi ID bằng Student ID và không giả định biết lịch sử camera frame 1–3.

| Case | Update timestamps F4/F5/F6 | Capture timestamps F4/F5/F6 | Processing F5 | F5 | F6 |
|---|---|---|---:|---|---|
| A | 100.0 / 104.0 / 107.0 | 98.0 / 100.1 / 104.1 | 3.80s | P1 | Chính P1, phụ P3 |
| B | 100.0 / 107.2 / 110.2 | 98.0 / 100.1 / 107.3 | 7.05s | P3 | Chính P3, phụ P4 |

Case B có **update gap 7.2s**, **capture gap 2.1s** và **Processing 7.05s** ở
F4→F5. Không dùng Processing làm đồng hồ tracker. Một test khác thay metadata
capture gap/processing và xác nhận output replay không đổi khi update timestamps
không đổi.

IoU được xác nhận từ dữ liệu thực của test:

| Cặp | IoU |
|---|---:|
| Người chính F4 → bbox F5 | 0.57709974 |
| Bbox F5 → người chính F6 | 0.58768886 |
| Bbox phụ F4 → bbox F5 | 0.10377988 |

Hai CLI replay trên các log tổng hợp A/B đã chạy thật: exit code 0, cả ba frame
`consistent=true`. Đây là tái hiện có kiểm soát, không phải replay timestamp
camera thực tế chưa được cung cấp.

## 6. Khi nào giữ P1 và khi nào tạo P3

- Gap update F4→F5 **không quá 5.0s**: P1 còn sống, IoU 0.5771 đạt ngưỡng nên
  giữ P1 qua bbox rộng và lúc thu nhỏ ở F6.
- Gap **vượt 5.0s**: P1/P2 bị xóa trước matching; F5 tạo ID tiếp theo P3. Log vẫn
  chỉ ra hình học P1/F5 đủ ngưỡng, nhưng candidate không được xét do expiry.
- Đúng 5.0s giữ P1; 5.001s tạo P3 trong replay seed P1/P2. Cả hai biên đã test.
- Case A: P2 bị xóa ở F5 vì unmatched dù chưa expired; detection phụ F6 nhận P3.
  Case B: P2 đã expired ở F5; detection phụ F6 nhận P4.

Kết quả xác nhận cơ chế có thể tạo đúng chuỗi ID đã thấy; chưa chốt nguyên nhân
camera cũ cho tới khi có updated_at thực tế. Không thay max_age hoặc IoU để ép ID.

## 7. Regression tests

| Bộ test | Đạt | Failure/Error |
|---|---:|---:|
| Diagnostic instrumentation + Live integration mới | 15 | 0 |
| Diagnostic bbox/offline replay mới | 7 | 0 |
| Hybrid worker diagnostic mới | 4 | 0 |
| PersonTracker | 17 | 0 |
| Association script nền hiện có | 5 | 0 |
| BehaviorDuration | 17 | 0 |
| CH-20 pipeline | 16 | 0 |
| CH-21 observer | 27 | 0 |
| CH-21 pipeline | 20 | 0 |
| Camera/Panel/Person integration | 14 | 0 |
| Layout | 8 | 0 |
| Snapshot/Overlay | 10 | 0 |
| Hybrid V7 | 34 | 0 |
| **Tổng** | **194** | **0** |

Association V4 thật chạy trong Camera, CH-20, CH-21 và diagnostic integration
harness, gồm ca heuristic. Script 5/5 hiện có kiểm tra association nền, không
gọi nhầm nó là suite V4 độc lập. Predictions đã lưu: 2/2 ảnh đạt, không chạy model
mới, không cộng vào số test trên.

Lệnh đã thực thi:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# Ran 151 tests, OK (lần cuối: 8.561s)
..\..\.venv-hybrid\Scripts\python.exe -m unittest test_hybrid_person_diagnostics -q
# Ran 4 tests, OK

Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -q
# Ran 34 tests, OK
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS 5/5
.\.venv-hybrid\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# 2/2 images
```

Diagnostic integration so sánh cùng input khi on/off: result keys, matches,
tracked people/faces, BehaviorFrame và WarningFrame bằng nhau. Kiểm thử lỗi
disk vẫn hoàn tất AI frames; mặc định không ghi ảnh, snapshot flag riêng không
tự bật master, quota frame/bytes có tác dụng, provenance sai frame bị từ chối.
Harness chặn SQLite, socket/API và khởi chạy model subprocess thật.

## 8. Hiệu năng và quyền riêng tư

Khi tắt, chỉ khởi tạo cấu hình và kiểm tra flag; không copy track state, tính
candidate IoU debug hoặc ghi file. Khi bật, copy state trước update; các cặp IoU
debug và ghi JSONL/PNG diễn ra sau update/result queue. Không thêm inference,
thread hoặc dependency.

Chi phí chính dự kiến là ma trận track×detection (giới hạn 100×100), serialize,
disk I/O và PNG encode nếu chủ động bật ảnh. Source sidecar nằm trong thời gian
Hybrid worker; ghi JSONL/PNG sau result không nằm trong `processing.elapsed`
đã đo nhưng có thể làm tăng gap của lần update tiếp theo. Debug không phải phép
đo hiệu năng không gây ảnh hưởng. Chưa đo FPS/latency camera thật; thời gian
unit test không đại diện hiệu năng iVCam.

Chỉ ghi Student ID/Face Track ID/Person Track ID và association cần chẩn đoán,
không embeddings/tên. Snapshot chứa toàn ảnh inference nên có thể có người/nền;
chỉ lưu khi chủ động bật và trong quota. Không gửi ra mạng, không DB. Session
cũ không bị tự xóa hoặc ghi đè.

## 9. PowerShell chạy camera với debug

Mặc định lệnh này chỉ lưu metadata; WARNING vẫn disabled:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_PERSON_DEBUG_SNAPSHOTS = '0'
$env:HTTM_PERSON_DEBUG_MAX_FRAMES = '10'
$env:HTTM_PERSON_DEBUG_MAX_BYTES = '20971520'
$env:HTTM_PERSON_DEBUG_DIR = Join-Path (Get-Location) 'ai_training\runs\person_tracker_debug'
try {
    .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
} finally {
    Remove-Item Env:HTTM_PERSON_DEBUG, Env:HTTM_PERSON_DEBUG_SNAPSHOTS, Env:HTTM_PERSON_DEBUG_MAX_FRAMES, Env:HTTM_PERSON_DEBUG_MAX_BYTES, Env:HTTM_PERSON_DEBUG_DIR -ErrorAction SilentlyContinue
}
```

**Nếu chủ động muốn lưu ảnh để xác minh P2/P4**, thay dòng SNAPSHOTS từ `'0'`
thành `'1'` trước khi chạy. Không có flag nào bật WARNING.

Đọc đường dẫn session từ `[PERSON DEBUG] ...` trên stderr. Replay hoàn toàn
offline, thay `<session>` bằng thư mục vừa tạo:

```powershell
.\.venv\Scripts\python.exe .\ai_training\integration\replay_person_tracker_diagnostics.py '.\ai_training\runs\person_tracker_debug\<session>\frames.jsonl'
```

CLI chỉ đọc JSONL, dùng `tracker_updated_at` và parameters đã ghi; không đọc ảnh,
không chạy model, không ghi file. Exit 0: mọi output nhất quán; 1: có mismatch;
2: log không hợp lệ/không đầy đủ. Cần toàn bộ session từ tracker ban đầu rỗng,
không dùng riêng đoạn frame 4–6 của một phiên camera đã chạy trước đó. Log cuối
bị ngắt giữa dòng do crash cần được kiểm tra, không tự sửa hoặc nội suy dữ liệu.

Để tắt debug ở các lần chạy khác, bỏ biến `HTTM_PERSON_DEBUG` hoặc đặt `'0'`.
Lệnh try/finally ở trên xóa các biến của lần chạy này khi Camera Live thoát.

## 10. Thông tin còn thiếu và bước tiếp theo

Chưa biết confidence/source/quality và ảnh inference thực tế của P2/P4 trong
camera cũ. Chưa thể kết luận false positive, duplicate cùng người, partial body
hay một người khác chỉ từ bbox/count. Những trường đó có thể thu ở lần chạy
mới; không thể phục hồi snapshot cũ từ frame.jpg đã bị ghi đè.

Người dùng chạy một session debug (chủ động bật snapshot nếu muốn kiểm tra vùng
phụ), gửi frames.jsonl cùng PNG frame liên quan nếu có. Đối chiếu update gap và
reason trước, rồi xem provenance/snapshot của bbox phụ. Chỉ sau bằng chứng mới
đề xuất tham số hoặc lọc detection ở nhiệm vụ riêng.

**Dừng sau Diagnostic A. Không sửa thuật toán, không tăng max_age, không bật
WARNING và không triển khai CH-22.**
