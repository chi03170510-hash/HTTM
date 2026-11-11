# FaceTracker Diagnostic A — Instrumentation + Offline Replay

Ngày: 2026-11-10. **219/219 test đạt, 0 failure, 0 error, 0 skip**: 25 test
Face Diagnostic mới và toàn bộ 194 regression test trước đó. Không chạy camera
hoặc inference model thật trong lượt này. Chưa triển khai phương án B/CH-22.

## 1. Files changed

| File trong `ai_training/integration` | Thay đổi |
|---|---|
| `face_tracker_diagnostics.py` | Mới: quan sát state, candidate IoU, decisions và mapping toàn bộ face |
| `replay_face_tracker_diagnostics.py` | Mới: CLI chỉ đọc shared JSONL và replay FaceTracker thật |
| `camera_coordinator_live.py` | Nhánh opt-in lưu timestamp được truyền vào update và state trước update |
| `person_tracker_diagnostics.py` | Thêm face flag và optional `face_tracker` block vào cùng JSONL/quota |
| `test_face_tracker_diagnostics.py` | 19 test offline/instrumentation/replay |
| `test_face_tracker_diagnostic_pipeline.py` | 6 test qua AI loop, V4, CH-20/21 và Panel thật |
| `FACE_TRACKER_DIAGNOSTIC_A_REPORT.md` | Báo cáo và lệnh kiểm thử lại |

Đã đọc Git status/source/tests trước khi sửa. So SHA-256 với 82 file có sẵn:
chỉ `camera_coordinator_live.py` và `person_tracker_diagnostics.py` thay đổi.
Giữ nguyên thay đổi chưa commit của người dùng, gồm `.gitignore`; thư mục
integration vốn untracked nên còn dùng bản đối chiếu ngoài Git diff.

**Không sửa** `recognition/face_tracker.py`, Face ID worker, PersonTracker,
Association V4, BehaviorDuration, WarningObserver, Hybrid V7, model/weights,
confidence threshold, camera index hoặc code CH-18/19. Giữ FaceTracker
`iou_threshold=0.3`, `max_age=2.0`, greedy matching và vòng đời track. WARNING
production tiếp tục disabled cho cả bốn behavior. Không thêm dependency,
commit hoặc push.

## 2. Bật/tắt debug và tính tương thích

Face debug cần đồng thời:

```text
HTTM_PERSON_DEBUG=1
HTTM_FACE_DEBUG=1
```

Mặc định cả hai tắt. Face flag riêng không bật session. Nếu chỉ bật Person debug,
JSONL giữ các trường cũ và không có face block. Với Face debug, root schema vẫn
là `person_tracker_diagnostic_v1`, thêm duy nhất optional key:

```json
{"face_tracker": {"schema": "face_tracker_diagnostic_v1"}}
```

AI result production và protocol worker không thêm trường. Person replay cũ
đọc được log có face block; mọi trường root cũ bằng nhau trong test on/off trên
cùng input. Không đổi file log của session camera cũ.

Dùng chung quota mặc định **10 frame, 20971520 byte (20 MiB)/session**. Face block
được tính vào chính quota JSONL + PNG cũ, không có quota/file persistence riêng.
Các cấu hình quota của Diagnostic A trước được giữ nguyên, không tự tăng giới hạn.
`HTTM_PERSON_DEBUG_SNAPSHOTS` vẫn mặc định tắt và độc lập với Face flag.

## 3. Metadata FaceTracker thu được

Khi opt-in, block của mỗi AI frame chứa:

- `frame_id`, `captured_at`, `face_tracker_updated_at`: chính timestamp truyền
  vào FaceTracker.update, không suy ra từ PersonTracker gap.
- `face_update_gap`, `person_tracker_updated_at`, `face_to_person_seconds`.
- `processing.face`, `processing.hybrid`: các khoảng đo worker đang có.
- `parameters` lấy từ tracker thật; `next_id_before/after`.
- `detections`: detection index, face_bbox, current_student_id và recognition_score
  thực có trong face worker reply. Không dùng stable label để thay current ID.
- `tracks_before/after`: Face ID dạng số, bbox, last_seen, age_seconds, stable_label,
  last_student_id, score thực có và updated_in_frame.
- Candidate IoU, threshold_passed, considered_by_tracker, selected.
- Decisions: expired / unmatched_retained / new / matched; reason chỉ rõ quá
  max_age, không đủ IoU hoặc thua ghép greedy.
- Mapping Face ID → detection indices → person_ref/Person Track/association status.
  Face không được association vẫn có mapping với `associations=[]`.
- Tổng số detection/track và cờ details_truncated.

State trước được copy khi debug bật. Toàn bộ tính IoU debug/đối chiếu post-state
diễn ra sau update thật, trong bước ghi diagnostic sau result queue. Không thay
đổi matching. Identity/score chỉ giúp suy ra detection nào đã được tiêu thụ từ
post-state; chúng không tham gia quyết định của thuật toán production.

IoU của track expired được ghi như dữ liệu phân tích, với
`considered_by_tracker=False`: không ngụ ý tracker đã ghép track quá hạn.
Nếu detection trùng hoàn toàn hoặc không phân biệt được index từ output,
`selected=null`/`selection_ambiguous=True`, không đoán index.

Giới hạn chi tiết giữ ở 100 face/track/match, nhất quán với hạ tầng debug hiện có.
Trong phạm vi đó ghi cả face không ghép được. Vượt giới hạn có count và cờ
truncated rõ ràng, không lọc output tracker; Face CLI từ chối replay dữ liệu thiếu.

Không log embedding, landmark vector, tên sinh viên, face crop hoặc trường tùy ý
từ worker reply. Thiếu ID/score thì bỏ trường, không điền Student ID hay score giả.
Face CLI từ chối exact replay nếu thiếu recognition metadata cần tái tạo state.

## 4. Replay A–H

| Case | Dữ liệu kiểm soát | Kết quả thực tế |
|---|---|---|
| A | Một bbox/SID, gap 1.5s | Giữ Face ID 1, matched |
| B | Cùng bbox/SID, gap đúng 2.0s | Giữ Face ID 1; age=2.0 chưa expire |
| C | Cùng bbox/SID, gap 2.001s | Face ID 1 expired, tạo ID 2 dù IoU=1 |
| D | Sáu gap 3.715/3.377/2.926/2.975/2.847/3.106s | Seed ID 1 rồi cấp ID 2→3→4→5→6→7; mỗi bước expired + new, bbox không đổi |
| E | Hai bbox/SID phân biệt, gap 1.5s, 10 frame | Giữ hai Face ID 1/2, không tráo; stable labels đủ phiếu |
| F | Hai bbox giữ vị trí, đảo thứ tự detection | Giữ Face ID theo bbox; detection indices đổi tương ứng |
| G | Pipeline thật, completion/Face gap 3s, P1/SV001 ổn định | Face IDs 1→2→3→4→5; duration 0→3→6→9→12s; cùng warning episode |
| H | Unknown, duplicate SID, mất association hoặc thiếu face | Duration/Warning ngắt episode không hợp lệ; khi hồi phục bắt đầu lại |

Case D dùng các con số gap đã nêu làm **Face update gaps tổng hợp có kiểm soát**.
Không khẳng định đây là timestamp FaceTracker đã đo trực tiếp trong camera cũ:
log cũ chỉ có Person update timestamps. Muốn xác minh trực tiếp camera cần chạy
session mới với Face flag.

Case G chạy real ai_loop → FaceTracker → Association V4 → PersonTracker →
BehaviorDuration → WarningObserver → main/Panel, chỉ mock worker replies/time/
camera/GUI. Fixture có captured_at cách completion 1.5s, đủ freshness; không để
test chỉ đi qua nhánh bị chặn. Mock config bật using_phone chỉ trong test.

Case H gồm bốn subcase, mỗi subcase 7 frame:

```text
Duration: 0, 3, 6, --, 0, 3, 6
Warning events: frame 3 và frame 7, episode IDs khác nhau
```

Trong invalid frame, mọi WarningView CLEAR. Không dùng Student ID đơn thuần
để giữ episode. Hai CLI trên shared log tổng hợp mới đều exit 0: Face replay
5/5 consistent và Person replay 5/5 consistent. Person CLI đọc log camera cũ
`session_90a6128ad12c41fb9d5c193362fac0a9/frames.jsonl` cũng consistent 10/10.

## 5. Biên expiry 2.0 / 2.001

Điều kiện source vẫn là `timestamp - last_seen > max_age` và thực hiện trước IoU.
Ở đúng 2.0, candidate được xét và IoU=1 giữ track. Ở 2.001, candidate debug vẫn
ghi IoU=1/threshold_passed=True nhưng considered_by_tracker=False; track cũ bị
xóa và ID mới được cấp. Không sửa epsilon, threshold hoặc expected result cũ.

Test bổ sung phân biệt face unmatched được **giữ nội bộ** tới hết tuổi với policy
PersonTracker xóa unmatched. Có cả ca thua cạnh tranh greedy và ca thiếu detection.

## 6. Tác động CH-20/CH-21 và schema production

Case G xác nhận Face ID đổi không reset episode khi P1/SV001 và các điều kiện
khác vẫn hợp lệ. Event đầu tiên ở frame 3, qualified span 6s; không phát lại cùng
episode dù ID tiếp tục đổi. Stable label các face track mới là Unknown, trong
khi current ID vẫn SV001; instrumentation ghi rõ hai trường này.

Case H xác nhận Unknown/trùng SID/association mất không được nối chuỗi cũ.
Case test on/off so sánh result keys, tracked faces/people, matches, BehaviorFrame,
WarningFrame và các mốc timing mock: bằng nhau. Renderer vẫn chỉ đọc result;
harness có cached repaint mỗi AI observation, không gọi observer từ render.

Production WARNING vẫn disabled. Không có Face flag nào bật cảnh báo. Không
sửa BehaviorDuration hoặc WarningObserver để làm các ca này đạt.

## 7. Regression thực thi

| Suite | Đạt | Failure/Error |
|---|---:|---:|
| Face instrumentation/offline mới | 19 | 0 |
| Face pipeline mới | 6 | 0 |
| Person Diagnostic instrumentation | 15 | 0 |
| Person Diagnostic replay | 7 | 0 |
| Hybrid worker diagnostics | 4 | 0 |
| PersonTracker | 17 | 0 |
| Association script nền | 5 | 0 |
| BehaviorDuration | 17 | 0 |
| CH-20 pipeline | 16 | 0 |
| CH-21 observer | 27 | 0 |
| CH-21 pipeline | 20 | 0 |
| Camera/Panel/Person integration | 14 | 0 |
| Layout | 8 | 0 |
| Snapshot/Overlay | 10 | 0 |
| Hybrid V7 | 34 | 0 |
| **Tổng** | **219** | **0** |

Không skip. Association V4 thật chạy trong integration harness, gồm ca matched
và heuristic; script nền 5/5 không được gọi nhầm là suite V4 độc lập. Predictions
đã lưu đạt thêm 2/2 ảnh, tính riêng, không chạy model mới.

Lệnh đã thực thi:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_face_tracker_diagnostics test_face_tracker_diagnostic_pipeline test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_behavior_duration test_behavior_overlay test_behavior_warning test_camera_person_tracking test_camera_live_layout test_person_tracker test_ch20_duration_pipeline test_ch21_warning_pipeline
# Ran 176 tests, OK (lần cuối: 9.935s)
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

`test_face_tracker_camera.py` là script camera thủ công; không chạy trong lượt
này. Các test FaceTracker offline mới dùng class production thật. Không đổi bất
kỳ expected result nào trong test cũ.

## 8. Hiệu năng và an toàn

Khi Face debug tắt, gọi trực tiếp bound method tracker.update đã chọn một lần
lúc khởi động; không kiểm tra Face flag trong vòng lặp, không thêm timer, copy
state, tính IoU, serialize, ảnh hoặc I/O Face debug. Không tuyên bố đã đo được
độ trễ tuyệt đối bằng 0 hoặc FPS camera không đổi.

Khi bật, copy state trước update và tính diagnostic sau result queue; chi phí
chính là track×detection, projection/mapping và JSONL, được giới hạn chi tiết và
quota. Không thêm inference, thread hoặc dependency. Ghi debug có thể tăng gap
lần tiếp theo; log trực tiếp phản ánh cả ảnh hưởng đó. Chưa benchmark camera.

Không phát sinh ảnh mới nếu snapshot flag tắt. JSONL + PNG dùng quota chung, lỗi
ghi/build diagnostic dừng debug nhưng AI vẫn hoàn tất các frame trong test.
Không SQLite/network/API; pipeline harness chặn các đường này. Không lưu
embedding hoặc đặc trưng sinh trắc. Chỉ log bbox và các ID/score được yêu cầu.

## 9. PowerShell chạy camera với Face debug

Chưa chạy camera thật trong lượt này. Lệnh sau chỉ ghi metadata, giới hạn 10
frame/20 MiB và giữ WARNING disabled:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_FACE_DEBUG = '1'
$env:HTTM_PERSON_DEBUG_SNAPSHOTS = '0'
$env:HTTM_PERSON_DEBUG_MAX_FRAMES = '10'
$env:HTTM_PERSON_DEBUG_MAX_BYTES = '20971520'
$env:HTTM_PERSON_DEBUG_DIR = Join-Path (Get-Location) 'ai_training\runs\person_tracker_debug'
try {
    .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
} finally {
    Remove-Item Env:HTTM_PERSON_DEBUG, Env:HTTM_FACE_DEBUG, Env:HTTM_PERSON_DEBUG_SNAPSHOTS, Env:HTTM_PERSON_DEBUG_MAX_FRAMES, Env:HTTM_PERSON_DEBUG_MAX_BYTES, Env:HTTM_PERSON_DEBUG_DIR -ErrorAction SilentlyContinue
}
```

Bỏ/đặt `HTTM_FACE_DEBUG=0` tắt Face block; bỏ/đặt master debug=0 tắt cả session.
Nếu chủ động muốn ảnh, người dùng phải bật snapshot flag riêng như Diagnostic A
cũ; tác vụ này không tự bật.

Đọc session path từ stderr `[PERSON DEBUG] ...`, dùng cùng file cho cả hai CLI:

```powershell
.\.venv\Scripts\python.exe .\ai_training\integration\replay_face_tracker_diagnostics.py '.\ai_training\runs\person_tracker_debug\<session>\frames.jsonl'
.\.venv\Scripts\python.exe .\ai_training\integration\replay_person_tracker_diagnostics.py '.\ai_training\runs\person_tracker_debug\<session>\frames.jsonl'
```

Face CLI chỉ đọc metadata, không đọc ảnh/model/DB. Exit 0: nhất quán; 1: mismatch;
2: không đủ dữ liệu/hỏng metadata. Cần session đầy đủ bắt đầu từ tracker rỗng;
từ chối frame mismatch, timestamp không tăng hoặc records truncated. Log camera
cũ chưa có face block không dùng để replay FaceTracker; không suy đoán lại
timestamp face từ person hoặc processing để điền vào.

## 10. Giá trị có thể cân nhắc ở bước B

**4.0s là ứng viên để kiểm thử offline ở bước B**, vì vượt sáu gap được nêu
(lớn nhất 3.715s) và cận trên Face gap khoảng 3.754s đã suy luận trong khảo sát.
5.0s có thể là ứng viên so sánh khả năng chịu spike với rủi ro retention cao hơn.
Đây không phải tham số production được chọn hoặc đã benchmark.

Chưa chạy thử/thay tham số B, không thêm environment flag điều chỉnh max_age.
Mọi replay/test Diagnostic A vẫn giữ 2.0/0.3. Session Face debug mới cần đo gap
trực tiếp, phân bố/spike và bbox IoU trước khi đề nghị giá trị chính thức.

## 11. Rủi ro nếu tăng max_age

Track khuôn mặt mất quan sát được giữ lâu hơn, có thể nối người thay thế vào
cùng vị trí cũ. Pure IoU không xét Student ID/embedding; hai người gần nhau/giao
nhau hoặc tie IoU có thể đổi identity. Test counterexample giữ nguyên hành vi
tie hiện tại: track ID lớn hơn thắng khi IoU bằng nhau, không hứa re-identification.

Recognition hiện tại/association đúng và đổi Student/Person sẽ reset episode
CH-20/21 theo thiết kế, nhưng recognition/association sai liên tục vẫn có rủi ro
gắn evidence sai. Không dùng stable label hoặc Student ID đơn thuần để kéo dài
duration qua identity không hợp lệ; tăng tuổi thọ không biến Warning thành xác
nhận vi phạm chính thức.

## 12. Bước tiếp theo và điểm dừng

Người dùng chạy một phiên Face debug, gửi shared JSONL mới. Đối chiếu
face_update_gap, age, expired reason, candidate IoU và tất cả mapping, gồm face
không association, để xác minh trực tiếp camera. Kiểm tra P1/SV001 và các state
CH-20/21 trên Panel theo điều kiện thực tế.

Sau dữ liệu đó mới phê duyệt riêng bước B và test departure/replacement/crossing.
**Dừng tại FaceTracker Diagnostic A. Không tự đổi max_age, thuật toán hoặc triển
khai CH-22. Không commit/push.**
