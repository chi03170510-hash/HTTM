# Person Confidence Trial 0.35

Ngày: 2026-11-11. Đã triển khai opt-in và kiểm thử tự động; **chưa chạy camera thật**.
Không train/cài dependency, bật WARNING, commit hoặc xóa diagnostic.

## A. Files changed

| File trong `ai_training/integration` | Nội dung |
|---|---|
| `hybrid_v7_stream_worker.py` | Đọc trial một lần khi khởi động; lọc sau fusion, trước reply/diagnostic sidecar; log stderr |
| `person_confidence_trial.py` (mới) | Parser environment, mapping confidence duy nhất, lọc giữ nguyên object/evidence và kiểm tra reference |
| `test_person_confidence_trial.py` (mới) | 16 test thuần Python, biên/metadata/reference và replay tracker/Association V4 |
| `test_hybrid_person_confidence_trial.py` (mới) | 12 test worker thật với inference mock, protocol và Camera Live/CH20 harness |
| `PERSON_CONFIDENCE_TRIAL_REPORT.md` (mới) | Báo cáo này |

Đã đọc source/Git status trước khi sửa. So SHA-256 của 192 file có sẵn thuộc source,
config, JSON/JSONL, báo cáo và ảnh diagnostic: chỉ worker thay đổi. Source FaceTracker,
PersonTracker, Association V4, BehaviorDuration/WarningObserver, Camera Live, config
production và dữ liệu cũ giữ nguyên. Không đổi YOLO threshold, TTL/IoU hoặc model/weights.

## B. Cơ chế

`HTTM_PERSON_TRIAL_MIN_CONF`:

- Không tồn tại: baseline, không cần confidence metadata, không chạy bộ lọc/reference
  validation hoặc ghi trial log theo frame. Các default detector giữ nguyên.
- Đúng literal `0.35`: giữ person có confidence gốc >=0.35.
- Giá trị khác (kể cả `0.350`, `.35`, chuỗi rỗng/khoảng trắng): `ValueError` có tên biến
  và hướng dẫn về baseline, trước khi load model trong worker.

Worker khởi động qua process con nên kế thừa environment của PowerShell. Cấu hình
được chốt một lần cho instance; thay environment không thay worker đã chạy.

Vị trí lọc: sau `spatial_v7()`, sau bước projection ra schema `people` hiện tại, trước
`write_detection_sidecar()` và reply tới Camera Live. PersonTracker/Association V4
chỉ nhận các person đã giữ. Không đổi crop, phone/food, inference hoặc fusion upstream.

Mapping dùng `output.person_ref == raw.detection_id` trong `obj.person_detections`
sau hợp nhất 640/high-res. Phải có đúng một raw record, output ref không trùng,
confidence là số hữu hạn trong [0,1]. Không lấy điểm behavior, person quality hoặc
highres_match_iou làm confidence thay thế; không tự chọn max high-res. Missing,
duplicate hoặc invalid metadata làm frame trial báo lỗi thay vì đoán.

Person được giữ giữ nguyên thứ tự/ref/bbox và behavior object; không tái tính hoặc
sửa evidence. Schema reply thành công vẫn gồm `frame_id`, `image`, `mode`,
`person_count`, `people`; schema mỗi person giữ nguyên bốn field. `person_count`
luôn bằng độ dài danh sách đã lọc. Không đưa raw object associations vào reply.

Để đáp ứng đồng thời không sửa behavior và không tạo dangling references, bộ lọc
kiểm tra các ID person nguyên chuỗi trong key/value/list của retained output. Nếu
evidence còn nhắc tới person vừa bị loại, trial **từ chối cả frame**, chỉ rõ đường
dẫn reference. Không xóa candidate/provenance hoặc chuyển ownership sang người khác.
Worker dùng error reply có sẵn (`frame_id`, `error`); Camera Live sẽ dừng khi nhận lỗi.
Đây là giới hạn có chủ đích của trial, không phải cơ chế tự sửa evidence.

Khi bật trial, console stderr thông báo enabled rồi ghi mỗi frame thành công:

```text
[PERSON CONFIDENCE TRIAL] {"frame_id": 2, "min_conf": 0.35, "before": 2, "after": 1, "rejected": [{"person_ref": "person_1", "confidence": 0.2858, "bbox": [...], "reason": "confidence_below_trial_threshold"}]}
```

Stdout protocol vẫn chỉ là JSON reply. Log không ghi Student ID/face embedding/ảnh.
Debug JSONL hiện có chỉ ghi detections được giữ; cần giữ stderr log để đối chiếu
detections bị loại. Không bật snapshot/debug bằng trial flag.

## C. Test results

**222/222 tests liên quan đạt; 0 failure, 0 error, 0 skip.** Gồm 28 test mới và
194 regression; không cộng thêm các lần rerun.

| Nhóm | Passed / total |
|---|---:|
| Bộ lọc/config/replay mới | 16/16 |
| Worker/protocol/Camera Live pipeline mới | 12/12 |
| Hybrid person diagnostics cũ | 4/4 |
| Hybrid V7 | 34/34 |
| PersonTracker và diagnostic/replay | 39/39 |
| Association script | 5/5 |
| Camera Person integration và Panel/Layout | 22/22 |
| BehaviorDuration/Overlay/Warning và CH20 pipeline | 70/70 |
| CH21 pipeline | 20/20 |
| **Tổng** | **222/222** |

Đã xác nhận: baseline thiếu provenance vẫn chạy như cũ; 0.3499 bị loại, 0.3500 và
0.3501 được giữ ở cả COCO 640/high-res-only; input, phone/food và behavior giữ nguyên;
metadata lỗi không được đoán; nested references bị đứt làm báo lỗi; log không phá
JSON stdout; toàn bộ person bị lọc trả danh sách rỗng hợp lệ.

Harness chạy worker process method thật, rồi luồng Camera Live thực → Association V4
→ PersonTracker → duration → snapshot → panel. Chỉ model replies/camera/GUI/time được
mock. Kiểm tra person bị lọc ngắt episode, quay lại nhận Person ID mới và duration 0;
render lại không tăng duration. WARNING production vẫn disabled; các đường model
process/SQLite/network/API bị guard trong harness.

Các lệnh regression đã chạy từ repository root:

```powershell
Push-Location .\ai_training\integration
..\..\.venv\Scripts\python.exe -m unittest -q test_person_confidence_trial test_behavior_duration test_behavior_overlay test_behavior_warning test_ch20_duration_pipeline test_ch21_warning_pipeline test_person_tracker test_person_tracker_diagnostics test_person_tracker_diagnostic_replay test_camera_person_tracking test_camera_live_layout
# 167 OK, 8.237s
..\..\.venv-hybrid\Scripts\python.exe -m unittest test_hybrid_person_confidence_trial test_hybrid_person_diagnostics -q
# 16 OK, 0.711s
Pop-Location
.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -q
# 34 OK
.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS 5/5
```

Association V4 được thực thi trong replay/harness; script riêng kiểm tra association
nền. Đây là phạm vi các bộ test liên quan đã chạy, không phải toàn bộ test repository.
Không chạy camera tự động, không inference lại checkpoint trên hai session.

## D. Replay baseline và candidate

Dùng toàn bộ frames.jsonl của hai session, giữ nguyên bbox, detection order,
confidence, Face/Person tracker update timestamps và nhận dạng được ghi. Baseline
được kiểm tra bằng replay FaceTracker/PersonTracker cũ, 10/10 frame/session khớp.

| Session | Frames | Baseline detections | Candidate detections | Association baseline / candidate |
|---|---:|---:|---:|---:|
| `session_5be088f4b254495c86ecd50a4f899ee3` | 10 | 19 | 10 | 10 / 10 |
| `session_e62b02db457743419e0352de635ee80a` | 10 | 24 | 21 | 17 / 17 |

Danh sách association theo từng frame (gồm Student/Face/Person Track và person_ref)
giữa baseline/candidate bằng nhau trong cả hai replay. Kiểm tra lại hai session qua
output gate của worker thật với spatial replies mock cũng cho 19→10 và 24→21.

Theo nhãn đối chiếu hình ảnh ở bước phân tích trước:

- Phiên một người: giữ 10/10 bbox người tiền cảnh, loại 9/9 bbox ghế/vải nghi giả đã
  xác nhận bằng snapshot. Candidate còn một Person Track mỗi frame.
- D3-T1: giữ 20/20 bbox của hai người tiền cảnh; loại .1805/.1259 ở F7 và .1743 ở F8.
  Ba bbox vùng người phía sau **chưa được tính là false positive**.
- Bbox rộng/composite .4361 ở F1 D3-T1 vẫn giữ. Trial không giải quyết bbox gộp hoặc
  trường hợp confidence cao nhưng detection sai.

JSONL không lưu đầy đủ behavior evidence, nên replay dùng `behaviors={}` (unknown),
không tạo detected evidence giả. Bảo toàn behavior/phone/food và reference closure
được kiểm tra bằng fixture có kiểm soát trong worker/pipeline tests. Chưa chứng minh
mọi evidence thực tế của hai session đều qua được reference guard.

Source SHA-256 giữ nguyên:

- Một người: `fe6d04bef166628ecf3a6a4218294fd0d5657f4f1e4822ab820b566efb6d46a8`.
- D3-T1: `09fa0a39befbec2c0f492f40772a579344bb86019604e8e3e923fca5cea2a882`.

## E. Rủi ro còn lại

- Person thật nhỏ/xa/che khuất có thể bị loại; với keep_unmatched=False một lần mất
  detection sẽ ngắt track/association/duration. Không bảo đảm Person ID giữ nguyên
  giữa các phiên A/B hoặc sau lần mất detection.
- Giảm tập person có thể làm Association V4 thay đổi lựa chọn. Replay hai session
  chưa thấy thay đổi association, không bảo đảm mọi cảnh đều vậy.
- Đây là lọc đầu ra: raw bbox giả vẫn có thể ảnh hưởng object ownership/fusion
  upstream. Không tuyên bố đã cải thiện behavior detection hoặc false-positive rate
  tổng quát của model.
- Nếu retained evidence tham chiếu person bị loại, trial dừng frame/session; cần xem
  error và rollback. Không tự sửa evidence để tiếp tục.
- Metadata hợp nhất hiện giữ confidence của base detection; raw high-res score bị
  ghép không được dùng để cứu bbox. Kết quả khác với chạy lại detector ở threshold cao.
- Chưa đo FPS/latency camera. Baseline chỉ thêm kiểm tra flag; candidate thêm mapping,
  lọc/reference scan và stderr log. Không cài hoặc yêu cầu dependency mới.

Chỉ đạt kiểm thử tự động cho cơ chế trial. Chờ camera A/B, đặc biệt vùng nền có người
và ca behavior thật, trước khi đánh giá ngưỡng hoặc chốt end-to-end HTTM.

## F. PowerShell chạy baseline/candidate và rollback

Baseline:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
Remove-Item Env:HTTM_PERSON_TRIAL_MIN_CONF -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Candidate:

```powershell
$env:HTTM_PERSON_TRIAL_MIN_CONF = '0.35'
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Giữ nguyên FaceTracker trial/debug/camera settings giữa A/B. Camera index hiện tại
vẫn 0, DSHOW → MSMF; xác nhận preview đúng iVCam. Mỗi phiên tối đa 10 AI results.
Không có biến môi trường bật WARNING production.

Nếu muốn lưu stderr/stdout cho candidate, dùng tên log mới, ví dụ trong TEMP:

```powershell
$personTrialLog = Join-Path $env:TEMP ('HTTM_person_trial_' + [guid]::NewGuid().ToString('N') + '.log')
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py 2>&1 | Tee-Object -FilePath $personTrialLog
```

Rollback: bấm Q và đợi worker đóng, sau đó:

```powershell
Remove-Item Env:HTTM_PERSON_TRIAL_MIN_CONF -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Chỉ bỏ biến trong shell không đổi worker đang chạy; phải mở phiên mới. Không dùng
setx, sửa JSON config hoặc xóa session diagnostic. Dừng tại trial này, chờ camera thực.
