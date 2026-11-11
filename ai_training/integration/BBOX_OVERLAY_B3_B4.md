# AI Snapshot Face / Behavior Overlay — B3+B4

Chỉ visualization, dùng chung opt-in `HTTM_PERSON_SNAPSHOT_OVERLAY=1` của B1+B2.
Mặc định OFF. Live Preview và AI Panel giữ nguyên. Không vẽ historical bbox lên
ảnh camera mới. Chưa nghiệm thu bằng camera thật.

## Luồng dữ liệu và đồng bộ

1. Coordinator lưu JPEG tạm theo observation, giữ decoded pixels trong AISnapshot.
2. Hai worker vẫn đọc đúng JPEG đó và trả JSON production với schema cũ.
3. Chỉ khi overlay ON, Face worker xuất `frame.face_visual.json`; Hybrid worker
   xuất `frame.hybrid_visual.json` cạnh JPEG tạm. Không chạy thêm inference.
4. Sidecar chỉ copy whitelist kết quả đã tính. Mỗi packet có `schema`, `role`,
   `frame_id`, `image` tuyệt đối, `source_size=[width,height]`, `jpeg_sha256`.
   Không có ảnh, embedding hoặc lịch sử identity. File tạm được dọn cùng phiên.
5. Sau khi DurationTracker và WarningObserver chạy xong, adapter xác minh frame,
   path, SHA256, kích thước và decoded pixels. Chỉ thay optional `render_metadata`
   trong snapshot bằng DTO frozen. Không thay observations hoặc result keys.
6. Cửa sổ riêng render Person layer rồi thêm detail layer/legend. Geometry được
   cache theo snapshot; chỉ repaint tuổi capture/completion. STALE dùng ngưỡng
   hiển thị B1+B2 hiện có, không thêm ngưỡng detector/tracker.

Metadata không ghép theo vị trí list. `person_ref` chỉ là khóa **cùng frame**,
phải duy nhất và trỏ tới PID hiện tại duy nhất. Không có history/retention ở đây.
SID, behavior state và duration chỉ từ `PersonBehaviorView` production.

## Nguồn bbox và confidence

| Lớp | Nguồn bbox | Nhãn/score |
|---|---|---|
| Person | `PersonBehaviorView.bbox` | PID + SID accepted/Unknown/Ambiguous; không thêm score |
| Face | Face worker `faces[].face_bbox` | `Face detection`; `Recognition`; `similarity` từ `score` |
| Sleeping | Hybrid `inf.detections[].bbox_xyxy`, class `sleeping` | `Behavior YOLO` từ chính candidate `confidence` |
| Using Phone candidate | Hybrid `inf.detections[].bbox_xyxy`, class `using_phone` | `Behavior YOLO` từ chính candidate `confidence` |
| COCO phone object | `obj.phone_detections[].bbox_xyxy` | `COCO score` từ chính object `confidence` |

Face similarity là độ tương đồng embedding, **không phải xác suất nhận diện đúng**.
Không có Face detection confidence trong worker output; không tự tạo giá trị này.
Score thiếu hiển thị N/A. Score YOLO/COCO không được xem là xác suất đã hiệu chuẩn.

FaceTrack ID chỉ hiện khi bbox + score ghép duy nhất với current tracked face,
track ID không trùng. Stable label của FaceTracker không được dùng để suy SID.
Raw Face SID chỉ kiểm tra sự nhất quán; không được hiển thị thay SID production.
Face Unknown hoặc SID mâu thuẫn không được hiển thị như production identity accepted.

Sleeping chỉ có nhãn `production detected box` khi candidate duy nhất, association
person hợp lệ, bbox trùng final decision và cả state decision/view đều detected.
Các candidate còn lại giữ nhãn YOLO candidate, kể cả khi person có production state
detected từ candidate khác. State/duration đi kèm luôn ghi `prod`.

Using Phone YOLO candidate và COCO object có màu, bbox, tag và score riêng.
Final `behaviors.using_phone.bounding_box` trong Hybrid V7 có thể là **COCO phone
bbox** trong khi model confidence là **behavior YOLO score**; adapter không ghép
nhầm hai giá trị này. Object chỉ có owner khi detection ID và association ID duy
nhất, status `ASSOCIATED_GEOMETRICALLY`, person_ref/PID hiện tại hợp lệ.
`phone support yes` nghĩa bbox object trùng final detected decision của person;
không biến bbox object thành vùng toàn bộ behavior.

Candidate chỉ liên kết người với status `associated` / `associated_by_person_roi`
và không có cờ association mơ hồ. Candidate chưa association giữ Unassigned;
không tìm người gần nhất hoặc tự suy SID. Duplicate/ambiguous SID thể hiện bằng chữ.
Không đọc hoặc sử dụng Shadow SID.

## Tọa độ và lỗi

Tất cả bbox sidecar dùng tọa độ ảnh JPEG gốc. Hybrid đã remap ROI/high-res về
ảnh gốc; adapter không áp lại letterbox inference. Chỉ renderer áp cùng phép
resize/letterbox như Person layer, dùng chuẩn raster B1: bao endpoint pixel,
clip về ảnh và đổi sang cận phải/dưới half-open chỉ trong lớp hiển thị.
Không sửa clip_box, detection, tracker hay association coordinates.

NaN/Inf, zero-area, malformed/outside box bị bỏ qua; layer lỗi không ngăn Person
layer hiển thị. Source mismatch/missing/corrupt, giới hạn diagnostic 1000 rows/2MiB,
renderer hoặc IO lỗi có thông báo `[BBOX VISUAL]` ở stderr. Packet lỗi bị bỏ qua;
không chế metadata thay thế. GUI lỗi chỉ tắt cửa sổ optional, Live tiếp tục.
OFF không xây/hash/ghi sidecar hoặc render detail layer.

## Chạy demo khi có camera

Trong PowerShell tại repository, đóng tiến trình camera cũ trước khi đổi env:

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'
$env:HTTM_PERSON_SNAPSHOT_OVERLAY = '1'
& .\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py
```

Không cần bật Shadow. Nếu đã bật, Shadow chỉ ghi diagnostic như trước; T_hold
vẫn None và không cấp SID production. WARNING vẫn disabled. Lệnh không đổi
camera index/backend, checkpoint, trial person/face hoặc các env đang có.

Tắt overlay/rollback chức năng ở lần chạy kế tiếp:

```powershell
Remove-Item Env:HTTM_PERSON_SNAPSHOT_OVERLAY -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py
```

## Kiểm thử offline thực chạy

```powershell
$env:PYTHONPATH = 'ai_training/integration;ai_training'
& .\.venv-hybrid\Scripts\python.exe -B -m unittest test_bbox_detail_overlay test_bbox_visual_workers test_person_snapshot_overlay -v
```

81/81 PASS: 40 detail tests + 6 worker tests + 35 B1+B2 tests.
Full regression: **608/608 PASS**, 0 fail/error/skip. Phạm vi: toàn bộ integration
unittest, 5 association function tests, 34 Hybrid V7 unit tests. Các script manual
camera/inference không có unittest không được chạy như camera test.

Model boundaries và VideoCapture được mock/guard. Ma trận Overlay OFF/ON ×
Shadow OFF/ON đi qua ai_loop/main thật bằng worker synthetic, so sánh deep-copy
input DurationTracker/WarningObserver, behavior/warning outputs, accepted SID,
snapshot pixels và AI Panel fields/geometry. Worker OFF/ON giữ nguyên protocol
JSON, threshold và số lần gọi inference boundaries. Test có reordered/mismatched
metadata, duplicate/missing/ambiguous SID, bbox invalid/clipping/resize, corrupt
sidecar, IO/renderer failure và confidence khác nguồn.

Artifacts mới, không ghi đè báo cáo cũ:
`ai_training/runs/bbox_b3_b4_validation_9ea11f43b45641ac9ad2c5c5c042fd3d/`.
Có full regression JSON/log từng module, hai PNG synthetic, PROVENANCE và bản
backup trước sửa của 5 file. Không dùng `git restore` trên integration đang có
thay đổi sẵn; nếu rollback source, đối chiếu backup/diff và chỉ hoàn tác B3+B4.

## Giới hạn nghiệm thu

- Preview là hình synthetic và score tự khai báo, không phải detection/model PASS.
- Chưa xác nhận độ khít bbox, độ ổn định FaceTrack, association/SID hoặc detection
  đúng với người thật, tốc độ/độ đọc nhãn trên iVCam, hay camera end-to-end PASS.
- Chỉ xuất candidate đã có trong `inf.detections` sau bước high-res promotion;
  không bổ sung toàn bộ raw high-res proposals đã bị detector pipeline loại.
- Nếu thiếu bằng chứng owner/metadata thì bỏ link/layer, vẫn hiển thị Person.
- Legend có giới hạn số dòng theo kích thước cửa sổ; ghi Showing N/total khi đông.
  Có thể resize cửa sổ riêng bằng OpenCV; không đổi layout Live/AI Panel.
- Overlay ON có thêm IO/hash/decoded-pixel verification và rendering. Cache chỉ
  tránh vẽ lại geometry; chưa đo performance camera thật. Thời gian production
  completion giữ thời điểm trước adapter, không bao gồm bước display projection.
