# Person BBox B1+B2 — separate historical snapshot, opt-in

Đã triển khai Person-only overlay trong cửa sổ riêng
`HTTM AI Snapshot - HISTORICAL (Person)`. Mặc định OFF. Cửa sổ Live Preview và
AI Panel giữ nguyên cấu trúc/pixel; không vẽ historical boxes lên ảnh live mới.
Không có Face BBox, Behavior BBox, confidence hoặc identity retention mới.

## Files

Sửa:

- `camera_coordinator_live.py`: đọc cờ hiển thị một lần trong main, gọi cửa sổ
  snapshot optional sau live imshow và chứa lỗi ở ranh giới hiển thị. ai_loop và
  draw_ai_panel giữ nguyên AST; không đổi result schema/production fields.
- `behavior_overlay.py`: thêm projection chỉ dùng để vẽ Person, nhãn production
  PID/SID, và keyword `person_only=False` cho renderer có sẵn. Legacy API giữ
  nguyên default rendering/label contract và các expected test cũ.

Mới:

- `person_snapshot_overlay.py`: adapter/cache, metadata validation, age/stale
  header, waiting/error placeholder và cửa sổ HighGUI riêng.
- `test_person_snapshot_overlay.py`: 35 test offline/mock.
- Báo cáo/hướng dẫn này.

Không sửa Hybrid V7/clip_box, FaceTracker, PersonTracker, Association V4,
DurationTracker, WarningObserver, AISnapshot hoặc dữ liệu AI Panel. Không thay
model/weights/threshold/TTL; T_hold=None, WARNING production vẫn disabled.

## B1 — contract tọa độ và projection

Source thực tế:

- Hybrid dùng xyxy trong hệ tọa độ pixel của ảnh JPEG nguồn, sau mapping crop
  về source. `hybrid_v3.clip_box` giới hạn endpoint tới W-1/H-1 và giữ float.
- PersonTracker copy geometry thành float, dùng diện tích (x2-x1)*(y2-y1), không
  cộng 1; không chứa flag inclusive/exclusive. Không thể coi đây là half-open
  integer crop rectangle mà không có một rasterization policy riêng.
- AISnapshot giữ ảnh BGR của đúng JPEG observation, cùng Frame ID/captured_at
  và BehaviorFrame. Source width/height lấy từ snapshot.image, không từ preview.

Projection **chỉ dành cho cửa sổ Person mới**:

1. Từ chối bbox malformed, NaN/Inf, đảo chiều, zero-area hoặc nằm hoàn toàn ngoài ảnh.
2. Rasterize các endpoint nguồn thành enclosing pixel bounds: floor near/far;
   far pixel được cộng 1 để tạo half-open raster bounds. Clip chỉ bản projection
   tới [0,W]/[0,H]. Không sửa tuple/list bbox gốc hoặc PersonBehaviorView.
3. Resize ảnh bằng cùng tỉ lệ fit-inside viewport; tính offset letterbox một lần.
4. Renderer half-open hiện có dùng floor near và ceil(far)-1 để chuyển sang
   inclusive coordinates cho OpenCV, rồi cộng offset.

Ví dụ full-source 100x50 đã được detector clip thành [0,0,99,49]: projection
thành [0,0,100,50]; ở viewport 100x50 bbox kết thúc ở pixel (99,49), không hụt
pixel cuối. Ảnh vẫn giữ nguyên, không kéo bbox ra ngoài viewport.

Đây là contract raster cho **display**, không thay contract geometry của YOLO,
tracker/association hoặc clip_box. Các float endpoint được rasterize nên hình
vẽ có thể bao thêm phần dưới một pixel nguồn. Legacy render_snapshot mặc định
giữ edge semantics cũ để tương thích; Camera Live mới gọi person_only=True.

Viewport mặc định 640x480 + header 80px; tỷ lệ nguồn được giữ, có letterbox.
Đổi độ phân giải/aspect giữa các observation làm render lại từ snapshot mới.

## B2 — đồng bộ và nhãn

Adapter chỉ nhận result, **không nhận frame live hoặc đường dẫn ảnh**. Nó kiểm tra
Frame ID/capture của result/AISnapshot, BehaviorFrame tương ứng và completion
time nếu có. Metadata không nhất quán sẽ hiện Render unavailable, không dùng
ảnh/bbox cache cũ. Không reload JPEG tạm đã có thể bị overwrite.

AISnapshot sở hữu bản copy pixel read-only. Pipeline test kiểm tra pixel đó bằng
pixel JPEG mà cả hai fake worker đã đọc; controller render nhận đúng instance
snapshot của result. Metadata là contract của producer đã test, không phải
chứng thực vật lý cho một bộ ảnh bên ngoài.

Mỗi bbox chỉ có `PID | SID`:

- SID được lấy từ PersonBehaviorView production: identity_status=matched và
  association được chấp nhận. Màu cyan biểu thị nhãn production hiện có, không
  phải confidence/ground truth.
- Thiếu/không hợp lệ/không được chấp nhận: `PID | Unknown`, màu xám.
- Identity ambiguous, duplicate SID/PID: `PID | Ambiguous`, màu amber.
- Không lấy SID từ matches, stable Face label hoặc Shadow proposal để điền chỗ trống.
- Không đọc behavior bbox/face bbox/confidence để vẽ trong giai đoạn này.

Header luôn có HISTORICAL, Frame ID và hai age:

- Capture age = now - captured_at.
- Completion age = now - completed_at; chưa có completion thì `--`.
- STALE ghi rõ capture/completion khi age tương ứng vượt giới hạn hiển thị
  `PERSON_TRACK_MAX_AGE` hiện có (5s). Đây không phải T_hold hoặc chính sách identity.
- Không có AI result: Waiting for first completed AI observation.

AI Panel cũ tiếp tục có định nghĩa freshness/behavior riêng, không bị thay dữ liệu
hoặc bố cục. Cửa sổ mới dùng ảnh lịch sử nên dù STALE, bbox vẫn nằm trên đúng ảnh
lịch sử; không có extrapolation sang vị trí hiện tại của người.

Geometry/labels cache theo Frame ID, capture time, snapshot instance và viewport.
Mỗi repaint chỉ copy canvas và cập nhật age header; không tăng duration hoặc gọi
inference. Renderer lỗi chỉ log một lần trên result lỗi và hiện placeholder;
result mới có thể thử render lại. Lỗi cửa sổ sẽ tắt riêng display cho phiên đó,
không dừng live camera/AI. main có thêm exception boundary cho visualization.

## Kiểm thử thực chạy

- **35/35 test mới PASS** bằng .venv.
- **562/562 regression offline PASS**, 0 failure/error/skip, bằng .venv-hybrid có sẵn.
- Bao gồm legacy overlay 10, layout 8, Camera/Person 14, Shadow 31, toolkit 66,
  CH20/CH21, backend, trackers, diagnostics và Hybrid V7 tests.
- Tổng gồm bộ integration unittest + 5 association functions + 34 Hybrid V7 unit
  tests; không cộng lại focused reruns. Không chạy script mở camera/model ở top level.

Coverage: source clip→display far edge, negative/partial/outside/fractional,
NaN/Inf/zero/reversed/malformed bbox, last source column, upscale/downscale,
letterbox/portrait/resolution changes, two people/reorder, accepted/Unknown/
Ambiguous/duplicate identity, waiting/stale/age, metadata mismatch, cache,
renderer/GUI exceptions và forced wrong Shadow proposal.

Ma trận Overlay OFF/ON × Shadow OFF/ON đối chiếu input DurationTracker và
WarningObserver, toàn bộ result fields/snapshot pixels và AI Panel samples:
không đổi. Using Phone/Sleeping và WARNING production disabled giữ nguyên.

Đối chiếu bản trước sửa: ai_loop và draw_ai_panel AST giống nhau; 12 pixel
comparisons của Panel (waiting/results, hai trang, ba aspect ratios) bằng nhau.
Source/test cũ không bị sửa expected để ép pass.

Artifacts mới:
`ai_training/runs/person_snapshot_overlay_b1_b2_validation_496334d543984d1e881e047892eecbec/`
gồm full regression/logs, production_parity.json, source patches/rollback backup
và ba PNG synthetic: accepted_two_people, ambiguous_two_people, unknown_person.
Đó là ảnh controlled BGR worker replies, không phải camera/face data thực tế.

## PowerShell khi có camera — chưa chạy trong tác vụ này

Baseline/OFF, giữ giao diện cũ:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
Remove-Item Env:HTTM_PERSON_SNAPSHOT_OVERLAY -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Bật Person snapshot demo:

```powershell
$env:HTTM_PERSON_SNAPSHOT_OVERLAY = '1'
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

Chỉ literal `1` bật overlay; unset/0/giá trị khác đều OFF. Cờ được đọc một lần
lúc main khởi động; đổi env cần chạy lại phiên. Không tự bật Shadow hoặc debug.
Camera index, DSHOW→MSMF, FaceTracker trial và model selection hiện có giữ nguyên.
Di chuyển hai cửa sổ để xem riêng Live và HISTORICAL; Q vẫn theo luồng OpenCV cũ.

Chạy test offline:

```powershell
& .\.venv\Scripts\python.exe -B -m unittest discover -s .\ai_training\integration -p test_person_snapshot_overlay.py -v
```

## Giới hạn chưa nghiệm thu

- Chưa mở webcam, inference hoặc GUI iVCam thật. Focus/đóng/di chuyển hai cửa sổ,
  FPS, độ trễ và khả năng đọc trên máy chiếu cần kiểm tra thực tế.
- SID production được hiển thị không chứng minh model nhận đúng actor. Vẫn cần
  actor truth cho crossing, overlap, replacement và recognition sai.
- Nhãn rất dài hoặc nhiều người trong vùng nhỏ vẫn cần đánh giá khả năng đọc;
  B1+B2 không thay toàn bộ label layout/legend.
- Phiên camera hiện vẫn giới hạn 10 AI observations. T_hold=None và Sleeping
  thiếu identity vẫn không được cộng duration bằng Shadow SID.
- Chưa triển khai Face/Behavior BBox hoặc confidence.

## Rollback

Tắt env overlay rồi chạy lại: production cũ và UI cũ vẫn dùng được, không cần
đổi model, tracker, association hoặc duration. Để rollback source hoàn toàn,
phục hồi riêng behavior_overlay.py và camera_coordinator_live.py từ rollback_before
trong thư mục validation sau khi kiểm tra sửa đổi mới hơn; ngừng dùng ba file mới.
Không reset repository, xóa diagnostic/báo cáo cũ, commit hoặc push.
