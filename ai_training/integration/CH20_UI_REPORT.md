# CH-20 UI — LIVE CAMERA | AI MONITORING

Ngày: 2026-11-10.

**Trạng thái:** Đã sửa bố cục và chạy kiểm thử tự động. Chưa chạy camera iPhone
trong lượt này. CH-20 chưa được nghiệm thu: duration thực tế chưa được xác nhận
qua chuỗi `detected` liên tiếp. Không chuyển CH-21.

Báo cáo này cập nhật phần giao diện ba vùng trong `CH20_IMPLEMENTATION_REPORT.md`;
cơ chế AI, duration và snapshot trong bộ nhớ ở báo cáo đó vẫn được giữ.

## Files changed

| File trong `ai_training/integration` | Thay đổi |
| --- | --- |
| `camera_coordinator_live.py` | Chỉ sửa phần dựng UI, phân trang và kích thước cửa sổ. Bỏ snapshot renderer khỏi vòng lặp hiển thị chính. |
| `test_camera_person_tracking.py` | Cập nhật kỳ vọng bố cục hai vùng; giữ kiểm tra snapshot đúng frame và khả năng debug. |
| `test_camera_live_layout.py` — mới | 8 kiểm thử UI, tỷ lệ ảnh, nội dung Panel, phím điều khiển và side effects. |
| `CH20_UI_REPORT.md` — mới | Báo cáo kết quả và hướng dẫn camera. |

Đã đọc Git status/diff và source trước sửa, lưu bản đối chiếu. Các thay đổi có sẵn
trong `.gitignore` và các file chưa được theo dõi không bị ghi đè bằng phiên bản Git.

## Bố cục

- Canvas có chiều rộng 1280 px: Live 896 px (70%), Panel 384 px (30%).
- Video được resize đồng tỷ lệ; không crop hoặc kéo giãn riêng chiều ngang/dọc.
  Header Live cao 48 px, không đè lên ảnh camera.
- Bỏ chiều cao tối thiểu 850 px và phần đệm đen của bố cục cũ. Với nguồn 640×480,
  bố cục là 1280×720; nguồn 1280×720 cho bố cục 1280×552. Nguồn dọc vẫn giữ tỷ lệ.
  Khi cần thêm chỗ cho văn bản rất dài, chỉ thêm khoảng đệm cần thiết để không cắt
  nội dung; không dùng bbox lịch sử để lấp vùng Live.
- Cửa sổ giữ tỷ lệ ảnh. Lần đầu hiển thị, kích thước cửa sổ được chọn theo canvas
  và chiều cao tối đa ban đầu 720 px; sau đó người dùng có thể resize.
- Panel hiển thị một sinh viên mỗi trang để đủ Frame ID, tuổi kết quả, processing
  time, Face Track ID, Student ID, Person Track ID, Association, identity status,
  bốn behavior và estimated duration.
- Face Track hiển thị theo đúng người đang chọn trong `BehaviorFrame`, không lấy
  face đầu tiên rồi ghép với thông tin của một người khác.
- N/P chuyển trang người, có quay vòng; Q thoát. Văn bản được wrap theo kích thước
  font thực tế, không cắt sau 48 ký tự như trước.
- Luôn ghi `Historical AI result - not live`. Khi chưa có kết quả, ghi
  `Waiting for AI...`; khi không có kết quả mới quá 5 giây từ completion, ghi
  `STALE`. Đây là trạng thái hiển thị, không phải WARNING hoặc thay đổi ngưỡng AI.

## Phần được giữ nguyên

Đã đối chiếu nội dung hàm `ai_loop()` với bản trước sửa: không thay đổi. Vì vậy
queue preview/worker, inference, Face ID, trackers, association, duration và cơ chế
tạo snapshot được giữ nguyên.

`result["snapshot"]` vẫn chứa ảnh riêng chỉ đọc, Frame ID, timestamp và behavior
cùng frame. `behavior_overlay.py` và `test_behavior_overlay.py` không sửa/xóa.
Debug vẫn có thể gọi `behavior_overlay.render_snapshot()` với metadata của result;
UI chính không gọi hàm này và không ghép snapshot vào Live/Panel.

Không sửa model/config/threshold/camera index, CH-18/CH-19, DB, API hoặc logic
cảnh báo. Duration vẫn chỉ đọc từ kết quả bất biến, không tăng khi render.

## Tests executed

Chạy từ root repository, tất cả đạt:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_live_layout.py -v
# 8 tests OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_person_tracking.py -v
# 14 tests OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_behavior_*.py -v
# 27 tests OK: duration 17 + snapshot 10

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_person_tracker.py -v
# 17 tests OK

.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# 5/5 association tests PASS

.\.venv\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# TOTAL MATCHED IMAGES: 2/2, predictions đã lưu + face tổng hợp

.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -v
# 34 tests OK
```

Tổng **105 test case đạt**, cộng association trên **2/2 ảnh đã lưu**. Association
V4 thật được chạy trong test Camera Live với worker giả lập; không chạy model mới.

Các test UI mới kiểm tra: tỷ lệ 70/30, nguồn 4:3/16:9/dọc, pixel Live resize đúng
tỷ lệ, không còn vùng snapshot, không có phần đệm 850 px, text không ra ngoài Panel,
ID dài không mất nội dung, bốn state/duration, trạng thái waiting/stale, N/P/Q,
không vẽ rectangle lên Live, không gọi snapshot renderer, không tăng duration khi
render lại và không phát sinh WARNING/DB/network.

Các test snapshot hiện có vẫn kiểm tra ảnh đúng JPEG của worker, file tạm bị xóa,
metadata Frame ID/timestamp, resize/clipping bbox và pixel snapshot bất biến.
Đã xem ảnh UI tổng hợp 1280×720 với lưới/hình tròn để kiểm tra tỷ lệ và nội dung;
ảnh tổng hợp không phải camera thực tế.

## Kiểm thử iPhone/iVCam

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py
```

1. Xác nhận chỉ có Live bên trái và AI Monitoring bên phải, không còn snapshot
   ở giữa. Video giữ tỷ lệ; không có bbox lịch sử trên Live.
2. Chờ đủ `Completed AI results: 10`. Kiểm tra Frame ID, result age, processing
   time, Face/P/Student ID và Association trên Panel.
3. Dùng N/P khi nhiều người; xác nhận mỗi trang đủ cả bốn behavior và duration.
   Kiểm tra trạng thái Unknown, waiting hoặc stale nếu xuất hiện. Q thoát sớm.
4. Xác nhận preview vẫn mượt. Không có số đo FPS camera mới trong lượt sửa UI này.
5. Duration `--` khi state chưa detected là đúng quy tắc hiện tại. Cần quan sát
   chuỗi detected thực tế trước khi nghiệm thu duration/CH-20; UI pass không thay
   thế điều kiện đó.

Dừng chờ người dùng kiểm thử camera, không commit/push hoặc chuyển CH-21.
