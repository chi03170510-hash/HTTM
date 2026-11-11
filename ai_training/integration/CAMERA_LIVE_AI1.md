# AI1 — Continuous standalone Camera Live

Phạm vi chỉ `ai_training/integration`: không tích hợp backend/frontend, API,
database hoặc WebSocket. Không mở webcam/inference khi triển khai và kiểm thử.

## Cấu hình

`HTTM_CAMERA_MAX_OBSERVATIONS`:

- Không khai báo: chạy liên tục đến Q, lỗi camera/worker hoặc Ctrl+C.
- Số nguyên dương ASCII, ví dụ `1`, `10`, `20`: dừng sau đúng số AI observations
  hoàn tất. Không phải số preview frames hoặc số lần polling queue.
- Rỗng, `0`, số âm, số thập phân, khoảng trắng hoặc chữ: ValueError trước khi mở
  camera/model/GUI. Không có thay đổi inference FPS, checkpoint, threshold/TTL/IoU.

Mặc định source `MAX_AI_FRAMES=None`. Hằng số giữ tên cũ để các harness đã có
tiếp tục inject phiên hữu hạn; production đọc env một lần trong main và truyền
nguyên giá trị vào AI thread. Camera index và thứ tự DSHOW→MSMF giữ nguyên.

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'

# Continuous: lệnh này chỉ dành cho người dùng chạy khi có camera.
Remove-Item Env:HTTM_CAMERA_MAX_OBSERVATIONS -ErrorAction SilentlyContinue
& .\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py

# Phiên kiểm thử đúng 20 observations; đổi thành 1 hoặc 10 nếu cần.
$env:HTTM_CAMERA_MAX_OBSERVATIONS = '20'
& .\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py

# BBox opt-in hiện có, không ảnh hưởng chế độ continuous/limited.
$env:HTTM_PERSON_SNAPSHOT_OVERLAY = '1'
```

Các env trial Person Confidence/Face TTL/checkpoint đang có không bị tự đặt hoặc
xóa. `WARNING_CONFIG` giữ nguyên disabled; T_hold=None. Cửa sổ snapshot historical
không vẽ kết quả AI lên live preview. AI Panel, layout và dữ liệu behavior giữ nguyên;
HUD chỉ đổi bộ đếm thành `AI results: N | continuous` hoặc `N/limit`.

## Các vị trí giới hạn cũ và thay đổi

Trước AI1: MAX_AI_FRAMES=10, for-range trong ai_loop, main dừng sau 10 results,
HUD N/10 và ngân sách Shadow lấy cùng hằng số. Poll frame_queue.Empty cũng tiêu
mất một lượt for-range, dù chưa thực hiện AI observation.

Sau AI1:

- Vòng AI dùng ngân sách hoàn tất riêng hoặc không có giới hạn. Frame ID tăng
  từ 1 sau khi thật sự lấy được frame; empty polling không tiêu ID/ngân sách.
- Main nhận finished/error từ AI thread; không dùng số result UI nhận được làm
  ngân sách inference. HUD dùng Frame ID cuối đã hoàn tất, kể cả UI bỏ backlog.
- Frame input vẫn là queue mới nhất maxsize=1. Inference vẫn face rồi hybrid,
  cùng JPEG, không thêm sleep/rate limiter hay thay đổi FPS inference.
- Result mailbox tối đa 3: bỏ data UI cũ khi cần, không block inference; vẫn có
  chỗ cho tối đa hai error controls và finished. Diagnostic nhận trực tiếp từng
  result trước khi observation tiếp theo được xử lý, không phụ thuộc backlog UI.
- Shadow diagnostic vẫn bounded: continuous mặc định tối đa 10 rows; phiên
  limited dùng ngân sách session, nhưng Shadow writer vẫn cap 100 rows/1MiB.
  Person/Face diagnostic giữ nguyên limits và env hiện có. Diagnostic hết ngân
  sách không làm dừng AI. Chạy continuous không có nghĩa log mọi frame vô hạn.

## Stop, lỗi và cleanup

`camera_live_runtime.LiveWorker` tái sử dụng protocol Worker cũ, chỉ thay queue
chờ reply bằng proxy cùng queue gốc. Timeout request 90s giữ nguyên; polling tối
đa 0.1s để thấy stop_event, không phải inference scheduling.

Q đặt stop_event; AI không mở request Hybrid tiếp nếu Q đến giữa hai request.
Nếu đang chờ subprocess inference, proxy kết thúc chờ mà không cần đợi đủ 90s.
`finally` dừng cả hai subprocess, terminate→wait(3s)→kill→wait(3s) nếu cần, đóng
pipes và join reader. Đây là hủy công việc còn dở ở cuối phiên, không làm inference
bổ sung. Hai worker và Shadow finish được cleanup độc lập, kể cả một close lỗi.

Main luôn release camera/destroy GUI và join AI thread, giới hạn shutdown 20s.
Cleanup lỗi đến sau Q vẫn được đọc từ terminal mailbox và báo RuntimeError.
Nếu hệ điều hành không dừng được process/thread trong giới hạn thì báo lỗi;
không tuyên bố cleanup thành công. Ctrl+C đi qua cùng finally.

Camera read lỗi, worker timeout/error hoặc protocol frame/image mismatch kết thúc
phiên; không retry model/camera vô hạn. Optional overlay init/render/show/metadata
exception chỉ tắt phần hiển thị tương ứng; AI tiếp tục khi dữ liệu AI vẫn hợp lệ.

## Snapshot, duration và memory

- Không đổi `draw_ai_panel`, `_wrap_panel_text`, renderer B1–B5 hoặc input schema
  DurationTracker. Repaint vẫn đọc immutable behavior view, không gọi update.
- Main giữ latest result, result mailbox tối đa 3 và input frame tối đa 1.
  BBox cache giữ snapshot hiện tại và một rendered image; không giữ danh sách
  toàn bộ lịch sử. JPEG và sidecars dùng lại trong một TemporaryDirectory.
- Test weakref/GC qua 30 snapshots xác nhận lịch sử bị release; test mailbox qua
  25 observations xác nhận tất cả 25 input vẫn đến DurationTracker, dù UI chỉ
  còn kết quả mới nhất trong mailbox.
- Đây là chứng minh cấu trúc/fixture, không thay thế đo RSS, GPU/native buffers
  hoặc tài nguyên camera trong một phiên dài thực tế.

## Test thực chạy

**30/30 test AI1 PASS; full regression 661/661 PASS**, 0 failure/error/skip.

Bao phủ: unlimited >20; limited 1/10/20; empty polling và unique Frame ID; Q khi
chờ protocol và giữa requests; timeout/error; worker initialization/cleanup lỗi;
camera read/GUI exception; bounded queue/terminal controls; protocol frame/path;
snapshot GC; Overlay OFF/ON × Shadow OFF/ON; duration inputs/views không đổi;
không phát warning. Existing CH20/CH21, identity và BBox tests cũng được chạy lại.

Có test chạy `main` và AI thread thật cùng mock camera/workers/GUI cho cả ba
limits, continuous Q sau >20 và timeout. Fake workers không load model; tất cả
VideoCapture/model constructor thật bị guard trong full regression runner.

```powershell
$env:PYTHONPATH = 'ai_training/integration;ai_training'
& .\.venv-hybrid\Scripts\python.exe -B -m unittest test_camera_live_continuous -v
```

Artifacts mới:
`ai_training/runs/camera_live_ai1_validation_a30300effa934d6fb9ac446524847961/`.
Có regression JSON/log từng module, verification, patch và backup trước AI1.

## Giới hạn và rollback

Chưa kiểm thử iVCam thật, latency khi Q giữa native model inference, model/device
resource cleanup trên Windows thực, độ mượt phiên dài hoặc memory native theo thời
gian. Chưa kết luận Camera Live end-to-end PASS. Không audit/sửa BE/FE trong AI1.

Rollback hành vi giới hạn cũ: đặt `HTTM_CAMERA_MAX_OBSERVATIONS='10'`; bỏ biến để
trở lại continuous mặc định mới. Rollback source: đối chiếu patch/backup, chỉ hoàn
tác `camera_coordinator_live.py` và bỏ hai module runtime/test mới cùng tài liệu
nếu không còn cần. Không dùng git reset/restore để ghi đè thay đổi người dùng.
