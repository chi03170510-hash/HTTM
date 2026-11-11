# AI2 — Runbook AI Camera Live độc lập trên Windows PowerShell

Phạm vi: Face Recognition + Hybrid V7 → Association V4 + PersonTracker →
BehaviorDurationTracker → AI Panel và AI Snapshot BBox B1–B5. Không BE/FE/database.
WARNING disabled; Identity Shadow chỉ diagnostic, T_hold=None. Các lệnh mở camera
bên dưới dành cho người dùng chạy khi đã có thiết bị; không chạy trong AI2.

## 1. Entrypoint và môi trường đúng

| Vai trò | Interpreter/file thực tế |
|---|---|
| Coordinator + GUI + Face worker | `.venv\Scripts\python.exe` |
| Hybrid worker (CPU) | `.venv-hybrid\Scripts\python.exe` — coordinator tự khởi chạy |
| Entrypoint hiện hành | `ai_training\integration\camera_coordinator_live.py` |
| Model selector | `ai_training\src\behavior_checkpoint.py` |
| Enrollment | `data\face_enrollment\SV*\embeddings.npz` |
| Face assets | Home của Face-worker user: `.insightface\models\buffalo_l\` |
| Camera | `camera_coordinator.py: CAMERA_INDEX=0`; Live thử DSHOW rồi MSMF |

`camera_coordinator.py` được import để dùng Worker/hằng số; chạy `main` của nó là
runner khác. `run_ai_v2.py`, `ai.behavior_detection.run_httm`, `integration.run_ch20`
và `run_video` là nhánh Round 3/legacy khác, không phải Hybrid V7 Camera Live này.
`run_hybrid_v7_images.py` chỉ dùng ảnh offline. Không sửa/xóa runner legacy.

Python 3.11.9 đã được inventory ở cả hai venv. Có thể kích hoạt Face/coordinator
venv, nhưng các lệnh luôn ghi rõ interpreter để không phụ thuộc activation:

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'
& .\.venv\Scripts\Activate.ps1
& .\.venv\Scripts\python.exe -B -c "import sys; print(sys.executable); print(sys.version)"
```

Nếu PowerShell chặn Activate.ps1, dùng các lệnh `& .\.venv\Scripts\python.exe`
trực tiếp. Không cần đổi ExecutionPolicy hoặc activate `.venv-hybrid` cho GUI.

Các model/config/source được production resolve từ vị trí file trong repository,
không từ working directory. Riêng custom diagnostic output tương đối sẽ resolve
theo working directory của tiến trình; runbook dùng đường dẫn tuyệt đối.

## 2. Preflight offline trước mỗi cấu hình demo

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'
$demoRoot = Join-Path (Get-Location).Path ('ai_training\runs\ai_demo_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $demoRoot -ErrorAction Stop | Out-Null
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_preflight.py --output (Join-Path $demoRoot 'preflight.json')
if ($LASTEXITCODE -ne 0) { throw 'Preflight FAIL/input error. Do not start camera; read the report.' }
$preflight = Get-Content -LiteralPath (Join-Path $demoRoot 'preflight.json') -Raw | ConvertFrom-Json
$preflight.checks | Where-Object { $_.status -ne 'PASS' } | Format-List
```

CLI tự tìm repository từ vị trí script; hỗ trợ `--root 'D:\Path With Spaces\HTTM-Chiv2'`
nếu cần kiểm tra checkout khác. Phải chạy bằng `.venv` thuộc checkout đó. `--json`
xuất JSON ra stdout. `--output` chỉ tạo file MỚI, từ chối ghi đè.

- PASS: những prerequisite được ghi trong check đã đạt.
- WARN: đọc tác động trước demo; không tự cài/xóa/đổi package hoặc config.
- FAIL: chưa đủ prerequisite; không dùng camera để tự tải model/cố sửa lỗi.
- Exit 0 = PASS hoặc WARN; exit 1 = có check FAIL; exit 2 = lỗi CLI/report IO.

Preflight chỉ dùng stdlib và chạy stdlib probe trong đúng hai interpreter: package
metadata + top-level module locations, Python version/prefix, AST cấu hình an toàn,
model existence/size/SHA256, cấu trúc archive enrollment, config/env/output write.
Không import cv2/YOLO/Torch/InsightFace/ONNX/MediaPipe, không tạo model/camera/GUI,
không tải hoặc cài gì. Chỉ tạo rồi xóa marker tạm riêng để kiểm tra quyền ghi;
không tạo diagnostic session. Hash đo được không phải danh sách hash xác thực.

**PASS/WARN không chứng minh DLL/ABI import được, model mapping đúng, embeddings
hợp lệ, camera mở được hoặc behavior/identity đúng.** Các check runtime đó vẫn còn.

## 3. Inventory dependency và model thực tế

Inventory ngày 11/11/2026, không phải yêu cầu tự cài các version này:

| Môi trường | Dependency đã quan sát |
|---|---|
| `.venv` | Python 3.11.9; NumPy 2.4.6; opencv-python 4.11.0.86; InsightFace 0.7.3; ONNX Runtime 1.31.1; ONNX 1.23.2; SciPy 1.17.1; scikit-image 0.26.0 |
| `.venv-hybrid` | Python 3.11.9; NumPy 2.2.6; Ultralytics 8.4.208; Torch 2.15.0; PyYAML 6.0.3; MediaPipe 1.2.2; opencv-python **và** opencv-python-headless 5.0.0.93 |

Hybrid cần COCO/YOLO/Torch/NumPy/OpenCV/PyYAML. MediaPipe landmarks optional theo
source; thiếu thì phải ghi nhận capability bị giảm. Coordinator cần OpenCV GUI.
`ai_training/requirements.txt` phục vụ image pipeline, có headless OpenCV và chưa
bao quát Face/coordinator venv; không dùng nó như lockfile chung cho cả hai venv.
AI2 không sửa requirements hoặc môi trường đã được thử nghiệm trước đây.

| Model | Bytes | SHA256 đo được |
|---|---:|---|
| `best_behavior_v1_4.pt` — mặc định | 5,467,994 | `6ce96c34ddc62327de37aa1b7ede9cd12863f4c0299f3d39b70f313263a04b47` |
| `best_behavior_v2.pt` — tùy chọn có sẵn | 5,473,818 | `4c932dc472c4480d7b1910203a5f4c09782ace770bd8fb8b8f20e8b638e3861e` |
| `pretrained_person/yolo11n_coco.pt` | 5,613,764 | `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1` |

Hai checkpoint behavior chỉ được selector chấp nhận theo tên trên. Worker kiểm tra
classes Sleeping/Using Phone khi load thật; preflight không load để xác nhận mapping.
Pose/Face `.task` optional cũng được inventory; hand asset bị worker chủ động chọn
`disabled_by_v6.task` không tồn tại. Không tải hand model cho demo này.

Face worker hiện dùng default InsightFace `buffalo_l` trong home của user chạy
worker, cần `det_10g.onnx` và `w600k_r50.onnx`. Preflight hash cả các `.onnx` khác
đang có trong cache vì FaceAnalysis liệt kê chúng trước khi lọc allowed modules.
Thiếu asset phải dừng chuẩn bị demo; không chạy camera để kích hoạt download.

Hiện có 2 enrollment archives. Preflight chỉ kiểm tra file ZIP và hai member
`student_id.npy`/`embeddings.npy`, không đọc/ghi/in các vectors. Runtime vẫn kiểm tra
SID khớp thư mục, Nx512, finite và norm. Có file không chứng minh nhận diện đúng.

Hai WARN thực tế ban đầu: nhiều distribution cùng ghi `cv2` trong Hybrid venv;
config `hybrid_v7.json` có duplicate keys (production đọc last-key-wins). Không tự
gộp/xóa key hay sửa package trong AI2. Lưu report để tái hiện cấu hình hiện tại.

## 4. Chạy liên tục và giới hạn 10 observations

Chạy preflight lại sau khi đổi env; dùng `--output` với tên mới nếu lưu report.

```powershell
# Liên tục; nhấn Q tại cửa sổ OpenCV để dừng.
Remove-Item Env:HTTM_CAMERA_MAX_OBSERVATIONS -ErrorAction SilentlyContinue
$consoleLog = Join-Path $demoRoot ('continuous_' + [guid]::NewGuid().ToString('N') + '.log')
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py 2>&1 | Tee-Object -FilePath $consoleLog

# Phiên ngắn để kiểm thử, tự kết thúc sau đúng 10 AI observations.
$env:HTTM_CAMERA_MAX_OBSERVATIONS = '10'
$consoleLog = Join-Path $demoRoot ('limited10_' + [guid]::NewGuid().ToString('N') + '.log')
& .\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py 2>&1 | Tee-Object -FilePath $consoleLog
```

Đây là số AI observations, không phải preview frames. Không có inference FPS mới.
Q phải đặt stop, release camera và đóng worker subprocess; Ctrl+C là đường dự phòng.
Ghi nhận mọi cleanup error; không coi đóng cửa sổ là bằng chứng worker đã thoát.

## 5. Overlay và Shadow diagnostic

```powershell
# Person/Face/Behavior BBox trên AI Snapshot historical riêng.
$env:HTTM_PERSON_SNAPSHOT_OVERLAY = '1'
# Tắt ở phiên tiếp theo:
Remove-Item Env:HTTM_PERSON_SNAPSHOT_OVERLAY -ErrorAction SilentlyContinue

# Bật Shadow diagnostic, không cấp SID cho production.
$env:HTTM_IDENTITY_SHADOW = '1'
$env:HTTM_IDENTITY_SHADOW_DIR = Join-Path $demoRoot 'shadow'
# Tắt ở phiên tiếp theo:
Remove-Item Env:HTTM_IDENTITY_SHADOW -ErrorAction SilentlyContinue
Remove-Item Env:HTTM_IDENTITY_SHADOW_DIR -ErrorAction SilentlyContinue
```

Chỉ chuỗi chính xác `'1'` bật flag. `'true'` không bật; preflight cảnh báo. Env chỉ
ảnh hưởng tiến trình khởi chạy sau khi đặt, không đổi tiến trình camera đang chạy.

Person/Face tracker debug, tùy chọn khi cần đối chiếu geometry:

```powershell
$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_FACE_DEBUG = '1'
$env:HTTM_PERSON_DEBUG_DIR = Join-Path $demoRoot 'person_debug'
# Chỉ bật nếu chủ động cần ảnh observation để gán actor truth:
# $env:HTTM_PERSON_DEBUG_SNAPSHOTS = '1'
```

Không mặc định bật ghi ảnh/embedding mới. Shadow không ghi ảnh/embedding. Các
session directory chỉ xuất hiện khi có dữ liệu được ghi; đọc console first-frame/
output-path/errors. Folder chưa có không tự chứng minh diagnostic bị lỗi.

Continuous Shadow mặc định tối đa 10 rows/1MiB; limited dùng ngân sách observations
nhưng cap 100 rows. Person debug mặc định 10 rows/20MiB, có clamp 100 rows/100MiB.
Hai giới hạn khác nhau. AI vẫn chạy sau khi logger hết ngân sách. Với test Sleeping,
bắt đầu kịch bản sớm trong phiên ngắn mới để không bỏ lỡ dữ liệu cần quan sát.

## 6. Thu thập và phân tích bằng chứng

Giữ trong cùng PowerShell `$demoRoot` đã tạo ở bước 2:

```powershell
# Tìm log đúng trong bộ output của demo này, không lấy đại session ở nơi khác.
$shadowLog = Get-ChildItem -LiteralPath $demoRoot -Filter 'frames.jsonl' -Recurse -File |
    Where-Object { $_.Directory.Parent.Name -eq 'shadow' } |
    Sort-Object LastWriteTime -Descending | Select-Object -First 1
if ($null -eq $shadowLog) { throw 'No Shadow JSONL found. Check enabled env, AI frames and console writer messages.' }

$analysisDir = Join-Path $demoRoot ('analysis_' + [guid]::NewGuid().ToString('N'))
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_test_toolkit.py analyze $shadowLog.FullName --provenance recorded --output-dir $analysisDir

# Xem dữ liệu production đã ghi, không biến proposal Shadow thành SID thật.
$rows = Get-Content -LiteralPath $shadowLog.FullName | ForEach-Object { $_ | ConvertFrom-Json }
$rows | Select-Object frame_id, captured_at, production_observation_status
$rows | ForEach-Object { $_.sleeping } | Format-List
$rows | ForEach-Object { $_.using_phone } | Format-List
```

Chọn session mới nhất chỉ để tìm file; không ghép hai logger bằng thứ tự dòng hoặc
timestamp gần đúng. Với identity safety, dùng [actor-truth guide](CAMERA_IDENTITY_TRUTH_GUIDE.md)
để bind exact session/frame/timestamps/hash. `--provenance recorded` chỉ khai báo
nguồn log thực; không tự tạo ground truth hoặc camera PASS.

Mỗi case ghi: actor/roster độc lập, expected behavior của người quan sát, Frame ID,
PID, SID production, association/identity status, behavior state, duration, capture/
completion age, model/hash/env, console/session paths, reviewer và verdict. Không
lấy predicted SID hoặc `detected` làm nhãn đúng. Dùng [checklist AI2](AI_CAMERA_LIVE_DEMO_CHECKLIST.md)
và checklist S1–S5/P1–P5/IC1–IC6 hiện có khi cần chi tiết.

## 7. Lỗi thường gặp và đường xử lý

| Hiện tượng | Kiểm tra tối thiểu, không sửa thuật toán |
|---|---|
| Sai Python/ModuleNotFound | Chạy lại bằng `.venv` path; đọc package check của cả hai venv. Không tự pip install |
| Missing checkpoint/cache | Đối chiếu path/hash preflight; dừng, yêu cầu cung cấp asset đã duyệt. Không tự tải |
| Camera không mở | Xem index/backend console; kết nối iVCam, chọn camera device đúng và đóng app đang chiếm thiết bị; chạy lại cùng cấu hình |
| DSHOW và MSMF đều fail | Preflight không mở hardware; cần xác minh Windows/device/index thực. Source hiện index 0, chưa có env đổi camera index; không dùng env giả `CAMERA_SOURCE`/`HTTM_CAMERA_INDEX` để sửa runner này |
| OpenCV GUI unavailable | Đọc headless/collision WARN; đây chưa phải bằng chứng GUI chạy. Dừng và báo môi trường, không tự gỡ/cài |
| Sleeping detected nhưng duration `--` | Xem SID/association/gap; detector không cần face, duration cần accepted SID. Không bật retention |
| Using Phone suspected/unknown | Xem behavior evidence và COCO object trong diagnostic; không thay threshold/model để cố PASS |
| Không thấy Shadow log | Xác nhận env trong đúng shell, first AI observation, output path, byte/frame limit và writer error |
| STALE | Đối chiếu Frame ID/capture age/completion age; nhãn historical là chủ đích, không phải bbox trên live frame |
| Q chậm/cleanup lỗi | Giữ console và thời điểm Q; AI1 có bounded join/terminate/kill. Chưa có kiểm chứng native stop latency bằng camera |

## 8. Env đã hỗ trợ và rollback cấu hình mặc định

| Env | Mặc định / giá trị thực sự hỗ trợ |
|---|---|
| `HTTM_BEHAVIOR_CHECKPOINT` | unset → V1.4; chỉ tên V1.4 hoặc V2 trong selector |
| `HTTM_CAMERA_MAX_OBSERVATIONS` | unset → continuous; số nguyên dương để giới hạn |
| `HTTM_FACE_TRACKER_TRIAL_MAX_AGE` | unset → 2.0s; chỉ `'4.0'` opt-in |
| `HTTM_PERSON_TRIAL_MIN_CONF` | unset → baseline; chỉ `'0.35'` opt-in |
| `HTTM_PERSON_SNAPSHOT_OVERLAY` | OFF; exact `'1'` bật BBox window |
| `HTTM_IDENTITY_SHADOW`, `HTTM_IDENTITY_SHADOW_DIR` | OFF; exact `'1'`, custom output optional |
| `HTTM_PERSON_DEBUG`, `HTTM_FACE_DEBUG` | OFF; Face debug cần Person debug ON |
| `HTTM_PERSON_DEBUG_SNAPSHOTS` | OFF; ghi ảnh opt-in, cần Person debug ON |
| `HTTM_PERSON_DEBUG_DIR`, `HTTM_PERSON_DEBUG_MAX_FRAMES`, `HTTM_PERSON_DEBUG_MAX_BYTES` | custom directory và bounded limits hiện có |

Không có env hợp lệ để bật WARNING hoặc T_hold. Các tên lạ được preflight báo
WARN, không biến thành quyền cấu hình production.

Sau khi đã dừng tiến trình camera, reset **chỉ các biến runner này hỗ trợ**, không
xóa mọi env HTTM của người dùng:

```powershell
$aiDemoVariables = @(
    'HTTM_BEHAVIOR_CHECKPOINT', 'HTTM_CAMERA_MAX_OBSERVATIONS',
    'HTTM_FACE_TRACKER_TRIAL_MAX_AGE', 'HTTM_PERSON_TRIAL_MIN_CONF',
    'HTTM_PERSON_SNAPSHOT_OVERLAY', 'HTTM_IDENTITY_SHADOW', 'HTTM_IDENTITY_SHADOW_DIR',
    'HTTM_PERSON_DEBUG', 'HTTM_FACE_DEBUG', 'HTTM_PERSON_DEBUG_SNAPSHOTS',
    'HTTM_PERSON_DEBUG_DIR', 'HTTM_PERSON_DEBUG_MAX_FRAMES', 'HTTM_PERSON_DEBUG_MAX_BYTES'
)
foreach ($aiDemoVariable in $aiDemoVariables) {
    Remove-Item -LiteralPath ('Env:' + $aiDemoVariable) -ErrorAction SilentlyContinue
}
& .\.venv\Scripts\python.exe -B .\ai_training\integration\camera_live_preflight.py
```

Reset không xóa log/ảnh/model/enrollment. Nếu cần giữ cấu hình trial đã duyệt,
lưu report/env trước và phục hồi đúng giá trị riêng; AI2 không tự áp trial.
Rollback AI2: ngừng dùng preflight/runbook/checklist và bỏ các file mới nếu cần;
không có source production hoặc thuật toán nào bị sửa.
