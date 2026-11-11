# Hướng dẫn kiểm thử PID/SID khi SV002 gục đầu

Tài liệu dành cho nhóm chạy camera thật trên nhánh `Chiv2`, thu Person Debug,
Face Debug và Identity Shadow, sau đó đối chiếu PID, SID, Sleeping và Duration.
Các lệnh tạo dữ liệu mới trong `ai_training/runs/`; không sửa source/config.

Giữ `T_hold=None`. Shadow không cấp SID production. Không đổi Association V4,
PersonTracker, FaceTracker, DurationTracker, model, threshold hoặc camera index.
Gục đầu là thao tác kiểm thử, không tự chứng minh sinh viên đang ngủ.

## 1. Chuẩn bị

- Mở PowerShell tại thư mục clone repository, đang ở nhánh `Chiv2`.
- Máy đã có `.venv\Scripts\python.exe` cho coordinator/Face và
  `.venv-hybrid\Scripts\python.exe` cho Hybrid; coordinator tự gọi cả hai.
- Đã chuẩn bị model, InsightFace assets và enrollment SV002 hợp lệ. Không dùng
  lần chạy này để tự cài package, tải model hoặc đăng ký lại khuôn mặt.
- Camera hiện tại dùng index `0`, thử DSHOW rồi MSMF. Đóng ứng dụng đang chiếm camera.
- Có người hỗ trợ bấm giờ/quan sát. Xác minh người tham gia là SV002 bằng danh sách
  và quan sát trực tiếp, độc lập với SID do AI báo. Lần đầu chỉ kiểm thử một người.
- Có thể tham khảo [runbook môi trường](AI_CAMERA_LIVE_RUNBOOK.md) nếu máy chưa sẵn sàng.

Entrypoint là `ai_training/integration/camera_coordinator_live.py`.
Không chạy `camera_coordinator.py`: đó là runner khác, không có pipeline Duration này.

### Cấu hình diagnostics có sẵn

| Biến môi trường | Giá trị dùng | Ý nghĩa |
|---|---|---|
| `HTTM_PERSON_DEBUG` | `1` | Bật Person Debug. |
| `HTTM_FACE_DEBUG` | `1` | Ghi block Face Debug trong Person log; cần Person Debug. |
| `HTTM_PERSON_DEBUG_SNAPSHOTS` | `1` | Lưu ảnh đúng observation để đối chiếu. |
| `HTTM_PERSON_DEBUG_DIR` | Thư mục riêng mỗi lần chạy | Gốc session Person/Face. |
| `HTTM_PERSON_DEBUG_MAX_FRAMES` | `100` | Mặc định 10, tối đa 100 observations. |
| `HTTM_PERSON_DEBUG_MAX_BYTES` | `104857600` | Mặc định 20 MiB, tối đa 100 MiB, gồm log và ảnh. |
| `HTTM_IDENTITY_SHADOW` | `1` | Bật Shadow chỉ để chẩn đoán. |
| `HTTM_IDENTITY_SHADOW_DIR` | Thư mục riêng mỗi lần chạy | Gốc session Shadow. |
| `HTTM_CAMERA_MAX_OBSERVATIONS` | `100` | Giới hạn camera 100 observations, đồng thời cấp budget Shadow 100. |

**Không chỉ tăng Person Debug:** nếu bỏ `HTTM_CAMERA_MAX_OBSERVATIONS`, camera chạy
liên tục nhưng Shadow chỉ ghi 10 observations. Shadow bị chặn ở 100 observations
hoặc 1 MiB; không có biến môi trường tăng giới hạn byte. Giá trị camera lớn hơn
100 không làm Shadow ghi quá 100.

100 observations không phải 100 giây. Nếu log hết budget trước khi đủ ba giai đoạn,
ghi nhận thiếu dữ liệu; không ghép nhiều lần khởi động thành một chuỗi PID.
Giới hạn byte có thể làm bỏ ảnh hoặc ngừng log trước giới hạn observation.

Các lệnh không đặt/xóa `HTTM_BEHAVIOR_CHECKPOINT`, `HTTM_PERSON_TRIAL_MIN_CONF`,
`HTTM_FACE_TRACKER_TRIAL_MAX_AGE`. Dùng cùng môi trường với cấu hình cần tái hiện.
Khi không có override, source chọn `best_behavior_v1_4.pt`; Hybrid V7 không đồng
nghĩa tự chọn checkpoint v2. Ghi lại model thực tế từ dòng `[BEHAVIOR MODEL]`.

## 2. PowerShell A: mở camera và ghi diagnostics

Chạy cả khối tại thư mục clone. Không cần activate venv hoặc đổi ExecutionPolicy.
Giữ cửa sổ A để dùng tiếp các biến khi trích xuất dữ liệu.

```powershell
$repo = git rev-parse --show-toplevel
if ($LASTEXITCODE -ne 0 -or -not $repo) {
    throw 'Hay mo PowerShell trong repository HTTM.'
}
$repo = $repo.Trim()
Set-Location -LiteralPath $repo
if ((git branch --show-current).Trim() -ne 'Chiv2') {
    throw 'Can chay tren nhanh Chiv2; kiem tra checkout truoc khi tiep tuc.'
}

$python = Join-Path $repo '.venv\Scripts\python.exe'
$hybridPython = Join-Path $repo '.venv-hybrid\Scripts\python.exe'
foreach ($path in @($python, $hybridPython)) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Khong tim thay: $path"
    }
}

$runId = 'sv002_sleep_{0}_{1}' -f `
    (Get-Date -Format 'yyyyMMdd_HHmmss'), `
    ([guid]::NewGuid().ToString('N').Substring(0, 8))
$runDir = Join-Path $repo "ai_training\runs\$runId"
New-Item -ItemType Directory -Path $runDir -ErrorAction Stop | Out-Null

$env:HTTM_PERSON_DEBUG = '1'
$env:HTTM_FACE_DEBUG = '1'
$env:HTTM_PERSON_DEBUG_SNAPSHOTS = '1'
$env:HTTM_PERSON_DEBUG_MAX_FRAMES = '100'
$env:HTTM_PERSON_DEBUG_MAX_BYTES = '104857600'
$env:HTTM_PERSON_DEBUG_DIR = Join-Path $runDir 'person'
$env:HTTM_IDENTITY_SHADOW = '1'
$env:HTTM_IDENTITY_SHADOW_DIR = Join-Path $runDir 'shadow'
$env:HTTM_CAMERA_MAX_OBSERVATIONS = '100'
$env:PYTHONDONTWRITEBYTECODE = '1'

Write-Host "RUN_DIR=$runDir" -ForegroundColor Cyan
$overrides = foreach ($name in @(
    'HTTM_BEHAVIOR_CHECKPOINT',
    'HTTM_PERSON_TRIAL_MIN_CONF',
    'HTTM_FACE_TRACKER_TRIAL_MAX_AGE'
)) {
    $value = [Environment]::GetEnvironmentVariable($name, 'Process')
    [pscustomobject]@{ Name = $name; Value = $value }
}
$overrides | Format-Table -AutoSize
$overrides | ConvertTo-Json | Set-Content `
    -LiteralPath (Join-Path $runDir 'runtime_overrides.json') -Encoding UTF8

$cameraProcess = Start-Process `
    -FilePath $python `
    -ArgumentList @('-B', '-u', 'ai_training/integration/camera_coordinator_live.py') `
    -WorkingDirectory $repo `
    -NoNewWindow `
    -RedirectStandardOutput (Join-Path $runDir 'stdout.log') `
    -RedirectStandardError (Join-Path $runDir 'stderr.log') `
    -PassThru `
    -Wait

Write-Host "Camera exit code: $($cameraProcess.ExitCode)"
Write-Host "RUN_DIR=$runDir"
```

Biến môi trường chỉ áp dụng cho PowerShell này và tiến trình con, không dùng `setx`.
Giá trị override rỗng trong bảng/JSON nghĩa là dùng mặc định source.
Camera mở cửa sổ OpenCV; console output đi vào hai file riêng. Nhấn `Q` trong cửa sổ
camera để dừng bình thường; không đóng cưỡng bức tiến trình giữa lúc ghi log.

### Các file sẽ được tạo

```text
ai_training/runs/sv002_sleep_<thoi-gian>_<id>/
├── runtime_overrides.json
├── stdout.log
├── stderr.log
├── phase_markers.jsonl                 # Bước 3 tạo
├── person/
│   └── session_<uuid-person>/
│       ├── frames.jsonl                # Person + block face_tracker
│       └── obs_001_frame_000001.png ...
└── shadow/
    └── session_<uuid-shadow>/
        └── frames.jsonl                # Shadow + behavior/duration production
```

Hai UUID độc lập, không có file Face Debug riêng. Logger tạo session khi ghi được
observation đầu tiên; chưa có file thì kiểm tra `stderr.log`. Ảnh có thể không được
lưu ở một số observation; đọc `snapshot.saved/reason` trước khi dùng làm bằng chứng.

## 3. PowerShell B: ba giai đoạn 15–30–15 giây

Mở PowerShell thứ hai, dán đúng `RUN_DIR` mà A đã in, không lấy session cũ gần nhất.

```powershell
$runDir = Read-Host 'Dan RUN_DIR tu cua so A, khong them dau ngoac kep'
$repo = (Resolve-Path -LiteralPath (Join-Path $runDir '..\..\..')).Path
$python = Join-Path $repo '.venv\Scripts\python.exe'
Get-Content -LiteralPath (Join-Path $runDir 'stderr.log') -Tail 25 -Encoding UTF8
```

Đường dẫn checkout ở B được lấy từ RUN_DIR do A tạo. Chờ model tải xong, AI Panel có kết quả nhận SV002
khi ngồi bình thường; log có `[PERSON DEBUG]` và `[IDENTITY SHADOW] first frame written`.
Chưa bấm giờ trong lúc khởi động model. Warm-up vẫn tiêu thụ observation budget.

Sau đó chạy khối sau. Người hỗ trợ đọc tín hiệu để sinh viên đổi tư thế.

```powershell
Read-Host 'AI da san sang va SV002 dang ngoi binh thuong: nhan Enter de bat dau'

$phaseCode = @'
import json
import sys
import time
from pathlib import Path

path = Path(sys.argv[1])
phases = [(0, 'upright'), (15, 'head_down'), (45, 'head_up'), (60, 'end')]
with path.open('x', encoding='utf-8') as stream:
    started = time.perf_counter()
    for offset, phase in phases:
        time.sleep(max(0.0, started + offset - time.perf_counter()))
        row = {
            'phase': phase,
            'planned_offset_seconds': offset,
            'at_perf_counter': time.perf_counter(),
            'clock': 'perf_counter',
        }
        stream.write(json.dumps(row) + '\n')
        stream.flush()
        print(str(offset) + 's: ' + phase, flush=True)
'@

& $python -B -u -c $phaseCode (Join-Path $runDir 'phase_markers.jsonl')
```

| Tín hiệu | Thao tác |
|---|---|
| `0s: upright` | Ngồi bình thường, mặt hướng camera trong 15 giây. |
| `15s: head_down` | Gục đầu xuống bàn, giữ 30 giây. |
| `45s: head_up` | Ngẩng lên, ngồi bình thường 15 giây. |
| `60s: end` | Tiếp tục ngồi bình thường, chờ thêm khoảng hai kết quả AI hoàn tất rồi nhấn `Q`. |

Mốc dùng `perf_counter` trên Windows, cùng loại đồng hồ với capture timestamp của
coordinator. Đây là thời điểm phát tín hiệu, không phải nhãn tư thế/identity tự động.
Xem ảnh/video để xác nhận phản ứng thực tế, nhất là observation sát mốc chuyển pha.
AI Panel là kết quả lịch sử; không dùng ảnh live hiện tại để gắn nhãn observation cũ.

Để xem camera còn sinh kết quả, có thể mở PowerShell khác, dán RUN_DIR rồi chạy:

```powershell
$runDir = Read-Host 'Dan RUN_DIR'
Get-Content -LiteralPath (Join-Path $runDir 'stdout.log') -Tail 10 -Wait
```

`Ctrl+C` tại cửa sổ xem log chỉ dừng lệnh theo dõi. Sau bước 60 giây phải kiểm tra
capture timestamp cuối đã qua mốc kết thúc; kết quả hoàn tất muộn có thể dùng ảnh cũ.

## 4. PowerShell A: nạp log sau khi camera đã đóng

Dùng tiếp cửa sổ A. Nếu mở cửa sổ mới, khối dưới sẽ hỏi đúng RUN_DIR rồi khôi phục
các đường dẫn; không chọn tự động theo ngày mới nhất.

```powershell
if (-not (Get-Variable -Name runDir -ValueOnly -ErrorAction SilentlyContinue)) {
    $runDir = Read-Host 'Dan dung RUN_DIR cua phien can phan tich'
}
$repo = (Resolve-Path -LiteralPath (Join-Path $runDir '..\..\..')).Path
Set-Location -LiteralPath $repo
$python = Join-Path $repo '.venv\Scripts\python.exe'
$runId = Split-Path -Path $runDir -Leaf

$personLogs = @(
    Get-ChildItem -LiteralPath (Join-Path $runDir 'person') `
        -Filter 'frames.jsonl' -Recurse -File
)
$shadowLogs = @(
    Get-ChildItem -LiteralPath (Join-Path $runDir 'shadow') `
        -Filter 'frames.jsonl' -Recurse -File
)
if ($personLogs.Count -ne 1 -or $shadowLogs.Count -ne 1) {
    throw 'Can dung mot Person log va mot Shadow log trong RUN_DIR nay.'
}
$personLog = $personLogs[0].FullName
$shadowLog = $shadowLogs[0].FullName
$personRows = @(
    Get-Content -LiteralPath $personLog -Encoding UTF8 |
        Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
)
$shadowRows = @(
    Get-Content -LiteralPath $shadowLog -Encoding UTF8 |
        Where-Object { $_.Trim() } | ForEach-Object { $_ | ConvertFrom-Json }
)
if ($personRows.Count -eq 0 -or $shadowRows.Count -eq 0) {
    throw 'Log rong; kiem tra stdout.log va stderr.log.'
}
$invalidShadow = @($shadowRows | Where-Object {
    $_.mode -ne 'SHADOW_ONLY' -or $_.production_applied -ne $false -or
    $null -ne $_.parameters.T_hold -or $null -ne $_.simulation_hold_seconds
})
if ($invalidShadow.Count -gt 0) {
    throw 'Shadow metadata khong dung T_hold=None / SHADOW_ONLY.'
}

Write-Host "Person observations: $($personRows.Count)"
Write-Host "Shadow observations: $($shadowRows.Count)"
$personRows[0].parameters | Format-List
$personRows[0].face_tracker.parameters | Format-List
$shadowRows[0].parameters | Format-List
Select-String -LiteralPath (Join-Path $runDir 'stderr.log') `
    -Pattern 'BEHAVIOR MODEL|TRIAL|SHADOW|PERSON DEBUG|byte limit|stopped|failed'
```

## 5. Xuất bảng so sánh theo observation

### SID production, Sleeping và Duration

Không lọc `student_id == SV002`: sẽ bỏ mất đoạn SID Unknown cần điều tra.
Đoạn dưới ghép đúng frame và timestamp, không dùng nearest-time hoặc thứ tự dòng.
Việc xác nhận hai nguồn cùng lần capture vẫn do người kiểm thử thực hiện.

```powershell
$markers = @(
    Get-Content -LiteralPath (Join-Path $runDir 'phase_markers.jsonl') `
        -Encoding UTF8 | ForEach-Object { $_ | ConvertFrom-Json }
)
$phaseStart = @($markers | Where-Object { $_.phase -eq 'upright' })
$phaseEnd = @($markers | Where-Object { $_.phase -eq 'end' })
if ($phaseStart.Count -ne 1 -or $phaseEnd.Count -ne 1) {
    throw 'Thieu moc bat dau/ket thuc; khong tu suy doan timeline.'
}
$t0 = [double]$phaseStart[0].at_perf_counter

function Get-TestPhase([double]$captureTime) {
    $previous = @(
        $markers | Where-Object { [double]$_.at_perf_counter -le $captureTime } |
            Sort-Object at_perf_counter
    )
    if ($previous.Count -eq 0) { return 'warmup' }
    return $previous[-1].phase
}

$personByFrame = @{}
foreach ($record in $personRows) {
    $key = [int]$record.frame_id
    if ($personByFrame.ContainsKey($key)) { throw "Trung Person frame $key" }
    $personByFrame[$key] = $record
}

$comparison = @(
    foreach ($record in $shadowRows) {
        $person = $personByFrame[[int]$record.frame_id]
        if ($null -eq $person -or
            $person.captured_at -ne $record.captured_at -or
            $person.tracker_updated_at -ne $record.tracker_updated_at -or
            $person.clock -ne $record.clock) {
            throw "Metadata khong khop tai frame $($record.frame_id)"
        }
        foreach ($sleep in $record.sleeping) {
            [pscustomobject]@{
                Frame           = $record.frame_id
                CapturedAt      = $record.captured_at
                RelativeSeconds = [Math]::Round(
                    ([double]$record.captured_at - $t0), 3)
                Phase           = Get-TestPhase ([double]$record.captured_at)
                PID             = $sleep.person_track_id
                ProductionSID   = $sleep.student_id
                IdentityStatus  = $sleep.identity_status
                Association     = $sleep.association_status
                FacePresent     = $sleep.current_face_present
                FaceSID         = $sleep.current_face_student_id
                ShadowSID       = $sleep.shadow_proposed_student_id
                ShadowStatus    = $sleep.identity_continuity_status
                ShadowReason    = $sleep.identity_continuity_reason
                Sleeping        = $sleep.sleeping_state
                SleepConfidence = $sleep.sleeping_model_confidence
                SleepReason     = $sleep.sleeping_decision_reason
                Seconds         = $sleep.sleeping_duration_seconds
                DurationEvent   = $sleep.duration_event
                Blockers        = $sleep.duration_blockers -join ';'
                ResetReasons    = $sleep.duration_reset_reasons -join ';'
                CaptureGap      = $person.captured_at_gap
                TrackerGap      = $person.tracker_update_gap
                SnapshotSaved   = $person.snapshot.saved
                SnapshotFile    = $person.snapshot.file
            }
        }
    }
)
$comparison | Export-Csv `
    -LiteralPath (Join-Path $runDir 'sleeping_comparison.csv') `
    -NoTypeInformation -Encoding UTF8
$comparison | Select-Object Frame, RelativeSeconds, Phase, PID, ProductionSID, `
    FaceSID, Association, Sleeping, Seconds | Format-Table -AutoSize

$shadowRows | Group-Object { Get-TestPhase ([double]$_.captured_at) } |
    Select-Object Name, Count | Format-Table -AutoSize
$lastCapture = ($shadowRows | Measure-Object captured_at -Maximum).Maximum
if ($lastCapture -lt [double]$phaseEnd[0].at_perf_counter) {
    Write-Warning 'Shadow chua ghi qua moc 60 giay: chua du coverage.'
}
foreach ($phase in @('upright', 'head_down', 'head_up')) {
    $count = @($shadowRows | Where-Object {
        (Get-TestPhase ([double]$_.captured_at)) -eq $phase
    }).Count
    if ($count -eq 0) { Write-Warning "Khong co Shadow observation o pha $phase" }
}
```

`Seconds` rỗng là duration `None`/không khả dụng, không phải 0 giây. Không có person
thì frame vẫn có trong JSONL nhưng không tạo dòng person trong CSV. `FaceSID` trong
Shadow là thông tin mặt liên quan theo resolver, không thay thế raw Face Debug.
`DurationEvent/ResetReasons` là phân loại chẩn đoán từ output, không phải event nội
bộ của DurationTracker. Kiểm tra `details_truncated` trong log gốc nếu dữ liệu thiếu.

### Face Recognition và sự kiện PersonTracker

```powershell
$faceRows = @(
    foreach ($record in $personRows) {
        foreach ($face in $record.face_tracker.detections) {
            $mapping = @($record.face_tracker.mapping | Where-Object {
                $_.detection_indices -contains $face.detection_index
            })
            [pscustomobject]@{
                Frame          = $record.frame_id
                CapturedAt     = $record.captured_at
                Phase          = Get-TestPhase ([double]$record.captured_at)
                DetectionIndex = $face.detection_index
                CurrentSID     = $face.current_student_id
                Similarity     = $face.recognition_score
                FaceBBox       = $face.face_bbox -join ','
                MappingJSON    = ConvertTo-Json -InputObject $mapping -Depth 8 -Compress
            }
        }
    }
)
$faceRows | Export-Csv -LiteralPath (Join-Path $runDir 'face_recognition.csv') `
    -NoTypeInformation -Encoding UTF8

$trackEvents = @(
    foreach ($record in $personRows) {
        foreach ($decision in $record.decisions) {
            [pscustomobject]@{
                Frame      = $record.frame_id
                CapturedAt = $record.captured_at
                Phase      = Get-TestPhase ([double]$record.captured_at)
                PID        = $decision.track_id
                Action     = $decision.action
                Reason     = $decision.reason
            }
        }
    }
)
$trackEvents | Export-Csv -LiteralPath (Join-Path $runDir 'person_track_events.csv') `
    -NoTypeInformation -Encoding UTF8
$trackEvents | Where-Object { $_.Action -ne 'matched' } | Format-Table -AutoSize
```

So sánh bằng `Frame` và `CapturedAt`; `MappingJSON` giữ cả thông tin mơ hồ và liên
kết face track → PID khi có. Mặt chưa association vẫn có dòng Face Debug, không tự
gán cho PID gần nhất. Xem `candidate_ious`, `tracks_before/after` trong Person JSONL
để truy nguyên track bị xóa/tạo; reason từ chối V4 chi tiết chưa được ghi đầy đủ.

### Báo cáo bằng toolkit có sẵn (tùy chọn)

```powershell
& $python -B 'ai_training/integration/camera_live_test_toolkit.py' `
    analyze $shadowLog --provenance recorded `
    --output-dir (Join-Path $runDir 'analysis_shadow')

& $python -B 'ai_training/integration/camera_live_test_toolkit.py' `
    analyze $personLog --provenance recorded `
    --output-dir (Join-Path $runDir 'analysis_person')
```

Mỗi thư mục có `analysis.json`, `analysis.md`, `observations.csv`, `events.csv`,
`frames.csv`. Output directory phải chưa tồn tại; đổi tên khi phân tích lại.
Đây là kiểm tra log, không xác nhận actor hay nghiệm thu model.

## 6. Đối chiếu actor thật bằng camera_identity_truth.py

Chỉ tạo template khi camera đã đóng vì công cụ khóa nguồn bằng SHA256.

```powershell
$truthPath = Join-Path $runDir 'actor_truth.json'
& $python -B 'ai_training/integration/camera_identity_truth.py' init `
    --person $personLog --shadow $shadowLog `
    --capture-session-id $runId --provenance recorded --output $truthPath
Write-Host "Template: $truthPath"
```

Template để trống actor/annotation và chưa xác nhận pairing. Người kiểm thử mở file
JSON rồi điền sau khi xem đúng ảnh/video:

| Phần | Cần điền |
|---|---|
| `pairing` | `confirmed=true`, `reviewer`, `evidence` chứng minh hai nguồn cùng lần capture. Timestamp/clock vẫn phải khớp. |
| `actors` | Actor ID độc lập; `expected_student_id=SV002` từ xác minh trực tiếp/danh sách; `basis=manual_roster`, `independent_of_ai=true`, `confirmed=true`, reviewer/evidence. |
| `annotations` | `kind=frame` hoặc `segment`; actor đã xem trong ảnh; `basis=manual_visual`, `independent_of_ai=true`, `confirmed=true`, reviewer/evidence. |
| `members` | Copy đúng `person_frame_id`, `person_track_id`, `captured_at`, `tracker_updated_at` từ `observation_catalog` sau khi xác minh actor. Segment liệt kê từng member và có capture_start/end. |

Không lấy SID AI làm roster, không mặc định P1 là SV002, không tự gán nhãn cho mọi
frame cùng PID. PID chỉ là khóa định vị observation để người kiểm thử gắn nhãn.
Nếu không đủ bằng chứng thì để observation chưa gắn nhãn.
Xem schema minh họa trong [Camera Identity Truth Guide](CAMERA_IDENTITY_TRUTH_GUIDE.md).

Sau khi gắn nhãn thủ công:

```powershell
$truthReportDir = Join-Path $runDir (
    'identity_review_' + [guid]::NewGuid().ToString('N')
)
& $python -B 'ai_training/integration/camera_identity_truth.py' evaluate `
    $truthPath --output-dir $truthReportDir
```

Kết quả: `identity.json`, `identity.csv`, `identity.md`, `events.csv`,
`unmatched_shadow.csv`. Đọc `status` và `issues`: exit 0 có thể là INCONCLUSIVE,
không tự coi là camera PASS. Không sửa JSONL nguồn sau khi tạo template.

## 7. Cách kết luận và bàn giao

| Quan sát | Điều có thể kết luận / cần kiểm tra |
|---|---|
| PID giữ nguyên, mất mặt, SID production rỗng, Sleeping detected | Phù hợp nhánh thiếu bằng chứng nhận diện khi person tracking còn liên tục; xác minh cùng actor bằng ảnh. |
| `unmatched_removed` hoặc `expired`, rồi `created` | Track bị kết thúc/tạo lại; xem IoU, bbox, gaps và ảnh để biết nguyên nhân cụ thể. |
| Face nhận SV002 nhưng SID production rỗng | Xem association và identity_status; raw association SID không phải SID production đã được chấp nhận. |
| SID phục hồi nhưng Duration về 0 | Đối chiếu observation làm episode đứt; timer không nối episode đã mất identity. |
| PID không đổi | Chưa chứng minh cùng người; cùng vị trí có thể thay actor. |
| Thiếu pha, log chạm cap hoặc không đủ ảnh/nhãn | Thiếu bằng chứng/INCONCLUSIVE; không suy diễn thành PID ổn định. |

Diagnostics/PNG có thêm I/O. Xem `TrackerGap` và `CaptureGap`, không chỉ elapsed
inference; log có thể làm thay đổi khoảng cách cập nhật dù thuật toán giữ nguyên.

Bàn giao: run ID, commit đang chạy, runtime_overrides.json, model thực tế, hai đường
dẫn JSONL, số observation mỗi pha, CSV xuất ra, các frame PID/SID đổi, cùng bằng
chứng actor thủ công. Nêu rõ phần nào từ code/log, phần nào đã được người quan sát
xác minh và phần nào chưa đủ dữ liệu. Chỉ chia sẻ ảnh/log camera qua kênh nhóm đã
thống nhất; không kèm dữ liệu camera/enrollment vào commit tài liệu hướng dẫn.

Đóng hai cửa sổ PowerShell khi hoàn tất để không mang biến diagnostics sang phiên
khác. Các lệnh trên không thay cấu hình môi trường ở mức user/machine.

## 8. Source dùng để kiểm tra hướng dẫn

- [camera_coordinator_live.py](camera_coordinator_live.py): ai_loop, shadow_budget,
  FaceTracker trial setting, capture và AI Panel.
- [camera_live_runtime.py](camera_live_runtime.py): HTTM_CAMERA_MAX_OBSERVATIONS.
- [person_tracker_diagnostics.py](person_tracker_diagnostics.py): flags, bounds,
  snapshots và Person JSONL.
- [face_tracker_diagnostics.py](face_tracker_diagnostics.py): detections, score, mapping.
- [identity_continuity_diagnostics.py](identity_continuity_diagnostics.py): Shadow
  flags, T_hold=None, bounds, production behavior rows.
- [behavior_checkpoint.py](../src/behavior_checkpoint.py): checkpoint mặc định/override.
- [camera_live_test_toolkit.py](camera_live_test_toolkit.py) và
  [camera_identity_truth.py](camera_identity_truth.py): CLI phân tích offline.

Tài liệu là quy trình chuẩn bị và chạy; không phải báo cáo một phiên camera đã thực hiện.
