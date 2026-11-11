# Kiểm thử độ ổn định PersonTracker — 2026-11-10

**Task:** PT-01–PT-08 và sửa Person Track ID thay đổi do nhịp xử lý AI chậm.

**Trạng thái:** Đã sửa và kiểm thử tự động. Người dùng đã báo camera iPhone/iVCam
xử lý được 10 frame trước sửa, nhưng bị đổi P2 → P3. Chưa kiểm thử camera sau sửa;
dừng chờ người dùng xác nhận, chưa chuyển CH-20.

## Khảo sát source và Git

- Các đường dẫn tích hợp trong yêu cầu đều tồn tại. Không tìm thấy `AGENTS.md`
  trong repository hoặc các thư mục cha được kiểm tra.
- Trước khi sửa, `.gitignore` đã có thay đổi; toàn bộ `ai_training/integration`,
  `ai_training/recognition`, `ai_training/enrollment` và một số script camera là
  dữ liệu chưa được Git theo dõi. Không coi các tệp này là do task này tạo ra.
- Trước lượt PT-01–PT-08 đầu tiên chưa có log `[PERSON TRACKING]` hoặc bộ test
  tracker. Chúng đã được thêm ở lượt đó và được giữ/mở rộng trong bản sửa này.
- `camera_coordinator.py` cung cấp `Worker`, `ROOT`, `INTEGRATION`,
  `CAMERA_INDEX=2`, `WORKER_TIMEOUT=90`; hàm `main()` của nó không dùng
  PersonTracker. Entry point kiểm thử tracking là `camera_coordinator_live.py`.
- Trong mã đang chạy, PersonTracker được khởi tạo và cập nhật ở `ai_loop()`
  của Camera Live. Một bản backup `camera_coordinator_live_before_person_panel.py`
  cũng chứa lời gọi tracker; không chỉnh sửa backup.

## Dòng dữ liệu Track ID

1. Camera Live giữ frame mới nhất trong queue, ghi ảnh tạm và gọi hai stream
   worker lần lượt: Face ID bằng `.venv`, Hybrid V7 bằng `.venv-hybrid`.
   Không có inference song song trong `ai_loop()` hiện tại.
2. `face_id_stream_worker.py` trả bbox, Student ID và score. `FaceTracker`
   theo dõi bbox, trả ID số nguyên và `stable_label` sau tối thiểu 3 quan sát
   cùng nhãn. Camera Live giữ nhận dạng hiện tại và nhãn ổn định riêng;
   đầu vào Association V4 dùng nhận dạng hiện tại. Panel hiển thị Face Track
   dưới dạng `T1`, `T2`, ...
3. Hybrid V7 lấy `person_ref` từ `spatial["people"]`; nguồn tham chiếu là
   `detection_id` của bộ phát hiện object (`person_{i}`), được tạo lại theo ảnh.
   Stream worker trả `people`, bbox clipped/raw và behaviors. `person_ref`
   không phải danh tính bền vững giữa các frame.
4. Association V4 liên kết face track với người theo hình học, trả face
   `track_id`, Student ID, `person_ref` và behaviors. Sau đó PersonTracker
   cập nhật danh sách người bằng IoU, trả `track_id="P1"`, `person_ref`, bbox.
5. Camera Live tạo `person_track_map[person_ref] = track_id` của frame hiện tại
   và gắn `match["person_track_id"]`. Không dùng face track ID làm person ID.
6. `tracked_people`, `tracked_faces`, `matches` được đưa vào result queue.
   `draw_ai_panel()` đọc `matches[].person_track_id` để hiển thị Person Track.
   Panel giữ kết quả AI lịch sử, chỉ hiển thị tối đa 1 face track và 2 matches;
   log mới ghi toàn bộ person tracks của mỗi frame được xử lý thành công.

## Root cause đã tái hiện

Frame 8 có timestamp `245676.420614`, bbox `[1.3, 92.81, 405.04, 415.77]`.
Frame 9 có timestamp `245680.0724245`, bbox `[8.61, 75.21, 377.65, 373.33]`.
Khoảng cập nhật là `3.6518105` giây; IoU tính bằng hàm trong source là
`0.7562669951159354`, cao hơn ngưỡng `0.3`.

Trong `update()`, track có `timestamp - last_seen > max_age` bị xóa trước khi
tạo các cặp IoU. Phép tái hiện đã cấp P1 rồi để hết hạn để frame 8 bắt đầu ở P2,
giữ đúng timestamp/bbox đã cung cấp. Quan sát số lần gọi hàm IoU ở frame 9:

| Cấu hình | Frame 8 → 9 | Số lần tính IoU ở frame 9 |
| --- | --- | --- |
| `max_age=3.0` trước sửa | P2 → P3 | 0: P2 bị xóa trước matching |
| `max_age=5.0` | P2 → P2 | 1: IoU đủ điều kiện |

Regression Camera Live trước sửa cũng thất bại với kết quả P2 → P3; sau sửa
đã chạy đạt với P2 → P2. Đây là tái hiện từ dữ liệu log, không phải chạy lại camera.

`person_updated_at` lấy từ `time.perf_counter()` sau khi Face ID và Hybrid V7
hoàn tất tuần tự. Tuổi track được đo giữa các lần hoàn tất xử lý; do đó thời gian
inference, chờ frame và chi phí xử lý đều có thể làm khoảng cập nhật vượt 3 giây.
Đổi sang giờ chụp hoặc di chuyển bước xóa track sau matching không giải quyết
được nhu cầu giới hạn tuổi của một bbox cũ. Bản sửa giữ nguyên đồng hồ và thứ tự
xóa track hết hạn trước matching.

## Giải pháp và tác động tới người mới/người quay lại

- Chỉ Camera Live dùng `PERSON_TRACK_MAX_AGE=5.0`, tạo dư địa khoảng 1.3 giây
  so với nhịp đã quan sát gần 3.7 giây. Đây là giới hạn cố định, không tự tăng
  theo thời gian chờ worker hoặc hồi sinh track quá hạn.
- Thêm tùy chọn `keep_unmatched=False` cho Camera Live. Sau matching, track
  không được ghép trong kết quả AI hiện tại bị xóa ngay, kể cả khi vẫn còn người
  khác hoặc bbox đầu vào không hợp lệ. Bước này diễn ra trước khi tạo track mới.
  Không kéo dài thời gian giữ track đã quan sát thấy mất lên 5 giây.
- Bbox tiếp tục ghép bằng IoU `>=0.3`, một track cho tối đa một detection.
  Không dùng Student ID hoặc `person_ref` để ép giữ ID. Track đã xóa không được
  tái sử dụng; bộ đếm ID chỉ tăng.
- Người hiện diện liên tục được giữ ID khi IoU đạt ngưỡng và khoảng cập nhật
  không quá 5 giây. Người ở vị trí mới không đạt IoU sẽ nhận ID mới. Người quay
  lại sau một kết quả AI vắng mặt, hoặc sau khoảng quá 5 giây, nhận ID mới ngay
  cả khi bbox/reference trùng trước đó.
- Lựa chọn này ưu tiên tránh gán lại track đã mất; đổi lại một lần detector bỏ
  sót/che khuất cũng có thể làm đổi ID khi người còn trong phòng.
- API mặc định `PersonTracker()` vẫn dùng `max_age=3.0`, `keep_unmatched=True`
  để giữ hành vi cho caller cũ. FaceTracker, Hybrid V7, InsightFace, model weights,
  dữ liệu Face ID và các file backup không được sửa.

## Files changed và implementation

| Tệp trong `ai_training/integration` | Thay đổi và lý do |
| --- | --- |
| `camera_coordinator_live.py` | Dùng tuổi thọ 5 giây và bỏ track không còn ghép được. Giữ log JSON `[PERSON TRACKING]` và đồng hồ cập nhật đã có. |
| `person_tracker.py` | Thêm tùy chọn loại track không ghép được; giữ mặc định cũ, matching IoU và schema kết quả. Giữ kiểm tra bbox đã sửa ở lượt trước. |
| `test_person_tracker.py` | 17 test độc lập: 9 test cũ và 8 regression cho frame 8–9, nhịp chậm, mất người, thay người, nhiều người và giới hạn thời gian. |
| `test_camera_person_tracking.py` | 8 test tích hợp với worker/đồng hồ giả; bao gồm frame 8–9 chính xác, vắng mặt, nhiều người và cập nhật chậm. |
| `PERSON_TRACKER_TEST_REPORT.md` | Báo cáo và hướng dẫn kiểm thử camera. |

Log có dạng sau; các giá trị minh họa này không phải bằng chứng camera:

```text
[PERSON TRACKING] {"frame_id": 1, "updated_at": 100.0, "tracked_people": [{"track_id": "P1", "person_ref": "person_0", "bbox": [20.0, 10.0, 120.0, 220.0]}]}
```

`updated_at` là số giây từ `time.perf_counter()`, dùng để so chênh lệch thời gian
cập nhật, không phải giờ lịch hay timestamp lúc chụp ảnh. Frame không có người
vẫn ghi `tracked_people: []`. Không thêm ảnh, embeddings hoặc Student ID vào log.

## Tests executed

Tất cả các kết quả dưới đây là kiểm thử mô phỏng/hồi quy không dùng camera.
Không cài thêm dependency; hai môi trường hiện có đều dùng Python 3.11.9.

Chạy từ `D:\Study\HTTM\HTTM-Chiv2`:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_person_tracker.py -v
# Ran 17 tests; OK

.\.venv\Scripts\python.exe -m unittest discover -s ai_training/integration -p test_camera_person_tracking.py -v
# Ran 8 tests; OK

.\.venv\Scripts\python.exe ai_training/integration/test_person_face_association.py
# PASS: 5/5 association tests

.\.venv\Scripts\python.exe ai_training/integration/test_hybrid_v7_real_output.py
# TOTAL MATCHED IMAGES: 2/2 (predictions đã lưu + face giả lập)

.\.venv-hybrid\Scripts\python.exe -m unittest ai_training.tests.test_hybrid_v7 -v
# Ran 34 tests; OK
```

| Mã | Tình huống mô phỏng | Kết quả đã chạy |
| --- | --- | --- |
| PT-01 | Một người đứng yên, 10 lần cập nhật cách 1 giây | Giữ P1; không sửa đầu vào. |
| PT-02 | Di chuyển nhẹ, 10 lần cập nhật | Giữ P1 và cập nhật bbox. |
| PT-03 | Reference đổi, thứ tự detection đảo, reference dùng lại cho người khác | Ghép theo bbox; ID theo đúng vị trí, không bám reference cũ. |
| PT-04 | Vắng mặt rồi trở lại trước/sau hết hạn | Trước hạn giữ P1; sau hạn tạo P2. Không trả track vắng mặt. |
| PT-05 | Hai người cùng lúc, đổi thứ tự detection | Hai ID riêng P1/P2 ổn định qua 10 lần cập nhật. |
| PT-06 | Hai người giao nhau, bbox chồng lấn/giống nhau | Mỗi detection có ID riêng. Quỹ đạo giao nhau đã mô phỏng gây hoán đổi ID; không xác nhận duy trì danh tính. |
| PT-07 | 17 kiểu bbox không hợp lệ, trộn với một detection hợp lệ | Bỏ qua bbox lỗi, không dừng tracker hay tạo track từ dữ liệu lỗi. |
| PT-08 | API mặc định vẫn có tuổi thọ 3 giây; khoảng cập nhật 1.53, 3.0, 3.001, 3.74 giây | 1.53/3.0 giữ P1; 3.001/3.74 tạo P2. Camera Live dùng cấu hình riêng được kiểm thử bên dưới. |

Tám regression độc lập mới xác nhận: frame 8–9 giữ P2 với giới hạn 5 giây;
10 lần cập nhật cách khoảng 1.53–3.74 giây giữ ID; đúng 5 giây còn giữ ID nhưng
5.001/30 giây tạo ID mới; một lần kết quả trống xóa track; chỉ một người mất
trong nhóm thì chỉ track đó bị xóa; người ở vị trí mới/reference dùng lại không
được ép nhận ID cũ; detection không hợp lệ không giữ track sống.

Tám test tích hợp kiểm tra: 10 frame với reference thay đổi; hai sinh viên giả
được gắn đúng P1/P2 khi đảo thứ tự ở nhịp 1/3.74 giây; log cho frame trống;
3.74/5 giây giữ ID nhưng quá 5 giây tạo mới; frame 8–9 dùng đúng số frame,
timestamp và bbox; người quay lại sau kết quả trống nhận ID mới; nhiều người
ở nhịp chậm chỉ loại track đã mất; panel thật vẽ Person Track từ result.
Worker, ghi ảnh và đồng hồ được giả lập; không nạp InsightFace/YOLO.

Test `test_hybrid_v7_real_output.py` dùng predictions cục bộ đã lưu từ lần chạy
trước và vị trí face tổng hợp, đạt 2/2 ảnh; không phải inference hoặc nhận dạng
camera mới. Toàn bộ test Association liên quan đã tìm thấy và có thể chạy không
cần camera đều được chạy; V4 được thực thi trong bộ tích hợp Camera Live.

Lần chạy trước sửa đã tái hiện lỗi `TypeError` với bbox scalar/boolean,
`OverflowError` với số quá lớn và nhận nhầm chuỗi/dictionary thành bbox.
Các lỗi này được sửa và PT-07 đã chạy lại đạt. Trong lúc xây test, fixture
giao nhau và trường `behaviors` cũng được điều chỉnh cho đúng contract thực tế;
không sửa thuật toán để ép giữ danh tính ở tình huống giao nhau.

## Limitations

- Chưa chạy camera sau bản sửa, inference model thật, che khuất thực tế hoặc
  lớp đông người trong task này. Test tổng hợp không thay thế camera.
- Camera Live có giới hạn 5 giây; xử lý lâu hơn vẫn đổi ID, đây là hành vi có
  chủ đích để không ghép một bbox quá cũ.
- Một lần mất detection làm mất ID trong Camera Live. Nếu A rời và B đứng vào
  vị trí gần như trùng bbox giữa hai lần AI, không có frame AI nào quan sát sự
  vắng mặt, IoU vẫn có thể nhầm B là A. Không thể bảo đảm danh tính trong tình
  huống này chỉ bằng bbox; bản sửa không tuyên bố giải quyết giới hạn đó.
- FaceTracker vẫn có `max_age=2.0`; cũng có nguy cơ hết hạn khi cập nhật chậm.
  Chưa sửa trong task PersonTracker.
- IoU chỉ dùng vị trí bbox; không có mô hình chuyển động/re-identification để
  bảo đảm giữ danh tính khi giao nhau, che khuất lâu hoặc bbox nhảy mạnh.
- Bản sửa bbox thuộc PersonTracker. Association V1–V4 có kiểm tra đầu vào riêng;
  chưa làm cứng toàn bộ pipeline trước phản hồi worker sai cấu trúc.
- Hồi quy Hybrid V7 kiểm tra các quy tắc hiện có, không đo độ chính xác model.
  Task này chưa xác nhận timer, cảnh báo vi phạm, API Backend hay CH-01–CH-19.

## Manual testing trên Windows/iPhone

Kết nối iPhone với iVCam, kiểm tra camera index 2 và đóng ứng dụng khác đang
chiếm camera. Lệnh dùng các môi trường và worker hiện có, không cài thư viện:

```powershell
Set-Location D:\Study\HTTM\HTTM-Chiv2
$ptLog = Join-Path $env:TEMP ('httm-person-tracking-{0}.log' -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
.\.venv\Scripts\python.exe -u .\ai_training\integration\camera_coordinator_live.py 2>&1 |
    Tee-Object -FilePath $ptLog
```

Lần 1: một người ngồi/đứng yên cho đủ `Completed AI results: 10`.
Lần 2: chạy lại, di chuyển nhẹ. Lần 3: rời rồi trở lại khung hình. Lần 4:
hai người đồng thời; sau đó giao nhau để quan sát giới hạn, không mặc định
ID phải được giữ khi giao nhau. Mỗi lần chạy chỉ xử lý tối đa 10 frame AI;
theo dõi số frame để thực hiện động tác trong thời gian đó. Nhấn Q để thoát sớm.
Ở lần 3, chờ ít nhất một log có `tracked_people: []` trước khi trở lại; kỳ vọng
ID mới. Ở lần 4, để một người rời trong khi người còn lại đứng yên; chỉ người
quay lại hoặc người thay thế tại vị trí cũ nhận ID mới.

Đọc log của lần chạy gần nhất:

```powershell
$ptFrames = @(Get-Content -LiteralPath $ptLog |
    Where-Object { $_.StartsWith('[PERSON TRACKING] ') } |
    ForEach-Object { $_.Substring('[PERSON TRACKING] '.Length) | ConvertFrom-Json })
'AI frames logged: {0}' -f $ptFrames.Count
$ptFrames | ForEach-Object {
    $ptFrame = $_
    foreach ($ptPerson in $ptFrame.tracked_people) {
        [pscustomobject]@{
            Frame = $ptFrame.frame_id
            Track = $ptPerson.track_id
            Reference = $ptPerson.person_ref
            Bbox = ($ptPerson.bbox -join ', ')
            UpdatedAt = $ptFrame.updated_at
        }
    }
} | Format-Table -AutoSize
for ($ptIndex = 1; $ptIndex -lt $ptFrames.Count; $ptIndex++) {
    'Frame {0}: update gap = {1:N3}s' -f $ptFrames[$ptIndex].frame_id,
        ($ptFrames[$ptIndex].updated_at - $ptFrames[$ptIndex - 1].updated_at)
}
```

Đối chiếu 10 dòng log và 10 kết quả AI; ghi lại lần ID thay đổi, bbox và khoảng
cập nhật tương ứng. Với một người đứng yên/di chuyển nhẹ, kỳ vọng một ID khi
IoU đạt ngưỡng, không có lần mất detection và khoảng cập nhật không quá 5 giây.
ID mới sau khoảng hơn 5 giây là giới hạn cấu hình hiện tại; mất detection cũng
chủ đích tạo ID mới khi quay lại. Kiểm tra panel
hiển thị cùng Person Track với log cho frame đang xem; panel là kết quả lịch sử.
Log nằm trên máy cục bộ; chỉ cần báo kết quả, không cần gửi ảnh hoặc dữ liệu mặt.

**Next task:** Dừng chờ người dùng xác nhận camera sau bản sửa. Không tự động
chuyển CH-20.
