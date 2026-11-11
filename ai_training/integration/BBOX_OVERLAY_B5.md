# BBox B5 — Visualization hardening, offline finalization

Phạm vi: chỉ renderer/label layout/cache của cửa sổ AI Snapshot opt-in. Không đổi
Live Preview, AI Panel hoặc dữ liệu production. Mặc định OFF qua biến hiện có
`HTTM_PERSON_SNAPSHOT_OVERLAY`. Không cần biến cấu hình mới.

## Audit và lỗi đã có bằng chứng

Những phần B1–B4 đã đúng được giữ lại: kiểm tra Frame ID/capture timestamp/JPEG
SHA256/source dimensions/pixels; bbox chỉ nằm trên historical snapshot; scale và
letterbox cùng transform; nguồn score tách biệt; SID/state/duration từ production
view; cache geometry và nhãn capture age/completion age; cô lập lỗi GUI.

| Lỗi tái hiện | Bằng chứng | Xử lý B5 |
|---|---|---|
| Person label đè nhau khi không còn vertical slot | Cảnh 20 bbox trùng nhau có 28 cặp plate overlap | Tìm slot theo hai trục; không có chỗ thì bỏ label và báo trong legend |
| SID dài bị cắt không báo | `_fit_text()` bỏ đuôi SID trên plate/legend | Đặt SID trước PID ở legend; `[SID clipped]`, `[PID clipped]`, `...` hoặc `!` tùy diện tích |
| Legend nhiều Face/metadata che mất thông tin người/hành vi | 80 detail boxes chỉ hiển thị 10 dòng đầu, không có bảng identity ưu tiên | Bảng SID/PID + state trước, duration sau; confidence/FT ở detail; báo số ID/detail/tag bị ẩn |
| Detail strokes che SID và tag chồng | Detail renderer vẽ sau Person plates tại cùng tọa độ | Bảo vệ vùng plate; tag không overlap, chỉ đặt trong bbox của chính nó; thiếu chỗ thì ẩn có số đếm |
| Nét bbox dày tràn letterbox | Ca far-edge có 400 pixel detail stroke ngoài image ROI | Vẽ stroke vào ROI ảnh; không sửa source bbox/projection contract |
| Clock lỗi làm cùng observation không phục hồi | `now < captured_at` một lần, lần sau `now` hợp lệ vẫn placeholder | Revalidate timestamp mỗi lần; chỉ memoize lỗi renderer |
| Rủi ro cache dựa vào `id(snapshot)` có thể tái sử dụng | Cache cũ không giữ source object | Giữ source snapshot và đưa metadata object vào cache key; validate output detail renderer |
| Metadata/view không nhất quán có thể còn SID cũ | Test thay production view thành duplicate SID nhưng giữ metadata cũ | Render guard bỏ detail có owner/SID/state mâu thuẫn với current production view |

Identity mismatch guard chỉ quyết định **có vẽ hay không**, không cấp/hủy/chuyển
SID production. Không dùng Shadow SID. Unknown/Ambiguous vẫn có chữ rõ ràng.

## Tọa độ và layout

- Contract B1/B3 giữ nguyên: source pixel xyxy; raster hóa bao endpoint rồi đổi
  sang half-open chỉ ở rendering. Clip stroke thickness bằng ROI; không sửa
  `clip_box`, detector, tracker hoặc association coordinates.
- NaN/Inf, bbox zero-area/reversed/outside bị bỏ qua. Aspect ratio và letterbox
  không đổi khi nguồn đổi độ phân giải.
- Person plates được vẽ sau các Person outlines/connectors. Detail renderer
  khôi phục vùng plate trước khi đặt tag. Không có hai plate/tag chiếm cùng ô.
- Tag chỉ nằm trong bbox của nó; không dời sang người khác để cố hiển thị đủ.
- Legend 540px giữ kích thước B3+B4. Với viewport 640×480, 20 identity rows vẫn
  hiển thị đủ; detail phụ bị lược có số đếm. View nhỏ hơn hoặc SID quá dài có
  dấu cắt/count rõ. Không tự thu nhỏ font vô hạn hoặc thay Live Preview layout.
- Header nhỏ rút gọn `C` = capture age, `Done` = completion age. STALE giữ chữ
  riêng trên viewport rất hẹp. Không suy ra clock offset hoặc thời gian bị thiếu.
- Cache key gồm Frame ID, capture timestamp, snapshot object, render metadata
  object và viewport. DTO frozen/owned pixels giữ contract B1–B4; không hỗ trợ
  mutation cưỡng ép bằng `object.__setattr__` hoặc bỏ bảo vệ ndarray.

## File thay đổi

Sửa:

- `behavior_overlay.py`: display plates, draw order, ROI stroke clipping.
- `person_snapshot_overlay.py`: cache/input-error recovery, output validation,
  header fitting, giữ source snapshot.
- `snapshot_detail_overlay.py`: production display guard, identity-first legend,
  ROI strokes, collision-free box-local tags và hidden counts.
- `test_person_snapshot_overlay.py`, `test_bbox_detail_overlay.py`: assert tọa
  độ display sau đổi drawing target sang ROI, giữ expected geometry cũ.

Tạo:

- `snapshot_label_layout.py`: helper placement/text fitting chỉ visualization.
- `test_snapshot_overlay_hardening.py`: 23 test mới.
- `benchmark_snapshot_overlay.py`: benchmark và preview synthetic không model.
- Tài liệu này và validation artifacts trong thư mục mới bên dưới.

`camera_coordinator_live.py`, `bbox_visual_metadata.py`, các worker/model/config,
FaceTracker, PersonTracker, Association V4, DurationTracker, WarningObserver và
Identity Continuity giữ nguyên byte theo audit. T_hold=None; WARNING disabled.

## Kiểm thử thực chạy

**631/631 regression offline PASS; 0 failures/errors/skips.** Bao gồm 23 test B5,
35 B1/B2, 40 B3/B4, 6 visual worker tests và các module integration/Hybrid liên
quan. Ma trận Overlay OFF/ON × Shadow OFF/ON có metadata thật từ worker fixture
đi qua ai_loop/main thật bằng mock: input DurationTracker/WarningObserver, output
production, AI Panel fields/geometry và snapshot pixels không đổi.

Các script manual camera/inference không có unittest không được chạy như camera
test. VideoCapture/model constructors được guard; không webcam, inference hoặc
training. Không cài dependency. Không commit/push.

Audit 966 file cũ: chỉ 5 file dự kiến trên bị sửa, không mất file; các file cũ
khác và report cũ nguyên vẹn. Git status trước/sau giống nhau vì integration đã
untracked từ trước; đây không phải tuyên bố working tree sạch. Có backup ban đầu,
backup sau lần người dùng xác nhận tiếp tục trên source đã hoàn nguyên và diff
theo file để rollback có chọn lọc.

## Benchmark cuối

Windows 10 build 19045, Intel64 Family 6 Model 142, Python 3.11.9 (.venv), OpenCV
4.11.0 (8 threads), NumPy 2.4.6. Nguồn 1280×720, viewport ảnh 640×480, mỗi người
4 detail boxes. Mỗi phép đo: 10 warmup + 100 mẫu, perf_counter_ns, p95 theo NumPy.

| Người | Cold render median / p95 (ms) | Cached repaint median / p95 (ms) |
|---:|---:|---:|
| 1 | 16.75 / 23.86 | 0.93 / 1.69 |
| 5 | 20.37 / 35.43 | 0.90 / 1.20 |
| 10 | 14.73 / 16.30 | 0.73 / 1.50 |
| 20 | 17.52 / 20.75 | 0.79 / 1.18 |

Cold render gồm Person/detail layout, metadata pixel hash validation, vẽ và copy
header; cached repaint chỉ copy kết quả cache và cập nhật header tuổi observation.
Không tính snapshot construction, inference, worker sidecar/JPEG IO hoặc GUI
imshow. **Không quy đổi thành FPS end-to-end.** Thời gian dao động theo tải máy;
không dùng số này để suy ra thứ hạng tốc độ theo số người hoặc camera performance.

Profile 30 cold renders/20 người tìm thấy `place_label` tốn ~0.346s trong tổng
0.897s profiled. Tối ưu hẹp: thử slot trực tiếp trước, bỏ các occupied regions
hoàn toàn ngoài vùng đặt tag. Không thay lịch inference hoặc số AI observations.
Test chứng minh plate ngoài vùng không làm thay đổi local placement. Chưa có
bằng chứng cần đổi các phần pipeline khác; không tối ưu thêm.

## Chạy lại và demo khi có camera

```powershell
Set-Location 'D:\Study\HTTM\HTTM-Chiv2'
$env:PYTHONPATH = 'ai_training/integration;ai_training'
& .\.venv-hybrid\Scripts\python.exe -B -m unittest test_snapshot_overlay_hardening test_person_snapshot_overlay test_bbox_detail_overlay test_bbox_visual_workers -v

# Output phải là thư mục MỚI; CLI từ chối ghi đè thư mục có sẵn.
& .\.venv\Scripts\python.exe -B .\ai_training\integration\benchmark_snapshot_overlay.py --output .\ai_training\runs\bbox_b5_benchmark_new --iterations 100 --warmup 10

# Chỉ chạy khi người dùng có camera; không được chạy trong lượt B5 này.
$env:HTTM_PERSON_SNAPSHOT_OVERLAY = '1'
& .\.venv\Scripts\python.exe .\ai_training\integration\camera_coordinator_live.py
```

Tắt toàn bộ optional overlay ở lần chạy sau:

```powershell
Remove-Item Env:HTTM_PERSON_SNAPSHOT_OVERLAY -ErrorAction SilentlyContinue
```

Để rollback source B5 về B4, đối chiếu 5 patch và copies `resume_before/`, chỉ
hoàn tác phần B5; bỏ các file mới nếu không còn được import/sử dụng. Không dùng
git reset/restore toàn bộ integration vốn có thay đổi người dùng chưa commit.

## Artifacts và giới hạn nghiệm thu

Thư mục:
`ai_training/runs/bbox_b5_validation_8517815f6ada44a9ac818f550c6bc252/`.

- `optimized/benchmark.json`: số cuối cho source được bàn giao.
- `optimized/synthetic_{1,2,5,10,20}_people.png`, `synthetic_overlap_20.png`,
  `synthetic_long_sid_small.png`, `synthetic_portrait.png`.
- `full_offline_regression.json`, log từng module, `final_verification.json`,
  audit reproduction, profile, backups và patches. Các thư mục before/after/final
  là số đo trung gian đã giữ lại; không dùng làm kết quả source cuối.

Mọi preview là geometry/SID/state/score tự khai báo, không phải model accuracy
hay camera end-to-end PASS. Camera thật còn cần xác minh bbox alignment, motion/
occlusion, identity/behavior accuracy, độ đọc nhãn trên màn hình thực, thời gian
stale và GUI overhead. 20 bbox trùng nhau có thể không hiện đủ tag/chi tiết;
hidden counts không thay thế bằng chứng nhận diện actor. Dừng phát triển BBox
offline sau B5; chỉ xử lý tiếp nếu có lỗi mới có bằng chứng.
