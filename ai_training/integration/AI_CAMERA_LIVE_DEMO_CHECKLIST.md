# AI2 — Demo readiness / camera acceptance checklist

**Tất cả camera cases dưới đây là NOT RUN khi bàn giao AI2.** Preflight, unit tests
hoặc synthetic preview không được chuyển case camera thành PASS. WARNING disabled,
T_hold=None; không điều chỉnh model/threshold/tracking/association/duration trong demo.

## Ghi trước phiên

- Case ID; ngày giờ địa phương; người kiểm thử/reviewer; mục tiêu của phiên.
- Actor A/B và roster SID độc lập với AI prediction; người chưa đăng ký ghi rõ.
- Link `preflight.json`, model/hash, hai Python paths, env và console log.
- Camera device/backend/index thực, nguồn sáng/vị trí và phiên log chính xác.
- Expected action/behavior do người quan sát xác nhận; không suy từ confidence,
  `detected`, predicted SID hay PID ổn định.

Kết quả:

- **PASS**: đủ bằng chứng đúng frame/actor và mọi điều kiện mong đợi của case đạt.
- **FAIL**: có bằng chứng đúng frame cho kết quả trái điều kiện mong đợi, ghi lỗi cụ thể.
- **INCONCLUSIVE**: đã chạy nhưng thiếu ground truth/log/frame linkage, logger hết
  ngân sách, không quan sát được điều kiện mục tiêu hoặc dữ liệu mơ hồ.
- **NOT RUN**: chưa chạy case camera thật.

## Các case tối thiểu

| ID / trạng thái ban đầu | Mục tiêu và thao tác | Kết quả mong đợi | Bằng chứng / PASS–FAIL–INCONCLUSIVE |
|---|---|---|---|
| A1 — NOT RUN | Một actor đã enrollment ngồi thẳng, di chuyển nhẹ trong khung | SID accepted khớp roster; PID/association và hành vi thuộc đúng actor | Frame/PID/SID/status + người quan sát. Sai SID → FAIL; không đủ nhận diện/actor truth → INCONCLUSIVE |
| A2 — NOT RUN | Hai actor có roster khác nhau cùng trong ảnh, cử động lần lượt | Không gán chéo SID/hành vi; duration độc lập | Ghi từng actor trên cùng AI frame. Đúng PID/SID trên nhiều frame không tự chứng minh actor; cần quan sát độc lập |
| P1 — NOT RUN | Một người thực sự dùng điện thoại; người thứ hai không dùng | Using Phone detected thuộc đúng actor khi evidence hỗ trợ; COCO phone object khác vùng candidate behavior | Ghi thao tác thật + decision/evidence/state/bbox. Không dùng detected làm ground truth; sai actor → FAIL; thiếu evidence → INCONCLUSIVE |
| P2 — NOT RUN | Đặt điện thoại xuống, cầm sách/vở hoặc ngồi bình thường | Không trình bày hành vi không xảy ra là using_phone đã xác nhận; duration ngắt theo production state | Ghi hành động và observation trước/sau. Không có positive/negative đủ rõ → INCONCLUSIVE; đối chiếu checklist P1–P5 cũ |
| S1 — NOT RUN | Nhận diện khi ngồi thẳng → gục đầu → ngẩng đầu lại | Sleeping state được đánh giá theo nhãn thao tác; SID thiếu/mơ hồ thì duration `--` đúng contract; không cấp Shadow SID | Người quan sát và Shadow Sleeping rows. Detector đúng/sai tách khỏi duration policy. Không nhìn panel được thì xem log, không tự nhớ/ước đoán frame |
| S2 — NOT RUN | Cúi đầu viết/đọc sách và ngồi bình thường làm đối chứng | Kiểm tra false positive Sleeping bằng nhãn thao tác độc lập | Chỉ kết luận trong các frame có nhãn; confidence cao không đủ PASS. Dùng S1–S5 cũ khi đánh giá đầy đủ |
| I1 — NOT RUN | Người chưa enrollment hoặc face recognition Unknown | Student Unknown, duration `--` nếu thiếu accepted SID; không suy SID từ FT/PID | Production row và panel cùng frame. Không có Unknown trong phiên → INCONCLUSIVE cho case này |
| I2 — NOT RUN | Hai người gần nhau/overlap/đổi chỗ, quan sát association mơ hồ nếu xuất hiện | Không biến ambiguous thành SID accepted; Panel Student Unknown + Identity ambiguous, BBox ghi Ambiguous khi tương ứng | Không cố tạo duplicate bằng sửa enrollment/threshold. Nếu không xảy ra ambiguity → INCONCLUSIVE; các test offline vẫn chỉ là evidence phần mềm |
| D1 — NOT RUN | Giữ behavior và identity hợp lệ qua ít nhất hai AI observations mới | Duration đầu từ 0 rồi tăng theo capture timestamps; repaint cùng frame không tăng; state/identity loss reset/ngắt đúng | Ghi Frame ID/capture/state/SID/duration. Gap phải đối chiếu max_gap hiện có 5s; không đặt công thức/threshold mới. Chỉ một frame đủ điều kiện → INCONCLUSIVE |
| B1 — NOT RUN | Bật Snapshot Overlay, lần lượt một/hai người và behavior/object | Person/Face/behavior bbox đúng JPEG observation; PID/SID/FT/score đúng nguồn; candidate không bị gọi là accepted; live preview không có historical boxes | Đối chiếu Frame ID và source size; không ghép theo list order. Không có loại detection cần kiểm → phần đó INCONCLUSIVE |
| F1 — NOT RUN | Quan sát trong lúc inference chậm/cũ và khi có result mới | Frame ID tăng theo observation; capture/completion age riêng; historical/STALE rõ; geometry cache không nhảy sang JPEG khác | Không sửa clock hoặc treo worker để ép demo. Không xuất hiện stale tự nhiên → nhánh stale INCONCLUSIVE; offline tests không thay camera evidence |
| Q1 — NOT RUN | Nhấn Q khi đang chạy, kiểm tra cửa sổ/console và khả năng mở lại camera | Camera/worker cleanup, log kết thúc hoặc lỗi rõ, không còn tiến trình worker của phiên; phiên mới mở lại được | Ghi thời điểm Q/exit, console, session/process evidence nếu có. Không tự kill process khác để tạo PASS |
| L1 — NOT RUN | Chạy continuous vượt 20 AI observations và đến thời lượng mục tiêu do tester ghi trước | Không tự dừng ở 10; không tăng queue/history snapshot vô hạn; UI/stop hoạt động; không WARNING | Theo dõi process memory/latency trong phiên thực, không suy từ synthetic benchmark; chưa đo đủ thời lượng/memory → INCONCLUSIVE cho ổn định dài hạn |

Duration/identity safety được chấm riêng với detector accuracy. Sleeping detected
và duration `--` khi SID không hợp lệ có thể **đúng policy**, không tự là bug.

## Đối chiếu kết quả sau mỗi phiên

1. Gắn đúng console/preflight/session vào case; không ghép theo timestamp gần đúng.
2. Kiểm tra logger còn ngân sách: continuous Shadow chỉ ghi tối đa 10 rows mặc định.
3. Chạy analyzer cho Sleeping/Using Phone; dữ liệu thiếu giữ INCONCLUSIVE.
4. Actor truth nhập tay độc lập; dùng `camera_identity_truth.py` khi cần xác minh
   đổi người/đổi PID/duplicate. BBox và dự đoán SID không phải nhãn actor.
5. Ghi verdict, reviewer, evidence paths và phạm vi đã quan sát. Không báo mAP/
   precision/recall hay phần trăm accuracy khi chưa có bộ nhãn phù hợp.

Template ghi kết quả (copy cho từng case):

```text
Case:
Status: NOT RUN
Reviewer / time:
Expected action / actor roster (independent):
Preflight / console / session:
Frame IDs / timestamps / PID / production SID / association:
Behavior state / duration / evidence:
Observed result:
PASS/FAIL criteria checked:
Missing evidence / reason INCONCLUSIVE:
```

## Điều kiện bàn giao demo

- Không còn preflight FAIL; WARN được ghi nhận tác động, không tự thay môi trường.
- Có kết quả camera thật cho phần muốn trình bày là PASS.
- Chưa chạy/thiếu ground truth phải nói rõ NOT RUN/INCONCLUSIVE.
- WARNING tiếp tục disabled, T_hold=None và Shadow không cấp identity production.
- Không triển khai AI3/AI4 để bù dữ liệu còn thiếu trong checklist này.
