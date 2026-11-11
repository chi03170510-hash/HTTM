# Checklist Camera Live HTTM

Checklist này dành cho kiểm thử thủ công khi có camera. Toolkit không mở camera hoặc chạy inference. Chưa test ngoài thực tế thì để NOT RUN.

## Quy ước chung

- Ghi case ID, session/file, Frame ID của giai đoạn cần đánh giá và người quan sát. Có thể dùng `frame_ids` trong manifest để chọn giai đoạn; không tự ghép hai session bằng Frame ID giống nhau.
- Ground truth do người quan sát xác nhận bằng hành động thực tế và bằng chứng gắn với frame/session. Không dùng detected, confidence, Face ID hoặc Person ID làm ground truth.
- Sleeping trong test là nhãn hành vi/tư thế theo quy ước đề tài; gục đầu hoặc nhắm mắt không chứng minh người đó đang ngủ sinh lý.
- Thống nhất trước thế nào là “sử dụng điện thoại”. P4 chỉ là negative nếu quy ước yêu cầu có tương tác/sử dụng, không chỉ cầm thiết bị.
- Không đặt thời lượng hay threshold mới. Để kiểm tra duration tăng cần ít nhất hai observation AI liên tiếp đủ điều kiện theo DurationTracker hiện tại. Giới hạn gap phải lấy từ log; không tự điền nếu thiếu.
- Camera hiện có 10 kết quả AI mỗi phiên. Không bảo đảm một phiên bao phủ mọi giai đoạn. Không cộng duration giữa các lần khởi động.
- Phase 1 không giữ SID khi mất face. Sleeping detected + identity không hợp lệ + duration null là hành vi bảo vệ hiện tại; không được coi riêng điều này là lỗi công thức duration.
- PASS case = các kiểm tra log bắt buộc PASS + người kiểm thử xác nhận ground truth PASS. FAIL phải nêu điều kiện log hoặc bằng chứng người quan sát. Thiếu evidence là INCONCLUSIVE; chưa chạy là NOT RUN.
- Phiên mới bật Shadow ghi cả Sleeping và Using Phone state/duration cùng blockers/reset reasons. Log cũ chỉ có Sleeping vẫn INCONCLUSIVE cho các điều kiện thiếu Using Phone; không hủy kết quả camera PASS đã được xác nhận trước đây. State detected trong log vẫn cần nhãn thủ công để kết luận detector đúng.

## Sleeping

### S1 — Gục đầu xuống bàn

- **Mục tiêu:** kiểm tra detector với tư thế mô phỏng và điều kiện duration khi face bị che.
- **Thao tác:** nhận diện khi ngồi thẳng; gục đầu; ngẩng đầu lại. Giữ cùng vị trí, không yêu cầu số giây mới.
- **Mong đợi:** ghi state thực tế; duration chỉ tăng khi SID/association/timestamp đủ điều kiện. Khi mất identity, duration không được tiếp tục gán cho SID cũ.
- **Bằng chứng:** session, các Frame ID ba giai đoạn, Sleeping state/confidence, PID/SID, association, continuity, duration/blockers và người xác nhận tư thế.
- **PASS/FAIL:** PASS phần detector cần người kiểm thử xác nhận nhãn đúng; PASS phần tăng duration cần cặp observation đủ điều kiện thực sự tăng. FAIL nếu có evidence gán nhầm SID hoặc duration trái điều kiện production.
- **INCONCLUSIVE:** mất SID nên không có cặp đủ điều kiện tăng duration; thiếu một giai đoạn, log hoặc ground truth. Không coi tư thế đơn lẻ là bằng chứng ngủ thật.

### S2 — Nhắm mắt, cúi đầu nhẹ

- **Mục tiêu:** đánh giá tình huống ít rõ hơn gục xuống bàn, có thể còn thấy mặt.
- **Thao tác:** ngồi thẳng được nhận diện; nhắm mắt/cúi nhẹ; trở lại tư thế ban đầu.
- **Mong đợi:** ghi đúng state và confidence mà model trả; không giả định face/eye landmarks là điều kiện detector bắt buộc.
- **Bằng chứng:** nhãn thủ công theo Frame ID, tình trạng mặt quan sát được, Sleeping state, SID/association và duration.
- **PASS/FAIL:** PASS khi nhãn đã thống nhất khớp kết quả và duration đáp ứng điều kiện; FAIL khi có bằng chứng sai theo nhãn đó hoặc gán nhầm người.
- **INCONCLUSIVE:** quy ước “Sleeping” chưa rõ, không xác nhận được nhắm mắt ở observation, hoặc không có cặp đủ điều kiện duration.

### S3 — Cúi đầu viết bài

- **Mục tiêu:** kiểm tra false positive Sleeping trong hoạt động viết.
- **Thao tác:** viết thật vào vở, có người quan sát xác nhận hoạt động và giai đoạn tương ứng.
- **Mong đợi:** không gán Sleeping cho các frame được gán nhãn viết theo quy ước. Duration không được tăng từ state không detected.
- **Bằng chứng:** Frame ID hoạt động viết, ground truth, Sleeping state/duration và SID.
- **PASS/FAIL:** PASS khi bằng chứng thủ công và log thỏa điều kiện; FAIL detector khi Sleeping detected được đối chiếu với nhãn viết đủ tin cậy, không chỉ suy ra từ confidence.
- **INCONCLUSIVE:** chỉ biết “cúi đầu” mà không xác nhận đang viết, thiếu state hoặc không đối chiếu được frame.

### S4 — Cúi đầu đọc sách

- **Mục tiêu:** kiểm tra false positive Sleeping khi đọc.
- **Thao tác:** đọc nội dung sách/vở thật; ghi giai đoạn hoạt động.
- **Mong đợi:** không gán Sleeping cho frame đã được người kiểm thử xác nhận là đọc theo quy ước.
- **Bằng chứng:** nhãn đọc, Frame ID, state, SID/association và duration.
- **PASS/FAIL:** đánh giá theo đối chiếu thủ công; FAIL nếu có false positive đã gán nhãn hoặc duration gán nhầm người.
- **INCONCLUSIVE:** không xác nhận được hoạt động đọc hoặc log không ghi state/duration.

### S5 — Ngồi bình thường

- **Mục tiêu:** lấy đối chứng không thực hiện Sleeping.
- **Thao tác:** ngồi bình thường, mặt rõ, không cố tạo các tư thế test khác.
- **Mong đợi:** không tạo Sleeping duration cho giai đoạn được xác nhận không Sleeping.
- **Bằng chứng:** Frame ID đối chứng, state/duration, SID/association và xác nhận của người quan sát.
- **PASS/FAIL:** PASS cần cả log và ground truth; FAIL nếu ghi nhận false positive hoặc duration vi phạm điều kiện.
- **INCONCLUSIVE:** người quan sát chưa xác nhận, mặt/người bị che hoặc thiếu log.

## Using Phone

### P1 — Sử dụng điện thoại liên tục

- **Mục tiêu:** kiểm tra detection và duration với hoạt động sử dụng thật.
- **Thao tác:** tương tác thật với điện thoại, giữ thiết bị nhìn thấy và thuộc vùng của người sử dụng.
- **Mong đợi:** state được đối chiếu với nhãn sử dụng; duration tăng trên cùng PID/SID khi đủ điều kiện hiện có.
- **Bằng chứng:** các Frame ID, nhãn sử dụng, phone evidence nếu đã ghi, state/duration, SID/association.
- **PASS/FAIL:** PASS cần evidence detector và cặp duration hợp lệ; FAIL nếu có mismatch ground truth đã xác nhận hoặc attribution sai.
- **INCONCLUSIVE:** JSONL thiếu Using Phone, không thấy thiết bị trong observation, thiếu SID hoặc ground truth. Có điện thoại không tự chứng minh đang sử dụng.

### P2 — Đặt điện thoại xuống

- **Mục tiêu:** kiểm tra kết thúc episode khi ngừng dùng điện thoại.
- **Thao tác:** sử dụng rồi đặt điện thoại xuống; xác định Frame ID sau khi ngừng tương tác.
- **Mong đợi:** episode cũ không kéo dài qua state không detected; model không nhầm thiết bị trên bàn với đang sử dụng theo quy ước.
- **Bằng chứng:** mốc trước/sau, nhãn thủ công, state và duration cùng PID/SID.
- **PASS/FAIL:** PASS khi transition/reset và nhãn phù hợp; FAIL nếu timer tiếp tục qua state không đủ điều kiện hoặc false positive được xác nhận.
- **INCONCLUSIVE:** thiếu log Using Phone hoặc không xác định được frame sau khi đặt xuống.

### P3 — Cầm sách hoặc vở

- **Mục tiêu:** kiểm tra nhầm sách/vở thành điện thoại hoặc hành vi sử dụng điện thoại.
- **Thao tác:** cầm và thao tác với sách/vở, không dùng điện thoại trong giai đoạn được đánh giá.
- **Mong đợi:** không gán Using Phone theo nhãn đối chứng.
- **Bằng chứng:** người quan sát xác nhận vật thể/hành động, Frame ID và state/duration; object evidence nếu có sẵn.
- **PASS/FAIL:** FAIL detector chỉ khi đối chiếu được false positive với nhãn; technical checks riêng không kết luận đúng vật thể.
- **INCONCLUSIVE:** không xác nhận được vật thể, thiếu Using Phone state hoặc frame mapping.

### P4 — Cầm điện thoại nhưng không sử dụng

- **Mục tiêu:** phân biệt hiện diện điện thoại với hành vi sử dụng theo quy ước đề tài.
- **Thao tác:** cầm điện thoại không tương tác; ghi rõ tư thế và định nghĩa “không sử dụng”.
- **Mong đợi:** không tự coi COCO thấy phone là bằng chứng sử dụng; kiểm tra kết quả model theo nhãn đã thống nhất.
- **Bằng chứng:** quy ước nhãn, Frame ID, xác nhận không tương tác, state/duration và object evidence nếu có.
- **PASS/FAIL:** PASS/FAIL do người kiểm thử đối chiếu quy ước; log detected một mình không xác nhận sử dụng thật.
- **INCONCLUSIVE:** định nghĩa chưa thống nhất hoặc thiếu Using Phone evidence. Logic hiện tại có thể chưa phân biệt được trường hợp này.

### P5 — Hai học sinh, chỉ một người dùng điện thoại

- **Mục tiêu:** kiểm tra gán đúng hành vi cho đúng người.
- **Thao tác:** hai người được nhận diện; chỉ một người sử dụng điện thoại, người kia làm đối chứng.
- **Mong đợi:** không gán chéo SID; chỉ người được gán nhãn sử dụng có episode tương ứng khi điều kiện hợp lệ.
- **Bằng chứng:** actor↔SID do người quan sát xác nhận, các PID/Frame ID, state/duration của cả hai và association.
- **PASS/FAIL:** FAIL khi có attribution sai đã xác minh. Nhiều track hoặc SID ổn định không đủ để tự PASS identity.
- **INCONCLUSIVE:** thiếu actor mapping, chỉ có một track, SID trùng/Unknown hoặc thiếu log Using Phone.

## Identity Continuity

### IC1 — Nhận diện khi ngồi thẳng

- **Mục tiêu:** ghi nhận anchor từ face hiện tại.
- **Thao tác:** người đã biết SID ngồi rõ mặt, chờ kết quả AI hoàn tất.
- **Mong đợi:** association trực tiếp hợp lệ và shadow face_confirmed; không dùng stable label cũ thay current recognition.
- **Bằng chứng:** Frame ID, người thật↔SID được xác nhận, association và continuity source/status.
- **PASS/FAIL:** PASS chỉ khi log có xác nhận trực tiếp và người kiểm thử xác nhận đúng người; FAIL khi ghi nhận SID sai có ground truth.
- **INCONCLUSIVE:** chỉ có shadow proposal hoặc thiếu face/association evidence.

### IC2 — Cúi đầu khiến mặt bị che

- **Mục tiêu:** kiểm tra xử lý mất mặt và bảo vệ duration trong Phase 1.
- **Thao tác:** sau IC1, che mặt bằng tư thế cúi đầu; không dùng Sleeping state để suy đoán identity.
- **Mong đợi:** ghi face absence thực tế khi có dữ liệu; T_hold chưa duyệt nên shadow không cấp SID cho production; duration null nếu identity không hợp lệ.
- **Bằng chứng:** Frame ID, face_present (false khác null), PID/SID, production duration và shadow status.
- **PASS/FAIL:** FAIL nếu shadow thay SID/duration production hoặc có attribution sai đã xác nhận. PASS cần người quan sát xác nhận cùng người và đúng việc che mặt.
- **INCONCLUSIVE:** face_present=null, chỉ có log association rỗng hoặc không biết actor nào trong observation.

### IC3 — Ngẩng đầu nhận diện lại

- **Mục tiêu:** kiểm tra xác nhận lại từ face mới.
- **Thao tác:** tiếp sau IC2, ngẩng đầu để mặt nhìn thấy trở lại.
- **Mong đợi:** face_confirmed trên cùng track hoặc face_recovered khi track đổi; không nối duration cũ qua đoạn identity không hợp lệ.
- **Bằng chứng:** chuỗi Frame ID mất mặt→có mặt, SID/association, continuity source và duration trước/sau.
- **PASS/FAIL:** FAIL nếu khôi phục SID không có face mới hoặc kế thừa duration sai; PASS cần manual identity review.
- **INCONCLUSIVE:** track đổi và log không đủ để liên kết các giai đoạn; thiếu pha ngẩng đầu. Analyzer không tự ghép hai actor chỉ vì cùng SID được báo.

### IC4 — Hai người ngồi gần nhau

- **Mục tiêu:** kiểm tra xung đột identity và tính duy nhất SID.
- **Thao tác:** hai người ngồi gần nhau, lần lượt rõ/che mặt trong điều kiện quan sát được.
- **Mong đợi:** không cấp cùng SID cho hai production tracks; ambiguous được giữ rõ, không bị cache che.
- **Bằng chứng:** manual actor mapping, cả hai PID/SID, association, continuity và production attribution.
- **PASS/FAIL:** log có duplicate production SID là FAIL kỹ thuật; identity đúng cần manual review.
- **INCONCLUSIVE:** chỉ có raw duplicate association claims, thiếu production attribution hoặc không phân biệt được actor.

### IC5 — Hai người đổi vị trí

- **Mục tiêu:** phát hiện hoán đổi hoặc nối nhầm identity.
- **Thao tác:** hai người đổi vị trí, có người quan sát ghi actor trước/sau và các giai đoạn giao nhau.
- **Mong đợi:** trạng thái mơ hồ không bị giả thành face-confirmed; không tự chuyển SID giữa track khi thiếu face mới.
- **Bằng chứng:** actor labels độc lập, Frame ID, các track, SID/source và association.
- **PASS/FAIL:** FAIL khi gán nhầm actor đã được đối chiếu. Không tự PASS từ tập track IDs không đổi.
- **INCONCLUSIVE:** crossing diễn ra giữa các observation, thiếu face hoặc nhãn actor; IoU-only không bảo đảm identity.

### IC6 — Người rời, người khác vào

- **Mục tiêu:** kiểm tra nguy cơ người mới kế thừa identity cũ.
- **Thao tác:** một người rời khung hình, người khác vào vùng cũ; ghi rõ diễn biến bằng quan sát độc lập.
- **Mong đợi:** người mới không tự nhận SID/duration cũ khi thiếu face xác nhận; sự thay đổi track chỉ là evidence kỹ thuật.
- **Bằng chứng:** Frame ID trước/rời/vào, actor labels, track/SID, source và duration.
- **PASS/FAIL:** FAIL khi có inheritance sai đã xác minh; PASS cần evidence người thay thế và dữ liệu production đầy đủ.
- **INCONCLUSIVE:** thay người hoàn toàn giữa hai observation hoặc không có frame/evidence phân biệt; không tuyên bố cùng người chỉ từ bbox giống nhau.
