# best.pt chính thức — YOLO11n Round 3

Cập nhật **27/10/2026**. Người dùng đã quyết định chọn **YOLO11n Round 3** làm checkpoint chính thức của HTTM. Việc thay file đã thành công, không training, inference hoặc sửa dataset.

| Class ID | Nhãn | Ý nghĩa |
|---|---|---|
| 0 | `sleeping` | Học sinh ngủ/gục theo quy ước nhãn dataset |
| 1 | `using_phone` | Học sinh sử dụng điện thoại theo quy ước nhãn dataset |

**Chính thức là quyết định chọn artifact của dự án, không phải chứng nhận đủ chính xác cho mọi lớp học.** Round 3 có bằng chứng cân bằng hai class tốt nhất trong các lượt đánh giá chung đã thực hiện; model vẫn bỏ sót nhiều sleeping và có nhiều phone FP.

## 1. Định danh và sao lưu

Các đường dẫn tính từ project root `E:\TTT\Project\HTTM`.

| Vai trò | Đường dẫn |
|---|---|
| Chính thức | `ai/behavior_detection/models/best.pt` |
| Nguồn bất biến | `ai/behavior_detection/experiments/yolo_resolution_architecture/references/round3_original_best.pt` |
| Backup V2 | `ai/behavior_detection/models/backups/best_v2_before_round3_20261027_125235_4dea81.pt` |

SHA256 của **best.pt chính thức và nguồn Round 3**:

```text
6821a02bbe1dde7deea8a7936766d93a99bc07b6b77db804ade563d497492965
```

SHA256 của **V2 cũ và backup**:

```text
6b967848c718e700e05c28b7ebd095bbeb2b8a1da4a0beab4c3548f18de15dc2
```

Đã kiểm tra nguồn, đích, backup và file tạm trước khi thay nguyên tử, rồi kiểm tra lại hash sau thay. Không ghi đè backup cũ hoặc checkpoint nguồn. [Biên bản thay checkpoint](../reports/official_round3_install_20261027_125235_4dea81/MODEL_REPLACEMENT_REPORT.md).

Các báo cáo cũ có `DO_NOT_PROMOTE_RETAIN_ROUND3` là quyết định tại thời điểm thí nghiệm. Người dùng đã cho phép đặt Round 3 thành `models/best.pt` sau đó; không sửa lại lịch sử báo cáo.

## 2. Recipe Round 3 đã khôi phục

- YOLO11n khởi tạo từ COCO pretrained, không phải V2 cũ.
- Train **2.002 ảnh**, 3.861 sleeping box, 5.237 phone box, 12 ảnh nhãn rỗng. Val **390 ảnh**, 164 sleeping GT và 279 phone GT.
- `imgsz=640`, `batch=16`, `nbs=64`, `seed=42`, AMP bật, không freeze; tối đa 100 epoch, `patience=25`.
- Chạy thực tế **37 epoch**, best CSV **epoch 12**; dừng sau 25 epoch không cải thiện.
- Optimizer yêu cầu `auto`; thực tế **AdamW**, LR đầu **0.001667**, bias warmup bắt đầu **0**, beta1 **0.9**; linear schedule, `lrf=0.01`, warmup 3 epoch.
- Augmentation: mosaic 1.0, scale 0.5, translate 0.1, HSV 0.015/0.7/0.4, fliplr 0.5, rotation 0. Loss box/cls/DFL: 7.5/0.5/1.5.

Giá trị yêu cầu `lr0=0.01`, `warmup_bias_lr=0.1` không phải LR thực tế sau override của auto optimizer. A đã kiểm chứng recipe và LR. [Recipe gốc](../experiments/round3_training_recipe_audit/ROUND3_RECIPE.md).

## 3. Kết quả tham chiếu

Cùng frozen Val **390 ảnh/GT**: AP ở conf 0.001, FP32, imgsz 640, rect=False, batch 16, NMS IoU 0.70, max_det 300, không TTA. TP/FP/FN/P/R/F1 lấy từ predict riêng **conf 0.25**, batch 1; matching cùng class IoU ≥ 0.50, greedy giảm confidence; tie IoU chọn dòng GT lớn hơn theo evaluator lưu trữ.

| Class | AP50 | AP50–95 | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| sleeping | 46,96% | 16,85% | 43 | 20 | 121 | 68,25% | 26,22% | 37,89% |
| using_phone | 50,92% | 26,42% | 157 | 237 | 122 | 39,85% | 56,27% | 46,66% |

Macro AP50 **48,94%**, macro AP50–95 **21,64%**, macro F1 **42,27%**. Không gọi các số này là accuracy. FP theo matching không tự chứng minh mọi trường hợp là nhận nhầm đồ vật.

T4 common comparison ghi mean latency khoảng **11,15 ms/ảnh**: predecoded BGR, batch 1 FP32, gồm preprocessing/inference/NMS, không gồm đọc ảnh hoặc load model. Không phải tốc độ toàn hệ thống camera. [Bằng chứng](../experiments/historical_checkpoint_comparison/received_20261027_124752_e6eb2e/ARTIFACT_REVIEW_REPORT.md).

## 4. Các lần thử chính

Bảng dưới dùng **common 640, fixed conf 0.25**. A/B replay khác với A/B kiến trúc 960 ở phần sau.

| Lần thử | Sleeping TP/FP/FN | Phone TP/FP/FN | Kết luận |
|---|---|---|---|
| Round 3 | 43/20/121 | 157/237/122 | Checkpoint chính thức |
| C1: fine-tune Round 3 trên Round 6, augmentation bảo thủ | 15/2/149 | 56/13/223 | Loại: FP giảm nhưng recall sụt mạnh |
| C2: chỉ giảm warmup_bias_lr 0.1 → 0.0001 so với C1 | 16/2/148 | 95/64/184 | Loại: phone hồi phục một phần, sleeping vẫn yếu |
| A: khởi tạo COCO, replay recipe Round 3 | 43/20/121 | 157/237/122 | Tái lập thành công, không cải thiện |
| B: recipe A, chỉ đổi sáu thao tác nhãn Round 6 đã duyệt | 16/16/148 | 130/209/149 | Loại: sleeping AP/recall giảm, phone cũng mất TP |
| V1.4 phục hồi từ Git | 11/6/153 | 94/30/185 | Không thay Round 3: sleeping recall 6,71% |
| V2 cũ, trước đây là models/best.pt | 7/2/157 | 97/22/182 | Phone AP/precision tốt hơn nhưng sleeping recall 4,27% |

**C1:** 13 epoch, best CSV 3, khoảng 9,57 phút T4. **C2:** 12 epoch, best CSV 2, khoảng 8,79 phút. Cả hai khởi tạo từ Round 3, AdamW LR 0.0001, bỏ mosaic và scale/translate. C1 đồng thời đổi recipe và revision nhãn so với training Round 3 nên không cô lập một biến. C2 cô lập warmup trong cấu hình so với C1, nhưng không khôi phục recall về baseline. [C1 review](../experiments/round3_quality_finetune/received_20261027_030236/ARTIFACT_REVIEW_REPORT.md) · [C2 review](../experiments/round3_quality_finetune/received_candidate2_20261027_034519/CANDIDATE2_ARTIFACT_REVIEW.md).

**A:** 37 epoch, best CSV 12, khoảng 12,43 phút. Prediction, metric và lịch sử số từng epoch tái lập Round 3; 503 serialized tensor-storage records giống byte dù hash toàn file khác do archive/metadata. **Đây là control thành công, không phải thất bại kỹ thuật.** [A review](../experiments/round3_recipe_replay_A/received_20261027_094216_ce0317/ARTIFACT_REVIEW_REPORT.md).

**B:** 37 epoch, best CSV 12, khoảng 12,13 phút. Sleeping AP50 18,16%, phone AP50 42,01%; mỗi class mất ròng 27 TP. Sleeping mất 41 TP cũ, có 14 TP mới; phone mất 32 TP cũ, có 5 TP mới. Không suy ra sáu sửa nhãn là sai hoặc tự đảo phê duyệt. [B review](../experiments/round3_recipe_replay_B/received_20261027_113237_86bb32/ARTIFACT_REVIEW_REPORT.md).

**Model cũ 70–79%:** V1.4 phục hồi nguyên byte từ Git; metadata lưu mAP50 **79,716% trên Val cũ**, README cũ ghi sleeping AP50 **75,2%**. V2 lưu mAP50 **76,104% trên dataset V2 cũ**. Không tương đương frozen Val hiện tại. Khi chạy chung, V2 phone AP50 **66,47%** tốt hơn Round 3 **50,92%**, nhưng phone và sleeping recall giảm. V2 macro AP50–95 cao hơn cũng không đủ để chọn chỉ theo aggregate. [Truy nguồn điểm cao](../experiments/historical_checkpoint_recovery/HIGH_SCORE_70_75_AUDIT.md).

## 5. Những nhánh khác đã thực hiện

- **Baseline nội bộ, Round 2, development, V2 Round 2:** đã nhận và audit artifact. Một số run dùng Val 21/119 ảnh hoặc revision khác; không trộn với frozen 390. Có sleeping yếu/FP nền nhưng chưa chứng minh cùng một nguyên nhân cho mọi run. [Inventory đầy đủ](../experiments/historical_checkpoint_recovery/HISTORICAL_TRAINING_INVENTORY.csv).
- **Round 4:** sleeping TP 43 → 9, phone TP 157 → 111 trên cùng 390 Val. Sửa nhãn được duyệt sau đó chưa có trong revision này; nguyên nhân học cụ thể chưa cô lập. [Audit](../reports/yolo11n_round4_error_audit_20261025/AUDIT_REPORT.md).
- **Round 6 lịch sử:** sáu thao tác trong năm file Train, ảnh/Val giữ nguyên; sleeping TP 43 → 16. B kiểm tra lại dưới recipe A hoạt động đúng. Phê duyệt nhãn vẫn giữ. [Diagnostic](../reports/round3_vs_round6_diagnostic_20261026/ROUND3_VS_ROUND6_DIAGNOSTIC_REPORT.md).
- **YOLO11n 960 / YOLO11s 960:** sleeping TP 20 mỗi model, phone TP 165/210. Phone tốt hơn ở nhánh lớn nhưng sleeping giảm, không chọn. Đây là kết quả 960, không giả làm protocol 640. Rectangular validation ảnh hưởng AP rõ, đã kiểm tra riêng trên T4. [Investigation](../experiments/yolo_resolution_architecture/results/t4_investigation_20261026/T4_INVESTIGATION_REPORT.md).
- **Candidate 3 hard examples:** đã làm giao diện duyệt và nhận 89 quyết định: 62 negative toàn ảnh, 21 giữ nhãn, 5 cần sửa, 1 không chắc. Trong 62 negative có 8 đã ở Train; **54 ảnh mới bị chặn** vì session/holdout coverage chưa xác minh. Không phải chưa duyệt thị giác. Không nhập ảnh hoặc train C3. [Source recovery](../experiments/round3_quality_finetune/candidate3_data_preparation/provenance_recovery/SOURCE_RECOVERY_REPORT.md).
- **Person-aware filter:** fail-open giữ 457 prediction, loại 0 FP; strict loại 33 FP nhưng mất 13/43 sleeping TP. Không bật vì chưa có lợi ích bảo toàn recall được chứng minh. [Decision](../experiments/round3_person_filter/FILTER_DECISION.md).
- **Confidence theo class:** sweep cache tìm phương án thăm dò sleep 0.07 / phone 0.50; thêm 84 sleeping FP, mất 31 phone TP. Không có cặp khác baseline giữ cả hai số TP mà không tăng tổng FP. Không áp dụng production. [Recommendation](../experiments/round3_confidence_analysis/RECOMMENDATION.md).
- **Video tự quay:** đã duyệt GT và đánh giá 27 frame chọn lọc: 14 sleeping/13 phone GT, cả hai TP=0 tại conf 0.25 theo box/IoU đã duyệt. Video cận một người/góc trên xuống/scope box khác; không đại diện toàn cảnh hoặc accuracy toàn video. Orientation nguồn giữ nguyên. [GT report](../experiments/round3_self_recorded_video/runs/self_20261026_234155_2bfe5a/evaluations/selected_gt_20261027_003722_181bb6/GT_EVALUATION_REPORT.md).
- **Video lớp học độc lập:** dừng vì chưa có video toàn cảnh đủ bằng chứng nguồn, quyền dùng và độc lập. Không tạo video/kết quả giả. [Blocked comparison](../experiments/round3_video_threshold_comparison/COMPARISON_REPORT.md).

## 6. Nguyên nhân: điều đã biết và điều chưa xác minh

### Có bằng chứng quan sát

- **Confidence giảm góp phần làm mất TP C1/C2.** Trong GT từng là TP Round 3, C1 có 25/41 sleeping và 59/103 phone GT có box cùng class IoU≥0.5 dưới conf 0.25; C2 tương ứng 34/41 và 53/64. Cờ bằng chứng không loại trừ nhau; không bảo đảm hạ ngưỡng sẽ phục hồi từng TP khi matching một-một. Dưới floor 0.01/pre-NMS vẫn chưa biết.
- **Bias warmup C1 cao so với LR danh định:** epoch 1 bias LR 0.0669643 so với lr0 0.0001. C2 thực sự áp dụng 0.0001 nhưng recall vẫn yếu; warmup không đủ giải thích toàn bộ regression.
- **A tái lập đúng recipe; B đổi revision nhãn và cho kết quả kém ở seed/run này.** Chưa xác định thao tác nhãn nào gây thay đổi, nhãn sửa sai hay dữ liệu nhiễu nói chung tốt hơn dữ liệu sạch.
- **Domain/scale/scope khác:** mẫu Train/Val thường người nhỏ trong cảnh rộng, video tự quay cận một người và box lớn. Scope cũng góp phần vào chênh lệch. Mẫu có chủ đích không ước lượng được phân bố toàn dataset hoặc nguyên nhân toàn bộ lỗi.
- **Protocol đánh giá ảnh hưởng metric:** trainer-final/rect=True và common rect=False cho AP khác rõ trong một số run; đã tách và kiểm tra T4. Không thay số common bằng số tốt hơn từ protocol khác.

### Chưa đủ bằng chứng kết luận

- Fine-tune ít LR, bỏ augmentation và đổi revision nhãn có thể làm lệch score/khả năng tổng quát. Train loss giảm, Val classification loss tăng sau best cho thấy khoảng cách tổng quát hóa, chưa chứng minh catastrophic forgetting hoặc một augmentation cụ thể là thủ phạm.
- Chưa chứng minh train quá ít epoch là nguyên nhân chính. Các run chọn best trước khi metric suy giảm và có early stopping; tăng epoch không được bảo đảm chữa lỗi.
- Điểm cũ cao có thể liên quan split dễ/cảnh tương quan; chưa kết luận chắc chắn leakage chỉ từ điểm cao. Mapping Train/session cũ với Val hiện tại còn thiếu.
- Người dùng xác nhận 257 sự kiện FP Round 3; các nhóm tóc/áo/bàn/ghế do AI đề xuất chưa thành tỷ lệ nguyên nhân được người dùng phân loại đầy đủ. Không mặc định mọi unmatched prediction là vật thể không phải người; còn có sai class, định vị hoặc scope/GT chưa rõ.

Hypothesis report trước A/B là tài liệu tại thời điểm viết. Kết quả A/B mới bổ sung bằng chứng; không coi câu “chưa có matched control” trong báo cáo cũ là trạng thái hiện tại. [Hypothesis audit lịch sử](../experiments/round3_training_recipe_audit/REGRESSION_HYPOTHESES.md).

## 7. Tham chiếu trong ứng dụng

- Từ project root, dùng `ai/behavior_detection/models/best.pt`.
- `scripts/test_image.py` và `scripts/test_video.py` đã dùng `BASE_DIR/models/best.pt`; lần load tiếp theo đọc Round 3. Không chạy các script trong thao tác thay model/tạo README.
- `behavior_detector.py` nhận `model_path` từ caller; phải truyền đúng đường dẫn. Không tự chứng nhận một tiến trình đang chạy đã reload.
- Tiến trình đã load V2 có thể giữ trọng số trong RAM tới khi reload/restart; không có service nào được restart tự động.
- README cấp module cũ còn nhắc V1.4 tại dòng 25, 133, 276. Đó là thông tin lịch sử chưa cập nhật; **không dùng metric V1.4 ở đó làm kết quả best.pt hiện tại**. README này mô tả artifact chính thức trong models.

Conf 0.25 trong bảng là điểm đánh giá tham chiếu, không phải ngưỡng tối ưu đã xác nhận cho mọi video. Chưa thay confidence/NMS hoặc bật person filter trong cấu hình ứng dụng.

## 8. Giới hạn và hướng tiếp theo

Frozen Val tập trung cảnh/phòng, các frame tương quan; không coi 390 ảnh là 390 quan sát độc lập. Nguồn/session Train cũ và holdout chưa được ánh xạ đầy đủ. Chưa có bằng chứng tổng quát trên video toàn cảnh độc lập hoặc toàn bộ box đều đúng đầu + thân trên. Không dùng frozen Test để chọn model/ngưỡng.

Ưu tiên video toàn cảnh được phép sử dụng, phiên quay độc lập, có GT toàn ảnh được duyệt nếu muốn đo TP/FP/FN. Hard negatives/hard positives mới cần annotation đầy đủ và provenance rõ; không nhập 54 ảnh bị chặn hoặc Val vào Train. Không tự mở thêm training chỉ để tìm con số cao hơn.

**Trạng thái:** Round 3 đã được người dùng chọn làm `best.pt` chính thức. Checkpoint khác giữ làm artifact tham chiếu, không tự thay thế. Quyết định này không xóa các giới hạn chất lượng đã đo được.
