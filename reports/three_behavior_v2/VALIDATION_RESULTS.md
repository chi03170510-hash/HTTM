# YOLO11s V2 Validation results

Checkpoint: `httm_three_behavior_v2_best.pt`
Classes: `0 sleeping`, `1 using_phone`, `2 raising_hand`
Selection: best checkpoint selected by the group-held-out Validation split. Kaggle run: imgsz 640, 100 epochs maximum, batch 16, patience 25, seed 42, optimizer auto. Training ran on Kaggle GPU; it was not repeated locally.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| sleeping | 0.7670 | 0.8348 | 0.8281 | 0.3991 |
| using_phone | 0.7044 | 0.5698 | 0.6547 | 0.3256 |
| raising_hand | 0.8583 | 0.8728 | 0.9094 | 0.4345 |
| **Macro mean of class mAP50 values** | — | — | **0.7974** | — |

These are diagnostic Validation metrics, not a Test result. Annotation completeness is **UNVERIFIED**, so these values do not establish independent generalization or accuracy. `TEST_NOT_AVAILABLE`; see [`TEST_STATUS.md`](TEST_STATUS.md). Confusion matrices, curves, and Validation prediction samples are in the Kaggle training artifact ZIP; they are not data to commit to the source repository.
