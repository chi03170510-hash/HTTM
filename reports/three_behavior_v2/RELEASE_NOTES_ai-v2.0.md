# ai-v2.0 release notes

## HTTM YOLO11s Three-Behavior V2

Checkpoint: `httm_three_behavior_v2_best.pt`
SHA-256: `75073a9e3a429e8f09fdff618e2ad044c4ae7a967ba6e11fcf828e229cf21b7a`

Class mapping: 0 `sleeping`, 1 `using_phone`, 2 `raising_hand`.

Trained on Kaggle GPU from YOLO11s pretrained weights; the selected checkpoint uses the group-held-out Validation split. Validation annotation completeness is **UNVERIFIED**. The package has no credible independent Test set (`TEST_NOT_AVAILABLE`). Validation metrics are diagnostics only and must not be described as Test accuracy or verified generalization.

The two source dataset versions declared CC BY 4.0 in their source metadata: [Classroom Behavior v1](https://universe.roboflow.com/behaviour-detection-tlowy/classroom-behavior-05jvk/dataset/1) and [Classroom Student Dataset v16](https://universe.roboflow.com/classroom-na2vo/classroom-student-dataset/dataset/16). Attribution and mapping details are in [`THIRD_PARTY_ATTRIBUTION.md`](THIRD_PARTY_ATTRIBUTION.md). No dataset images or labels are included in this release.
