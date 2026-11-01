# HTTM AI V2 Integration Guide

## Checkpoint and integrity

The default model is YOLO11s V2, stored at:

`ai/behavior_detection/models/experimental/three_behavior_v2_best.pt`

The file is distributed as the `httm_three_behavior_v2_best.pt` asset in GitHub Release [`ai-v2.0`](https://github.com/chi03170510-hash/HTTM/releases/tag/ai-v2.0). Download it from the release asset into the path above, then verify its SHA-256:

```powershell
New-Item -ItemType Directory -Force ai/behavior_detection/models/experimental | Out-Null
Invoke-WebRequest -Uri "https://github.com/chi03170510-hash/HTTM/releases/download/ai-v2.0/httm_three_behavior_v2_best.pt" -OutFile ai/behavior_detection/models/experimental/three_behavior_v2_best.pt
(Get-FileHash ai/behavior_detection/models/experimental/three_behavior_v2_best.pt -Algorithm SHA256).Hash
```

Expected SHA-256: `75073a9e3a429e8f09fdff618e2ad044c4ae7a967ba6e11fcf828e229cf21b7a`.

Install Python dependencies from the repository root:

```powershell
python -m pip install -r ai/behavior_detection/requirements_ai_v2.txt
```

For GPU inference, install a PyTorch build compatible with the machine's CUDA runtime before installing the requirements. CPU works for short inference but is slower.

## Frame inference API

```python
import cv2
from ai.behavior_detection import BehaviorDetector

detector = BehaviorDetector()  # loads V2 once
cap = cv2.VideoCapture("lesson.mp4")
try:
    ok, frame_bgr = cap.read()  # OpenCV BGR
    if ok:
        for detection in detector.predict(frame_bgr):
            print(detection)
finally:
    cap.release()
    detector.reset()  # stateless reset hook when switching videos
```

Each returned detection has `class_id`, `class_name`, `confidence`, `bbox_xyxy` (pixel coordinates), `track_id`, and `status`. Mapping is `0 sleeping`, `1 using_phone`, `2 raising_hand`. `track_id` is `None` in this frame-only API. `status="MODEL_DETECTION"` means a model box passed the configured confidence cutoff; it is not ground truth or a confirmed behavior event. `BehaviorDetector.detect(frame_bgr)` remains available for the legacy output schema.

V1 remains selectable with `BehaviorDetector(model_version="v1")` or by passing an explicit model path. The V1 checkpoint is not included in the V2 release.

## Hybrid tracking

Install the dependency file, then run Hybrid from the repository root:

```powershell
python -m ai.behavior_detection.integration.hybrid_behavior_inference --source "lesson.mp4" --device 0
```

V2 is the default. Select V1 with `--behavior-version v1`. Hybrid keeps its existing YOLO11s-Pose, person/phone detector, ByteTrack, and temporal/geometry heuristics. These heuristic statuses are candidate dispositions, not ground truth; Hybrid has not been shown to outperform the baseline. The auxiliary weights must also be present at `ai/behavior_detection/models/pose/yolo11s-pose.pt` and `ai/behavior_detection/models/pretrained_person/yolo11n_coco.pt` for Hybrid use. They are not included in the V2 checkpoint asset.

## Kaggle and evaluation status

- Training notebook: [`HTTM_ThreeBehavior_V2_Kaggle.ipynb`](experiments/accuracy_first_development/HTTM_ThreeBehavior_V2_Kaggle.ipynb).
- Evaluation-only notebook: [`HTTM_ThreeBehavior_V2_Kaggle_Test_Evaluation.ipynb`](experiments/accuracy_first_development/HTTM_ThreeBehavior_V2_Kaggle_Test_Evaluation.ipynb).
- Validation results: [`VALIDATION_RESULTS.md`](../../reports/three_behavior_v2/VALIDATION_RESULTS.md).
- Independent Test status: [`TEST_STATUS.md`](../../reports/three_behavior_v2/TEST_STATUS.md), **TEST_NOT_AVAILABLE**.

V2 was selected on Validation. Annotation completeness is **UNVERIFIED**, there is no credible independent Test, and the Validation metrics are diagnostics only. Do not report them as Test accuracy or verified generalization.
