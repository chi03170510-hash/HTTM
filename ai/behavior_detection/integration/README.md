# AI V2 inference integration

The default `BehaviorDetector` and legacy video adapter use YOLO11s V2, with class mapping 0 `sleeping`, 1 `using_phone`, 2 `raising_hand`. The model path is `ai/behavior_detection/models/experimental/three_behavior_v2_best.pt`; obtain the file from the `ai-v2.0` release as described in `AI_INTEGRATION_GUIDE.md`. Set `--model` on `run_video` or pass `model_version="v1"` to compare with V1.

Install dependencies with `python -m pip install -r ai/behavior_detection/requirements_ai_v2.txt`.

## Video baseline

```powershell
python -m ai.behavior_detection.integration.run_video --source "lesson.mp4" --sample-fps 2 --pipeline legacy --output "results/lesson_overlay.mp4"
```

The baseline calls `BehaviorDetector.predict(frame_bgr)` through the compatibility adapter. It reports model detections; there is no per-frame track ID in this baseline. The experimental `--pipeline unified` option belongs to a separate prototype and needs additional model assets.

## Hybrid

```powershell
python -m ai.behavior_detection.integration.hybrid_behavior_inference --source "lesson.mp4" --device 0
```

Hybrid uses V2 by default. Pass `--behavior-version v1` to select V1. YOLO11s-Pose, ByteTrack, and the existing temporal/geometry rules remain as implemented. Person/phone and pose auxiliary weights are separate from the V2 artifact; see the integration guide. Hybrid dispositions are not ground-truth events or a validated accuracy result.

Annotation completeness for V2 Validation remains UNVERIFIED. There is no independent Test; the current status is `TEST_NOT_AVAILABLE`.
