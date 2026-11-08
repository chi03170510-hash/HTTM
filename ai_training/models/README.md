# Inference model assets

`MANIFEST.json` records the SHA-256, size, task, and class mapping for the handoff assets.

- `best_behavior_v2.pt` is the primary YOLO11n two-class behavior checkpoint.
- `best_behavior_v1_4.pt` is the reference two-class checkpoint.
- `pretrained_person/yolo11n_coco.pt` supplies COCO person, phone, and supported food object proposals for Hybrid V7.
- `face_landmarker.task` and `pose_landmarker_full.task` supply optional face and pose landmarks. If MediaPipe cannot initialize, its evidence is unavailable and the pipeline records that limitation.

The hand-landmarker task is optional and is not bundled. The image inference runner does not train or modify any model.
