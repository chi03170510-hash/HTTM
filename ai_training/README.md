# Classroom behavior inference handoff

Image-mode inference for four classroom behaviors:

- `sleeping`
- `using_phone`
- `turning_back`
- `eating`

The primary behavior checkpoint is **YOLO11n**, `models/best_behavior_v2.pt`. Its verified class mapping is `{0: sleeping, 1: using_phone}`. The `turning_back` and `eating` behaviors do not exist as classes in that checkpoint: V7 derives them from separate pose/face and COCO object evidence with spatial rules. The V1.4 behavior checkpoint is included as a comparison/reference model.

## Clone and install

```powershell
git clone --branch feature/httm-ai-v2 https://github.com/chi03170510-hash/HTTM.git
cd HTTM
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r ai_training/requirements.txt
```

The repository contains the behavior checkpoints and the COCO, MediaPipe Face, and MediaPipe Pose assets used by the image runner. No dataset, test images, or videos are required to import the inference code; supply an image directory at runtime.

## Run Hybrid V7 on images

The runner uses the V2 checkpoint by default. Pass `--weights` to select V1.4 or another compatible two-class checkpoint.

```powershell
# Primary V2 checkpoint
python ai_training/run_hybrid_v7_images.py `
  --images .\classroom_images `
  --weights ai_training\models\best_behavior_v2.pt `
  --config ai_training\config\hybrid_v7.json `
  --output-dir ai_training\runs\my_v2_image_run

# Reference V1.4 checkpoint
python ai_training/run_hybrid_v7_images.py `
  --images .\classroom_images `
  --weights ai_training\models\best_behavior_v1_4.pt `
  --config ai_training\config\hybrid_v7.json `
  --output-dir ai_training\runs\my_v1_4_image_run
```

The run directory contains `images_clean/`, `images_debug/`, `predictions.json`, `predictions.csv`, `raw_detections.json`, and `summary.json`. Each person's behavior result includes its state, model score where applicable, evidence, missing evidence, and decision reason. For a single image, temporal confirmation is not applied.

The V7 config is in `config/hybrid_v7.json`; V1.4 spatial settings are in `config/hybrid_behavior_v1_4.json`. The model score is a raw detector score, not a calibrated probability for the complete hybrid system.

## V2 Kaggle test results

Reported V2 Kaggle test metrics: **mAP50 0.6927**, **mAP50-95 0.3443**, **precision 0.7064**, and **recall 0.6206**. These are the supplied Kaggle test results; do not infer performance for other camera conditions from them.

V2 has not been confirmed better than V1.4 in every real-world scenario. In the 15-image regression comparison, V2 preserved some phone-use cases through Hybrid V7 evidence but missed one sleeping detection that V1.4 produced on image 7; the set has no independent labels and does not establish general superiority. Keep the weights selectable and compare on labeled data before changing a deployed default.

## Source and model layout

- `models/best_behavior_v2.pt`: primary behavior detector.
- `models/best_behavior_v1_4.pt`: reference behavior detector.
- `models/pretrained_person/yolo11n_coco.pt`: person, phone, and supported food object evidence.
- `models/face_landmarker.task`, `models/pose_landmarker_full.task`: optional per-person face/pose landmarks used by V7.
- `src/hybrid_v7.py`, `src/hybrid_v6_1.py`, and their V2–V5 dependencies: image spatial fusion and evidence rules.
- `src/object_evidence.py`: local COCO/MediaPipe adapter; hand landmarks remain optional when the hand asset is absent.
- `run_hybrid_v7_images.py`: image-only entry point.

Tests that exercise the V7 rule contracts can be run with:

```powershell
python -m unittest ai_training.tests.test_hybrid_v7
```

No model training is performed by this repository's image inference command.
