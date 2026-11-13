
"""
HTTM - YOLO Behavior Detector
Behavior V1.4 Model

Model: ai/behavior_detection/models/best_behavior_v1_4.pt

Supported behaviors:
    0 - sleeping
    1 - using_phone

This detector does not support raising_hand.
"""

from dataclasses import asdict
import math
from pathlib import Path
import time

import numpy as np
from ultralytics import YOLO

if __package__:
    from .diagnostics import emit_debug
else:  # Preserve imports from scripts run inside ai/behavior_detection.
    from diagnostics import emit_debug

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = BASE_DIR / "models" / "best_behavior_v1_4.pt"

EXPECTED_CLASSES = {
    0: "sleeping",
    1: "using_phone",
}
CANDIDATE_CONFIDENCE = 0.10


class BehaviorDetector:
    """
    Detect student behaviors using the V1.4 YOLO model.

    Compatible with:
        - detector_adapter.py
        - integration.run_video
        - legacy pipeline
    """

    def __init__(
        self,
        model_path=None,
        sleeping_conf=0.40,
        phone_conf=0.40,
        imgsz=640,
        device=None,
        *,
        debug_detections=False,
    ):
        # 1. Select V1.4 checkpoint
        self.model_path = (
            Path(model_path)
            if model_path is not None
            else DEFAULT_MODEL_PATH
        )

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {self.model_path}"
            )

        # 2. Load YOLO model
        self.model = YOLO(str(self.model_path))

        # 3. Read class names from checkpoint
        names = self.model.names

        if isinstance(names, dict):
            name_items = names.items()
        else:
            name_items = enumerate(names)

        self.class_names = {
            int(class_id): str(name).strip().lower()
            for class_id, name in name_items
        }

        # 4. Validate V1.4 classes
        if self.class_names != EXPECTED_CLASSES:
            raise RuntimeError(
                "Incorrect YOLO checkpoint classes.\n"
                f"Expected: {EXPECTED_CLASSES}\n"
                f"Actual: {self.class_names}\n"
                f"Model: {self.model_path}"
            )

        self.supported_behaviors = tuple(
            self.class_names.values()
        )

        # 5. Configure thresholds
        self.thresholds = {
            "sleeping": float(sleeping_conf),
            "using_phone": float(phone_conf),
        }

        for behavior, threshold in self.thresholds.items():
            if not 0.0 <= threshold <= 1.0:
                raise ValueError(
                    f"Invalid confidence threshold for "
                    f"{behavior}: {threshold}"
                )

        self.imgsz = imgsz
        self.device = device
        self.debug_detections = debug_detections
        self._debug_context = None
        self.last_debug_candidates = None

    def set_debug_context(self, metadata):
        """Associate opt-in reports with the original frame, on the AI thread."""
        self._debug_context = asdict(metadata)

    def _predict_yolo(self, frame):
        return self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            conf=CANDIDATE_CONFIDENCE,
            iou=0.70,
            device=self.device,
            verbose=False,
        )[0]

    def warmup(self, frame_shape=(480, 640, 3)):
        """Prepare the predictor before capture; dummy output is never an event."""
        frame = np.zeros(frame_shape, dtype=np.uint8)
        emit_debug("yolo_warmup_start", input_shape=list(frame.shape), model=str(self.model_path))
        started = time.perf_counter()
        self._predict_yolo(frame)
        elapsed_ms = (time.perf_counter() - started) * 1000
        emit_debug("yolo_warmup", input_shape=list(frame.shape),
                   duration_ms=round(elapsed_ms, 3), model=str(self.model_path),
                   counted_as_detection=False)
        return elapsed_ms

    def predict(self, frame):
        """
        Run YOLO inference on one OpenCV BGR frame.

        Returns a list of detection dictionaries.
        """

        # Reset per-call debug state so an error cannot reuse the previous boxes.
        self.last_debug_candidates = None
        # Validate input frame
        if (
            not isinstance(frame, np.ndarray)
            or getattr(frame, "ndim", None) != 3
            or frame.shape[2] != 3
            or frame.shape[0] == 0
            or frame.shape[1] == 0
        ):
            raise ValueError(
                "frame must be a non-empty BGR image (HxWx3)"
            )
        if frame.dtype != np.uint8:
            raise ValueError("frame must use OpenCV BGR uint8 values in the range 0..255")

        # Run YOLO inference
        predictor = getattr(self.model, "predictor", None)
        cold_start = not bool(predictor is not None and getattr(predictor, "done_warmup", False))
        started = time.perf_counter()
        result = self._predict_yolo(frame)
        predict_ms = (time.perf_counter() - started) * 1000

        detections = []
        candidates = []

        # Process bounding boxes
        for box in result.boxes if result.boxes is not None else ():
            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id)
            confidence = float(box.conf[0])
            threshold = self.thresholds.get(class_name)
            reason = ("unsupported_class_id" if class_name is None else
                      "below_class_threshold" if confidence < threshold else "accepted")
            bbox_xyxy = ([float(value) for value in box.xyxy[0].tolist()]
                         if self.debug_detections or reason == "accepted" else None)
            if self.debug_detections:
                x1, y1, x2, y2 = bbox_xyxy
                candidates.append({
                    "class_id": class_id, "class_name": class_name,
                    "confidence": confidence, "bbox_xyxy": bbox_xyxy,
                    "official_threshold": threshold, "decision": reason,
                    "bbox_in_source_bounds": (
                        all(math.isfinite(v) for v in bbox_xyxy)
                        and 0 <= x1 <= x2 <= frame.shape[1]
                        and 0 <= y1 <= y2 <= frame.shape[0]
                    ),
                })
            if reason != "accepted":
                continue

            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": round(confidence, 4),
                "bbox_xyxy": bbox_xyxy,
                "track_id": None,
                "status": "MODEL_DETECTION",
            })

        if self.debug_detections:
            # Consumed on the same AI thread by the optional image recorder.
            self.last_debug_candidates = candidates
            speed = getattr(result, "speed", {}) or {}
            emit_debug(
                "yolo_candidates", metadata=self._debug_context,
                stage="post_nms_before_official_threshold",
                candidate_confidence_floor=CANDIDATE_CONFIDENCE,
                thresholds=self.thresholds, model=str(self.model_path), classes=self.class_names,
                input={"shape_hwc": list(frame.shape), "dtype": str(frame.dtype),
                       "pixel_format": "BGR", "contiguous": bool(frame.flags.c_contiguous),
                       "strides": list(frame.strides),
                       "value_range": [int(frame.min()), int(frame.max())]},
                requested_imgsz=self.imgsz,
                yolo_original_shape=list(getattr(result, "orig_shape", frame.shape[:2])),
                bbox_coordinate_space="original_input_pixels_xyxy",
                cold_start=cold_start, predict_wall_ms=round(predict_ms, 3),
                yolo_stage_ms={k: float(v) for k, v in speed.items()},
                unprofiled_ms=round(max(0.0, predict_ms - sum(speed.values())), 3),
                candidates=candidates, accepted_count=len(detections),
                empty_reason=("no_yolo_candidates_at_conf_0.10" if not candidates else
                              "all_candidates_filtered" if not detections else None),
            )
        return detections

    def detect(self, frame):
        """
        Legacy compatibility method.

        Output format expected by detector_adapter.py:
            class
            class_id
            confidence
            bbox
        """

        predictions = self.predict(frame)

        return [
            {
                "class": item["class_name"],
                "class_id": item["class_id"],
                "confidence": item["confidence"],
                "bbox": [
                    int(value)
                    for value in item["bbox_xyxy"]
                ],
            }
            for item in predictions
        ]

    def reset(self):
        """
        Reset detector state.

        This YOLO detector is stateless.
        Tracking is handled by another component.
        """
        return None
