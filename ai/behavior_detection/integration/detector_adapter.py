
"""
HTTM - Detector Adapter

Default model: best_behavior_v1_4.pt
Supported behaviors:
    0 - sleeping
    1 - using_phone

Converts BehaviorDetector output into integration events.
"""

from pathlib import Path

from ..behavior_detector import BehaviorDetector, DEFAULT_MODEL_PATH
from ..diagnostics import emit_debug
from .event_schema import DetectionEvent, FrameResult


DEFAULT_CHECKPOINT = DEFAULT_MODEL_PATH

SUPPORTED_BEHAVIORS = {
    0: "sleeping",
    1: "using_phone",
}


class DetectorAdapter:
    def __init__(self, detector, *, debug_detections=False, debug_frame_recorder=None):
        if not callable(getattr(detector, "detect", None)):
            raise TypeError(
                "Detector must provide a callable detect(frame) method"
            )
        if debug_frame_recorder is not None and not debug_detections:
            raise ValueError("debug frame saving requires debug_detections=True")

        self.detector = detector
        self.debug_detections = debug_detections
        self.debug_frame_recorder = debug_frame_recorder

    @classmethod
    def from_default_checkpoint(
        cls,
        model_path=None,
        device=None,
        *,
        debug_detections=False,
    ):
        checkpoint = (
            Path(model_path)
            if model_path is not None
            else DEFAULT_CHECKPOINT
        )

        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"YOLO checkpoint not found: {checkpoint}"
            )

        detector = BehaviorDetector(
            model_path=str(checkpoint),
            device=device,
            debug_detections=debug_detections,
        )

        for class_id, expected_name in SUPPORTED_BEHAVIORS.items():
            actual_name = detector.class_names.get(class_id)

            if actual_name != expected_name:
                raise ValueError(
                    f"Invalid model class {class_id}: "
                    f"expected {expected_name}, got {actual_name}"
                )

        if len(detector.class_names) != len(SUPPORTED_BEHAVIORS):
            raise ValueError(
                "Round 3 adapter expects exactly 2 behavior classes"
            )

        print("[HTTM] Detector initialized")
        print(f"[HTTM] Model: {checkpoint}")
        print(f"[HTTM] Classes: {detector.class_names}")

        return cls(detector, debug_detections=debug_detections)

    def warmup(self):
        return self.detector.warmup()

    def process(self, frame, metadata):
        try:
            if self.debug_detections:
                set_context = getattr(self.detector, "set_debug_context", None)
                if callable(set_context):
                    set_context(metadata)
            raw_detections = self.detector.detect(frame)

            detections = []
            filtered = []

            for item in raw_detections:
                class_id = int(item["class_id"])
                behavior = str(item["class"])

                if class_id not in SUPPORTED_BEHAVIORS:
                    if self.debug_detections:
                        filtered.append({"detection": item, "reason": "unsupported_class_id"})
                    continue

                if behavior != SUPPORTED_BEHAVIORS[class_id]:
                    if self.debug_detections:
                        filtered.append({"detection": item, "reason": "class_name_id_mismatch"})
                    continue

                detection = DetectionEvent(
                    behavior=behavior,
                    class_id=class_id,
                    confidence=float(item["confidence"]),
                    bbox_xyxy_px=tuple(
                        int(value) for value in item["bbox"]
                    ),
                    frame_index=metadata.frame_index,
                    timestamp_ms=metadata.timestamp_ms,
                )

                detections.append(detection)

            if self.debug_detections:
                emit_debug("adapter_detections", metadata=metadata.__dict__,
                           detector_output_count=len(raw_detections),
                           accepted_count=len(detections), filtered=filtered)
            result = FrameResult(
                status="ok",
                metadata=metadata,
                detections=tuple(detections),
            )

        except Exception as exc:
            if self.debug_detections:
                emit_debug("adapter_error", metadata=metadata.__dict__,
                           error=f"{type(exc).__name__}: {exc}")
            result = FrameResult(
                status="inference_error",
                metadata=metadata,
                error=f"{type(exc).__name__}: {exc}",
            )

        # The worker owns this frame. Image encoding and disk I/O never run in
        # the video loop and cannot change its original frame or event schema.
        if self.debug_frame_recorder is not None:
            candidates = (getattr(self.detector, "last_debug_candidates", None)
                          if result.status == "ok" else None)
            self.debug_frame_recorder.record(frame, metadata, candidates, result)
        return result
