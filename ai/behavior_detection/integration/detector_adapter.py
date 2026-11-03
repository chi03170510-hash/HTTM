"""Normalize the existing BehaviorDetector output to the integration schema."""

from pathlib import Path
from typing import Any, Optional

from .event_schema import DetectionEvent, FrameMetadata, FrameResult


DEFAULT_CHECKPOINT = (Path(__file__).resolve().parents[1] / "models" / "experimental" / "three_behavior_v2_best.pt")


class DetectorAdapter:
    def __init__(self, detector: Any):
        if not callable(getattr(detector, "detect", None)):
            raise TypeError("detector must provide detect(frame)")
        self.detector = detector

    @classmethod
    def from_default_checkpoint(
        cls, model_path: Optional[str | Path] = None, device: Optional[str] = None
    ) -> "DetectorAdapter":
        from ..behavior_detector import BehaviorDetector

        detector = BehaviorDetector(
            model_path=str(Path(model_path) if model_path else DEFAULT_CHECKPOINT),
            device=device,
        )
        return cls(detector)

    def process(self, frame: Any, metadata: FrameMetadata) -> FrameResult:
        try:
            raw_detections = self.detector.detect(frame)
            detections = tuple(
                DetectionEvent(
                    behavior=str(item["class"]),
                    class_id=int(item["class_id"]),
                    confidence=float(item["confidence"]),
                    bbox_xyxy_px=tuple(int(value) for value in item["bbox"]),
                    frame_index=metadata.frame_index,
                    timestamp_ms=metadata.timestamp_ms,
                )
                for item in raw_detections
            )
            return FrameResult(status="ok", metadata=metadata, detections=detections)
        except Exception as exc:
            return FrameResult(
                status="inference_error", metadata=metadata, error=f"{type(exc).__name__}: {exc}"
            )
