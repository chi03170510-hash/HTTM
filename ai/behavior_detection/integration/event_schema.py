"""Typed records shared by the video adapter and its callers."""

from dataclasses import asdict, dataclass
from typing import Literal, Optional


FrameStatus = Literal["ok", "read_error", "inference_error"]


@dataclass(frozen=True)
class FrameMetadata:
    source_id: str
    frame_index: int
    timestamp_ms: int
    frame_width: int
    frame_height: int


@dataclass(frozen=True)
class DetectionEvent:
    behavior: str
    class_id: int
    confidence: float
    bbox_xyxy_px: tuple[int, int, int, int]
    frame_index: int
    timestamp_ms: int
    track_id: Optional[str] = None

    def to_dict(self) -> dict:
        result = asdict(self)
        result["bbox_xyxy_px"] = list(self.bbox_xyxy_px)
        return result


@dataclass(frozen=True)
class FrameResult:
    status: FrameStatus
    metadata: Optional[FrameMetadata]
    detections: tuple[DetectionEvent, ...] = ()
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "metadata": asdict(self.metadata) if self.metadata else None,
            "detections": [event.to_dict() for event in self.detections],
            "error": self.error,
        }
