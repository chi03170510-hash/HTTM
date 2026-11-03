"""Video integration adapter for HTTM behavior detection."""

from .detector_adapter import DEFAULT_CHECKPOINT, DetectorAdapter
from .duration_limit import DurationLimit
from .event_schema import DetectionEvent, FrameMetadata, FrameResult
from .frame_sampler import FrameSampler
from .pipeline import VideoBehaviorPipeline
from .video_source import FramePacket, VideoSource, VideoSourceError

__all__ = [
    "DEFAULT_CHECKPOINT", "DetectionEvent", "DetectorAdapter", "DurationLimit", "FrameMetadata",
    "FramePacket", "FrameResult", "FrameSampler", "VideoBehaviorPipeline",
    "VideoSource", "VideoSourceError",
]
