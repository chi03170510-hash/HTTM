"""OpenCV video/webcam reader with zero-based frame indices."""

from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any, Callable, Optional, Union

from .event_schema import FrameMetadata


class VideoSourceError(RuntimeError):
    """The source could not be opened or a frame could not be read."""


@dataclass(frozen=True)
class FramePacket:
    frame: Any
    metadata: FrameMetadata


Source = Union[str, Path, int]


class VideoSource:
    def __init__(
        self,
        source: Source,
        source_id: Optional[str] = None,
        capture_factory: Optional[Callable[[Source], Any]] = None,
        cv2_module: Any = None,
        clock_ms: Callable[[], float] = lambda: monotonic() * 1000,
    ):
        self.source = source
        self.source_id = source_id or str(source)
        self._capture_factory = capture_factory
        self._cv2 = cv2_module
        self._clock_ms = clock_ms
        self._capture = None
        self._frame_index = 0
        self._started_ms = None
        self._is_webcam = isinstance(source, int)

    def open(self) -> "VideoSource":
        if self._capture is not None:
            return self
        if self._capture_factory is None:
            if self._cv2 is None:
                import cv2
                self._cv2 = cv2
            self._capture_factory = self._cv2.VideoCapture
        self._capture = self._capture_factory(self.source)
        if not self._capture.isOpened():
            self.release()
            raise VideoSourceError(f"Could not open video source: {self.source}")
        self._started_ms = self._clock_ms()
        return self

    def read(self) -> Optional[FramePacket]:
        if self._capture is None:
            self.open()
        ok, frame = self._capture.read()
        if not ok or frame is None:
            if not self._is_webcam:
                count = int(self._capture.get(self._cv2.CAP_PROP_FRAME_COUNT)) if self._cv2 else 0
                position = int(self._capture.get(self._cv2.CAP_PROP_POS_FRAMES)) if self._cv2 else 0
                if count > 0 and position >= count:
                    return None
                # OpenCV does not distinguish a corrupt file frame from EOF when
                # frame count metadata is unavailable. Treat that case as EOF.
                if count <= 0:
                    return None
            raise VideoSourceError(f"Could not read frame {self._frame_index} from {self.source}")

        height, width = frame.shape[:2]
        timestamp_ms = self._timestamp_ms()
        packet = FramePacket(
            frame=frame,
            metadata=FrameMetadata(
                source_id=self.source_id,
                frame_index=self._frame_index,
                timestamp_ms=timestamp_ms,
                frame_width=int(width),
                frame_height=int(height),
            ),
        )
        self._frame_index += 1
        return packet

    def _timestamp_ms(self) -> int:
        if not self._is_webcam and self._cv2 is not None:
            position_ms = float(self._capture.get(self._cv2.CAP_PROP_POS_MSEC))
            if position_ms >= 0:
                return int(position_ms)
        now = self._clock_ms()
        return max(0, int(now - (self._started_ms if self._started_ms is not None else now)))

    def release(self) -> None:
        if self._capture is not None:
            self._capture.release()
            self._capture = None

    def __enter__(self) -> "VideoSource":
        return self.open()

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.release()
