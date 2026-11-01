"""Optional OpenCV debug overlay for normalized frame results."""

from typing import Any

from .event_schema import FrameResult


def draw_detections(frame: Any, result: FrameResult, cv2_module: Any = None) -> Any:
    if cv2_module is None:
        import cv2 as cv2_module
    for detection in result.detections:
        x1, y1, x2, y2 = detection.bbox_xyxy_px
        label = f"{detection.behavior} {detection.confidence:.2f}"
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2_module.putText(
            frame, label, (x1, max(0, y1 - 8)), cv2_module.FONT_HERSHEY_SIMPLEX,
            0.5, (0, 220, 0), 1, cv2_module.LINE_AA,
        )
    return frame
