"""Display-only, sample-and-hold boxes on live video (no tracking or inference).

Coordinates and labels always come from the same completed observation. Boxes
have bounded capture and completion ages; delayed geometry is explicitly labeled.
Cached results cannot extend duration.
"""

import math
from collections import deque

import cv2
import numpy as np

from behavior_duration import behavior_label, duration_text


AI_TARGET_FPS = 8.0  # Upper bound, not a promise of model throughput.
BBOX_MAX_CAPTURE_AGE = 10.0  # Hard limit; slow CPU results can still be displayed.
BBOX_MAX_RESULT_AGE = 5.0  # Hide if inference stops producing results.
BBOX_DELAY_WARNING = 0.6  # Beyond this, geometry is explicitly historical/delayed.
HEADER_HEIGHT = 100


class InferencePacer:
    """Wait before fetching the latest frame; never delay the camera/UI thread."""

    def __init__(self, fps=AI_TARGET_FPS):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("AI target FPS must be finite and positive")
        self.interval = 1.0 / fps
        self.next_start = None

    def wait(self, stop_event, now):
        delay = 0.0 if self.next_start is None else max(0.0, self.next_start - now)
        return stop_event.wait(delay) if delay > 0 else stop_event.is_set()

    def started(self, now):
        # Slow inference never accumulates catch-up requests.
        self.next_start = now + self.interval


class FrameRateMeter:
    """Bounded recent event rate, with idle time included (no synthetic frames)."""

    def __init__(self, window=3.0):
        self.window = window
        self.timestamps = deque(maxlen=120)

    def mark(self, timestamp):
        if not self.timestamps or timestamp > self.timestamps[-1]:
            self.timestamps.append(timestamp)

    def fps(self, now):
        recent = [stamp for stamp in self.timestamps if 0 <= now - stamp <= self.window]
        if len(recent) < 2:
            return 0.0
        elapsed = now - recent[0]
        return (len(recent) - 1) / elapsed if elapsed > 0 else 0.0


def result_age(result, now, field="captured_at"):
    if result is None:
        return None
    try:
        age = now - float(result[field])
    except (KeyError, TypeError, ValueError):
        return None
    return age if math.isfinite(age) and age >= 0 else None


def _source_size(result):
    size = result.get("source_size")
    if size is None:
        snapshot = result.get("snapshot")
        image = getattr(snapshot, "image", None)
        if image is not None:
            size = (image.shape[1], image.shape[0])
    try:
        width, height = map(float, size)
        if all(math.isfinite(v) and v > 0 for v in (width, height)):
            return width, height
    except (TypeError, ValueError):
        pass
    return None


def _clip_box(box, source_size, width, height):
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    sw, sh = source_size
    x1, x2 = max(0., x1), min(sw, x2)
    y1, y2 = max(0., y1), min(sh, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return (min(width - 1, round(x1 * width / sw)), min(height - 1, round(y1 * height / sh)),
            min(width - 1, round(x2 * width / sw)), min(height - 1, round(y2 * height / sh)))


def _text(image, text, origin, color, scale=.48, outline=3):
    # Fit long IDs to the available width without drawing outside the frame.
    available = image.shape[1] - origin[0] - 4
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > available:
        text = text[:-1]
    if text:
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), outline, cv2.LINE_AA)
        cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_live_bboxes(frame, result, *, now, camera_fps=0.0, ai_fps=0.0,
                     max_age=BBOX_MAX_CAPTURE_AGE, view_width=1280):
    """Render live pixels with last-observation boxes, never a historical image.

    No motion compensation is claimed. Different aspect ratios invalidate the
    geometry; proportional resolution changes are scaled in source coordinates.
    """
    height, width = frame.shape[:2]
    output = frame.copy()
    age = result_age(result, now)
    size = _source_size(result) if result is not None else None
    same_geometry = size is not None and math.isclose(width / height, size[0] / size[1], rel_tol=.01)
    completion_age = result_age(result, now, "completed_at")
    valid_times = (age is not None and completion_age is not None and completion_age <= age)
    usable = (valid_times and age <= max_age and completion_age <= BBOX_MAX_RESULT_AGE and same_geometry)
    delayed = usable and age > BBOX_DELAY_WARNING
    observation = result.get("behavior_frame") if result is not None else None
    drawn_count = 0
    if usable and observation is not None:
        for person in observation.people:
            box = _clip_box(person.bbox, size, width, height)
            if box is None:
                continue
            states = {behavior.state for behavior in person.behaviors}
            color = ((70, 100, 255) if "detected" in states else
                     (0, 210, 255) if "suspected" in states else (80, 220, 100))
            x1, y1, x2, y2 = box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            drawn_count += 1
            labels = [str(person.person_track_id)]
            for behavior in person.behaviors:
                if behavior.state not in ("suspected", "detected"):
                    continue
                duration = duration_text(behavior)
                suffix = f" {duration}" if duration != "--" else ""
                labels.append(f"{behavior_label(behavior.name)}: {behavior.state}{suffix}")
            # Keep the whole label block visible, including boxes at the bottom.
            y = max(10, min(y1 + 11, height - 6 - 12 * (len(labels) - 1)))
            for label in labels:
                _text(output, label, (max(2, x1 + 3), y), color, scale=.28, outline=2)
                y += 12
                if y > height - 4:
                    break

    scaled_height = max(1, round(height * view_width / width))
    preview = cv2.resize(output, (view_width, scaled_height), interpolation=cv2.INTER_AREA
                         if view_width < width else cv2.INTER_LINEAR)
    canvas = np.full((scaled_height + HEADER_HEIGHT, view_width, 3), 24, dtype=np.uint8)
    canvas[HEADER_HEIGHT:] = preview
    age_text = "--" if age is None else f"{age:.2f}s"
    status = ("Waiting for AI" if result is None else
              "STALE / incompatible frame - BBoxes hidden" if not usable else
              "No valid person boxes in latest AI observation" if drawn_count == 0 else
              f"DELAYED AI BBoxes: {age:.2f}s old - positions/labels may not match current people" if delayed else
              "BBoxes: last AI observation (may lag motion)")
    _text(canvas, f"LIVE + AI | Camera FPS: {camera_fps:.1f} | AI FPS: {ai_fps:.1f} / target <={AI_TARGET_FPS:g} | Age: {age_text}",
          (12, 22), (235, 235, 235), .52)
    _text(canvas, status, (12, 44), (0, 210, 255), .5)
    runtime = result.get("inference_runtime") if result else None
    backend = runtime.get("backend", "unknown") if isinstance(runtime, dict) else "unknown"
    if backend == "cuda":
        precisions = {m.get("precision", "unknown") for m in runtime.get("models", [])}
        precision = "FP16" if precisions == {"float16"} else "FP32" if precisions == {"float32"} else "mixed/unknown precision"
        device_text = f"YOLO: GPU / CUDA - {runtime.get('device_name', 'unknown')} ({precision})"
    elif backend == "cpu":
        device_text = "YOLO: CPU"
    elif backend == "mixed":
        device_text = "YOLO: MIXED GPU/CPU"
    else:
        device_text = "YOLO: checking actual backend..."
    if result is not None and not usable:
        device_text += " [last result]"
    face_info = result.get("face_runtime") if result else None
    face_backend = face_info.get("backend", "unknown") if isinstance(face_info, dict) else "unknown"
    face_text = "GPU / CUDA" if face_backend == "cuda" else "CPU" if face_backend == "cpu" else "checking..."
    _text(canvas, device_text + f" | Face: {face_text} | MediaPipe: CPU", (12, 65), (120, 235, 180), .50)
    _text(canvas, "Q: Quit | I: diagnostics | Labels are AI estimates, not confirmed violations", (12, 87),
          (200, 200, 200), .46)
    return canvas
