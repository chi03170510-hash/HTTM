"""Opt-in historical BBox window. No live pixels or AI decisions as input."""

import math
import os
import sys
import time

import cv2
import numpy as np

from behavior_overlay import AISnapshot, SNAPSHOT_HEADER, _fit_text, render_snapshot
from snapshot_detail_overlay import render_details
from snapshot_label_layout import fit_marked


OVERLAY_ENV = "HTTM_PERSON_SNAPSHOT_OVERLAY"
WINDOW_NAME = "HTTM AI Snapshot - HISTORICAL (BBox)"


def overlay_from_environment(*, stale_after):
    if os.environ.get(OVERLAY_ENV) != "1":
        return None
    overlay = PersonSnapshotOverlay(stale_after=stale_after)
    print(f"[PERSON SNAPSHOT OVERLAY] enabled; separate historical window; {WINDOW_NAME}", flush=True)
    return overlay


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class PersonSnapshotOverlay:
    def __init__(self, *, stale_after, max_size=(640, 480)):
        if not finite(stale_after) or stale_after <= 0:
            raise ValueError("stale_after must use an existing finite positive display limit")
        if len(max_size) != 2 or any(type(v) is not int or v < 1 for v in max_size):
            raise ValueError("max_size must contain positive integer dimensions")
        self.stale_after, self.max_size = stale_after, tuple(max_size)
        self._cached = self._key = None
        self._failed_key = object()
        self._window_open = self._display_failed = False
        self._window_size = None
        self._source_snapshot = None  # Keep identity alive; Python id cannot recycle.
        self._reported_failure = None

    def _placeholder(self, message):
        width, height = self.max_size
        image = np.full((SNAPSHOT_HEADER + height, width, 3), 25, dtype=np.uint8)
        for text, y in (("AI SNAPSHOT - HISTORICAL", 22), (message, 46)):
            cv2.putText(image, _fit_text(text, width - 16, .43), (8, y), cv2.FONT_HERSHEY_SIMPLEX,
                        .43, (0, 220, 255), 1, cv2.LINE_AA)
        return image

    def _failure(self, key, exc, *, cache_failure=True):
        report = (key, type(exc).__name__, str(exc))
        if self._reported_failure != report:
            print(f"[PERSON SNAPSHOT OVERLAY] render unavailable; live continues; "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        self._reported_failure = report
        self._failed_key = key if cache_failure else object()
        self._cached = self._key = None

    def render(self, result, *, now=None):
        """Pure display projection: snapshot-only pixels; no mutation of result."""
        if result is None:
            return self._placeholder("Waiting for first completed AI observation")
        key, rendering = None, False
        try:
            if not isinstance(result, dict):
                raise ValueError("AI display result must be a mapping")
            snapshot = result.get("snapshot")
            key = (result.get("frame_id"), result.get("captured_at"), id(snapshot),
                   id(getattr(snapshot, "render_metadata", None)), self.max_size)
            now = time.perf_counter() if now is None else now
            snapshot = result["snapshot"]
            frame_id, capture = result["frame_id"], result["captured_at"]
            if (not isinstance(snapshot, AISnapshot) or type(frame_id) is not int or frame_id < 1
                    or not finite(capture) or not finite(now) or now < capture
                    or snapshot.frame_id != frame_id or snapshot.captured_at != capture
                    or snapshot.observations != result["behavior_frame"]):
                raise ValueError("snapshot/JPEG/production observation metadata mismatch")
            completion = result.get("completed_at")
            if completion is not None and (not finite(completion) or not capture <= completion <= now):
                raise ValueError("completion timestamp invalid or outside capture/current time")
            if key == self._failed_key:
                return self._placeholder("Render unavailable; live preview continues")
            rendering = True
            self._source_snapshot = snapshot
            if self._key != key:
                rendered = render_snapshot(snapshot, expected_frame_id=frame_id, expected_captured_at=capture,
                                           max_size=self.max_size, person_only=True)
                rendered.validate(frame_id, capture)
                width, height = self.max_size
                if (rendered.image.dtype != np.uint8 or rendered.image.shape != (SNAPSHOT_HEADER + height, width, 3)):
                    raise ValueError("renderer output size/type mismatch")
                rendered = render_details(snapshot, rendered, max_size=self.max_size)
                rendered.validate(frame_id, capture)
                if (rendered.image.dtype != np.uint8 or rendered.image.ndim != 3 or rendered.image.shape[2] != 3
                        or rendered.image.shape[0] != SNAPSHOT_HEADER + height
                        or rendered.image.shape[1] not in {width, width + 540}):
                    raise ValueError("detail renderer output size/type mismatch")
                self._cached, self._key = rendered, key
            self._cached.validate(frame_id, capture)
            image = self._cached.image.copy()  # Only the header ages change on repaint.
            capture_age = now - capture
            completion_age = None if completion is None else now - completion
            status = []
            if capture_age > self.stale_after:
                status.append("capture")
            if completion_age is not None and completion_age > self.stale_after:
                status.append("completion")
            completion_text = "--" if completion_age is None else f"{completion_age:.1f}s"
            texts = [(f"Capture age: {capture_age:.1f}s | Completion age: {completion_text}", 59),
                     ("STALE: " + " + ".join(status) if status else "HISTORICAL observation; not the live camera frame", 75)]
            for text, y in texts:
                if text.startswith("STALE") and self.max_size[0] < 180:
                    text = "STALE"
                elif text.startswith("Capture age:") and self.max_size[0] < 350:
                    text = f"C:{capture_age:.1f}s Done:{completion_text}"
                cv2.putText(image, fit_marked(text, self.max_size[0] - 16, .38), (8, y),
                            cv2.FONT_HERSHEY_SIMPLEX, .38, (0, 220, 255), 1, cv2.LINE_AA)
            return image
        except Exception as exc:
            self._failure(key, exc, cache_failure=rendering)
            return self._placeholder("Render unavailable; live preview continues")

    def show(self, result, *, now=None):
        """GUI boundary is optional: its failure cannot stop Camera Live."""
        if self._display_failed:
            return False
        try:
            image = self.render(result, now=now)
            if not self._window_open:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
                self._window_open = True
            size = (image.shape[1], image.shape[0])
            if size != self._window_size:
                cv2.resizeWindow(WINDOW_NAME, *size)
                self._window_size = size
            cv2.imshow(WINDOW_NAME, image)
            return True
        except Exception as exc:
            self._display_failed = True
            print(f"[PERSON SNAPSHOT OVERLAY] window unavailable; live continues; "
                  f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            return False
