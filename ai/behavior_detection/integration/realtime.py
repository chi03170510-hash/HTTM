"""Latest-frame inference worker and wall-clock playback/FPS helpers.

OpenCV capture and GUI stay on the caller's thread. The worker owns inference
and its optional result sink; neither work queue nor result history is built up.
"""

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import math
import threading
import time
from typing import Any

from ..diagnostics import emit_debug
from .event_schema import FrameMetadata
from .video_source import FramePacket


class RateMeter:
    """Measure completed operations over a recent wall-clock window."""

    def __init__(self, window_seconds=5.0):
        self.window_seconds = window_seconds
        self._times = deque()

    def record(self, now):
        self._times.append(now)
        self._prune(now)

    def _prune(self, now):
        while self._times and self._times[0] < now - self.window_seconds:
            self._times.popleft()

    def fps(self, now):
        self._prune(now)
        if len(self._times) < 2:
            return 0.0
        elapsed = now - self._times[0]
        return (len(self._times) - 1) / elapsed if elapsed > 0 else 0.0


class FramePacer:
    """Schedule file frames against a fixed monotonic origin, without drift."""

    def __init__(self, fps, clock=time.monotonic):
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("playback fps must be finite and greater than zero")
        self.fps = fps
        self._clock = clock
        self._origin = None
        self._first_index = None

    def remaining(self, frame_index):
        now = self._clock()
        if self._origin is None:
            self._origin = now
            self._first_index = frame_index
        deadline = self._origin + (frame_index - self._first_index) / self.fps
        return max(0.0, deadline - now)


@dataclass(frozen=True)
class InferenceSnapshot:
    result: Any
    metadata: FrameMetadata
    captured_at: float
    started_at: float
    completed_at: float

    @property
    def latency_ms(self):
        return (self.completed_at - self.started_at) * 1000.0

    def is_fresh(self, metadata, now, ttl_seconds):
        """Age starts at frame capture, including inference time, not completion."""
        return (
            metadata.source_id == self.metadata.source_id
            and metadata.frame_width == self.metadata.frame_width
            and metadata.frame_height == self.metadata.frame_height
            and metadata.frame_index >= self.metadata.frame_index
            and 0 <= now - self.captured_at <= ttl_seconds
            and 0 <= metadata.timestamp_ms - self.metadata.timestamp_ms <= ttl_seconds * 1000
        )


class LatestFrameWorker:
    """One inference at a time, at most target_fps starts per second.

    New frames replace the single pending frame. Slow inference never triggers
    catch-up jobs. All array copies and inference run outside the shared lock.
    stop() waits for the current native inference/sink call before returning.
    """

    def __init__(self, process, target_fps=1.0, on_result=None, clock=time.monotonic,
                 *, debug_detections=False):
        if not math.isfinite(target_fps) or target_fps <= 0:
            raise ValueError("target_fps must be finite and greater than zero")
        self._process = process
        self._interval = 1.0 / target_fps
        self._on_result = on_result
        self._clock = clock
        self._condition = threading.Condition(threading.Lock())
        self._pending = None
        self._snapshot = None
        self._stopping = False
        self._error = None
        self._thread = None
        self._started_at = None
        self._stopped_at = None
        self._processed_frames = 0
        self._inference_seconds = 0.0
        self._rate = RateMeter()
        self._debug_detections = debug_detections
        self._first_inference_ms = None

    def start(self):
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("AI worker can only be started once")
            self._started_at = self._clock()
            self._thread = threading.Thread(target=self._run, name="HTTM-AI", daemon=False)
            self._thread.start()
        return self

    def publish(self, packet, captured_at=None):
        captured_at = self._clock() if captured_at is None else captured_at
        owned_packet = FramePacket(frame=packet.frame.copy(), metadata=packet.metadata)
        with self._condition:
            if not self._stopping:
                self._pending = (owned_packet, captured_at)
                self._condition.notify()

    def latest(self):
        with self._condition:
            snapshot = self._snapshot
        # Callers cannot mutate a dictionary result retained by the worker.
        return deepcopy(snapshot)

    def stats(self):
        now = self._clock()
        with self._condition:
            elapsed = ((self._stopped_at or now) - self._started_at
                       if self._started_at is not None else 0.0)
            return {
                "processed_frames": self._processed_frames,
                "inference_seconds": self._inference_seconds,
                "ai_fps": self._rate.fps(now),
                "mean_ai_fps": self._processed_frames / elapsed if elapsed > 0 else 0.0,
                "first_inference_ms": self._first_inference_ms,
            }

    def raise_if_failed(self):
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError(f"AI worker failed: {error}") from error

    def stop(self):
        with self._condition:
            self._stopping = True
            self._pending = None
            self._condition.notify_all()
            thread = self._thread
        if thread is not None:
            thread.join()

    def _run(self):
        last_key = None
        next_start = float("-inf")
        previous_started = None
        previous_metadata = None
        try:
            while True:
                with self._condition:
                    while not self._stopping:
                        now = self._clock()
                        pending = self._pending
                        key = ((pending[0].metadata.source_id, pending[0].metadata.frame_index)
                               if pending is not None else None)
                        if pending is not None and key != last_key and now >= next_start:
                            break
                        timeout = max(0.0, next_start - now) if key != last_key and pending else None
                        self._condition.wait(timeout=timeout)
                    if self._stopping:
                        return
                    packet, captured_at = pending
                frame = packet.frame.copy()
                started = self._clock()
                next_start = started + self._interval
                last_key = key
                result = self._process(frame, packet.metadata)
                completed = self._clock()
                snapshot = InferenceSnapshot(
                    deepcopy(result), packet.metadata, captured_at, started, completed
                )
                with self._condition:
                    self._snapshot = snapshot
                    self._processed_frames += 1
                    self._inference_seconds += completed - started
                    if self._first_inference_ms is None:
                        self._first_inference_ms = (completed - started) * 1000
                    self._rate.record(completed)
                output_started = self._clock()
                if self._on_result is not None:
                    self._on_result(result, packet.metadata)
                if self._debug_detections:
                    emit_debug(
                        "ai_sample_timing", metadata=packet.metadata.__dict__,
                        target_interval_ms=self._interval * 1000,
                        source_gap_ms=(packet.metadata.timestamp_ms - previous_metadata.timestamp_ms
                                       if previous_metadata is not None else None),
                        skipped_source_frames=(max(0, packet.metadata.frame_index - previous_metadata.frame_index - 1)
                                               if previous_metadata is not None else 0),
                        start_gap_ms=((started - previous_started) * 1000
                                      if previous_started is not None else None),
                        input_age_at_start_ms=max(0.0, (started - captured_at) * 1000),
                        processing_ms=(completed - started) * 1000,
                        result_output_ms=(self._clock() - output_started) * 1000,
                    )
                previous_started = started
                previous_metadata = packet.metadata
        except Exception as exc:
            with self._condition:
                self._error = exc
                self._stopping = True
                self._condition.notify_all()
        finally:
            with self._condition:
                self._stopped_at = self._clock()
