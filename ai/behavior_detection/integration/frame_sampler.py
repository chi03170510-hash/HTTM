"""Timestamp-based sampler. The caller chooses the target FPS."""

class FrameSampler:
    def __init__(self, target_fps: float):
        if target_fps <= 0:
            raise ValueError("target_fps must be greater than zero")
        self.interval_ms = 1000.0 / float(target_fps)
        self._start_ms = None
        self._next_ms = None

    def should_sample(self, timestamp_ms: int) -> bool:
        timestamp_ms = int(timestamp_ms)
        if self._start_ms is None:
            self._start_ms = timestamp_ms
            self._next_ms = timestamp_ms
        if timestamp_ms + 1e-6 < self._next_ms:
            return False
        elapsed = max(0, timestamp_ms - self._start_ms)
        steps = int(elapsed // self.interval_ms) + 1
        self._next_ms = self._start_ms + steps * self.interval_ms
        return True
