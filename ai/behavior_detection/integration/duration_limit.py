"""Source-timeline duration guard for sampled video processing."""


class DurationLimit:
    """Allow frames before a duration boundary measured from the first frame."""

    def __init__(self, max_duration_s=None):
        if max_duration_s is not None and max_duration_s <= 0:
            raise ValueError("max_duration_s must be greater than zero")
        self.max_duration_ms = (
            None if max_duration_s is None else float(max_duration_s) * 1000.0
        )
        self._start_ms = None
        self._expired = False

    def allows(self, timestamp_ms: int) -> bool:
        """Return whether a frame timestamp is inside the configured time window."""
        if self.max_duration_ms is None:
            return True
        if self._expired:
            return False

        timestamp_ms = int(timestamp_ms)
        if self._start_ms is None:
            self._start_ms = timestamp_ms
        if timestamp_ms - self._start_ms >= self.max_duration_ms:
            self._expired = True
            return False
        return True
