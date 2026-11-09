
import time
from dataclasses import dataclass


@dataclass
class TimerState:
    track_id: int
    behavior: str
    start_time: float
    elapsed: float = 0.0
    warned: bool = False
    paused: bool = False
    last_update: float = 0.0


class BehaviorTimer:

    def __init__(
        self,
        warning_seconds=5.0,
        max_count_interval=0.5
    ):
        self.warning_seconds = warning_seconds
        self.max_count_interval = max_count_interval
        self.timers = {}

    def update(self, states, now=None):
        if now is None:
            now = time.monotonic()

        warnings = []
        active_ids = set()

        for track_id, state in states.items():

            if (
                state.status != "CONFIRMED"
                or state.behavior is None
            ):
                continue

            active_ids.add(track_id)

            timer = self.timers.get(track_id)

            if (
                timer is None
                or timer.behavior != state.behavior
            ):
                timer = TimerState(
                    track_id=track_id,
                    behavior=state.behavior,
                    start_time=now,
                    last_update=now
                )
                self.timers[track_id] = timer

            delta = max(0.0, now - timer.last_update)

            # Count only when current evidence supports
            # the active confirmed behavior.
            evidence_active = getattr(
                state,
                "evidence_active",
                True
            )

            if evidence_active:
                timer.elapsed += min(
                    delta,
                    self.max_count_interval
                )
                timer.paused = False
            else:
                timer.paused = True

            timer.last_update = now

            if (
                timer.elapsed >= self.warning_seconds
                and not timer.warned
            ):
                timer.warned = True

                warnings.append({
                    "track_id": track_id,
                    "behavior": timer.behavior,
                    "elapsed": timer.elapsed,
                    "timestamp": now
                })

        # Reset if no longer confirmed.
        for track_id in list(self.timers):
            if track_id not in active_ids:
                del self.timers[track_id]

        return self.timers, warnings

    def get_timer(self, track_id):
        return self.timers.get(track_id)
