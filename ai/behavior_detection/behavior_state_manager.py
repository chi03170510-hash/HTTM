
import time
from dataclasses import dataclass


@dataclass
class BehaviorState:
    track_id: int
    status: str = "NORMAL"
    behavior: str | None = None
    confidence: float = 0.0
    consecutive_hits: int = 0
    last_behavior_seen: float = 0.0
    last_face_seen: float = 0.0
    last_inference_id: int = -1
    status_before_lost: str = "NORMAL"

    # Evidence for a competing behavior
    candidate_behavior: str | None = None
    candidate_hits: int = 0

    # True only when the latest YOLO inference
    # supports the active confirmed behavior.
    evidence_active: bool = False


class BehaviorStateManager:

    def __init__(
        self,
        min_confirm_frames=3,
        max_behavior_missing=0.8,
        max_face_missing=1.5
    ):
        self.min_confirm_frames = min_confirm_frames
        self.max_behavior_missing = max_behavior_missing
        self.max_face_missing = max_face_missing
        self.states = {}

    def update(
        self,
        visible_tracks,
        matches,
        inference_id,
        now=None
    ):
        if now is None:
            now = time.monotonic()

        visible_ids = set(visible_tracks)

        for track_id in visible_ids:
            if track_id not in self.states:
                self.states[track_id] = BehaviorState(
                    track_id=track_id,
                    last_face_seen=now
                )

            state = self.states[track_id]
            state.last_face_seen = now

            if state.status == "LOST":
                state.status = state.status_before_lost

        for track_id in visible_ids:
            state = self.states[track_id]

            if state.last_inference_id == inference_id:
                continue

            state.last_inference_id = inference_id
            detection = matches.get(track_id)

            if detection is None:
                state.evidence_active = False
                state.candidate_behavior = None
                state.candidate_hits = 0
                continue

            label = detection["label"]
            confidence = float(detection["confidence"])

            if state.behavior is None:
                state.behavior = label
                state.consecutive_hits = 1
                state.confidence = confidence
                state.last_behavior_seen = now
                state.status = "SUSPECTED"
                state.evidence_active = True
                continue

            if label == state.behavior:
                state.candidate_behavior = None
                state.candidate_hits = 0
                state.consecutive_hits += 1
                state.confidence = confidence
                state.last_behavior_seen = now
                state.evidence_active = True

                if state.consecutive_hits >= self.min_confirm_frames:
                    state.status = "CONFIRMED"
                else:
                    state.status = "SUSPECTED"

            else:
                # Different prediction: do not switch immediately.
                state.evidence_active = False

                if state.candidate_behavior != label:
                    state.candidate_behavior = label
                    state.candidate_hits = 1
                else:
                    state.candidate_hits += 1

                if state.candidate_hits >= self.min_confirm_frames:
                    state.behavior = label
                    state.confidence = confidence
                    state.consecutive_hits = (
                        state.candidate_hits
                    )
                    state.last_behavior_seen = now
                    state.status = "CONFIRMED"
                    state.evidence_active = True
                    state.candidate_behavior = None
                    state.candidate_hits = 0

        # Timeouts run on every displayed frame.
        for track_id, state in list(self.states.items()):

            if track_id not in visible_ids:
                if now - state.last_face_seen > self.max_face_missing:
                    del self.states[track_id]
                    continue

                if state.status != "LOST":
                    state.status_before_lost = state.status
                    state.status = "LOST"

                state.evidence_active = False
                continue

            if (
                state.behavior is not None
                and now - state.last_behavior_seen
                > self.max_behavior_missing
            ):
                state.status = "NORMAL"
                state.behavior = None
                state.confidence = 0.0
                state.consecutive_hits = 0
                state.candidate_behavior = None
                state.candidate_hits = 0
                state.evidence_active = False

        return self.states

    def get_state(self, track_id):
        return self.states.get(track_id)
