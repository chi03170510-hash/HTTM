"""Experimental, in-memory alerts. All behaviors are DISABLED by default.

No model, rendering, persistence, API or legacy timer dependencies. Qualified
observation spans are research signals, never official violation confirmation.
"""

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from behavior_duration import BEHAVIORS, BehaviorFrame


WARNING_SECONDS = 5.0
MAX_CAPTURE_AGE = 5.0
MAX_OBSERVATION_GAP = 5.0


@dataclass(frozen=True)
class WarningConfig:
    sleeping: bool = False
    using_phone: bool = False
    turning_back: bool = False
    eating: bool = False

    def __post_init__(self):
        if any(type(getattr(self, name)) is not bool for name in BEHAVIORS):
            raise ValueError("experimental behavior flags must be explicit booleans")

    @property
    def enabled_behaviors(self):
        return tuple(name for name in BEHAVIORS if getattr(self, name))


DEFAULT_WARNING_CONFIG = WarningConfig()


@dataclass(frozen=True)
class WarningEvent:
    episode_id: int
    person_track_id: str
    student_id: str
    behavior: str
    frame_id: int
    captured_at: float
    estimated_seconds: float
    qualified_seconds: float
    observation_count: int


@dataclass(frozen=True)
class WarningView:
    person_track_id: str
    student_id: str | None
    behavior: str
    state: str
    reason: str
    episode_id: int | None = None
    qualified_seconds: float = 0.0
    observation_count: int = 0
    event_emitted: bool = False


@dataclass(frozen=True)
class WarningFrame:
    frame_id: int
    captured_at: float
    evaluated_at: float
    enabled_behaviors: tuple[str, ...]
    views: tuple[WarningView, ...]
    events: tuple[WarningEvent, ...] = ()
    valid: bool = True

    def is_fresh(self, now, frame_id, captured_at):
        return (self.valid and self.frame_id == frame_id and self.captured_at == captured_at
                and _finite(now) and 0 <= now - self.captured_at <= MAX_CAPTURE_AGE)


@dataclass
class _Episode:
    episode_id: int
    student_id: str
    last_capture: float
    last_duration: float
    qualified_start: float | None = None
    observation_count: int = 0
    emitted: bool = False


def _finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _score(value):
    return _finite(value) and 0 <= value <= 1


def _student(value):
    return (isinstance(value, str) and bool(value.strip())
            and value.strip().casefold() not in {"unknown", "none", "null"})


def _box(value):
    return (isinstance(value, (list, tuple)) and len(value) == 4
            and all(_finite(v) for v in value) and value[2] > value[0] and value[3] > value[1])


def evidence_check(behavior, decision):
    """Validate existing V7 evidence contracts without lowering model thresholds."""
    if not isinstance(decision, dict) or decision.get("state") != "detected":
        return False, "missing_detected_evidence"
    if decision.get("status") != "IMAGE_ONLY" or decision.get("decision") != "detected":
        return False, "inconsistent_image_decision"
    support = decision.get("supporting_evidence")
    contradict = decision.get("contradicting_evidence")
    evidence = decision.get("evidence")
    if not isinstance(support, (list, tuple)) or not isinstance(contradict, (list, tuple)) or not isinstance(evidence, dict):
        return False, "missing_evidence"
    if contradict:
        return False, "contradicting_evidence"
    if evidence.get("missing_evidence", []):
        return False, "missing_evidence"
    if behavior == "sleeping":
        threshold = evidence.get("sleeping_yolo_threshold")
        valid = ("behavior_yolo_sleeping_candidate" in support
                 and _score(threshold) and threshold > 0
                 and _score(decision.get("model_confidence"))
                 and decision["model_confidence"] >= threshold
                 and _score(evidence.get("sleeping_yolo_confidence"))
                 and evidence["sleeping_yolo_confidence"] >= threshold
                 and _box(evidence.get("sleeping_yolo_box")))
    elif behavior == "using_phone":
        geometry = evidence.get("phone_object_geometry", {})
        limits = geometry.get("limits", {}) if isinstance(geometry, dict) else {}
        geometry_valid = isinstance(geometry, dict) and isinstance(limits, dict) and all(
            _finite(geometry.get(key)) and _finite(limits.get(key))
            and 0 <= geometry[key] <= limits[key] and limits[key] > 0
            for key in ("area_ratio", "height_ratio", "width_ratio")
        )
        boxes = evidence.get("phone_boxes")
        ids = evidence.get("phone_detection_ids")
        valid = ("behavior_yolo_using_phone_candidate" in support
                 and "unique_same_person_coco_phone_object" in support
                 and _score(decision.get("model_confidence"))
                 and evidence.get("phone_association_status") == "ASSOCIATED_GEOMETRICALLY"
                 and isinstance(ids, (list, tuple)) and bool(ids)
                 and all(isinstance(value, str) and bool(value.strip()) for value in ids)
                 and len(set(ids)) == len(ids)
                 and isinstance(boxes, (list, tuple)) and bool(boxes) and all(_box(b) for b in boxes)
                 and len(ids) == len(boxes)
                 and geometry_valid)
    elif behavior == "eating":
        linked = evidence.get("food_mouth_matches")
        valid = ("unique_same_person_supported_food_object" in support
                 and "food_box_near_visible_mouth" in support
                 and evidence.get("same_person_food_association") is True
                 and evidence.get("mouth_source") == "face_landmark"
                 and isinstance(linked, (list, tuple)) and bool(linked)
                 and all(isinstance(row, dict) and isinstance(row.get("food_detection_id"), str)
                         and bool(row["food_detection_id"].strip())
                         and isinstance(row.get("class_name"), str) and bool(row["class_name"].strip())
                         and _score(row.get("object_confidence"))
                         and _finite(row.get("mouth_distance_head_height"))
                         and _finite(row.get("proximity_threshold_head_height"))
                         and row["proximity_threshold_head_height"] > 0
                         and 0 <= row["mouth_distance_head_height"] <= row["proximity_threshold_head_height"]
                         for row in linked))
    elif behavior == "turning_back":
        # Current spatial V7 returns suspected/unknown here, so enabling this flag
        # alone cannot promote those states. Require the existing calibration data.
        valid = ("head_orientation_matches_calibrated_rear_direction" in support
                 and evidence.get("rear_direction_calibrated") is True
                 and all(_finite(evidence.get(key)) for key in
                         ("head_yaw_deg", "rear_yaw_target_deg", "rear_direction_tolerance_deg"))
                 and evidence["rear_direction_tolerance_deg"] > 0
                 and abs(evidence["head_yaw_deg"] - evidence["rear_yaw_target_deg"])
                 <= evidence["rear_direction_tolerance_deg"])
    else:
        return False, "unsupported_behavior"
    return (True, "qualified") if valid else (False, "insufficient_behavior_evidence")


class BehaviorWarningObserver:
    def __init__(self, config=DEFAULT_WARNING_CONFIG):
        if not isinstance(config, WarningConfig):
            raise TypeError("config must be an explicit WarningConfig")
        self.config = config
        self._episodes = {}
        self._next_episode = 1
        self._last_id = 0
        self._last_capture = None
        self._last_frame = None

    def _invalid(self, frame, now, reason):
        self._episodes.clear()
        if isinstance(frame.frame_id, int) and not isinstance(frame.frame_id, bool):
            self._last_id = max(self._last_id, frame.frame_id)
        views = tuple(WarningView(p.person_track_id, p.student_id, b.name, "CLEAR", reason)
                      for p in frame.people for b in p.behaviors)
        self._last_frame = WarningFrame(frame.frame_id, frame.captured_at, now,
                                        self.config.enabled_behaviors, views, valid=False)
        return self._last_frame

    def update(self, frame: BehaviorFrame, matches, *, observed_at):
        """Consume one AI observation; never call from a UI repaint.

        CH-20 episodes and the consecutive qualified span are separate. Evidence
        loss clears only the span, retaining the one-event latch in that episode.
        """
        if (not isinstance(frame.frame_id, int) or isinstance(frame.frame_id, bool) or frame.frame_id < 1
                or not frame.timestamp_valid or not _finite(frame.captured_at)
                or not _finite(observed_at) or observed_at < frame.captured_at):
            return self._invalid(frame, observed_at, "invalid_timestamp_or_metadata")
        if self._last_frame is not None and frame.frame_id == self._last_frame.frame_id:
            if frame.captured_at == self._last_frame.captured_at:
                # Cached status is safe, but an event must never be returned twice.
                return replace(self._last_frame, events=())
            return self._invalid(frame, observed_at, "duplicate_frame_metadata")
        if (frame.frame_id <= self._last_id or
                (self._last_capture is not None and frame.captured_at <= self._last_capture)):
            return self._invalid(frame, observed_at, "non_increasing_observation")
        self._last_id, self._last_capture = frame.frame_id, frame.captured_at

        person_counts = Counter(p.person_track_id for p in frame.people)
        student_counts = Counter(p.student_id for p in frame.people if _student(p.student_id))
        by_track, raw_claims = defaultdict(list), defaultdict(set)
        for match in matches:
            tid, sid = match.get("person_track_id"), match.get("student_id")
            by_track[tid].append(match)
            if tid is not None and _student(sid):
                raw_claims[sid].add(tid)
        next_episodes, views, events = {}, [], []
        fresh = observed_at - frame.captured_at <= MAX_CAPTURE_AGE
        for person in frame.people:
            tid, sid = person.person_track_id, person.student_id
            identity_ok = (isinstance(tid, str) and re.fullmatch(r"P[1-9]\d*", tid) is not None
                           and person_counts[tid] == 1 and _student(sid)
                           and student_counts[sid] == 1 and len(raw_claims[sid]) <= 1
                           and person.identity_status == "matched")
            candidates = by_track[tid]
            match = candidates[0] if len(candidates) == 1 else {}
            association_ok = (identity_ok and person.association_status == "matched"
                              and len(candidates) == 1 and person.face_track_id is not None
                              and match.get("association_status") == "matched"
                              and match.get("person_ref") == person.person_ref
                              and match.get("student_id") == sid
                              and match.get("track_id") == person.face_track_id)
            raw_behaviors = match.get("behaviors", {})
            if not isinstance(raw_behaviors, dict):
                raw_behaviors = {}
            for behavior in person.behaviors:
                name, duration = behavior.name, behavior.estimated_seconds
                if name not in self.config.enabled_behaviors:
                    views.append(WarningView(tid, sid, name, "CLEAR", "disabled"))
                    continue
                if not identity_ok or behavior.state != "detected" or not _finite(duration) or duration < 0:
                    views.append(WarningView(tid, sid, name, "CLEAR", "identity_or_behavior_invalid"))
                    continue
                key = (tid, name)
                episode = self._episodes.get(key)
                if (episode is None or episode.student_id != sid or duration == 0
                        or duration < episode.last_duration
                        or frame.captured_at - episode.last_capture > MAX_OBSERVATION_GAP):
                    episode = _Episode(self._next_episode, sid, frame.captured_at, duration)
                    self._next_episode += 1
                episode.last_capture, episode.last_duration = frame.captured_at, duration
                next_episodes[key] = episode
                qualified, reason = evidence_check(name, raw_behaviors.get(name))
                if not association_ok:
                    qualified, reason = False, "association_not_matched"
                if not fresh:
                    qualified, reason = False, "stale_capture"
                if not qualified:
                    episode.qualified_start, episode.observation_count = None, 0
                    views.append(WarningView(tid, sid, name, "CLEAR", reason, episode.episode_id,
                                             event_emitted=episode.emitted))
                    continue
                if episode.qualified_start is None:
                    episode.qualified_start = frame.captured_at
                episode.observation_count += 1
                span = frame.captured_at - episode.qualified_start
                state = "CANDIDATE"
                if span > WARNING_SECONDS and duration > WARNING_SECONDS:
                    state = "WARNING_EXPERIMENTAL"
                    if not episode.emitted:
                        events.append(WarningEvent(episode.episode_id, tid, sid, name, frame.frame_id,
                                                   frame.captured_at, duration, span, episode.observation_count))
                        episode.emitted = True
                views.append(WarningView(tid, sid, name, state, "qualified", episode.episode_id,
                                         span, episode.observation_count, episode.emitted))
        self._episodes = next_episodes
        self._last_frame = WarningFrame(frame.frame_id, frame.captured_at, observed_at,
                                        self.config.enabled_behaviors, tuple(views), tuple(events))
        return self._last_frame
