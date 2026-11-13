"""Estimated spans of consecutive image observations, with no warning side effects.

One instance owns one camera session. Only AI observations advance it; rendering
receives frozen values. These estimates are not temporal violation confirmation.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass


BEHAVIORS = ("sleeping", "using_phone", "turning_back", "eating")
STATES = frozenset(("detected", "suspected", "unknown", "not_detected"))
MATCHED_STATUSES = frozenset(("matched", "matched_v4_heuristic"))


@dataclass(frozen=True)
class BehaviorDuration:
    name: str
    state: str
    estimated_seconds: float | None


@dataclass(frozen=True)
class PersonBehaviorView:
    person_track_id: str
    person_ref: str
    bbox: tuple
    student_id: str | None
    face_track_id: int | str | None
    association_status: str
    identity_status: str
    behaviors: tuple[BehaviorDuration, ...]


@dataclass(frozen=True)
class BehaviorFrame:
    frame_id: int
    captured_at: float
    people: tuple[PersonBehaviorView, ...]
    timestamp_valid: bool = True


@dataclass(frozen=True)
class _Episode:
    student_id: str
    started_at: float
    last_seen: float


def _student_id(value):
    if not isinstance(value, str):
        return None
    value = value.strip()
    if not value or value.casefold() in {"unknown", "none", "null"}:
        return None
    return value


class BehaviorDurationTracker:
    def __init__(self, max_gap=5.0, anchor_sleeping_identity=False):
        if not math.isfinite(max_gap) or max_gap <= 0:
            raise ValueError("max_gap must be finite and positive")
        self.max_gap = max_gap
        self.anchor_sleeping_identity = bool(anchor_sleeping_identity)
        self._episodes = {}
        self._identity_anchors = {}
        self._last_frame = None
        self._last_capture = None

    def update(self, frame_id, captured_at, people, tracked_people, matches):
        """Consume a new AI frame; return all tracks, including unassociated ones.

        Duplicate frame IDs reuse the original immutable result. Older IDs are
        rejected. Non-increasing capture time ends all episodes and exposes no
        duration; the next increasing timestamp starts from zero.
        """
        if not isinstance(frame_id, int) or isinstance(frame_id, bool) or frame_id < 1:
            raise ValueError("frame_id must be a positive integer")
        if self._last_frame is not None:
            if frame_id == self._last_frame.frame_id:
                return self._last_frame
            if frame_id < self._last_frame.frame_id:
                raise ValueError("out-of-order AI frame")
        captured_at = float(captured_at)
        if not math.isfinite(captured_at):
            raise ValueError("captured_at must be finite")
        timestamp_valid = self._last_capture is None or captured_at > self._last_capture
        if (not timestamp_valid or (self._last_capture is not None
                                    and captured_at - self._last_capture > self.max_gap)):
            self._episodes.clear()
            self._identity_anchors.clear()
        if timestamp_valid:
            self._last_capture = captured_at

        by_ref = defaultdict(list)
        for person in people:
            by_ref[person["person_ref"]].append(person)
        track_counts = Counter(person["track_id"] for person in tracked_people)
        ref_counts = Counter(person["person_ref"] for person in tracked_people)
        by_track = defaultdict(list)
        student_claims = defaultdict(set)
        face_claims = defaultdict(set)
        for match in matches:
            tid = match.get("person_track_id")
            by_track[tid].append(match)
            if tid is not None and match.get("association_status") in MATCHED_STATUSES:
                sid = _student_id(match.get("student_id"))
                if sid is not None:
                    student_claims[sid].add(tid)
                if match.get("track_id") is not None:
                    face_claims[match["track_id"]].add(tid)

        next_episodes = {}
        views = []
        for track in tracked_people:
            tid, ref = track["track_id"], track["person_ref"]
            candidates = by_track[tid]
            source = by_ref[ref][0] if len(by_ref[ref]) == 1 else {}
            match = candidates[0] if len(candidates) == 1 else {}
            status = match.get("association_status", "unmatched")
            sid = _student_id(match.get("student_id"))
            identity_status = "matched"
            if (track_counts[tid] != 1 or ref_counts[ref] != 1 or len(by_ref[ref]) != 1
                    or len(candidates) > 1):
                identity_status = "ambiguous_association"
            elif not match or status not in MATCHED_STATUSES or match.get("person_ref") != ref:
                identity_status = "unassociated"
            elif sid is None:
                identity_status = "unknown"
            elif len(student_claims[sid]) > 1:
                identity_status = "ambiguous_student"
            elif len(face_claims[match.get("track_id")]) > 1:
                identity_status = "ambiguous_face"
            if identity_status != "matched":
                sid = None
                if (
                    self.anchor_sleeping_identity
                    and timestamp_valid
                    and identity_status in ("unassociated", "unknown")
                    and (source.get("behaviors", {}).get("sleeping", {}) or {}).get("state") == "detected"
                ):
                    anchor = self._identity_anchors.get(tid)
                    if anchor is not None and captured_at - anchor["last_seen"] <= self.max_gap:
                        ab = anchor["bbox"]
                        tb = tuple(track["bbox"])
                        x1, y1 = max(ab[0], tb[0]), max(ab[1], tb[1])
                        x2, y2 = min(ab[2], tb[2]), min(ab[3], tb[3])
                        inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                        union = (max(0.0, ab[2] - ab[0]) * max(0.0, ab[3] - ab[1])
                                 + max(0.0, tb[2] - tb[0]) * max(0.0, tb[3] - tb[1]) - inter)
                        iou = inter / union if union > 0 else 0.0
                        anchored_sid = anchor["student_id"]
                        if iou >= 0.65 and len(student_claims[anchored_sid]) == 0:
                            sid = anchored_sid
                            identity_status = "anchored_sleeping"
                            self._identity_anchors[tid] = {
                                "student_id": sid,
                                "bbox": tb,
                                "last_seen": captured_at,
                            }
            elif timestamp_valid and sid is not None:
                self._identity_anchors[tid] = {
                    "student_id": sid,
                    "bbox": tuple(track["bbox"]),
                    "last_seen": captured_at,
                }

            durations = []
            for name in BEHAVIORS:
                decision = source.get("behaviors", {}).get(name, {})
                state = decision.get("state", "unknown") if isinstance(decision, dict) else "unknown"
                if state not in STATES:
                    state = "unknown"
                elapsed = None
                if timestamp_valid and sid is not None and state == "detected":
                    previous = self._episodes.get((tid, name))
                    if (previous is None or previous.student_id != sid
                            or captured_at - previous.last_seen > self.max_gap):
                        previous = _Episode(sid, captured_at, captured_at)
                    elapsed = captured_at - previous.started_at
                    next_episodes[(tid, name)] = _Episode(sid, previous.started_at, captured_at)
                durations.append(BehaviorDuration(name, state, elapsed))
            views.append(PersonBehaviorView(
                tid, ref, tuple(track["bbox"]), sid, match.get("track_id"),
                status, identity_status, tuple(durations),
            ))

        # Anything absent, non-detected, or ambiguously identified ends here.
        self._episodes = next_episodes
        self._last_frame = BehaviorFrame(frame_id, captured_at, tuple(views), timestamp_valid)
        return self._last_frame


def duration_text(behavior):
    return "--" if behavior.estimated_seconds is None else f"~{behavior.estimated_seconds:.1f}s"


def behavior_label(name):
    return name.replace("_", " ").title()
