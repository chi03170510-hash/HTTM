"""Read-only identity hypotheses. Nothing in this module authorizes production use.

No hold period is configured in Camera Live. A finite simulation_hold_seconds
may be supplied explicitly by tests; it is never read from the environment.
Recognition IDs are the current worker's already-thresholded results. They are
not independent ground truth. Behaviors, images and embeddings are not inputs.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, replace

from person_tracker import calculate_iou


def known_student(value):
    return (value.strip() if isinstance(value, str) and value.strip()
            and value.strip().casefold() not in {"unknown", "none", "null"} else None)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def box(value):
    if not isinstance(value, (tuple, list)) or len(value) != 4 or not all(finite(v) for v in value):
        raise ValueError("shadow requires a finite xyxy box")
    result = tuple(float(v) for v in value)
    if result[2] <= result[0] or result[3] <= result[1]:
        raise ValueError("shadow requires a positive xyxy box")
    return result


@dataclass(frozen=True)
class PriorTrack:
    person_track_id: str
    bbox: tuple
    last_seen: float


@dataclass(frozen=True)
class CurrentPerson:
    person_track_id: str
    person_ref: str
    bbox: tuple


@dataclass(frozen=True)
class CurrentFace:
    bbox: tuple
    student_id: str
    score: float


@dataclass(frozen=True)
class CurrentAssociation:
    person_track_id: str
    person_ref: str
    face_track_id: int | str | None
    student_id: str | None
    status: str
    face_bbox: tuple | None
    face_score: float | None


@dataclass(frozen=True)
class IdentityObservation:
    person_track_id: str
    face_student_id: str | None
    proposed_student_id: str | None
    status: str
    identity_source: str
    association_status: str
    track_continuity: str
    face_present: bool | None
    reason: str
    last_face_age_seconds: float | None


@dataclass(frozen=True)
class ShadowFrame:
    frame_id: int
    captured_at: float | None
    tracker_updated_at: float | None
    timestamp_valid: bool
    simulation_hold_seconds: float | None
    reset_reason: str | None
    observations: tuple[IdentityObservation, ...]


@dataclass(frozen=True)
class _Anchor:
    student_id: str
    captured_at: float


def capture_prior_tracks(tracker):
    """Copy real tracker state before update; no tracker mutation or new clock."""
    return tuple(PriorTrack(f"P{key}", box(track.bbox), track.last_seen)
                 for key, track in tracker.tracks.items())


def project_current(tracked_people, faces, matches):
    """Whitelist scalars/tuples, excluding behaviors and all biometric payloads."""
    people = tuple(CurrentPerson(p["track_id"], p["person_ref"], box(p["bbox"])) for p in tracked_people)
    current_faces = (None if faces is None else tuple(
        CurrentFace(box(f["face_bbox"]), f["student_id"], f["score"]) for f in faces))
    associations = tuple(CurrentAssociation(
        m.get("person_track_id"), m.get("person_ref"), m.get("track_id"), m.get("student_id"),
        m.get("association_status", "unmatched"),
        box(m["face_bbox"]) if m.get("face_bbox") is not None else None, m.get("face_score")) for m in matches)
    return people, current_faces, associations


class IdentityContinuityResolver:
    def __init__(self, *, iou_threshold, max_update_gap, simulation_hold_seconds=None):
        if not finite(iou_threshold) or not 0 < iou_threshold <= 1:
            raise ValueError("invalid existing tracker IoU threshold")
        if not finite(max_update_gap) or max_update_gap <= 0:
            raise ValueError("invalid existing tracker max_age")
        if simulation_hold_seconds is not None and (
                not finite(simulation_hold_seconds) or simulation_hold_seconds <= 0):
            raise ValueError("simulation_hold_seconds must be explicitly finite and positive")
        self.iou_threshold, self.max_update_gap = iou_threshold, max_update_gap
        self.simulation_hold_seconds = simulation_hold_seconds
        self._anchors, self._last_faces, self._history = {}, {}, {}
        self._last_capture = self._last_update = self._last_frame = None
        self._last_valid_id = None

    def _age(self, tid, timestamp):
        anchor = self._last_faces.get(tid)
        return max(0., timestamp - anchor.captured_at) if anchor and finite(timestamp) else None

    def update(self, frame_id, captured_at, tracker_updated_at, before, people, faces, matches):
        """Consume immutable projections. None faces means unavailable, not absent."""
        if (self._last_frame is not None and frame_id == self._last_frame.frame_id
                and captured_at == self._last_frame.captured_at
                and tracker_updated_at == self._last_frame.tracker_updated_at):
            return self._last_frame
        valid = (isinstance(frame_id, int) and not isinstance(frame_id, bool) and frame_id > 0
                 and finite(captured_at) and finite(tracker_updated_at)
                 and captured_at <= tracker_updated_at
                 and all(finite(p.last_seen) and p.last_seen <= tracker_updated_at for p in before)
                 and (self._last_valid_id is None or frame_id > self._last_valid_id)
                 and (self._last_capture is None or captured_at > self._last_capture)
                 and (self._last_update is None or tracker_updated_at > self._last_update))
        reset = None
        if not valid:
            reset = "invalid_or_non_increasing_timestamp_or_frame"
            self._anchors.clear()
            self._last_faces.clear()
            self._history.clear()
        elif self._last_capture is not None and captured_at - self._last_capture > self.max_update_gap:
            reset = "capture_gap_exceeds_existing_limit"
            self._anchors.clear()
        if valid:
            self._last_capture, self._last_update = captured_at, tracker_updated_at
            self._last_valid_id = frame_id

        previous = {p.person_track_id: p for p in before}
        counts = Counter(p.person_track_id for p in people)
        refs = Counter(p.person_ref for p in people)
        current = {p.person_track_id: p for p in people}
        by_track, face_claims, sid_claims = defaultdict(list), Counter(), defaultdict(set)
        for m in matches:
            by_track[m.person_track_id].append(m)
            if m.face_track_id is not None:
                face_claims[m.face_track_id] += 1
            if known_student(m.student_id):
                sid_claims[known_student(m.student_id)].add(m.person_track_id)
        raw_sid_counts = Counter(known_student(f.student_id) for f in faces or () if known_student(f.student_id))
        blocked_sids = {sid for sid, tids in sid_claims.items() if len(tids) > 1}
        blocked_sids.update(sid for sid, count in raw_sid_counts.items() if count > 1)
        expired = {tid for tid, p in previous.items() if valid and (
            not finite(p.last_seen) or tracker_updated_at - p.last_seen > self.max_update_gap)}
        edges = {(old.person_track_id, new.person_track_id) for old in before for new in people
                 if old.person_track_id not in expired
                 and calculate_iou(old.bbox, new.bbox) >= self.iou_threshold}
        old_degrees = Counter(a for a, _ in edges)
        new_degrees = Counter(b for _, b in edges)
        rows, pending, next_anchors, direct_owners = {}, {}, {}, {}

        def observation(person, status, reason, *, sid=None, source="none", present=None, face_sid=None):
            tid = person.person_track_id
            association = by_track[tid][0].status if len(by_track[tid]) == 1 else (
                "ambiguous" if by_track[tid] else "unmatched")
            continuity = "expired" if tid in expired else "same_track" if tid in previous else "new_track"
            return IdentityObservation(tid, face_sid, sid, status, source, association, continuity,
                                       present, reason, self._age(tid, captured_at))

        # Resolve ALL direct face claims first, so cached owners cannot win by iteration order.
        for tid, person in current.items():
            ms = by_track[tid]
            relevant = [f for f in faces or () if calculate_iou(f.bbox, person.bbox) > 0
                        or any(m.face_bbox == f.bbox for m in ms)]
            present = None if faces is None else bool(relevant)
            face_sid = relevant[0].student_id if len(relevant) == 1 else None
            row = lambda status, reason: observation(person, status, reason, present=present, face_sid=face_sid)
            if not valid:
                rows[tid] = row("unknown", reset)
            elif counts[tid] != 1 or refs[person.person_ref] != 1 or len(ms) > 1:
                rows[tid] = row("ambiguous", "duplicate_track_reference_or_association")
            elif faces is None:
                rows[tid] = row("unknown", "current_face_data_unavailable")
            elif len(relevant) > 1:
                rows[tid] = row("ambiguous", "multiple_current_faces_for_person")
            elif relevant and not known_student(face_sid):
                rows[tid] = row("unknown", "current_face_unknown")
            elif ms:
                m = ms[0]
                sid = known_student(m.student_id)
                if sid in blocked_sids or (m.face_track_id is not None and face_claims[m.face_track_id] > 1):
                    rows[tid] = row("ambiguous", "duplicate_student_or_face_claim")
                elif m.person_ref != person.person_ref or m.status != "matched":
                    rows[tid] = row("ambiguous" if m.status != "matched_v4_heuristic" else "unknown",
                                    "heuristic_association_cannot_seed" if m.status == "matched_v4_heuristic"
                                    else "association_not_unique_or_matched")
                elif (sid is None or m.face_track_id is None or len(relevant) != 1
                      or relevant[0].bbox != m.face_bbox or known_student(face_sid) != sid
                      or not finite(relevant[0].score) or relevant[0].score != m.face_score):
                    rows[tid] = row("unknown", "current_recognition_not_verified")
                else:
                    recovered = sid in self._history and self._history[sid] != tid
                    anchor = _Anchor(sid, captured_at)
                    next_anchors[tid] = self._last_faces[tid] = anchor
                    direct_owners[sid] = tid
                    self._history[sid] = tid
                    rows[tid] = replace(observation(
                        person, "face_recovered" if recovered else "face_confirmed",
                        "fresh_face_on_new_track" if recovered else "unique_current_face_match",
                        sid=sid, source="current_face_recognition", present=True, face_sid=face_sid),
                        last_face_age_seconds=0.)
            elif present:
                rows[tid] = row("ambiguous", "current_face_without_unique_association")
            else:
                pending[tid] = person

        unlocalized_face = any(not any(calculate_iou(f.bbox, p.bbox) > 0 for p in people) for f in faces or ())
        for tid, person in pending.items():
            anchor = self._anchors.get(tid)
            competing = any(other.person_track_id != tid and calculate_iou(person.bbox, other.bbox) > 0
                            for other in people)
            ambiguous = (competing or old_degrees[tid] > 1 or new_degrees[tid] > 1)
            same = ((tid, tid) in edges and tid not in expired)
            reason, status = None, "unknown"
            if unlocalized_face:
                reason, status = "unlocalized_current_face_evidence", "ambiguous"
            elif ambiguous:
                reason, status = "spatial_assignment_ambiguous", "ambiguous"
            elif tid in expired:
                reason = "track_expired"
            elif anchor and (anchor.student_id in blocked_sids or anchor.student_id in direct_owners
                             or raw_sid_counts[anchor.student_id] > 0):
                reason, status = "student_claimed_elsewhere", "ambiguous"
            elif reset:
                reason = reset
            elif anchor is None or not same:
                reason = "no_valid_same_track_anchor"
            elif self.simulation_hold_seconds is None:
                reason = "hold_period_unapproved"
            elif captured_at - anchor.captured_at > self.simulation_hold_seconds:
                reason = "identity_hold_expired"
            if reason:
                rows[tid] = observation(person, status, reason, present=None if unlocalized_face else False)
            else:
                next_anchors[tid] = anchor  # Never refresh the direct-face timestamp from a bbox.
                rows[tid] = observation(person, "track_retained", "simulation_only_same_track",
                                        sid=anchor.student_id, source="person_track_simulation", present=False)

        for tid in sorted(set(self._anchors) | set(previous)):
            if tid not in current:
                rows[tid] = IdentityObservation(
                    tid, None, None, "unknown", "none", "unmatched", "expired" if tid in expired else "missing",
                    None, "track_expired" if tid in expired else "person_detection_missing", self._age(tid, captured_at))
        # Final invariant applies across direct and simulated proposals together.
        proposed = Counter(r.proposed_student_id for r in rows.values() if r.proposed_student_id)
        for tid, r in tuple(rows.items()):
            if r.proposed_student_id and proposed[r.proposed_student_id] > 1:
                rows[tid] = replace(r, proposed_student_id=None, status="ambiguous", identity_source="none",
                                    reason="duplicate_student_proposal")
                next_anchors.pop(tid, None)
        self._anchors = next_anchors
        self._last_faces = {tid: anchor for tid, anchor in self._last_faces.items() if tid in current}
        self._last_frame = ShadowFrame(
            frame_id, captured_at if finite(captured_at) else None,
            tracker_updated_at if finite(tracker_updated_at) else None, bool(valid),
            self.simulation_hold_seconds, reset, tuple(rows.values()))
        return self._last_frame
