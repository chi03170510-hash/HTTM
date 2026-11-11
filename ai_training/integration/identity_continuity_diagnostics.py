"""Opt-in shadow-only JSONL; never changes production results or writes images."""

import json
import os
import sys
import uuid
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from identity_continuity import IdentityContinuityResolver, capture_prior_tracks, finite, project_current


SHADOW_ENV = "HTTM_IDENTITY_SHADOW"
SHADOW_DIR_ENV = "HTTM_IDENTITY_SHADOW_DIR"
MAX_DETAILS = 100
MAX_BYTES = 1024 * 1024
DIAGNOSTIC_BEHAVIORS = ("sleeping", "using_phone")


def shadow_from_environment(root, tracker, max_frames):
    if os.environ.get(SHADOW_ENV) != "1":
        return None
    output_dir = os.environ.get(SHADOW_DIR_ENV, str(Path(root) / "ai_training/runs/identity_continuity_shadow"))
    try:
        return IdentityContinuityShadow(
            output_dir, iou_threshold=tracker.iou_threshold, max_update_gap=tracker.max_age, max_frames=max_frames)
    except Exception as exc:
        print(f"[IDENTITY SHADOW] initialization failed; AI continues; output_dir={output_dir}; "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        return None


def behavior_diagnostic(frame, raw_people, previous, max_gap, shadow_frame, *, names=DIAGNOSTIC_BEHAVIORS):
    """Classify frozen production outputs; never run or modify a duration timer.

    Reset reasons are diagnostic inferences from adjacent logged observations,
    not access to DurationTracker internals or proof of physical identity.
    """
    sources = defaultdict(list)
    for person in raw_people:
        sources[person.get("person_ref")].append(person)
    old_people = {p.person_track_id: p for p in previous.people} if previous is not None else {}
    shadow_people = {p.person_track_id: p for p in shadow_frame.observations}
    gap = frame.captured_at - previous.captured_at if previous is not None else None
    duplicate = previous is not None and frame.frame_id == previous.frame_id
    result = {"duration_capture_gap_seconds": gap, "duration_max_gap": max_gap,
              "production_timestamp_valid": frame.timestamp_valid,
              "previous_logged_frame_id": previous.frame_id if previous is not None else None,
              "duration_missing_person_track_ids": sorted(set(old_people) - {p.person_track_id for p in frame.people}),
              "duration_reason_source": "diagnostic_classification_of_production_outputs"}
    for name in names:
        result[name] = _behavior_rows(name, frame, sources, old_people, shadow_people, previous, gap, max_gap, duplicate)
    return result


def _behavior_rows(name, frame, sources, old_people, shadow_people, previous, gap, max_gap, duplicate):
    rows = []
    for person in frame.people:
        value = next((b for b in person.behaviors if b.name == name), None)
        state = value.state if value is not None else None
        seconds = value.estimated_seconds if value is not None else None
        source = sources[person.person_ref]
        decisions = source[0].get("behaviors") if len(source) == 1 else None
        decision = decisions.get(name, {}) if isinstance(decisions, dict) else {}
        decision = decision if isinstance(decision, dict) else {}
        confidence = decision.get("model_confidence")
        confidence = confidence if finite(confidence) else None
        shadow = shadow_people.get(person.person_track_id)
        prior = old_people.get(person.person_track_id)
        prior_behavior = (next((b for b in prior.behaviors if b.name == name), None)
                          if prior is not None else None)
        blockers, resets = [], []
        if not frame.timestamp_valid:
            blockers.append("capture_timestamp_invalid")
        if person.identity_status != "matched":
            blockers.append(f"identity_{person.identity_status}")
        elif person.student_id is None:
            blockers.append("student_id_missing")
        if state != "detected":
            blockers.append(f"{name}_observation_missing" if state is None else f"{name}_{state}")
        if previous is not None:
            if finite(gap) and finite(max_gap) and gap > max_gap:
                resets.append("capture_gap_exceeded")
            if not previous.timestamp_valid:
                resets.append("previous_capture_timestamp_invalid")
            if prior is not None:
                if prior.student_id != person.student_id:
                    resets.append("reported_student_id_changed")
                if prior.identity_status != "matched":
                    resets.append("previous_identity_ineligible")
                if prior_behavior is None or prior_behavior.state != "detected":
                    resets.append(f"previous_{name}_not_detected")
            else:
                resets.append("new_person_track")
                if person.student_id is not None and any(
                        p.student_id == person.student_id for p in previous.people):
                    resets.append("person_track_changed_for_reported_student")
        if duplicate:
            event = "duplicate_observation"
        elif seconds is None:
            event = "unavailable"
            if not blockers:
                blockers.append("not_explained_by_available_outputs")
        elif seconds == 0:
            event = "started_or_restarted"
            if previous is None:
                resets.append("first_logged_observation")
            elif not resets:
                resets.append("start_not_explained_by_available_outputs")
        elif (prior is not None and prior.student_id == person.student_id and prior_behavior is not None
              and prior_behavior.estimated_seconds is not None):
            event = "increased" if seconds > prior_behavior.estimated_seconds else "unchanged_or_decreased"
        else:
            event = "positive_duration_observed"
        if duplicate or (seconds is None and (prior_behavior is None or prior_behavior.estimated_seconds is None)):
            # Repeated ineligible observations do not prove another episode was reset.
            resets = []
        rows.append({
            "person_track_id": person.person_track_id, "student_id": person.student_id,
            "association_status": person.association_status, "identity_status": person.identity_status,
            "identity_continuity_status": shadow.status if shadow else None,
            "identity_continuity_reason": shadow.reason if shadow else None,
            "current_face_student_id": shadow.face_student_id if shadow else None,
            "current_face_present": shadow.face_present if shadow else None,
            "shadow_proposed_student_id": shadow.proposed_student_id if shadow else None,
            f"{name}_state": state, f"{name}_model_confidence": confidence,
            f"{name}_decision_reason": (decision.get("decision_reason")
                                      if isinstance(decision.get("decision_reason"), str) else None),
            f"{name}_duration_seconds": seconds, "duration_event": event,
            "duration_blockers": blockers, "duration_reset_reasons": resets,
        })
    return rows


def sleeping_diagnostic(frame, raw_people, previous, max_gap, shadow_frame):
    """Compatibility entry point for callers reading the original Sleeping shape."""
    return behavior_diagnostic(frame, raw_people, previous, max_gap, shadow_frame, names=("sleeping",))


class IdentityContinuityShadow:
    def __init__(self, output_dir, *, iou_threshold, max_update_gap, max_frames):
        # Deliberately no environment/configuration path for T_hold.
        self.resolver = IdentityContinuityResolver(iou_threshold=iou_threshold, max_update_gap=max_update_gap)
        self.output_dir = Path(output_dir).resolve()
        self.max_frames = min(max_frames, 100)
        self.frames_written = self.bytes_written = 0
        self.session_dir = None
        self._stopped = False
        self._stop_reason = None
        self._finished = False
        self._previous_behavior_frame = None
        print(f"[IDENTITY SHADOW] enabled; output_dir={self.output_dir}; "
              "waiting for first completed AI result; no frames written yet; T_hold unapproved",
              file=sys.stderr, flush=True)

    @property
    def active(self):
        return (not self._stopped and not self._finished and self.frames_written < self.max_frames
                and self.bytes_written < MAX_BYTES)

    def _stop(self, exc):
        self._stopped = True
        self._stop_reason = f"{type(exc).__name__}: {exc}"
        print(f"[IDENTITY SHADOW] stopped; AI continues; output_dir={self.output_dir}; "
              f"session_dir={self.session_dir}; {self._stop_reason}", file=sys.stderr, flush=True)

    def before_update(self, tracker):
        if not self.active:
            return None
        try:
            if len(tracker.tracks) > MAX_DETAILS:
                raise ValueError("shadow detail limit")
            return capture_prior_tracks(tracker)
        except Exception as exc:
            self._stop(exc)
            return None

    def analyze(self, frame_id, captured_at, updated_at, before, tracked_people, faces, matches):
        if before is None or not self.active:
            return None
        try:
            if max(len(tracked_people), len(faces), len(matches)) > MAX_DETAILS:
                raise ValueError("shadow detail limit")
            people, current_faces, associations = project_current(tracked_people, faces, matches)
            return self.resolver.update(frame_id, captured_at, updated_at, before, people, current_faces, associations)
        except Exception as exc:
            self._stop(exc)
            return None

    def record(self, frame, *, result=None, raw_people=(), duration_max_gap=None):
        if frame is None or not self.active:
            return False
        try:
            data = {"schema": "identity_continuity_shadow_v1", "behavior_diagnostic_version": 2,
                    "mode": "SHADOW_ONLY", "clock": "perf_counter",
                    "parameters": {"person_iou_threshold": self.resolver.iou_threshold,
                                   "person_max_age": self.resolver.max_update_gap, "T_hold": None},
                    "production_applied": False, "hold_period_approved": False, **asdict(frame)}
            behavior_frame = None
            if result is not None:
                behavior_frame = result["behavior_frame"]
                if (behavior_frame.frame_id != frame.frame_id or behavior_frame.captured_at != frame.captured_at
                        or result["frame_id"] != frame.frame_id or result["captured_at"] != frame.captured_at):
                    raise ValueError("shadow/production frame metadata mismatch")
                if not finite(duration_max_gap) or duration_max_gap <= 0:
                    raise ValueError("missing existing DurationTracker max_gap")
                data.update(behavior_diagnostic(behavior_frame, raw_people, self._previous_behavior_frame,
                                                duration_max_gap, frame))
                data["production_observation_status"] = "available"
            else:
                data.update(production_observation_status="not_provided", sleeping=[], using_phone=[])
            payload = (json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
            if self.bytes_written + len(payload) > MAX_BYTES:
                self._stopped = True
                self._stop_reason = "byte_limit_reached"
                print(f"[IDENTITY SHADOW] byte limit reached; AI continues; "
                      f"output_dir={self.output_dir}; frames_written={self.frames_written}",
                      file=sys.stderr, flush=True)
                return False
            if self.session_dir is None:
                self.session_dir = self.output_dir / f"session_{uuid.uuid4().hex}"
                self.session_dir.mkdir(parents=True, exist_ok=False)
                mode = "xb"
            else:
                mode = "ab"
            with (self.session_dir / "frames.jsonl").open(mode) as stream:
                stream.write(payload)
            self.frames_written += 1
            self.bytes_written += len(payload)
            self._previous_behavior_frame = behavior_frame
            if self.frames_written == 1:
                print(f"[IDENTITY SHADOW] first frame written; SHADOW ONLY; {self.session_dir / 'frames.jsonl'}",
                      file=sys.stderr, flush=True)
            return True
        except Exception as exc:
            self._stop(exc)
            return False

    def finish(self):
        if self._finished:
            return
        self._finished = True
        if self.frames_written:
            message = f"frames_written={self.frames_written}; log={self.session_dir / 'frames.jsonl'}"
        else:
            message = f"frames_written=0; no diagnostic frames written; output_dir={self.output_dir}"
        print(f"[IDENTITY SHADOW] finished; {message}; stop_reason={self._stop_reason or 'none'}",
              file=sys.stderr, flush=True)
