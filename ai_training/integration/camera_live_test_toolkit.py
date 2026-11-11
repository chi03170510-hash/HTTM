"""Offline Camera Live log analysis. Standard library only; never imports AI workers.

Schemas are explicit: person_tracker_diagnostic_v1, identity_continuity_shadow_v1,
and camera_test_fixture_v1 (synthetic test data ONLY). No session auto-merging,
identity reconstruction, inference, camera access, or production configuration.
"""

import argparse
import csv
import hashlib
import json
import math
import sys
import uuid
from collections import Counter, defaultdict
from pathlib import Path


STATES = {"detected", "suspected", "unknown", "not_detected"}
MATCHED = {"matched", "matched_v4_heuristic"}
BEHAVIORS = ("sleeping", "using_phone")
CASES = {
    "S1": ("Gục đầu xuống bàn", "sleeping", "duration_progress"),
    "S2": ("Nhắm mắt, cúi đầu nhẹ", "sleeping", "duration_progress"),
    "S3": ("Cúi đầu viết bài", "sleeping", None),
    "S4": ("Cúi đầu đọc sách", "sleeping", None),
    "S5": ("Ngồi bình thường", "sleeping", None),
    "P1": ("Sử dụng điện thoại liên tục", "using_phone", "duration_progress"),
    "P2": ("Đặt điện thoại xuống", "using_phone", None),
    "P3": ("Cầm sách hoặc vở", "using_phone", None),
    "P4": ("Cầm điện thoại nhưng không sử dụng", "using_phone", None),
    "P5": ("Hai học sinh, chỉ một người dùng điện thoại", "using_phone", "multiple_tracks_seen"),
    "IC1": ("Nhận diện khi ngồi thẳng", None, "face_confirmation_seen"),
    "IC2": ("Cúi đầu che khuôn mặt", None, "face_absence_seen"),
    "IC3": ("Ngẩng đầu nhận diện lại", None, "face_reconfirmation_seen"),
    "IC4": ("Hai học sinh ngồi gần nhau", None, "multiple_tracks_seen"),
    "IC5": ("Hai học sinh đổi vị trí", None, "multiple_tracks_seen"),
    "IC6": ("Người rời camera, người khác vào", None, "track_turnover_seen"),
}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sid(value):
    return value.strip() if isinstance(value, str) and value.strip() and value.strip().lower() not in {
        "unknown", "none", "null"} else None


def strict_json(text):
    def reject(value):
        raise ValueError(f"Non-finite JSON number: {value}")
    def unique_keys(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(text, parse_constant=reject, object_pairs_hook=unique_keys)


def issue(issues, code, line, detail, severity="missing"):
    issues.append({"code": code, "line": line, "detail": detail, "severity": severity})


def checked_sid(value, issues, line):
    if value is not None and not isinstance(value, str):
        issue(issues, "invalid_student_id_type", line, "student_id must be string or null", "error")
    return sid(value)


def array(record, key, issues, line):
    if key not in record:
        issue(issues, "missing_field", line, key)
        return None
    if not isinstance(record[key], list) or any(not isinstance(x, dict) for x in record[key]):
        issue(issues, "invalid_array", line, key, "error")
        return None
    return record[key]


def person(pid):
    return {"person_track_id": pid, "student_id": None, "student_id_available": False,
            "student_id_source": "missing", "association_status": None, "association_available": False,
            "identity_status": None, "shadow_status": None, "face_present": None, "behaviors": {}}


def behavior(record, state_key, duration_key, confidence_key, issues, line):
    state = record.get(state_key)
    if state is not None and (not isinstance(state, str) or state not in STATES):
        issue(issues, "invalid_behavior_state", line, str(state), "error")
        state = None
    duration = record.get(duration_key)
    valid_duration = duration is None or (number(duration) and duration >= 0)
    if not valid_duration:
        issue(issues, "invalid_duration", line, str(duration), "error")
        duration = None
    confidence = record.get(confidence_key)
    reason = record.get(state_key.removesuffix("state") + "decision_reason")
    return {"state": state, "state_available": state is not None,
            "duration": duration, "duration_available": duration_key in record and valid_duration,
            "confidence": confidence if number(confidence) else None,
            "decision_reason": reason if isinstance(reason, str) else None,
            "logged_event": record.get("duration_event"),
            "blockers": record.get("duration_blockers") if isinstance(record.get("duration_blockers"), list) else [],
            "reset_reasons": record.get("duration_reset_reasons") if isinstance(record.get("duration_reset_reasons"), list) else []}


def shadow_production_people(record, issues, line):
    """Join additive per-behavior rows by PID, never by row order or Student ID."""
    tables = {}
    for name in BEHAVIORS:
        if name in record or name == "sleeping" or record.get("behavior_diagnostic_version") == 2:
            tables[name] = array(record, name, issues, line)
    complete = bool(tables) and all(rows is not None for rows in tables.values())
    counts = {name: Counter(row.get("person_track_id") for row in rows or []
                            if isinstance(row.get("person_track_id"), str)) for name, rows in tables.items()}
    duplicated = {pid for counter in counts.values() for pid, count in counter.items() if count > 1}
    if duplicated:
        issue(issues, "duplicate_behavior_person_track", line, sorted(duplicated), "error")
        complete = False
    if (len(counts) == 2 and all(rows is not None for rows in tables.values())
            and set(counts["sleeping"]) != set(counts["using_phone"])):
        issue(issues, "behavior_person_track_sets_differ", line, "sleeping/using_phone PID sets differ", "error")
        complete = False
    people, joined, metadata = [], {}, {}
    for name, rows in tables.items():
        for row in rows or []:
            pid = row.get("person_track_id")
            can_join = isinstance(pid, str) and bool(pid) and pid not in duplicated
            p = joined.get(pid) if can_join else None
            if p is None:
                p = person(pid)
                p.update(student_id=checked_sid(row.get("student_id"), issues, line),
                         student_id_available="student_id" in row, student_id_source="production",
                         association_status=row.get("association_status"), association_available="association_status" in row,
                         identity_status=row.get("identity_status"))
                people.append(p)
                if can_join:
                    joined[pid], metadata[pid] = p, row
            else:
                fields = ("student_id", "association_status", "identity_status")
                prior = metadata[pid]
                conflict = [key for key in fields if key in prior and key in row and prior[key] != row[key]]
                missing = [key for key in fields if key not in prior or key not in row]
                if conflict or missing:
                    issue(issues, "behavior_identity_conflict" if conflict else "behavior_identity_missing", line,
                          f"{pid}: conflicting={conflict}, missing={missing}", "error" if conflict else "missing")
                    # Retain measured states/durations, but do not choose either identity claim.
                    p.update(student_id=None, student_id_available=False, student_id_source="missing",
                             association_status=None, association_available=False, identity_status=None)
                    complete = False
            p["behaviors"][name] = behavior(row, f"{name}_state", f"{name}_duration_seconds",
                                             f"{name}_model_confidence", issues, line)
    return people, complete


def normalize(record, line, issues):
    schema = record.get("schema")
    if not isinstance(schema, str) or schema not in {
            "person_tracker_diagnostic_v1", "identity_continuity_shadow_v1", "camera_test_fixture_v1"}:
        issue(issues, "unsupported_schema", line, repr(schema), "error")
        return None
    fid, capture = record.get("frame_id"), record.get("captured_at")
    if not isinstance(fid, int) or isinstance(fid, bool) or fid < 1:
        issue(issues, "invalid_frame_id" if "frame_id" in record else "missing_field", line, "frame_id",
              "error" if "frame_id" in record else "missing")
        fid = None
    if not number(capture):
        issue(issues, "invalid_timestamp" if capture is not None else "missing_field", line, "captured_at",
              "error" if capture is not None else "missing")
        capture = None
    result = {"line": line, "frame_id": fid, "captured_at": capture, "schema": schema,
              "people": [], "people_complete": False, "duration_max_gap": None,
              "timestamp_declared_valid": record.get("production_timestamp_valid", record.get("timestamp_valid")),
              "synthetic": schema == "camera_test_fixture_v1", "expiry_count": 0}
    if schema == "person_tracker_diagnostic_v1":
        tracks = array(record, "tracks_after", issues, line)
        associations = array(record, "associations", issues, line)
        updated = record.get("tracker_updated_at")
        if not number(updated):
            issue(issues, "missing_field", line, "tracker_updated_at")
        complete = record.get("details_truncated") is False
        if not complete:
            issue(issues, "incomplete_details", line, "details_truncated false not confirmed")
        result["people_complete"] = tracks is not None and number(updated) and complete
        for track in tracks or []:
            if not number(track.get("last_seen")):
                issue(issues, "invalid_track_timestamp" if "last_seen" in track else "missing_field", line,
                      "tracks_after.last_seen", "error" if "last_seen" in track else "missing")
                result["people_complete"] = False
                continue
            if number(updated) and track.get("last_seen") != updated:
                continue  # Retained but unobserved tracker entries are not current detections.
            p = person(track.get("track_id"))
            matches = [m for m in associations or [] if m.get("person_track_id") == p["person_track_id"]]
            if len(matches) == 1:
                p.update(student_id=checked_sid(matches[0].get("student_id"), issues, line), student_id_available="student_id" in matches[0],
                         student_id_source="association", association_status=matches[0].get("association_status"),
                         association_available="association_status" in matches[0])
            elif len(matches) > 1:
                p.update(association_status="ambiguous", association_available=True)
            elif associations is not None and complete:
                p.update(student_id_available=True, student_id_source="association", association_status="unmatched",
                         association_available=True)
            result["people"].append(p)
        decisions = record.get("decisions", [])
        if isinstance(decisions, list):
            result["expiry_count"] = sum(isinstance(d, dict) and d.get("action") == "expired" for d in decisions)
    elif schema == "identity_continuity_shadow_v1":
        observations = array(record, "observations", issues, line)
        shadows = {o["person_track_id"]: o for o in observations or [] if isinstance(o.get("person_track_id"), str)}
        if record.get("production_observation_status") == "available":
            result["people"], result["people_complete"] = shadow_production_people(record, issues, line)
            result["duration_max_gap"] = record.get("duration_max_gap") if number(record.get("duration_max_gap")) else None
        else:
            result["people_complete"] = observations is not None
            for row in observations or []:
                if row.get("track_continuity") == "expired" or row.get("track_continuity") == "missing":
                    continue
                p = person(row.get("person_track_id"))
                p.update(association_status=row.get("association_status"),
                         association_available="association_status" in row)
                result["people"].append(p)
        for p in result["people"]:
            shadow = shadows.get(p["person_track_id"], {})
            p.update(shadow_status=shadow.get("status"), face_present=shadow.get("face_present"),
                     face_student_id=shadow.get("face_student_id"), shadow_proposed_student_id=shadow.get("proposed_student_id"))
            # A shadow proposal is NEVER substituted for production Student ID.
    else:
        if record.get("synthetic") is not True:
            issue(issues, "fixture_not_marked_synthetic", line, "synthetic must be true", "error")
        rows = array(record, "people", issues, line)
        result["people_complete"] = rows is not None
        result["duration_max_gap"] = record.get("duration_max_gap") if number(record.get("duration_max_gap")) else None
        for row in rows or []:
            p = person(row.get("person_track_id"))
            p.update(student_id=checked_sid(row.get("student_id"), issues, line), student_id_available="student_id" in row,
                     student_id_source="production", association_status=row.get("association_status"),
                     association_available="association_status" in row, identity_status=row.get("identity_status"),
                     shadow_status=row.get("identity_continuity_status"), face_present=row.get("face_present"))
            decisions = row.get("behaviors", {})
            if isinstance(decisions, dict):
                for name in BEHAVIORS:
                    if isinstance(decisions.get(name), dict):
                        p["behaviors"][name] = behavior(decisions[name], "state", "estimated_seconds",
                                                        "model_confidence", issues, line)
            result["people"].append(p)
    for p in result["people"]:
        if not isinstance(p["person_track_id"], str) or not p["person_track_id"]:
            issue(issues, "missing_person_track_id", line, "person_track_id",
                  "missing" if p["person_track_id"] is None else "error")
            p["person_track_id"] = None
            result["people_complete"] = False
        for key in ("association_status", "identity_status", "shadow_status"):
            if p[key] is not None and not isinstance(p[key], str):
                issue(issues, "invalid_field_type", line, key, "error")
                p[key] = None
                if key == "association_status":
                    p["association_available"] = False
        if p["association_status"] is None:
            p["association_available"] = False
        if p["face_present"] is not None and type(p["face_present"]) is not bool:
            issue(issues, "invalid_field_type", line, "face_present", "error")
            p["face_present"] = None
    if "duration_max_gap" in record and record["duration_max_gap"] is None:
        issue(issues, "missing_field", line, "duration_max_gap is unavailable")
    elif "duration_max_gap" in record and (not number(record["duration_max_gap"]) or record["duration_max_gap"] <= 0):
        issue(issues, "invalid_duration_max_gap", line, "duration_max_gap must be finite and positive", "error")
        result["duration_max_gap"] = None
    if result["timestamp_declared_valid"] is not None and type(result["timestamp_declared_valid"]) is not bool:
        issue(issues, "invalid_field_type", line, "timestamp validity flag", "error")
        result["timestamp_declared_valid"] = None
    return result


def resolve_log(path):
    path = Path(path)
    if path.is_dir():
        if (path / "frames.jsonl").is_file():
            return path / "frames.jsonl"
        logs = list(path.glob("*.jsonl"))
        if len(logs) != 1:
            raise ValueError("Specify one session/JSONL file; directory has zero or multiple JSONL files")
        return logs[0]
    if not path.is_file():
        raise ValueError(f"Log does not exist: {path}")
    return path


def analyze(path, provenance="unverified", frame_ids=None):
    if not isinstance(provenance, str) or provenance not in {"recorded", "synthetic", "unverified"}:
        raise ValueError("Invalid declared provenance")
    source = resolve_log(path)
    payload = source.read_bytes()
    issues, frames, seen, total_objects, duplicates = [], [], {}, 0, 0
    for line, text in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        try:
            record = strict_json(text)
            if not isinstance(record, dict):
                raise ValueError("JSONL record must be an object")
        except ValueError as exc:
            issue(issues, "invalid_json", line, str(exc), "error")
            continue
        total_objects += 1
        normalized = normalize(record, line, issues)
        if normalized is None:
            continue
        fid = normalized["frame_id"]
        fingerprint = json.dumps(record, sort_keys=True, ensure_ascii=True)
        if fid is not None and fid in seen:
            if seen[fid] == fingerprint:
                duplicates += 1
                issue(issues, "duplicate_record_ignored", line, str(fid), "warning")
            else:
                issue(issues, "conflicting_frame_id", line, str(fid), "error")
            continue
        if fid is not None:
            seen[fid] = fingerprint
        if frame_ids is None or fid in frame_ids:
            frames.append(normalized)
    if frame_ids is not None:
        for fid in sorted(set(frame_ids) - {f["frame_id"] for f in frames}):
            issue(issues, "requested_frame_missing", None, str(fid))
    if len({f["schema"] for f in frames}) > 1:
        issue(issues, "mixed_schemas", None, "One file mixes source schemas; no cross-source continuity inferred", "error")
    effective = "synthetic" if any(f["synthetic"] for f in frames) else provenance
    events = []

    def event(kind, frame, pid=None, name=None, basis="observed", detail=""):
        events.append({"kind": kind, "basis": basis, "frame_id": frame["frame_id"], "line": frame["line"],
                       "person_track_id": pid, "behavior": name, "detail": detail})

    previous = None
    stats = {name: {"states": Counter(), "state_rows": 0, "duration_rows": 0,
                    "null_durations": 0, "numeric_durations": 0, "reported_events": Counter()} for name in BEHAVIORS}
    tracks, students, association_counts = Counter(), Counter(), Counter()
    person_rows = production_sid_rows = 0
    for frame in frames:
        people = frame["people"]
        current = {p["person_track_id"]: p for p in people if p["person_track_id"] is not None}
        old = {p["person_track_id"]: p for p in previous["people"] if p["person_track_id"] is not None} if previous else {}
        sequential = bool(previous and frame["frame_id"] is not None and previous["frame_id"] is not None
                          and frame["schema"] == previous["schema"]
                          and frame["frame_id"] == previous["frame_id"] + 1
                          and number(frame["captured_at"]) and number(previous["captured_at"])
                          and frame["captured_at"] > previous["captured_at"])
        if previous:
            if (number(frame["captured_at"]) and number(previous["captured_at"])
                    and frame["captured_at"] <= previous["captured_at"]):
                event("timestamp_not_increasing", frame, detail=f"{previous['captured_at']} -> {frame['captured_at']}")
            if frame["frame_id"] is not None and previous["frame_id"] is not None and frame["frame_id"] <= previous["frame_id"]:
                event("frame_id_not_increasing", frame)
            if frame["people_complete"] and previous["people_complete"] and set(current) != set(old):
                event("person_track_set_changed", frame, detail=json.dumps({"removed": sorted(set(old)-set(current)),
                                                                            "added": sorted(set(current)-set(old))}))
            if (frame["people_complete"] and previous["people_complete"]
                    and all(p["association_available"] for p in people + previous["people"])):
                before_matches = sum(p["association_status"] in MATCHED for p in previous["people"])
                now_matches = sum(p["association_status"] in MATCHED for p in people)
                if now_matches < before_matches:
                    event("association_match_count_dropped", frame, detail=json.dumps({
                        "before": before_matches, "after": now_matches, "actor_correspondence_inferred": False}))
        if frame["timestamp_declared_valid"] is False:
            event("producer_timestamp_invalid", frame)
        if frame["expiry_count"]:
            event("track_expiry_logged", frame, basis="logged_diagnostic", detail=str(frame["expiry_count"]))
        claims = defaultdict(list)
        for p in people:
            person_rows += 1
            pid, student = p["person_track_id"], p["student_id"]
            if pid is not None:
                tracks[pid] += 1
            if student:
                students[student] += 1
                claims[(p["student_id_source"], student)].append(pid)
            if p["student_id_available"] and p["student_id_source"] == "production":
                production_sid_rows += 1
            if p["association_available"]:
                association_counts[p["association_status"]] += 1
            prior = old.get(pid) if sequential else None
            if sequential and student and prior is None:
                old_ids = [o["person_track_id"] for o in previous["people"]
                           if o["student_id"] == student and o["person_track_id"] is not None]
                if old_ids:
                    event("reported_student_changed_track", frame, pid, basis="inferred",
                          detail=json.dumps({"reported_student_id": student, "previous_track_ids": old_ids,
                                             "physical_identity_verified": False}))
            if prior:
                if (prior["association_available"] and p["association_available"]
                        and prior["association_status"] in MATCHED and p["association_status"] not in MATCHED):
                    event("association_lost", frame, pid)
                if prior["student_id"] and p["student_id_available"] and student is None:
                    event("student_id_lost", frame, pid)
                if prior["student_id"] and student and prior["student_id"] != student:
                    event("reported_student_id_changed", frame, pid, detail=f"{prior['student_id']} -> {student}")
            for name in BEHAVIORS:
                value = p["behaviors"].get(name, {})
                stat = stats[name]
                if value.get("state_available"):
                    stat["states"][value["state"]] += 1
                    stat["state_rows"] += 1
                if value.get("duration_available"):
                    stat["duration_rows"] += 1
                    stat["null_durations"] += value["duration"] is None
                    stat["numeric_durations"] += number(value["duration"])
                if isinstance(value.get("logged_event"), str):
                    stat["reported_events"][value["logged_event"]] += 1
                    event("logged_duration_classification", frame, pid, name, "logged_diagnostic", value["logged_event"])
                if value.get("blockers"):
                    event("logged_duration_blocked", frame, pid, name, "logged_diagnostic", json.dumps(value["blockers"]))
                if value.get("reset_reasons"):
                    event("logged_duration_start_or_reset_reasons", frame, pid, name, "logged_diagnostic",
                          json.dumps(value["reset_reasons"]))
                if value.get("duration_available") and value.get("duration") is None and not value.get("blockers"):
                    context = []
                    if p["student_id_available"] and student is None:
                        context.append("reported_student_id_unavailable")
                    if p["identity_status"] is not None and p["identity_status"] != "matched":
                        context.append(f"reported_identity_{p['identity_status']}")
                    if value.get("state_available") and value["state"] != "detected":
                        context.append(f"reported_state_{value['state']}")
                    if context:
                        event("duration_unavailable_context", frame, pid, name, "inferred", json.dumps(context))
                old_value = prior["behaviors"].get(name, {}) if prior else {}
                if value.get("duration_available") and old_value.get("duration_available"):
                    a, b = old_value.get("duration"), value.get("duration")
                    if number(a) and number(b):
                        if b > a:
                            event("duration_increased", frame, pid, name)
                        elif b == a and value.get("state") == "detected":
                            event("duration_not_increased", frame, pid, name)
                        elif b == 0 and a > 0:
                            event("duration_reset_candidate", frame, pid, name, "inferred", "positive duration -> zero; not proof of same actor")
                    elif number(a) and b is None:
                        event("duration_became_unavailable", frame, pid, name)
        for (claim_source, student), ids in claims.items():
            if len(ids) > 1:
                event("duplicate_production_sid" if claim_source == "production" else "duplicate_association_sid",
                      frame, detail=json.dumps({"student_id": student, "tracks": ids}))
        valid_ids = [p["person_track_id"] for p in people if p["person_track_id"] is not None]
        if len(set(valid_ids)) != len(valid_ids):
            event("duplicate_person_track_id", frame)
        previous = frame
    # Missing behavior is never converted into a zero detection count or not_detected.
    for name, stat in stats.items():
        stat["states"] = dict(stat["states"]) if stat["state_rows"] else None
        stat["reported_events"] = dict(stat["reported_events"])
        stat["missing_state_rows"] = person_rows - stat["state_rows"]
        stat["missing_duration_rows"] = person_rows - stat["duration_rows"]
    return {"source": str(source.resolve()), "source_sha256": hashlib.sha256(payload).hexdigest(),
            "declared_provenance": provenance, "effective_provenance": effective,
            "provenance_independently_verified": False, "valid_json_objects": total_objects,
            "ai_frame_count": len({f["frame_id"] for f in frames if f["frame_id"] is not None}),
            "duplicate_records_ignored": duplicates, "person_observations": person_rows,
            "production_sid_rows": production_sid_rows, "reported_person_tracks": dict(tracks),
            "reported_student_ids": dict(students), "association_statuses": dict(association_counts),
            "behaviors": stats, "expiry_events_logged": sum(f["expiry_count"] for f in frames),
            "issues": issues, "events": events, "frames": frames,
            "ground_truth": "not_inferred_from_detector_or_ids"}


def check(check_id, status, reason):
    return {"check": check_id, "status": status, "reason": reason}


def technical_checks(analysis, name=None, special=None):
    frames, issues, events = analysis["frames"], analysis["issues"], analysis["events"]
    kinds = {e["kind"] for e in events}
    checks = [check("log_integrity", "FAIL" if any(i["severity"] == "error" for i in issues) else
                    "INCONCLUSIVE" if not frames or any(i["severity"] == "missing" for i in issues) else "PASS",
                    "JSON/schema validation; missing fields are not guessed")]
    known_times = frames and all(number(f["captured_at"]) and f["frame_id"] is not None for f in frames)
    checks.append(check("observation_timestamps", "FAIL" if kinds & {
        "timestamp_not_increasing", "frame_id_not_increasing", "producer_timestamp_invalid"} else
        "PASS" if known_times else "INCONCLUSIVE", "Observed frame/capture order only"))
    checks.append(check("person_track_unique", "FAIL" if "duplicate_person_track_id" in kinds else
                        "PASS" if frames and all(f["people_complete"] for f in frames) else "INCONCLUSIVE",
                        "Unique reported track IDs within each complete observation"))
    checks.append(check("production_sid_unique", "FAIL" if "duplicate_production_sid" in kinds else
                        "PASS" if analysis["person_observations"] > 0 and analysis["production_sid_rows"] == analysis["person_observations"]
                        else "INCONCLUSIVE", "Association SID and shadow proposals are not production attribution"))
    if name:
        stat = analysis["behaviors"][name]
        checks.append(check("behavior_fields_available", "PASS" if analysis["person_observations"] > 0
                            and stat["missing_state_rows"] == stat["missing_duration_rows"] == 0 else "INCONCLUSIVE",
                            f"{name}: persisted state and duration keys required; null duration is distinct from missing"))
        tested = violations = 0
        eligible_pairs = increases = 0
        previous = None
        for frame in frames:
            for p in frame["people"]:
                value = p["behaviors"].get(name, {})
                if (value.get("state_available") and value.get("duration_available") and p["student_id_available"]
                        and p["student_id_source"] == "production" and p["identity_status"] is not None):
                    tested += 1
                    ineligible = (value["state"] != "detected" or p["student_id"] is None
                                  or p["identity_status"] != "matched" or frame["timestamp_declared_valid"] is False)
                    violations += ineligible and value["duration"] is not None
                    violations += (not ineligible and frame["timestamp_declared_valid"] is True
                                   and value["duration"] is None)
                if previous and number(frame["captured_at"]) and number(previous["captured_at"]):
                    prior = next((o for o in previous["people"] if o["person_track_id"] == p["person_track_id"]), None)
                    old_value = prior["behaviors"].get(name, {}) if prior else {}
                    if (prior and p["person_track_id"] is not None and frame["schema"] == previous["schema"]
                            and frame["frame_id"] is not None and previous["frame_id"] is not None
                            and frame["frame_id"] == previous["frame_id"] + 1
                            and p["student_id"] is not None and p["student_id"] == prior["student_id"]
                            and p["identity_status"] == prior["identity_status"] == "matched"
                            and value.get("state") == old_value.get("state") == "detected"
                            and number(value.get("duration")) and number(old_value.get("duration"))
                            and frame["timestamp_declared_valid"] is not False
                            and previous["timestamp_declared_valid"] is not False
                            and number(frame["duration_max_gap"])
                            and 0 < frame["captured_at"] - previous["captured_at"] <= frame["duration_max_gap"]):
                        eligible_pairs += 1
                        increases += value["duration"] > old_value["duration"]
            previous = frame
        checks.append(check("duration_identity_gate", "FAIL" if violations else "PASS" if tested else "INCONCLUSIVE",
                            "Persisted duration must be null when reported identity/state is ineligible"))
        if special == "duration_progress":
            checks.append(check("duration_progress", "PASS" if eligible_pairs and increases == eligible_pairs else
                                "FAIL" if eligible_pairs else "INCONCLUSIVE",
                                f"Eligible adjacent pairs={eligible_pairs}; increases={increases}; max gap must be logged"))
    if special and special != "duration_progress":
        observed = False
        if special == "multiple_tracks_seen":
            observed = any(len({p["person_track_id"] for p in f["people"]
                                if p["person_track_id"] is not None}) >= 2 for f in frames)
        elif special == "face_confirmation_seen":
            observed = any(p["shadow_status"] in {"face_confirmed", "face_recovered"} for f in frames for p in f["people"])
        elif special == "face_absence_seen":
            observed = any(p["face_present"] is False for f in frames for p in f["people"])
        elif special == "face_reconfirmation_seen":
            absent = set()
            for f in frames:
                for p in f["people"]:
                    if p["person_track_id"] in absent and p["shadow_status"] in {"face_confirmed", "face_recovered"}:
                        observed = True
                    if p["face_present"] is False:
                        absent.add(p["person_track_id"])
        elif special == "track_turnover_seen":
            departure_seen = False
            for e in events:
                if e["kind"] == "person_track_set_changed":
                    detail = json.loads(e["detail"])
                    departure_seen = departure_seen or bool(detail["removed"])
                    observed = observed or (departure_seen and bool(detail["added"]))
        checks.append(check(special, "PASS" if observed else "INCONCLUSIVE",
                            "Reported workflow evidence only; not proof of physical actors or model correctness"))
    return checks


def new_manifest():
    return {"schema": "httm_camera_test_manifest_v1", "cases": [
        {"case_id": key, "run": False, "log": None, "provenance": "unverified", "frame_ids": [],
         "ground_truth": {"confirmed": False, "verdict": "INCONCLUSIVE", "reviewer": "", "evidence": ""}}
        for key in CASES]}


def evaluate_manifest(path):
    path = Path(path)
    manifest = strict_json(path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or manifest.get("schema") != "httm_camera_test_manifest_v1":
        raise ValueError("Expected httm_camera_test_manifest_v1")
    entries = manifest.get("cases")
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise ValueError("cases must be an array of objects")
    by_id = {}
    for entry in entries:
        cid = entry.get("case_id")
        if not isinstance(cid, str) or cid not in CASES or cid in by_id or type(entry.get("run")) is not bool:
            raise ValueError("Unknown/duplicate case_id or invalid run boolean")
        by_id[cid] = entry
    results = []
    for cid, (title, name, special) in CASES.items():
        entry = by_id.get(cid, {})
        result = {"case_id": cid, "title": title, "status": "NOT RUN", "checks": [],
                  "manual_review": entry.get("ground_truth", {}), "reason": "No run declared", "source": None}
        if entry.get("run"):
            result["status"] = "INCONCLUSIVE"
            try:
                if not isinstance(entry.get("log"), str) or not entry["log"].strip():
                    raise ValueError("No log supplied for declared run")
                selected = entry.get("frame_ids", [])
                if not isinstance(selected, list) or any(type(x) is not int or x < 1 for x in selected):
                    raise ValueError("frame_ids must be positive integers")
                data = analyze(path.parent / entry["log"], entry.get("provenance", "unverified"), selected or None)
                result.update(source=data["source"], source_sha256=data["source_sha256"],
                              effective_provenance=data["effective_provenance"],
                              checks=technical_checks(data, name, special))
                gt = entry.get("ground_truth", {})
                reviewed = (isinstance(gt, dict) and gt.get("confirmed") is True
                            and isinstance(gt.get("reviewer"), str) and bool(gt["reviewer"].strip())
                            and isinstance(gt.get("evidence"), str) and bool(gt["evidence"].strip())
                            and isinstance(gt.get("verdict"), str)
                            and gt["verdict"] in {"PASS", "FAIL", "INCONCLUSIVE"})
                statuses = {c["status"] for c in result["checks"]}
                if data["effective_provenance"] != "recorded":
                    result["reason"] = "Synthetic/unverified evidence cannot PASS a real camera case"
                elif "FAIL" in statuses:
                    result.update(status="FAIL", reason="A verifiable technical log condition failed; not a model accuracy verdict")
                elif reviewed and gt["verdict"] == "FAIL":
                    result.update(status="FAIL", reason="Human-reviewed ground truth verdict")
                elif "INCONCLUSIVE" in statuses or not reviewed or gt["verdict"] != "PASS":
                    result["reason"] = "Missing log evidence or explicit human ground-truth confirmation"
                else:
                    result.update(status="PASS", reason="Log checks PASS plus explicit human confirmation; no ground truth inferred")
            except (OSError, ValueError, UnicodeError) as exc:
                result["reason"] = f"Evidence unavailable: {exc}"
        results.append(result)
    return {"schema": "httm_camera_test_results_v1", "manifest": str(path.resolve()), "cases": results,
            "counts": {status: sum(r["status"] == status for r in results)
                       for status in ("PASS", "FAIL", "INCONCLUSIVE", "NOT RUN")},
            "limits": ["Ground truth comes only from explicit human review, never detector state or track IDs.",
                       "Recorded provenance is user-declared, not independently authenticated.",
                       "Synthetic technical tests cannot establish real-camera PASS."]}


def markdown_value(value):
    return str(value).replace("|", "\\|").replace("\n", " ")


def write_csv(path, fields, rows):
    with path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v for k, v in row.items()})


def write_analysis(data, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    checks = technical_checks(data)
    data["technical_checks"] = checks
    with (output_dir / "analysis.json").open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
    rows = []
    for frame in data["frames"]:
        for p in frame["people"]:
            for name in BEHAVIORS:
                value = p["behaviors"].get(name, {})
                rows.append({"line": frame["line"], "frame_id": frame["frame_id"], "captured_at": frame["captured_at"],
                             "person_track_id": p["person_track_id"], "student_id": p["student_id"],
                             "student_id_available": p["student_id_available"], "student_id_source": p["student_id_source"],
                             "association_status": p["association_status"], "identity_status": p["identity_status"],
                             "shadow_status": p["shadow_status"],
                             "behavior": name, "state": value.get("state"), "state_available": value.get("state_available", False),
                             "duration_seconds": value.get("duration"), "duration_available": value.get("duration_available", False),
                             "confidence": value.get("confidence"), "decision_reason": value.get("decision_reason"),
                             "logged_event": value.get("logged_event"),
                             "logged_blockers": value.get("blockers", []), "logged_reset_reasons": value.get("reset_reasons", [])})
    fields = ["line", "frame_id", "captured_at", "person_track_id", "student_id", "student_id_available", "student_id_source",
              "association_status", "identity_status", "shadow_status", "behavior", "state", "state_available", "duration_seconds",
              "duration_available", "confidence", "decision_reason", "logged_event", "logged_blockers", "logged_reset_reasons"]
    write_csv(output_dir / "observations.csv", fields, rows)
    write_csv(output_dir / "events.csv", ["kind", "basis", "frame_id", "line", "person_track_id", "behavior", "detail"], data["events"])
    write_csv(output_dir / "frames.csv", ["line", "frame_id", "captured_at", "people_complete", "person_track_ids"], [
        {"line": f["line"], "frame_id": f["frame_id"], "captured_at": f["captured_at"], "people_complete": f["people_complete"],
         "person_track_ids": [p["person_track_id"] for p in f["people"]]} for f in data["frames"]])
    lines = ["# Camera log analysis — offline", "", f"Source: {data['source']}", f"SHA256: `{data['source_sha256']}`", "",
             f"Declared/effective provenance: {data['declared_provenance']} / {data['effective_provenance']} (not independently verified).", "",
             "## Observed / persisted facts", "", f"Unique AI frame IDs: {data['ai_frame_count']}; JSON objects: {data['valid_json_objects']}; exact duplicates ignored: {data['duplicate_records_ignored']}.",
             f"Reported tracks: {data['reported_person_tracks']}", f"Reported SIDs: {data['reported_student_ids']} (not ground truth).",
             f"Association statuses: {data['association_statuses']}; logged expiry events: {data['expiry_events_logged']}.", "",
             "| Behavior | Persisted state rows | State counts | Duration keys / null values |", "|---|---:|---|---|"]
    for name, stat in data["behaviors"].items():
        lines.append(f"| {name} | {stat['state_rows']} | {markdown_value(stat['states'] if stat['states'] is not None else 'MISSING DATA')} | {stat['duration_rows']} / {stat['null_durations']} |")
    lines += ["", "## Persisted behavior observations", "",
              "Confidence/reason come from the same person's production behavior output, when logged; no inference rerun.", "",
              "| Frame | Track | SID | Behavior | State | Confidence | Seconds | Decision reason | Logged event | Blockers / reset reasons |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(markdown_value(r[k]) for k in
              ("frame_id", "person_track_id", "student_id", "behavior", "state", "confidence", "duration_seconds",
               "decision_reason", "logged_event")) + " | " + markdown_value({
                   "blockers": r["logged_blockers"], "reset_reasons": r["logged_reset_reasons"]}) + " |" for r in rows]
    lines += ["", "## Missing data and limits", "", "Missing state is not not_detected. Missing duration key is not recorded null duration.",
              "Shadow proposals are separate from production SID. Raw association claims do not prove production attribution.",
              "No files are merged by matching Frame ID. No model accuracy/precision/recall/mAP or physical-identity correctness is inferred.", "",
              "## Data validation issues", ""]
    lines += [f"- line {i['line']}: {i['severity']} / {i['code']} / {markdown_value(i['detail'])}" for i in data["issues"]] or ["- None recorded."]
    lines += ["", "## Timeline: observed, logged diagnostic, and inferred", "",
              "| Frame | Track | Behavior | Basis | Event | Detail |", "|---|---|---|---|---|---|"]
    lines += ["| " + " | ".join(markdown_value(e[k]) for k in ("frame_id", "person_track_id", "behavior", "basis", "kind", "detail")) + " |" for e in data["events"]]
    lines += ["", "## Technical checks only", ""]
    lines += [f"- {c['check']}: **{c['status']}** — {c['reason']}" for c in checks]
    lines += ["", "These checks do not PASS a behavior/camera case. Human ground-truth review is required."]
    with (output_dir / "analysis.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")
    return checks


def write_results(report, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "test_results.json").open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
    write_csv(output_dir / "test_results.csv", ["case_id", "title", "status", "reason", "source", "checks", "manual_review"], [
        {k: r.get(k) for k in ("case_id", "title", "status", "reason", "source", "checks", "manual_review")} for r in report["cases"]])
    lines = ["# Camera test results", "", f"Counts: {report['counts']}", "", "| Case | Scenario | Result | Reason |", "|---|---|---|---|"]
    lines += [f"| {r['case_id']} | {r['title']} | {r['status']} | {markdown_value(r['reason'])} |" for r in report["cases"]]
    lines += ["", "## Automatic checks versus human review", ""]
    for r in report["cases"]:
        if r["status"] != "NOT RUN":
            lines += [f"### {r['case_id']}", "", f"Evidence: {r['source']}", f"Human review: {r['manual_review']}"]
            lines += [f"- {c['check']}: {c['status']} — {c['reason']}" for c in r["checks"]]
    lines += ["", *report["limits"]]
    with (output_dir / "test_results.md").open("x", encoding="utf-8") as stream:
        stream.write("\n".join(lines) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    a = commands.add_parser("analyze", help="Read one saved session or JSONL")
    a.add_argument("input", type=Path)
    a.add_argument("--provenance", choices=("recorded", "synthetic", "unverified"), default="unverified")
    a.add_argument("--output-dir", type=Path)
    r = commands.add_parser("report", help="Aggregate declared runs and explicit human review")
    r.add_argument("manifest", type=Path)
    r.add_argument("--output-dir", type=Path)
    t = commands.add_parser("init-manifest", help="Create 16 NOT RUN cases, without overwriting")
    t.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init-manifest":
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(new_manifest(), stream, ensure_ascii=False, indent=2)
            print(str(args.output.resolve()))
            return 0
        output = args.output_dir or Path(__file__).resolve().parents[1] / "runs/camera_test_toolkit" / f"{args.command}_{uuid.uuid4().hex}"
        if args.command == "analyze":
            data = analyze(args.input, args.provenance)
            checks = write_analysis(data, output)
            print(json.dumps({"output": str(output.resolve()), "ai_frames": data["ai_frame_count"], "checks": checks}, ensure_ascii=False))
            return 1 if any(c["status"] == "FAIL" for c in checks) else 0
        report = evaluate_manifest(args.manifest)
        write_results(report, output)
        print(json.dumps({"output": str(output.resolve()), "counts": report["counts"]}))
        return 1 if report["counts"].get("FAIL", 0) else 0
    except (OSError, ValueError, UnicodeError) as exc:
        print(f"Toolkit error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
