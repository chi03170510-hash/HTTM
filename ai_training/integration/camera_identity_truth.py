"""Manual actor-truth comparison of saved logs. Standard library, offline only.

No camera/model/resolver calls. No SID autofill, line-order/nearest-time matching,
clock conversion, production attribution from raw association, or hold policy.
"""

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from camera_live_test_toolkit import number, resolve_log, shadow_production_people, sid, strict_json


SCHEMA = "httm_actor_truth_v1"
SOURCE_SCHEMAS = {"person": "person_tracker_diagnostic_v1", "shadow": "identity_continuity_shadow_v1"}


def reviewed(value):
    return (isinstance(value, dict) and value.get("confirmed") is True
            and all(isinstance(value.get(k), str) and value[k].strip() for k in ("reviewer", "evidence")))


def valid_id(value):
    return isinstance(value, str) and bool(value.strip())


def problem(issues, code, detail):
    issues.append({"code": code, "detail": detail})


def source_descriptor(path):
    path = resolve_log(path).resolve()
    return {"path": str(path), "session_id": path.parent.name,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_source(descriptor, role, base):
    if not isinstance(descriptor, dict):
        raise ValueError(f"Missing {role} source descriptor")
    path = resolve_log(base / descriptor["path"]).resolve()
    payload = path.read_bytes()
    if descriptor.get("session_id") != path.parent.name or descriptor.get("sha256") != hashlib.sha256(payload).hexdigest():
        raise ValueError(f"{role} source session/path/SHA256 mismatch; re-annotate changed data")
    frames, issues, prior = {}, [], None
    for line, text in enumerate(payload.decode("utf-8-sig").splitlines(), 1):
        if not text.strip():
            continue
        record = strict_json(text)
        if not isinstance(record, dict) or record.get("schema") != SOURCE_SCHEMAS[role]:
            raise ValueError(f"{role} line {line}: unexpected source schema")
        fid = record.get("frame_id")
        if type(fid) is not int or fid < 1 or fid in frames:
            raise ValueError(f"{role} line {line}: invalid/duplicate frame ID; no row-order fallback")
        capture, update = record.get("captured_at"), record.get("tracker_updated_at")
        valid = (number(capture) and number(update) and capture <= update and record.get("clock") == "perf_counter"
                 and record.get("timestamp_valid") is not False and record.get("production_timestamp_valid") is not False)
        if role == "shadow" and record.get("timestamp_valid") is not True:
            valid = False
        if prior is not None:
            valid = (valid and fid > prior[0] and number(prior[1]) and capture > prior[1]
                     and number(prior[2]) and update > prior[2])
        if not valid:
            problem(issues, "source_timestamp_or_clock_invalid", {"role": role, "frame_id": fid})
        frames[fid] = {"record": record, "valid": bool(valid), "line": line}
        prior = fid, capture, update
    return {"descriptor": descriptor, "frames": frames, "issues": issues,
            "synthetic": any(f["record"].get("synthetic") is True for f in frames.values())}


def person_rows(source, issues):
    rows = []
    for fid, frame in source["frames"].items():
        record = frame["record"]
        tracks, associations = record.get("tracks_after"), record.get("associations")
        if not isinstance(tracks, list) or any(not isinstance(p, dict) for p in tracks):
            problem(issues, "person_tracks_missing_or_invalid", fid)
            frame["valid"] = False
            continue
        if record.get("details_truncated") is not False:
            problem(issues, "person_details_incomplete", fid)
            frame["valid"] = False
        associations_valid = isinstance(associations, list) and all(isinstance(m, dict) for m in associations)
        if not associations_valid:
            problem(issues, "raw_associations_missing", fid)
        associations = associations if associations_valid else []
        current = []
        for track in tracks:
            if not number(track.get("last_seen")):
                problem(issues, "person_last_seen_missing", fid)
                frame["valid"] = False
                continue
            if track["last_seen"] != record.get("tracker_updated_at"):
                continue  # Retained but not detected, not a current observation.
            pid = track.get("track_id")
            if not valid_id(pid):
                problem(issues, "person_track_id_invalid", fid)
                frame["valid"] = False
                continue
            bbox = track.get("bbox")
            if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4 or not all(number(v) for v in bbox)
                    or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]):
                frame["valid"] = False
                problem(issues, "person_bbox_invalid", {"frame_id": fid, "person_track_id": pid})
            matches = [m for m in associations if m.get("person_track_id") == pid]
            current.append({"person_frame_id": fid, "captured_at": record.get("captured_at"),
                            "tracker_updated_at": record.get("tracker_updated_at"), "person_track_id": pid,
                            "bbox": track.get("bbox"), "association_sid": sid(matches[0].get("student_id")) if len(matches) == 1 else None,
                            "raw_association_status": matches[0].get("association_status") if len(matches) == 1 else
                                "ambiguous" if len(matches) > 1 else "unmatched" if associations_valid else "unavailable"})
        if len({r["person_track_id"] for r in current}) != len(current):
            problem(issues, "duplicate_person_track_id", fid)
            frame["valid"] = False
        rows.extend(current)
    return rows


def shadow_rows(source, issues):
    result = {}
    for fid, frame in source["frames"].items():
        record = frame["record"]
        observations = record.get("observations")
        if not isinstance(observations, list) or any(not isinstance(o, dict) for o in observations):
            observations = []
            frame["valid"] = False
            problem(issues, "shadow_observations_missing_or_invalid", fid)
        current = [o for o in observations if o.get("track_continuity") not in ("expired", "missing")]
        for observation in current:
            if (not isinstance(observation.get("status"), str)
                    or observation["status"] not in {"face_confirmed", "face_recovered", "track_retained", "unknown", "ambiguous"}
                    or (observation.get("proposed_student_id") is not None and not isinstance(observation["proposed_student_id"], str))):
                frame["valid"] = False
                problem(issues, "shadow_identity_metadata_invalid", fid)
        if any(not valid_id(o.get("person_track_id")) for o in current) or len({o.get("person_track_id") for o in current}) != len(current):
            frame["valid"] = False
            problem(issues, "shadow_person_id_ambiguous", fid)
        production = {}
        if record.get("production_observation_status") == "available":
            errors = []
            people, complete = shadow_production_people(record, errors, frame["line"])
            if errors or not complete:
                frame["valid"] = False
                problem(issues, "shadow_production_metadata_incomplete", {"frame_id": fid, "issues": errors})
            production = {p["person_track_id"]: p for p in people if valid_id(p["person_track_id"])}
            if set(production) != {o.get("person_track_id") for o in current}:
                frame["valid"] = False
                problem(issues, "shadow_production_person_sets_differ", fid)
        result[fid] = {}
        for observation in current:
            pid = observation.get("person_track_id")
            if not valid_id(pid):
                continue
            prod = production.get(pid, {})
            result[fid][pid] = {"shadow_frame_id": fid, "person_track_id": pid,
                "production_sid_available": prod.get("student_id_available", False),
                "production_sid": prod.get("student_id"), "identity_status": prod.get("identity_status"),
                "association_status": prod.get("association_status"), "shadow_status": observation.get("status"),
                "shadow_sid_available": "proposed_student_id" in observation,
                "shadow_sid": sid(observation.get("proposed_student_id")), "face_present": observation.get("face_present"),
                "shadow_reason": observation.get("reason")}
    return result


def make_template(person, *, shadow=None, capture_session_id, provenance="unverified"):
    if not valid_id(capture_session_id) or provenance not in {"recorded", "synthetic", "unverified"}:
        raise ValueError("Explicit capture session ID and valid provenance are required")
    sources = {"person": source_descriptor(person)}
    if shadow is not None:
        sources["shadow"] = source_descriptor(shadow)
    loaded = read_source(sources["person"], "person", Path.cwd())
    issues = list(loaded["issues"])
    catalog = [{**{k: r[k] for k in ("person_frame_id", "captured_at", "tracker_updated_at", "person_track_id", "bbox")},
                "snapshot_metadata": loaded["frames"][r["person_frame_id"]]["record"].get("snapshot")}
               for r in person_rows(loaded, issues)]
    return {"schema": SCHEMA, "capture_session_id": capture_session_id, "provenance": provenance,
            "sources": sources, "pairing": {"confirmed": False, "reviewer": "", "evidence": "", "frame_map": []},
            "actors": [], "annotations": [], "observation_catalog": catalog, "catalog_issues": issues}


def pair_frames(person, shadow, pairing, issues):
    if shadow is None:
        problem(issues, "shadow_source_not_provided", "Production SID not inferred from raw association")
        return {}
    if not reviewed(pairing):
        problem(issues, "source_pair_not_human_confirmed", "Different logger session UUIDs do not establish a shared camera run")
        return {}
    mapping = pairing.get("frame_map", [])
    if not isinstance(mapping, list):
        raise ValueError("pairing.frame_map must be a list")
    explicit = {}
    for entry in mapping:
        if (not isinstance(entry, dict) or type(entry.get("person_frame_id")) is not int
                or type(entry.get("shadow_frame_id")) is not int or not reviewed(entry)):
            raise ValueError("Frame mapping needs explicit IDs and human confirmation/evidence")
        pid, sid_ = entry["person_frame_id"], entry["shadow_frame_id"]
        if pid in explicit or pid not in person["frames"] or sid_ not in shadow["frames"]:
            raise ValueError("Duplicate or nonexistent explicit frame mapping")
        explicit[pid] = sid_
    candidates = {fid: explicit.get(fid, fid) for fid in person["frames"]}
    uses = Counter(candidates.values())
    paired = {}
    for fid, other_id in candidates.items():
        left, right = person["frames"][fid], shadow["frames"].get(other_id)
        if right is None:
            problem(issues, "shadow_frame_unmatched", fid)
        elif uses[other_id] > 1:
            problem(issues, "frame_mapping_not_one_to_one", {"person_frame_id": fid, "shadow_frame_id": other_id})
        elif not left["valid"] or not right["valid"]:
            problem(issues, "frame_integrity_prevents_pairing", fid)
        elif any(left["record"].get(k) != right["record"].get(k) for k in ("captured_at", "tracker_updated_at", "clock")):
            problem(issues, "frame_timestamp_or_clock_mismatch", {"person_frame_id": fid, "shadow_frame_id": other_id})
        else:
            paired[fid] = other_id
    return paired


def annotation_labels(manifest, rows, issues):
    actors, expected_claims = {}, defaultdict(list)
    for actor in manifest.get("actors", []):
        if not isinstance(actor, dict) or not valid_id(actor.get("actor_id")) or actor["actor_id"] in actors:
            raise ValueError("Actor IDs must be unique strings")
        valid = reviewed(actor) and actor.get("independent_of_ai") is True and actor.get("basis") == "manual_roster"
        expected = sid(actor.get("expected_student_id"))
        actors[actor["actor_id"]] = {"valid": valid, "expected_sid": expected}
        if valid and expected:
            expected_claims[expected].append(actor["actor_id"])
    for expected, owners in expected_claims.items():
        if len(owners) > 1:
            problem(issues, "actor_roster_sid_conflict", {"sid": expected, "actors": owners})
            for owner in owners:
                actors[owner]["valid"] = False
    current = {(r["person_frame_id"], r["person_track_id"]): r for r in rows}
    claims = defaultdict(list)
    for unit in manifest.get("annotations", []):
        if not isinstance(unit, dict) or not isinstance(unit.get("members"), list):
            raise ValueError("Annotations require an explicit members list")
        actor = actors.get(unit.get("actor_id"), {})
        valid = (reviewed(unit) and unit.get("independent_of_ai") is True and unit.get("basis") == "manual_visual"
                 and actor.get("valid") and unit.get("kind") in {"frame", "segment"} and bool(unit["members"]))
        if unit.get("kind") == "frame" and len(unit["members"]) != 1:
            valid = False
        if unit.get("kind") == "segment":
            start, end = unit.get("capture_start"), unit.get("capture_end")
            valid = valid and number(start) and number(end) and start <= end
        if not valid:
            problem(issues, "annotation_not_independently_confirmed", unit.get("actor_id"))
            continue
        for member in unit["members"]:
            if not isinstance(member, dict) or type(member.get("person_frame_id")) is not int or not valid_id(member.get("person_track_id")):
                problem(issues, "annotation_member_invalid", member)
                continue
            key = member["person_frame_id"], member["person_track_id"]
            row = current.get(key)
            aligned = row is not None and all(number(member.get(k)) and member[k] == row[k]
                                             for k in ("captured_at", "tracker_updated_at"))
            if unit["kind"] == "segment":
                aligned = aligned and unit["capture_start"] <= member.get("captured_at", float("inf")) <= unit["capture_end"]
            if not aligned:
                problem(issues, "annotation_observation_not_exactly_matched", member)
                continue
            claims[key].append({"actor_id": unit["actor_id"], "expected_sid": actor["expected_sid"],
                                "reviewer": unit["reviewer"], "evidence": unit["evidence"]})
    labels = {}
    for key, values in claims.items():
        if len(values) != 1:
            problem(issues, "annotation_conflict", {"observation": key, "actors": [v["actor_id"] for v in values]})
        else:
            labels[key] = values[0]
    owners = defaultdict(list)
    for key, value in labels.items():
        owners[(key[0], value["actor_id"])].append(key)
    for keys in owners.values():
        if len(keys) > 1:
            problem(issues, "actor_assigned_to_multiple_tracks_in_frame", keys)
            for key in keys:
                labels.pop(key, None)
    return labels


def verdict(available, prediction, expected, truth_valid):
    if not available:
        return "MISSING"
    if not truth_valid or expected is None:
        return "UNSCORABLE"
    if prediction is None:
        return "UNKNOWN"
    return "CORRECT" if prediction == expected else "WRONG"


def evaluate(path):
    path = Path(path).resolve()
    payload = path.read_bytes()
    manifest = strict_json(payload.decode("utf-8-sig"))
    if (not isinstance(manifest, dict) or manifest.get("schema") != SCHEMA
            or not valid_id(manifest.get("capture_session_id")) or manifest.get("provenance") not in {"recorded", "synthetic", "unverified"}):
        raise ValueError("Expected actor truth schema, explicit capture_session_id and provenance")
    sources = manifest.get("sources", {})
    person = read_source(sources.get("person"), "person", path.parent)
    shadow = read_source(sources["shadow"], "shadow", path.parent) if "shadow" in sources else None
    issues = list(person["issues"]) + (list(shadow["issues"]) if shadow else [])
    rows = person_rows(person, issues)
    shadows = shadow_rows(shadow, issues) if shadow else {}
    paired = pair_frames(person, shadow, manifest.get("pairing", {}), issues)
    pids = defaultdict(set)
    for row in rows:
        pids[row["person_frame_id"]].add(row["person_track_id"])
    for fid, other in tuple(paired.items()):
        if pids[fid] != set(shadows[other]):
            problem(issues, "paired_person_sets_differ", {"person_frame_id": fid, "shadow_frame_id": other})
            paired.pop(fid)
    labels = annotation_labels(manifest, rows, issues)
    events, output = [], []
    for row in rows:
        fid, pid = row["person_frame_id"], row["person_track_id"]
        label = labels.get((fid, pid), {})
        truth_valid = bool(label) and person["frames"][fid]["valid"]
        other_id = paired.get(fid)
        proposal = shadows.get(other_id, {}).get(pid, {})
        value = {**row, **proposal, "capture_session_id": manifest["capture_session_id"],
                 "person_session_id": person["descriptor"]["session_id"],
                 "shadow_session_id": shadow["descriptor"]["session_id"] if other_id is not None else None,
                 "pair_verified": other_id is not None, "actor_id": label.get("actor_id"),
                 "actor_truth_valid": truth_valid, "expected_sid": label.get("expected_sid"),
                 "truth_evidence": label.get("evidence"), "truth_reviewer": label.get("reviewer")}
        for channel in ("production", "shadow"):
            value[channel + "_verdict"] = verdict(proposal.get(channel + "_sid_available", False),
                proposal.get(channel + "_sid"), label.get("expected_sid"), truth_valid)
        value["false_identity_retention"] = value.get("shadow_status") == "track_retained" and value["shadow_verdict"] == "WRONG"
        value["unscorable_reason"] = ("person_source_integrity_invalid" if not person["frames"][fid]["valid"] else
            "no_valid_manual_actor_label" if not truth_valid else "actor_roster_sid_missing" if not label.get("expected_sid") else
            "shadow_frame_not_verified_or_missing" if other_id is None else
            "production_identity_not_persisted" if not proposal.get("production_sid_available") else None)
        for channel in ("production", "shadow"):
            if value[channel + "_verdict"] == "WRONG":
                events.append({"kind": "wrong_sid", "person_frame_id": fid, "person_track_id": pid,
                               "channel": channel, "detail": {"expected": value["expected_sid"], "predicted": proposal[channel + "_sid"]}})
        output.append(value)
    # Duplicate SID claims are source-local facts, even when ground truth/join is unavailable.
    duplicates = {"production": [], "shadow": [], "raw_association": []}
    groups = [(fid, list(values.values())) for fid, values in shadows.items()]
    for channel in ("production", "shadow"):
        for fid, observations in groups:
            claims = defaultdict(set)
            for value in observations:
                if value.get(channel + "_sid_available") and value.get(channel + "_sid"):
                    claims[value[channel + "_sid"]].add(value["person_track_id"])
            for student, ids in claims.items():
                if len(ids) > 1:
                    duplicates[channel].append({"shadow_frame_id": fid, "sid": student, "person_track_ids": sorted(ids)})
    for fid in person["frames"]:
        claims = defaultdict(set)
        for row in rows:
            if row["person_frame_id"] == fid and row["association_sid"]:
                claims[row["association_sid"]].add(row["person_track_id"])
        duplicates["raw_association"].extend({"person_frame_id": fid, "sid": student, "person_track_ids": sorted(ids)}
            for student, ids in claims.items() if len(ids) > 1)
    previous_actor, previous_pid = {}, {}
    for row in output:
        if not row["actor_truth_valid"]:
            continue
        actor, pid, fid = row["actor_id"], row["person_track_id"], row["person_frame_id"]
        old = previous_actor.get(actor)
        if old is not None and old["person_frame_id"] != fid:
            if old["person_track_id"] != pid:
                events.append({"kind": "labelled_actor_person_track_changed", "person_frame_id": fid, "person_track_id": pid,
                               "channel": "manual_truth", "detail": {"actor": actor, "previous_pid": old["person_track_id"]}})
            if old.get("face_present") is True and row.get("face_present") is False:
                events.append({"kind": "labelled_actor_face_lost", "person_frame_id": fid, "person_track_id": pid, "channel": "shadow", "detail": actor})
            if old.get("face_present") is False and row.get("face_present") is True:
                events.append({"kind": "labelled_actor_face_reappeared", "person_frame_id": fid, "person_track_id": pid, "channel": "shadow", "detail": actor})
        old_pid = previous_pid.get(pid)
        if old_pid is not None and old_pid["actor_id"] != actor:
            events.append({"kind": "different_labelled_actor_on_same_pid", "person_frame_id": fid, "person_track_id": pid,
                           "channel": "manual_truth", "detail": {"before": old_pid["actor_id"], "after": actor}})
        previous_actor[actor], previous_pid[pid] = row, row
    unmatched = [r for fid, values in shadows.items() if fid not in set(paired.values()) for r in values.values()]
    for fid in person["frames"]:
        frame_rows = [r for r in output if r["person_frame_id"] == fid]
        for index, left in enumerate(frame_rows):
            for right in frame_rows[index + 1:]:
                a, b = left["bbox"], right["bbox"]
                if (person["frames"][fid]["valid"] and min(a[2], b[2]) > max(a[0], b[0])
                        and min(a[3], b[3]) > max(a[1], b[1])):
                    events.append({"kind": "person_bbox_overlap", "person_frame_id": fid, "person_track_id": left["person_track_id"],
                                   "channel": "person_geometry", "detail": {"other_pid": right["person_track_id"], "identity_inferred": False}})
    metrics = {"person_ai_frames": len(person["frames"]), "shadow_ai_frames": len(shadows),
               "total_observations": len(output), "observations_with_valid_actor_truth": sum(r["actor_truth_valid"] for r in output),
               "paired_frames": len(paired), "unmatched_shadow_observations": len(unmatched),
               "unmatched_retention_observations": sum(r.get("shadow_status") == "track_retained" for r in unmatched),
               "false_identity_retention": sum(r["false_identity_retention"] for r in output),
               "retention_unscorable": sum(r.get("shadow_status") == "track_retained" and r["shadow_verdict"] not in {"CORRECT", "WRONG"} for r in output)}
    for channel in ("production", "shadow"):
        metrics[channel] = {key.lower(): sum(r[channel + "_verdict"] == key for r in output)
                            for key in ("CORRECT", "WRONG", "UNKNOWN", "MISSING", "UNSCORABLE")}
        metrics[channel]["ambiguous_status"] = sum(str(r.get("identity_status" if channel == "production" else "shadow_status", "")).startswith("ambiguous") for r in output)
        metrics[channel]["duplicate_sid_frames"] = len({d["shadow_frame_id"] for d in duplicates[channel]})
    failed = any(metrics[c]["wrong"] or duplicates[c] for c in ("production", "shadow"))
    incomplete = (not output or issues or unmatched or any(metrics[c][k] for c in ("production", "shadow")
                                                         for k in ("unknown", "missing", "unscorable", "ambiguous_status")))
    provenance = "synthetic" if person["synthetic"] or (shadow and shadow["synthetic"]) else manifest["provenance"]
    incomplete = incomplete or provenance == "unverified"
    return {"schema": "httm_camera_identity_evaluation_v1", "capture_session_id": manifest["capture_session_id"],
            "manifest": str(path), "manifest_sha256": hashlib.sha256(payload).hexdigest(), "sources": sources,
            "effective_provenance": provenance, "status": "FAIL" if failed else "INCONCLUSIVE" if incomplete else "PASS",
            "metrics": metrics, "issues": issues, "observations": output, "events": events,
            "duplicate_sid": duplicates, "unmatched_shadow_observations": unmatched, "production_applied": False,
            "limits": ["Human roster/visual evidence is declared, not independently authenticated by this tool.",
                       "PASS only compares supplied labels with logged identities; it does not approve retention or T_hold.",
                       "Total observations are current person detections; unmatched shadow rows are counted separately.",
                       "Missing production SID is never reconstructed from raw association or shadow proposal."]}


def write_report(report, directory):
    output = Path(directory)
    output.mkdir(parents=True, exist_ok=False)
    (output / "identity.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    columns = ["capture_session_id", "person_session_id", "shadow_session_id", "person_frame_id", "shadow_frame_id",
               "captured_at", "tracker_updated_at", "person_track_id", "bbox", "actor_id", "actor_truth_valid", "expected_sid",
               "association_sid", "raw_association_status", "production_sid", "identity_status", "production_verdict",
               "shadow_sid", "shadow_status", "shadow_reason", "face_present", "shadow_verdict", "pair_verified",
               "false_identity_retention", "unscorable_reason", "truth_reviewer", "truth_evidence"]
    for filename, rows, fields in (("identity.csv", report["observations"], columns),
                                  ("events.csv", report["events"], ["kind", "person_frame_id", "person_track_id", "channel", "detail"]),
                                  ("unmatched_shadow.csv", report["unmatched_shadow_observations"],
                                   sorted({k for row in report["unmatched_shadow_observations"] for k in row}) or ["shadow_frame_id"])):
        with (output / filename).open("x", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v for k, v in row.items() if k in fields})
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["# Camera identity comparison — offline", "", f"Session: {report['capture_session_id']}",
             f"Provenance: {report['effective_provenance']}; result: **{report['status']}**", "",
             "Human labels, production identity and shadow hypotheses are separate channels.", "",
             "## Counts", "", "```json", json.dumps(report["metrics"], indent=2), "```", "",
             "## Labelled observations", "", "| Frame | PID | Actor truth | Expected SID | Production | Result | Shadow | Status | Result |",
             "|---|---|---|---|---|---|---|---|---|"]
    for row in report["observations"]:
        lines.append("| " + " | ".join(cell(row.get(k)) for k in ("person_frame_id", "person_track_id", "actor_id", "expected_sid",
                     "production_sid", "production_verdict", "shadow_sid", "shadow_status", "shadow_verdict")) + " |")
    lines += ["", "## Events between labelled observations", "",
              "| Frame | PID | Channel | Event | Detail |", "|---|---|---|---|---|"]
    lines += ["| " + " | ".join(cell(e[k]) for k in ("person_frame_id", "person_track_id", "channel", "kind", "detail")) + " |"
              for e in report["events"]]
    lines += ["", "## Missing/mismatched evidence", "", *[f"- {cell(i)}" for i in report["issues"]],
              "", "## Duplicate SID claims", "", cell(report["duplicate_sid"]), "", "## Limits", "", *report["limits"]]
    (output / "identity.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init", help="Bind saved logs and create a BLANK human-label template")
    init.add_argument("--person", type=Path, required=True)
    init.add_argument("--shadow", type=Path)
    init.add_argument("--capture-session-id", required=True)
    init.add_argument("--provenance", choices=("recorded", "synthetic", "unverified"), default="unverified")
    init.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("evaluate", help="Compare exact observations with independent manual truth")
    run.add_argument("truth", type=Path)
    run.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "init":
            data = make_template(args.person, shadow=args.shadow, capture_session_id=args.capture_session_id, provenance=args.provenance)
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(data, stream, ensure_ascii=False, indent=2)
            print(str(args.output.resolve()))
            return 0
        report = evaluate(args.truth)
        write_report(report, args.output_dir)
        print(json.dumps({"output": str(args.output_dir.resolve()), "status": report["status"], "metrics": report["metrics"]}))
        return 1 if report["status"] == "FAIL" else 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Identity truth error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
