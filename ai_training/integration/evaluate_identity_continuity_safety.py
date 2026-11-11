"""Offline synthetic identity safety evaluation; never authorizes retention.

CLI only evaluates the current no-hold policy. Explicit simulation_hold_seconds
is an API for unit tests, not an environment variable, CLI or production policy.
Actor ground truth never enters the tracker, resolver or association inputs.
"""

import argparse
import ast
import csv
import hashlib
import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

# Allow the standalone CLI under -I -S without site-packages or environment paths.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from identity_continuity import IdentityContinuityResolver, capture_prior_tracks, project_current
from identity_continuity_safety_fixtures import ACTOR_STUDENTS, safety_fixtures
from person_tracker import PersonTracker


ROOT = Path(__file__).resolve().parents[2]


def production_tracker_policy():
    """Read existing literal tracker settings without importing the camera app."""
    source = ROOT / "ai_training/integration/camera_coordinator_live.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    constants = {target.id: node.value for node in tree.body if isinstance(node, ast.Assign)
                 for target in node.targets if isinstance(target, ast.Name)}
    loop = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "ai_loop")
    calls = [node for node in ast.walk(loop) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "PersonTracker"]
    if len(calls) != 1:
        raise ValueError("Cannot uniquely read existing Camera Live PersonTracker policy")
    values = {kw.arg: ast.literal_eval(constants[kw.value.id] if isinstance(kw.value, ast.Name) else kw.value)
              for kw in calls[0].keywords}
    if set(values) != {"iou_threshold", "max_age", "keep_unmatched"}:
        raise ValueError("Unknown Camera Live tracker policy shape; no settings guessed")
    return values


def observation_inputs(sample):
    """Strip ALL actor labels before production algorithms see the sample."""
    people, faces, matches = [], [], []
    unavailable = any(d.face_kind == "unavailable" for d in sample.detections)
    for index, detection in enumerate(sample.detections):
        if detection.face_kind not in {"present", "absent", "unavailable"}:
            raise ValueError("Unknown synthetic face availability")
        ref = f"person_{index}"
        people.append({"person_ref": ref, "bbox_xyxy": list(detection.bbox)})
        if detection.face_kind == "present":
            x1, y1, x2, y2 = detection.bbox
            bbox = [x1 + .3 * (x2 - x1), y1 + .05 * (y2 - y1),
                    x1 + .6 * (x2 - x1), y1 + .2 * (y2 - y1)]
            faces.append({"face_bbox": bbox, "student_id": detection.recognized_student_id, "score": .82})
            if detection.association_status is not None:
                matches.append({"person_ref": ref, "track_id": sample.frame_id * 100 + index,
                                "student_id": detection.recognized_student_id,
                                "association_status": detection.association_status,
                                "face_bbox": bbox, "face_score": .82})
    return people, None if unavailable else faces, matches


def evaluate_fixture(fixture, *, simulation_hold_seconds=None):
    if fixture.synthetic is not True:
        raise ValueError("Safety fixtures must be explicitly synthetic")
    if not fixture.frames:
        raise ValueError("An empty fixture cannot establish identity safety")
    policy = production_tracker_policy()
    tracker = PersonTracker(**policy)
    resolver = IdentityContinuityResolver(iou_threshold=tracker.iou_threshold, max_update_gap=tracker.max_age,
                                          simulation_hold_seconds=simulation_hold_seconds)
    metrics = Counter({key: 0 for key in (
        "false_identity_retentions", "false_fresh_face_proposals", "duplicate_sid_frames",
        "wrong_track_retention_transfers", "correct_retentions", "unresolved_identities",
        "unscorable_observations", "replacement_retentions", "current_observations", "wrong_identity_proposals")})
    trace, sid_origins, previous_actors = [], {}, {}
    seen_frames = set()
    for sample in fixture.frames:
        if sample.frame_id in seen_frames or len(sample.actor_labels) != len(sample.detections):
            raise ValueError("Fixture needs unique frame IDs and one independent actor label per detection")
        seen_frames.add(sample.frame_id)
        if any(actor is not None and actor not in ACTOR_STUDENTS for actor in sample.actor_labels):
            raise ValueError("Unknown actor label, not a recognized SID")
        before = capture_prior_tracks(tracker)
        people, faces, matches = observation_inputs(sample)
        tracked = tracker.update(people, sample.updated_at)
        ids = {p["person_ref"]: p["track_id"] for p in tracked}
        if len(tracked) != len(sample.detections) or len(ids) != len(sample.detections):
            raise ValueError("Synthetic detections were filtered or ambiguously mapped; cannot score actor truth")
        for match in matches:
            match["person_track_id"] = ids.get(match["person_ref"])
        result = resolver.update(sample.frame_id, sample.captured_at, sample.updated_at,
                                 before, *project_current(tracked, faces, matches))
        # Only evaluation sees this map. person_ref is a CURRENT-frame join,
        # never historical identity or a label derived from Student/Person ID.
        actors = {ids[f"person_{i}"]: actor for i, actor in enumerate(sample.actor_labels) if f"person_{i}" in ids}
        proposed = Counter(r.proposed_student_id for r in result.observations
                           if r.person_track_id in actors and r.proposed_student_id is not None)
        metrics["duplicate_sid_frames"] += any(n > 1 for n in proposed.values())
        rows = []
        for row in result.observations:
            current = row.person_track_id in actors
            actor = actors.get(row.person_track_id)
            expected = ACTOR_STUDENTS.get(actor)
            wrong = None if expected is None or row.proposed_student_id is None else row.proposed_student_id != expected
            transfer = (row.status == "track_retained" and row.proposed_student_id in sid_origins
                        and sid_origins[row.proposed_student_id] != row.person_track_id)
            replaced = (current and actor is not None and previous_actors.get(row.person_track_id) is not None
                        and previous_actors[row.person_track_id] != actor)
            if current:
                metrics["current_observations"] += 1
                metrics["unscorable_observations"] += expected is None
                metrics["unresolved_identities"] += row.proposed_student_id is None
                metrics["wrong_identity_proposals"] += wrong is True
                metrics["false_identity_retentions"] += row.status == "track_retained" and wrong is True
                metrics["false_fresh_face_proposals"] += row.identity_source == "current_face_recognition" and wrong is True
                metrics["correct_retentions"] += row.status == "track_retained" and wrong is False
                metrics["wrong_track_retention_transfers"] += transfer
                metrics["replacement_retentions"] += row.status == "track_retained" and replaced
            rows.append({**asdict(row), "current_person": current, "ground_truth_actor": actor,
                         "ground_truth_student_id": expected, "wrong_proposal": wrong,
                         "actor_changed_on_same_track": replaced, "cached_transfer_to_different_track": transfer})
        for row in result.observations:
            if row.identity_source == "current_face_recognition" and row.proposed_student_id is not None:
                sid_origins[row.proposed_student_id] = row.person_track_id
        trace.append({"frame_id": sample.frame_id, "captured_at": sample.captured_at,
                      "tracker_updated_at": sample.updated_at, "timestamp_valid": result.timestamp_valid,
                      "reset_reason": result.reset_reason, "observations": rows})
        previous_actors = actors
    failures = sum(metrics[key] for key in ("false_identity_retentions", "false_fresh_face_proposals",
                                           "duplicate_sid_frames", "wrong_track_retention_transfers"))
    status = "FAIL" if failures else "INCONCLUSIVE" if metrics["unscorable_observations"] else "PASS"
    return {"case_id": fixture.name, "description": fixture.description, "synthetic": True,
            "safety_status": status, "metrics": dict(metrics), "trace": trace,
            "limits": ["Safety status evaluates these labelled synthetic observations only, not camera/model accuracy.",
                       "PASS with no retained SID does not demonstrate usable continuity.",
                       "Recognition SID and Person Track ID never supply actor ground truth."]}


def evaluate_suite(*, simulation_hold_seconds=None):
    cases = [evaluate_fixture(case, simulation_hold_seconds=simulation_hold_seconds) for case in safety_fixtures()]
    return {"schema": "identity_continuity_safety_v1", "provenance": "synthetic",
            "profile": "unit_test_simulation" if simulation_hold_seconds is not None else "current_no_hold_shadow",
            "simulation_hold_seconds": simulation_hold_seconds, "production_T_hold": None,
            "production_applied": False, "tracker_policy_from_current_source": production_tracker_policy(),
            "counts": {status: sum(c["safety_status"] == status for c in cases) for status in ("PASS", "FAIL", "INCONCLUSIVE")},
            "metrics": {key: sum(c["metrics"][key] for c in cases) for key in cases[0]["metrics"]}, "cases": cases}


def audit_recorded_sessions(root=ROOT / "ai_training/runs/person_tracker_debug"):
    """Read timing/evidence coverage, without treating recorded SID as truth."""
    sessions, capture_gaps, update_gaps = [], [], []
    for path in sorted(Path(root).glob("session_*/frames.jsonl")):
        payload = path.read_bytes()
        records = [json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
        raw_face_frames = sum(isinstance(r.get("face_tracker"), dict)
                              and isinstance(r["face_tracker"].get("detections"), list) for r in records)
        for a, b in zip(records, records[1:]):
            capture_gaps.append(b["captured_at"] - a["captured_at"])
            update_gaps.append(b["tracker_updated_at"] - a["tracker_updated_at"])
        sessions.append({"source": str(path.resolve()), "source_sha256": hashlib.sha256(payload).hexdigest(),
                         "frames": len(records), "raw_face_diagnostic_frames": raw_face_frames,
                         "independent_actor_truth_supplied": False})
    def stats(values):
        return {"count": len(values), "min": min(values), "mean": sum(values) / len(values), "max": max(values)} if values else None
    return {"scope": "recorded_person_tracker_sessions_only; not synthetic validation folders",
            "sessions": sessions, "session_count": len(sessions), "frame_count": sum(s["frames"] for s in sessions),
            "capture_gap_seconds": stats(capture_gaps), "tracker_update_gap_seconds": stats(update_gaps),
            "production_T_hold": None, "T_hold_selection": "INCONCLUSIVE",
            "reason": "Timing gaps and recorded SID do not label actor continuity, recognition correctness or replacement risk.",
            "needed_evidence": ["Actor-labelled face-loss intervals and fresh re-identification, with frame/capture/update times.",
                                "Same-person and replacement/crossing/overlap episodes, including hidden changes between AI updates.",
                                "Using Phone/Sleeping diagnostics and identity eligibility, without treating behavior as identity evidence."]}


def write_report(report, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output / "safety.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    with (output / "safety.csv").open("x", encoding="utf-8-sig", newline="") as stream:
        fields = ["case_id", "safety_status", *report["metrics"]]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in report["cases"]:
            writer.writerow({"case_id": case["case_id"], "safety_status": case["safety_status"], **case["metrics"]})
    lines = ["# Identity Continuity — synthetic safety evaluation", "", f"Profile: {report['profile']}",
             f"Unit-test simulation hold: {report['simulation_hold_seconds']}; production T_hold: unapproved / None.",
             "No camera/model validation. Actor labels are independent of observed recognition and PID.", "",
             f"Counts: {report['counts']}", "", "| Case | Safety | False retained SID | False fresh face SID | Correct retentions | Unresolved |",
             "|---|---|---:|---:|---:|---:|"]
    for c in report["cases"]:
        m = c["metrics"]
        lines.append(f"| {c['case_id']} | {c['safety_status']} | {m['false_identity_retentions']} | {m['false_fresh_face_proposals']} | {m['correct_retentions']} | {m['unresolved_identities']} |")
    lines += ["", "Safety PASS with no retained SID does not establish continuity. Stable PID is never identity ground truth.",
              "FAIL measures wrong labelled proposals, including inherited recognition errors; a unit test detecting that FAIL can still pass.",
              "INCONCLUSIVE means actor truth is unavailable. No production T_hold recommendation is derived from this synthetic report."]
    (output / "safety.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="NEW directory; never overwritten")
    args = parser.parse_args(argv)
    try:
        report = evaluate_suite()  # Deliberately no CLI/environment access to a hold period.
        report["recorded_evidence_audit"] = audit_recorded_sessions()
        write_report(report, args.output_dir)
        print(json.dumps({"output": str(args.output_dir.resolve()), "counts": report["counts"], "production_T_hold": None}))
        return 1 if report["counts"]["FAIL"] else 0
    except (OSError, ValueError) as exc:
        parser.exit(2, f"Safety evaluation error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
