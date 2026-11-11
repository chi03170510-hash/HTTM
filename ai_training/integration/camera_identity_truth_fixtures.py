"""Authored synthetic logs/truth for testing the comparator, not live inference."""

import json
from pathlib import Path

from camera_identity_truth import make_template


BOXES = {"P1": [20, 40, 180, 420], "P2": [320, 40, 480, 420],
         "P3": [20, 40, 180, 420], "P4": [320, 40, 480, 420]}


def confirmation(evidence="authored synthetic evidence"):
    return {"confirmed": True, "reviewer": "synthetic_fixture_author", "evidence": evidence}


def make_logs(specifications):
    """Specifications contain literal AI claims; no truth roster is consulted."""
    person, shadow = [], []
    for index, specifications in enumerate(specifications, 1):
        capture, update = 100. + index, 200. + index
        tracks, associations, observations, behaviors = [], [], [], {"sleeping": [], "using_phone": []}
        for number_, spec in enumerate(specifications):
            pid, production, proposal, status = spec
            bbox = BOXES[pid]
            tracks.append({"track_id": pid, "person_ref": f"person_{number_}", "bbox": bbox, "last_seen": update})
            if production is not None:
                associations.append({"person_track_id": pid, "person_ref": f"person_{number_}",
                                     "student_id": production, "association_status": "matched"})
            present = status in {"face_confirmed", "face_recovered"}
            observations.append({"person_track_id": pid, "proposed_student_id": proposal, "status": status,
                                 "identity_source": "synthetic_controlled_output", "association_status": "matched" if present else "unmatched",
                                 "face_present": present, "track_continuity": "same_track" if index > 1 else "new_track",
                                 "face_student_id": production if present else None, "reason": "synthetic_fixture"})
            for name in behaviors:
                behaviors[name].append({"person_track_id": pid, "student_id": production,
                    "association_status": "matched" if production else "unmatched",
                    "identity_status": "matched" if production else "unassociated",
                    f"{name}_state": "detected", f"{name}_duration_seconds": float(index - 1) if production else None,
                    f"{name}_model_confidence": .8, f"{name}_decision_reason": "synthetic_evidence",
                    "duration_event": "synthetic", "duration_blockers": [] if production else ["identity_unassociated"],
                    "duration_reset_reasons": []})
        person.append({"schema": "person_tracker_diagnostic_v1", "synthetic": True, "frame_id": index,
                       "captured_at": capture, "tracker_updated_at": update, "clock": "perf_counter",
                       "details_truncated": False, "tracks_after": tracks, "associations": associations})
        shadow.append({"schema": "identity_continuity_shadow_v1", "synthetic": True,
                       "frame_id": index, "captured_at": capture, "tracker_updated_at": update,
                       "clock": "perf_counter", "timestamp_valid": True, "production_timestamp_valid": True,
                       "production_observation_status": "available", "behavior_diagnostic_version": 2,
                       "observations": observations, **behaviors})
    return person, shadow


def write_bundle(directory, *, adversarial=False):
    """Write a NEW bundle; manual fixture actor labels are separate from AI specs."""
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=False)
    if adversarial:
        specifications = [
            [("P1", "SV001", "SV001", "face_confirmed"), ("P2", "SV002", "SV002", "face_confirmed")],
            [("P1", None, "SV001", "track_retained"), ("P2", "SV002", "SV002", "face_confirmed")],
            [("P1", "SV001", "SV001", "track_retained"), ("P2", "SV002", "SV002", "track_retained")],
            [("P1", None, None, "ambiguous"), ("P2", None, None, "ambiguous")],
            [("P3", None, "SV001", "track_retained")],
            [("P3", "SV003", "SV003", "face_recovered")],
            [("P3", "SV001", "SV001", "face_confirmed"), ("P4", "SV001", "SV001", "face_confirmed")],
        ]
        # Independently authored physical actors; do NOT derive from AI SIDs.
        actors_by_frame = [["actor_a", "actor_b"], ["actor_a", "actor_b"], ["actor_b", "actor_a"],
                           ["actor_b", "actor_a"], ["actor_c"], ["actor_c"], ["actor_c", "actor_a"]]
    else:
        specifications = [[("P1", "SV001", "SV001", "face_confirmed"),
                           ("P2", "SV002", "SV002", "face_confirmed")] for _ in range(2)]
        actors_by_frame = [["actor_a", "actor_b"], ["actor_a", "actor_b"]]
    person, shadow = make_logs(specifications)
    if adversarial:
        # Observed overlap in frame 4 is geometry, not actor truth.
        person[3]["tracks_after"][1]["bbox"] = [100, 40, 260, 420]
    paths = {}
    for name, records in (("person", person), ("shadow", shadow)):
        folder = root / f"session_{name}_synthetic"
        folder.mkdir()
        path = folder / "frames.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        paths[name] = path
    manifest = make_template(paths["person"], shadow=paths["shadow"], capture_session_id="synthetic_camera_run", provenance="synthetic")
    manifest["pairing"].update(confirmation("authored paired synthetic log streams"))
    manifest["actors"] = [{"actor_id": actor, "expected_student_id": expected,
                            "basis": "manual_roster", "independent_of_ai": True, **confirmation("authored independent roster")}
                           for actor, expected in (("actor_a", "SV001"), ("actor_b", "SV002"), ("actor_c", "SV003"))]
    for record, labels in zip(person, actors_by_frame):
        for track, actor in zip(record["tracks_after"], labels):
            member = {"person_frame_id": record["frame_id"], "captured_at": record["captured_at"],
                      "tracker_updated_at": record["tracker_updated_at"], "person_track_id": track["track_id"]}
            manifest["annotations"].append({"kind": "frame", "actor_id": actor, "basis": "manual_visual",
                                            "independent_of_ai": True, "members": [member], **confirmation()})
    path = root / "actor_truth.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (root / "PROVENANCE.json").write_text(json.dumps({"synthetic": True, "source": "authored test data, no inference or camera",
                                                     "retention_outputs": "adversarial logged proposals, not enabled policy"}), encoding="utf-8")
    return path
