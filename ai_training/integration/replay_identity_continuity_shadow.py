"""Read-only projection of recorded identity evidence, NOT a full face replay.

PersonTracker is replayed exactly. Logged associations are not recomputed.
Without raw face boxes/scores we cannot validate recognition or seed continuity;
face_present remains None, never False. No synthetic face data is substituted.
"""

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from identity_continuity import IdentityContinuityResolver, capture_prior_tracks, project_current
from person_tracker import PersonTracker
from replay_person_tracker_diagnostics import replay_records as replay_person_records


def replay_logged_observations(records):
    records = tuple(records)
    consistency = replay_person_records(records)
    tracker = PersonTracker(**records[0]["parameters"])
    resolver = IdentityContinuityResolver(iou_threshold=tracker.iou_threshold, max_update_gap=tracker.max_age)
    frames = []
    for record in records:
        before = capture_prior_tracks(tracker)
        people = [{"person_ref": d["person_ref"], "bbox_xyxy": d["bbox"]} for d in record["detections"]]
        actual = tracker.update(people, record["tracker_updated_at"])
        matches = [{**m, "track_id": m.get("face_track_id")} for m in record["associations"]]
        projected = project_current(actual, None, matches)
        frame = resolver.update(record["frame_id"], record["captured_at"], record["tracker_updated_at"],
                                before, *projected)
        frames.append(asdict(frame))
    return {
        "scope": "partial_logged_identity_evidence_projection",
        "full_identity_replay": False,
        "association_geometry_recomputed": False,
        "missing_inputs": ["raw_current_face_detections", "face_bbox_and_recognition_score_for_validation"],
        "limitations": ["No raw face evidence is reconstructed from Student ID or Person ID.",
                        "Current face presence is unavailable, not an observed face miss.",
                        "No hold period is approved; no face-absent retention is proposed."],
        "person_tracker_consistent_frames": sum(r["consistent"] for r in consistency),
        "frame_count": len(frames), "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON file; existing files are never overwritten")
    args = parser.parse_args()
    payload = args.log.read_bytes()
    report = replay_logged_observations(json.loads(line) for line in payload.decode("utf-8-sig").splitlines() if line.strip())
    report.update(source=str(args.log.resolve()), source_sha256=hashlib.sha256(payload).hexdigest())
    text = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "frames"}, indent=2))
    return 0 if report["person_tracker_consistent_frames"] == report["frame_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
