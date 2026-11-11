"""Read-only replay of optional face_tracker blocks in a shared diagnostic JSONL."""

import argparse
import json
import math

from face_tracker_diagnostics import capture_tracks
from face_tracker import FaceTracker


def replay_records(records):
    tracker, parameters, previous, output = None, None, None, []
    for record in records:
        block = record.get("face_tracker")
        if (record.get("schema") != "person_tracker_diagnostic_v1" or not isinstance(block, dict)
                or block.get("schema") != "face_tracker_diagnostic_v1" or block.get("details_truncated")):
            raise ValueError("complete face_tracker blocks required; collect a new session with both debug flags")
        if ((record["frame_id"], record["captured_at"], record["tracker_updated_at"])
                != (block["frame_id"], block["captured_at"], block["person_tracker_updated_at"])):
            raise ValueError("face/person frame metadata mismatch")
        timestamp = block["face_tracker_updated_at"]
        if (not isinstance(timestamp, (int, float)) or isinstance(timestamp, bool) or not math.isfinite(timestamp)
                or (previous is not None and timestamp <= previous)):
            raise ValueError("face timestamps must be finite and increasing")
        if tracker is None:
            if block["tracks_before"] or block["next_id_before"] != 1:
                raise ValueError("replay requires the complete session starting from an empty face tracker")
            parameters = block["parameters"]
            tracker = FaceTracker(**parameters)
        if block["parameters"] != parameters:
            raise ValueError("face tracker parameters changed within session")
        before_matches = capture_tracks(tracker, timestamp) == block["tracks_before"] and tracker.next_id == block["next_id_before"]
        detections = []
        for detection in block["detections"]:
            if not all(key in detection for key in ("face_bbox", "current_student_id", "recognition_score")):
                raise ValueError("exact replay needs original recognition metadata; no missing values are invented")
            detections.append((tuple(detection["face_bbox"]), detection["current_student_id"], detection["recognition_score"]))
        actual = tracker.update(detections, timestamp)
        consistent = (before_matches and capture_tracks(tracker, timestamp) == block["tracks_after"]
                      and tracker.next_id == block["next_id_after"])
        output.append({"frame_id": block["frame_id"], "face_tracker_updated_at": timestamp,
                       "face_update_gap": None if previous is None else timestamp - previous,
                       "face_track_ids": [row[0] for row in actual], "consistent": consistent})
        previous = timestamp
    if not output:
        raise ValueError("diagnostic log is empty")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="complete shared frames.jsonl, with Face debug enabled")
    args = parser.parse_args()
    try:
        with open(args.log, encoding="utf-8") as stream:
            output = replay_records(json.loads(line) for line in stream if line.strip())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Replay rejected: {exc}\n")
    for row in output:
        print(json.dumps(row))
    return 0 if all(row["consistent"] for row in output) else 1


if __name__ == "__main__":
    raise SystemExit(main())
