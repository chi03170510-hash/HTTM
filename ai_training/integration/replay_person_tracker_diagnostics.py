"""Replay a complete diagnostic JSONL session through the unchanged tracker.

Read-only CLI: python replay_person_tracker_diagnostics.py path/to/frames.jsonl
Capture gaps and processing durations are reported, never used as update clocks.
"""

import argparse
import json
import math

from person_tracker import PersonTracker


def replay_records(records):
    tracker, parameters, results = None, None, []
    for record in records:
        if record.get("schema") != "person_tracker_diagnostic_v1" or record.get("details_truncated"):
            raise ValueError("replay requires complete diagnostic_v1 records without truncation")
        if tracker is None:
            if record["tracks_before"]:
                raise ValueError("replay requires the complete session beginning with an empty tracker")
            parameters = record["parameters"]
            tracker = PersonTracker(**parameters)
        if record["parameters"] != parameters:
            raise ValueError("tracker parameters changed inside diagnostic session")
        timestamp = record["tracker_updated_at"]
        if not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("invalid tracker timestamp")
        people = [{"person_ref": d["person_ref"], "bbox_xyxy": d["bbox"]} for d in record["detections"]]
        actual = tracker.update(people, timestamp)
        expected = [{key: item[key] for key in ("track_id", "person_ref", "bbox")}
                    for item in record["tracks_after"] if item["last_seen"] == timestamp]
        actual = json.loads(json.dumps(actual))
        expected.sort(key=lambda item: int(item["track_id"][1:]))
        results.append({"frame_id": record["frame_id"], "tracked_people": actual,
                        "consistent": actual == expected, "tracker_update_gap": record["tracker_update_gap"],
                        "captured_at_gap": record["captured_at_gap"], "processing": record["processing"]["elapsed"]})
    if not results:
        raise ValueError("diagnostic log is empty")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", help="frames.jsonl from a diagnostic session")
    args = parser.parse_args()
    try:
        with open(args.log, encoding="utf-8") as stream:
            results = replay_records(json.loads(line) for line in stream if line.strip())
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Replay rejected: {exc}\n")
    for result in results:
        print(json.dumps(result))
    return 0 if all(result["consistent"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
