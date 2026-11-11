"""Observe the unchanged FaceTracker; no image, embedding, model or I/O access."""

import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "recognition"))
from face_tracker import calculate_iou


MAX_FACE_DETAILS = 100


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def capture_tracks(tracker, timestamp):
    rows = []
    for track in list(tracker.tracks.values())[:MAX_FACE_DETAILS]:
        row = {"face_track_id": track.track_id, "face_bbox": list(track.bbox),
               "last_seen": track.last_seen, "age_seconds": timestamp - track.last_seen,
               "stable_label": track.stable_label, "updated_in_frame": track.last_seen == timestamp}
        if track.labels:
            row["last_student_id"] = track.labels[-1]
        if _number(track.score):
            row["recognition_score"] = track.score
        rows.append(row)
    return rows


def _fingerprint(detection):
    return (tuple(detection["face_bbox"]), detection.get("current_student_id"),
            detection.get("recognition_score"))


def _matches(track, detection):
    return (track["updated_in_frame"] and track["face_bbox"] == detection["face_bbox"]
            and ("current_student_id" not in detection or track.get("last_student_id") == detection["current_student_id"])
            and ("recognition_score" not in detection or track.get("recognition_score") == detection["recognition_score"]))


class FaceTrackerDiagnostics:
    def __init__(self):
        self._previous_update = None

    def before_update(self, tracker, timestamp):
        return {"face_tracker_updated_at": timestamp, "tracks": capture_tracks(tracker, timestamp),
                "track_count": len(tracker.tracks), "next_id": tracker.next_id,
                "parameters": {"iou_threshold": tracker.iou_threshold, "max_age": tracker.max_age}}

    def build(self, before, tracker, faces, result, person_updated_at):
        """Compute diagnostics AFTER update, inferring selections from actual state.

        Identity/score help identify the consumed detection in the output only;
        the production matching itself continues to use IoU alone.
        """
        timestamp = before["face_tracker_updated_at"]
        after = capture_tracks(tracker, timestamp)
        old_by_id = {t["face_track_id"]: t for t in before["tracks"]}
        new_by_id = {t["face_track_id"]: t for t in after}
        detections = []
        for index, face in enumerate(faces[:MAX_FACE_DETAILS]):
            row = {"detection_index": index, "face_bbox": list(face["face_bbox"])}
            if "student_id" in face:
                row["current_student_id"] = face["student_id"]
            if _number(face.get("score")):
                row["recognition_score"] = face["score"]
            detections.append(row)
        counts = Counter(_fingerprint(d) for d in detections)
        assignments = {t["face_track_id"]: [d["detection_index"] for d in detections if _matches(t, d)]
                       for t in after if t["updated_in_frame"]}
        pairs, decisions, live_eligible = [], [], set()
        for old in before["tracks"]:
            tid = old["face_track_id"]
            expired = old["age_seconds"] > before["parameters"]["max_age"]
            eligible = False
            for detection in detections:
                iou = calculate_iou(old["face_bbox"], detection["face_bbox"])
                passed = iou >= before["parameters"]["iou_threshold"]
                eligible = eligible or passed
                fingerprint = _fingerprint(detection)
                if not expired and passed:
                    live_eligible.add(fingerprint)
                selected = detection["detection_index"] in assignments.get(tid, [])
                pairs.append({"face_track_id": tid, "detection_index": detection["detection_index"],
                              "iou": iou, "threshold_passed": passed,
                              "considered_by_tracker": not expired,
                              "selected": None if counts[fingerprint] > 1 or len(assignments.get(tid, [])) > 1 else selected})
            if expired:
                decisions.append({"face_track_id": tid, "action": "expired", "reason": "age_exceeds_max_age"})
            elif tid in new_by_id and not new_by_id[tid]["updated_in_frame"]:
                decisions.append({"face_track_id": tid, "action": "unmatched_retained",
                                  "reason": "not_selected_by_greedy_matching" if eligible else "no_eligible_iou"})
        mapping = []
        for current in after:
            if not current["updated_in_frame"]:
                continue
            tid = current["face_track_id"]
            indices = assignments[tid]
            ambiguous = len(indices) != 1
            created = tid not in old_by_id
            eligible = any(_fingerprint(d) in live_eligible for d in detections if d["detection_index"] in indices)
            reason = ("eligible_tracks_assigned_elsewhere" if eligible else "no_eligible_live_track") if created else "selected_by_tracker"
            decisions.append({"face_track_id": tid, "action": "new" if created else "matched", "reason": reason})
            # Include unassociated faces, not only faces visible in the panel.
            links = []
            for match in result["matches"][:MAX_FACE_DETAILS]:
                if match.get("track_id") == tid:
                    link = {key: match[key] for key in ("person_ref", "person_track_id", "association_status", "student_id")
                            if key in match}
                    links.append(link)
            mapping.append({"face_track_id": tid, "detection_indices": indices,
                            "selection_ambiguous": ambiguous, "associations": links})
        block = {
            "schema": "face_tracker_diagnostic_v1", "frame_id": result["frame_id"],
            "captured_at": result["captured_at"], "face_tracker_updated_at": timestamp,
            "face_update_gap": None if self._previous_update is None else timestamp - self._previous_update,
            "person_tracker_updated_at": person_updated_at, "face_to_person_seconds": person_updated_at - timestamp,
            "processing": {"face": result["face_time"], "hybrid": result["hybrid_time"]},
            "parameters": before["parameters"], "next_id_before": before["next_id"], "next_id_after": tracker.next_id,
            "face_detection_count": len(faces), "track_count_before": before["track_count"],
            "track_count_after": len(tracker.tracks),
            "detections": detections, "tracks_before": before["tracks"], "tracks_after": after,
            "candidate_ious": pairs, "decisions": decisions, "mapping": mapping,
            "details_truncated": before["track_count"] > MAX_FACE_DETAILS or len(tracker.tracks) > MAX_FACE_DETAILS
                                 or len(faces) > MAX_FACE_DETAILS or len(result["matches"]) > MAX_FACE_DETAILS,
        }
        self._previous_update = timestamp
        return block
