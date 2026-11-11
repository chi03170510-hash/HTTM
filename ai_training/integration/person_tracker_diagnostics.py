"""Opt-in, bounded tracker diagnostics. Never changes matching or AI decisions.

Worker provenance travels in one temporary sidecar, not the production protocol.
Only the coordinator persists JSONL and explicitly requested inference snapshots.
"""

import hashlib
import json
import math
import os
import sys
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from person_tracker import calculate_iou


MAX_SOURCE_BYTES = 64 * 1024
MAX_DETAILS = 100


@dataclass(frozen=True)
class DiagnosticConfig:
    enabled: bool = False
    snapshots: bool = False
    output_dir: Path = Path("ai_training/runs/person_tracker_debug")
    max_frames: int = 10
    max_bytes: int = 20 * 1024 * 1024
    face_enabled: bool = False

    def __post_init__(self):
        if any(type(value) is not bool for value in (self.enabled, self.snapshots, self.face_enabled)):
            raise ValueError("diagnostic opt-in flags must be booleans")
        if not 0 <= self.max_frames <= 100 or not 0 <= self.max_bytes <= 100 * 1024 * 1024:
            raise ValueError("diagnostic limits: 0..100 frames, 0..100 MiB per session")

    @classmethod
    def from_environment(cls, root=None):
        if os.environ.get("HTTM_PERSON_DEBUG") != "1":
            return cls()
        def limit(name, default, maximum):
            try:
                return max(0, min(int(os.environ.get(name, default)), maximum))
            except ValueError:
                return default
        default_dir = (Path(root) if root is not None else Path.cwd()) / "ai_training/runs/person_tracker_debug"
        return cls(True, os.environ.get("HTTM_PERSON_DEBUG_SNAPSHOTS") == "1",
                   Path(os.environ.get("HTTM_PERSON_DEBUG_DIR", str(default_dir))),
                   limit("HTTM_PERSON_DEBUG_MAX_FRAMES", 10, 100),
                   limit("HTTM_PERSON_DEBUG_MAX_BYTES", 20 * 1024 * 1024, 100 * 1024 * 1024),
                   face_enabled=os.environ.get("HTTM_FACE_DEBUG") == "1")


def _box(person):
    value = person.get("person_bbox_xyxy_clipped") or person.get("bbox_xyxy")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        value = [float(v) for v in value]
    except (TypeError, ValueError, OverflowError):
        return None
    return value if all(math.isfinite(v) for v in value) and value[2] > value[0] and value[3] > value[1] else None


def detection_provenance(people, raw_people):
    """Project only actual COCO person metadata; never borrow behavior scores."""
    rows = []
    for person in people[:MAX_DETAILS]:
        ref = person.get("person_ref")
        matches = [p for p in raw_people if p.get("detection_id") == ref]
        row = {"person_ref": ref, "bbox": _box(person), "metadata_available": len(matches) == 1}
        if len(matches) == 1:
            raw = matches[0]
            for key in ("confidence", "class_id", "class_name", "highres_only", "highres_match_iou"):
                value = raw.get(key)
                if isinstance(value, (str, bool, int, float)) and (not isinstance(value, float) or math.isfinite(value)):
                    row[key] = value[:256] if isinstance(value, str) else value
            if isinstance(raw.get("sources"), list):
                row["sources"] = [s[:256] for s in raw["sources"][:16] if isinstance(s, str)]
            quality = raw.get("person_quality")
            if isinstance(quality, dict):
                row["person_quality"] = {key: quality[key] for key in (
                    "source", "confidence_object_detector", "area_frame_ratio", "width_frame_ratio",
                    "height_frame_ratio", "quality", "coarse_box_cannot_confirm_behavior") if key in quality}
        rows.append(row)
    return rows


def sidecar_path(image_path):
    return Path(image_path).with_suffix(".person_debug.json")


def write_detection_sidecar(config, image_path, frame_id, image_shape, people, raw_people):
    """One bounded, overwritten temporary file; errors never stop inference."""
    if not config.enabled or config.max_frames == 0 or config.max_bytes == 0:
        return False
    try:
        payload = json.dumps({
            "frame_id": frame_id, "image": str(Path(image_path).resolve()),
            "width": int(image_shape[1]), "height": int(image_shape[0]),
            "detections": detection_provenance(people, raw_people),
            "source_details_truncated": len(people) > MAX_DETAILS,
        }, allow_nan=False).encode("utf-8")
        if len(payload) > min(MAX_SOURCE_BYTES, config.max_bytes):
            return False
        sidecar_path(image_path).write_bytes(payload)
        return True
    except (OSError, ValueError, TypeError) as exc:
        print(f"[PERSON DEBUG] temporary provenance unavailable: {type(exc).__name__}", file=sys.stderr)
        return False


def _read_provenance(image_path, result):
    try:
        with sidecar_path(image_path).open("rb") as stream:
            payload = stream.read(MAX_SOURCE_BYTES + 1)
        if len(payload) > MAX_SOURCE_BYTES:
            return {}, "sidecar_size_limit"
        data = json.loads(payload)
        image = result["snapshot"].image
        if (data["frame_id"] != result["frame_id"] or data["image"] != str(Path(image_path).resolve())
                or (data["height"], data["width"]) != image.shape[:2]):
            return {}, "sidecar_metadata_mismatch"
        return data, "validated_sidecar"
    except FileNotFoundError:
        return {}, "not_provided_by_worker"
    except (OSError, ValueError, TypeError, KeyError):
        return {}, "invalid_sidecar"


def _tracks(tracker, timestamp):
    return [{"track_id": f"P{key}", "person_ref": track.person_ref, "bbox": list(track.bbox),
             "last_seen": track.last_seen, "age_seconds": timestamp - track.last_seen}
            for key, track in list(tracker.tracks.items())[:MAX_DETAILS]]


class PersonTrackerDiagnostics:
    def __init__(self, config):
        self.config = config
        self.session_dir = None
        self.frames_written = self.bytes_written = 0
        self._previous_update = self._previous_capture = None
        self._stopped = False
        self._face_diagnostics = None
        if config.enabled and config.face_enabled:
            from face_tracker_diagnostics import FaceTrackerDiagnostics
            self._face_diagnostics = FaceTrackerDiagnostics()

    @property
    def active(self):
        return (self.config.enabled and not self._stopped
                and self.frames_written < self.config.max_frames and self.bytes_written < self.config.max_bytes)

    def before_update(self, tracker, timestamp):
        if not self.active:
            return None
        # Copy only; all IoU calculations and disk I/O happen AFTER the real update.
        return {"tracker_updated_at": timestamp, "tracks": _tracks(tracker, timestamp),
                "track_count": len(tracker.tracks),
                "parameters": {"iou_threshold": tracker.iou_threshold, "max_age": tracker.max_age,
                               "keep_unmatched": tracker.keep_unmatched}}

    def before_face_update(self, tracker, timestamp):
        if not self.active or self._face_diagnostics is None:
            return None
        try:
            return self._face_diagnostics.before_update(tracker, timestamp)
        except Exception as exc:
            self._stopped = True
            print(f"[FACE DEBUG] stopped; AI continues: {type(exc).__name__}", file=sys.stderr)
            return None

    def record(self, before, tracker, people, result, image_path, *, face_context=None):
        if before is None or not self.active:
            return None
        try:
            return self._record(before, tracker, people, result, image_path, face_context=face_context)
        except Exception as exc:
            self._stopped = True
            print(f"[PERSON DEBUG] stopped; AI continues: {type(exc).__name__}", file=sys.stderr)
            return None

    def _record(self, before, tracker, people, result, image_path, *, face_context=None):
        updated = before["tracker_updated_at"]
        after = _tracks(tracker, updated)
        previous = {t["track_id"]: t for t in before["tracks"]}
        current = {t["track_id"]: t for t in after}
        data, status = _read_provenance(image_path, result)
        sources = data.get("detections", [])
        detections = []
        for index, person in enumerate(people[:MAX_DETAILS]):
            row = {"detection_index": index, "person_ref": person.get("person_ref"), "bbox": _box(person),
                   "metadata_available": False}
            source = [s for s in sources if s.get("person_ref") == row["person_ref"] and s.get("bbox") == row["bbox"]]
            if len(source) == 1:
                row.update({key: source[0][key] for key in (
                    "metadata_available", "confidence", "class_id", "class_name", "sources", "person_quality",
                    "highres_only", "highres_match_iou") if key in source[0]})
            detections.append(row)
        fingerprints = Counter((d["person_ref"], tuple(d["bbox"]) if d["bbox"] is not None else None) for d in detections)
        pairs, decisions, eligible_live_detections = [], [], set()
        for old in before["tracks"]:
            expired = old["age_seconds"] > before["parameters"]["max_age"]
            eligible = False
            for det in detections:
                if det["bbox"] is None:
                    continue
                overlap = calculate_iou(old["bbox"], det["bbox"])
                new = current.get(old["track_id"])
                selected = bool(new and new["last_seen"] == updated and new["person_ref"] == det["person_ref"]
                                and new["bbox"] == det["bbox"])
                fingerprint = (det["person_ref"], tuple(det["bbox"]))
                ambiguous = fingerprints[fingerprint] > 1
                threshold_passed = overlap >= before["parameters"]["iou_threshold"]
                eligible = eligible or threshold_passed
                if not expired and threshold_passed:
                    eligible_live_detections.add(fingerprint)
                pairs.append({"track_id": old["track_id"], "detection_index": det["detection_index"],
                              "person_ref": det["person_ref"], "iou": overlap,
                              "threshold_passed": threshold_passed,
                              "considered_by_tracker": not expired,
                              "selected": None if ambiguous else selected})
            if expired:
                decisions.append({"track_id": old["track_id"], "action": "expired", "reason": "age_exceeds_max_age"})
            elif old["track_id"] not in current:
                decisions.append({"track_id": old["track_id"], "action": "unmatched_removed",
                                  "reason": "not_selected_by_greedy_matching" if eligible else "no_eligible_iou"})
        for new in after:
            if new["last_seen"] != updated:
                continue
            created = new["track_id"] not in previous
            eligible = (new["person_ref"], tuple(new["bbox"])) in eligible_live_detections
            reason = ("eligible_tracks_assigned_elsewhere" if eligible else "no_eligible_live_track") if created else "selected_by_tracker"
            decisions.append({"track_id": new["track_id"], "person_ref": new["person_ref"],
                              "action": "created" if created else "matched", "reason": reason})
        image = result["snapshot"].image
        record = {
            "schema": "person_tracker_diagnostic_v1", "frame_id": result["frame_id"],
            "captured_at": result["captured_at"], "tracker_updated_at": updated,
            "completed_at": result["completed_at"], "clock": "perf_counter",
            "tracker_update_gap": None if self._previous_update is None else updated - self._previous_update,
            "captured_at_gap": None if self._previous_capture is None else result["captured_at"] - self._previous_capture,
            "image_dimensions": {"width": int(image.shape[1]), "height": int(image.shape[0])},
            "processing": {"elapsed": result["elapsed"], "face": result["face_time"], "hybrid": result["hybrid_time"]},
            "parameters": before["parameters"], "tracks_before": before["tracks"], "tracks_after": after,
            "detections": detections, "candidate_ious": pairs, "decisions": decisions,
            "detection_metadata_status": status,
            "details_truncated": before["track_count"] > MAX_DETAILS or len(tracker.tracks) > MAX_DETAILS
                                 or len(people) > MAX_DETAILS or len(result["matches"]) > MAX_DETAILS
                                 or data.get("source_details_truncated", False),
            "associations": [{"person_ref": m.get("person_ref"), "person_track_id": m.get("person_track_id"),
                              "association_status": m.get("association_status"), "face_track_id": m.get("track_id"),
                              "student_id": m.get("student_id")} for m in result["matches"][:MAX_DETAILS]],
            "snapshot": {"saved": False, "reason": "disabled"},
            "persistent_debug_write_excluded_from_processing": True,
        }
        if face_context is not None and self._face_diagnostics is not None:
            face_before, face_tracker, faces = face_context
            record["face_tracker"] = self._face_diagnostics.build(face_before, face_tracker, faces, result, updated)
        snapshot_bytes = None
        if self.config.snapshots:
            snapshot = result["snapshot"]
            if (snapshot.frame_id, snapshot.captured_at) != (result["frame_id"], result["captured_at"]):
                raise ValueError("diagnostic snapshot metadata mismatch")
            import cv2
            ok, encoded = cv2.imencode(".png", image)
            if ok:
                snapshot_bytes = encoded.tobytes()
                record["snapshot"] = {"saved": True, "file": f"obs_{self.frames_written + 1:03d}_frame_{result['frame_id']:06d}.png",
                                      "frame_id": snapshot.frame_id, "captured_at": snapshot.captured_at,
                                      "sha256": hashlib.sha256(snapshot_bytes).hexdigest()}
            else:
                record["snapshot"] = {"saved": False, "reason": "encode_failed"}
        def encode():
            return (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode("utf-8")
        payload = encode()
        if self.bytes_written + len(payload) + len(snapshot_bytes or b"") > self.config.max_bytes and snapshot_bytes is not None:
            snapshot_bytes = None
            record["snapshot"] = {"saved": False, "reason": "byte_limit"}
            payload = encode()
        if self.bytes_written + len(payload) > self.config.max_bytes:
            self._stopped = True
            print("[PERSON DEBUG] byte limit reached; AI continues", file=sys.stderr)
            return None
        if self.session_dir is None:
            self.session_dir = self.config.output_dir / ("session_" + uuid.uuid4().hex)
            self.session_dir.mkdir(parents=True, exist_ok=False)
            print(f"[PERSON DEBUG] {self.session_dir} (limits: {self.config.max_frames} frames, {self.config.max_bytes} bytes)", file=sys.stderr)
        if snapshot_bytes is not None:
            with (self.session_dir / record["snapshot"]["file"]).open("xb") as stream:
                stream.write(snapshot_bytes)
            self.bytes_written += len(snapshot_bytes)
        with (self.session_dir / "frames.jsonl").open("ab") as stream:
            stream.write(payload)
        self.bytes_written += len(payload)
        self.frames_written += 1
        self._previous_update, self._previous_capture = updated, result["captured_at"]
        return record
