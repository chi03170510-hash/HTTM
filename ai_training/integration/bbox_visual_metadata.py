"""Read-only display sidecars. No model calls, identity decisions or history.

Only the existing opt-in snapshot overlay exports these temporary files. Worker
protocol replies remain unchanged. Coordinates are original-image pixel xyxy.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import sys


SCHEMA = "bbox_visual_source_v1"
MAX_ROWS = 1000  # Diagnostic resource bound, not a detection threshold.
MAX_BYTES = 2 * 1024 * 1024


def enabled():
    return os.environ.get("HTTM_PERSON_SNAPSHOT_OVERLAY") == "1"


def diagnostic(message):
    print(f"[BBOX VISUAL] {message}", file=sys.stderr, flush=True)


def sidecar_path(image_path, role):
    if role not in {"face", "hybrid"}:
        raise ValueError("unknown visual source")
    return Path(image_path).with_suffix(f".{role}_visual.json")


def _fields(row, names):
    return {key: row[key] for key in names if key in row}


def write_source(image_path, frame_id, shape, role, *, faces=(), detections=(), objects=None, people=()):
    """Copy a whitelist of existing evidence; failures cannot fail inference."""
    try:
        if role == "face":
            rows = {"faces": [_fields(f, ("face_bbox", "student_id", "score")) for f in faces]}
        else:
            objects = objects or {}
            rows = {
                "candidates": [_fields(d, ("bbox_xyxy", "class_name", "confidence", "source", "person_ref",
                                            "association_status", "association_ambiguous"))
                               for d in detections if d.get("class_name") in {"sleeping", "using_phone"}],
                "phones": [_fields(d, ("detection_id", "bbox_xyxy", "confidence", "source"))
                           for d in objects.get("phone_detections", [])],
                "phone_associations": [_fields(d, ("phone_detection_id", "person_detection_id", "status"))
                                       for d in objects.get("v6_phone_person_associations", [])],
                "decisions": [{"person_ref": p.get("person_ref"), "name": name,
                               **_fields(b, ("state", "bounding_box"))}
                              for p in people for name, b in p.get("behaviors", {}).items()
                              if name in {"sleeping", "using_phone"}],
            }
        if sum(len(v) for v in rows.values()) > MAX_ROWS:
            raise ValueError("visual source row limit exceeded; layer omitted")
        path = Path(image_path)
        packet = {"schema": SCHEMA, "role": role, "frame_id": frame_id,
                  "image": str(path.resolve()), "jpeg_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                  "source_size": [int(shape[1]), int(shape[0])], **rows}
        data = json.dumps(packet, ensure_ascii=False, allow_nan=False).encode("utf-8")
        if len(data) > MAX_BYTES:
            raise ValueError("visual source byte limit exceeded; layer omitted")
        sidecar_path(path, role).write_bytes(data)
        return True
    except Exception as exc:
        diagnostic(f"frame={frame_id} {role} export skipped: {type(exc).__name__}: {exc}")
        return False


@dataclass(frozen=True)
class VisualBox:
    kind: str
    bbox: tuple
    score: float | None
    owner_pid: str | None = None
    identity: str = "Unknown"
    face_track_id: int | str | None = None
    recognition: str | None = None
    state: str | None = None
    duration: float | None = None
    selected: bool = False
    provenance: str | None = None


@dataclass(frozen=True)
class VisualMetadata:
    frame_id: int
    captured_at: float
    source_size: tuple
    pixels_sha256: str
    jpeg_sha256: str
    boxes: tuple[VisualBox, ...]
    issues: tuple[str, ...] = ()

    def validate(self, snapshot):
        if (self.frame_id != snapshot.frame_id or self.captured_at != snapshot.captured_at
                or self.source_size != (snapshot.image.shape[1], snapshot.image.shape[0])
                or self.pixels_sha256 != hashlib.sha256(snapshot.image.tobytes()).hexdigest()):
            raise ValueError("display metadata/frame/decoded JPEG pixels mismatch")


def _box(value):
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in value):
        return None
    return tuple(value) if value[2] > value[0] and value[3] > value[1] else None


def _score(value):
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def _known(value):
    return isinstance(value, str) and value.strip() and value.strip().casefold() not in {"unknown", "none", "null"}


def _read(path, role, frame_id, size, jpeg_hash):
    source = sidecar_path(path, role)
    if source.stat().st_size > MAX_BYTES:
        raise ValueError("visual source byte limit exceeded")
    packet = json.loads(source.read_text(encoding="utf-8"))
    if (packet.get("schema") != SCHEMA or packet.get("role") != role or type(packet.get("frame_id")) is not int
            or packet["frame_id"] != frame_id or packet.get("image") != str(Path(path).resolve())
            or packet.get("source_size") != list(size) or packet.get("jpeg_sha256") != jpeg_hash):
        raise ValueError("visual source frame/path/size/JPEG hash mismatch")
    keys = ("faces",) if role == "face" else ("candidates", "phones", "phone_associations", "decisions")
    if any(not isinstance(packet.get(k), list) or any(not isinstance(r, dict) for r in packet[k]) for k in keys):
        raise ValueError("invalid visual source rows")
    if sum(len(packet[k]) for k in keys) > MAX_ROWS:
        raise ValueError("visual source row limit exceeded")
    return packet


def collect(snapshot, image_path, face_result, tracked_faces):
    """Join within ONE observation using verified keys, never list positions.

    person_ref is a same-frame key only. Production PersonBehaviorView is the
    sole source of displayed SID/state/duration; raw face SID is never displayed.
    """
    from behavior_overlay import _production_person_labels
    import cv2
    import numpy as np

    size = (snapshot.image.shape[1], snapshot.image.shape[0])
    decoded = cv2.imread(str(image_path))
    if decoded is None or not np.array_equal(decoded, snapshot.image):
        raise ValueError("snapshot pixels do not match worker JPEG")
    jpeg_hash = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
    issues, boxes, sources = [], [], {}
    for role in ("face", "hybrid"):
        try:
            sources[role] = _read(image_path, role, snapshot.frame_id, size, jpeg_hash)
        except Exception as exc:
            issues.append(f"{role}: {type(exc).__name__}: {exc}")
    people = snapshot.observations.people
    refs = defaultdict(list)
    for p in people:
        refs[p.person_ref].append(p)
    pids = Counter(p.person_track_id for p in people)
    labels = _production_person_labels(people)

    def owner(ref):
        rows = refs.get(ref, ())
        if len(rows) == 1 and pids[rows[0].person_track_id] == 1:
            return rows[0]
        if ref is not None:
            issues.append(f"person_ref={ref}: missing/duplicate person key; owner omitted")
        return None

    def attach(item, p, name=None):
        if p is None:
            return item
        # Reuse the accepted production identity policy from the Person layer.
        identity = labels[p.person_track_id][0][0].split(" | ", 1)[1]
        states = [b for b in p.behaviors if b.name == name]
        b = states[0] if len(states) == 1 else None
        return replace(item, owner_pid=p.person_track_id, identity=identity,
                       state=b.state if b else None, duration=b.estimated_seconds if b else None)

    def valid_box(row, key):
        b = _box(row.get(key))
        if b is None:
            issues.append(f"invalid {key}; box omitted")
        return b

    face_source = sources.get("face")
    if face_source:
        # Check that the worker reply and sidecar are the same recognition output.
        expected = [_fields(f, ("face_bbox", "student_id", "score")) for f in face_result.get("faces", [])]
        if (face_result.get("frame_id") != snapshot.frame_id
                or face_result.get("image") != str(Path(image_path).resolve()) or expected != face_source["faces"]):
            issues.append("face reply/sidecar mismatch; face layer omitted")
        else:
            face_counts = Counter(_box(f.get("face_bbox")) for f in expected)
            track_ids = Counter(t[0] for t in tracked_faces)
            face_claims = Counter(p.face_track_id for p in people if p.face_track_id is not None)
            for f in expected:
                b = valid_box(f, "face_bbox")
                if b is None:
                    continue
                matches = [t for t in tracked_faces if _box(t[1]) == b and t[3] == f.get("score")]
                tid = (matches[0][0] if len(matches) == 1 and face_counts[b] == 1
                       and track_ids[matches[0][0]] == 1 else None)
                ps = [p for p in people if tid is not None and p.face_track_id == tid]
                p = owner(ps[0].person_ref) if len(ps) == 1 and face_claims[tid] == 1 else None
                item = VisualBox("face", b, _score(f.get("score")), face_track_id=tid,
                                 recognition="recognized; SID not accepted" if _known(f.get("student_id")) else "Unknown")
                if p and (p.identity_status.startswith("ambiguous") or not _known(f.get("student_id"))
                          or (p.student_id and p.student_id != f.get("student_id"))):
                    item = replace(item, identity="Ambiguous" if p.identity_status.startswith("ambiguous") else "Unknown")
                else:
                    item = attach(item, p)
                    if _known(item.identity) and item.identity != "Ambiguous":
                        item = replace(item, recognition="recognized; production SID accepted")
                boxes.append(item)

    hybrid = sources.get("hybrid")
    if hybrid:
        decisions = defaultdict(list)
        for d in hybrid["decisions"]:
            decisions[(d.get("person_ref"), d.get("name"))].append(d)

        def selected(item, p, name):
            ds = decisions.get((p.person_ref, name), ()) if p else ()
            if len(ds) == 1 and ds[0].get("state") == item.state == "detected":
                return replace(item, selected=_box(ds[0].get("bounding_box")) == item.bbox)
            return item

        candidates = hybrid["candidates"]
        counts = Counter((d.get("person_ref"), d.get("class_name"), _box(d.get("bbox_xyxy"))) for d in candidates)
        for d in candidates:
            name = d.get("class_name")
            if name not in {"sleeping", "using_phone"}:
                issues.append("unsupported candidate kind omitted")
                continue
            b = valid_box(d, "bbox_xyxy")
            if b is None:
                continue
            associated = (d.get("association_status") in {"associated", "associated_by_person_roi"}
                          and not d.get("association_ambiguous"))
            p = owner(d.get("person_ref")) if associated else None
            item = VisualBox(name + "_candidate", b, _score(d.get("confidence")),
                             identity="Ambiguous" if d.get("association_ambiguous") or d.get("association_status") == "ambiguous" else "Unknown",
                             provenance=d.get("source"))
            item = attach(item, p, name)
            # Only a unique Sleeping candidate can be the accepted decision box.
            if name == "sleeping" and counts[(d.get("person_ref"), name, b)] == 1:
                item = selected(item, p, name)
            boxes.append(item)
        ids = Counter(p.get("detection_id") for p in hybrid["phones"])
        assoc = defaultdict(list)
        for a in hybrid["phone_associations"]:
            assoc[a.get("phone_detection_id")].append(a)
        for d in hybrid["phones"]:
            b = valid_box(d, "bbox_xyxy")
            if b is None:
                continue
            key = d.get("detection_id")
            rows = assoc.get(key, ())
            unique = key is not None and ids[key] == 1 and len(rows) == 1
            p = owner(rows[0].get("person_detection_id")) if unique and rows[0].get("status") == "ASSOCIATED_GEOMETRICALLY" else None
            ambiguous = (ids[key] > 1 or len(rows) > 1
                         or any(a.get("status") in {"AMBIGUOUS", "UNCERTAIN"} for a in rows))
            item = VisualBox("phone_object", b, _score(d.get("confidence")),
                             identity="Ambiguous" if ambiguous else "Unknown", provenance=d.get("source"))
            boxes.append(selected(attach(item, p, "using_phone"), p, "using_phone"))
    return VisualMetadata(snapshot.frame_id, snapshot.captured_at, size,
                          hashlib.sha256(snapshot.image.tobytes()).hexdigest(), jpeg_hash, tuple(boxes), tuple(issues))


def attach_to_result(result, image_path, face_result, tracked_faces):
    """Display-only replacement AFTER production duration/warning calculation."""
    try:
        metadata = collect(result["snapshot"], image_path, face_result, tracked_faces)
        for issue in metadata.issues:
            diagnostic(f"frame={result['frame_id']} {issue}")
        result["snapshot"] = replace(result["snapshot"], render_metadata=metadata)
    except Exception as exc:
        diagnostic(f"frame={result.get('frame_id')} metadata omitted: {type(exc).__name__}: {exc}")
