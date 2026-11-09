"""Production copies of the existing CH-18 matching and CH-20M identity helpers.

Extracted without changing their algorithm/constants. Legacy test entrypoints
stay intact; importing this module opens no logs, database, camera, or model.
"""
import logging
import math
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

FACE_CONFIDENCE = 0.60
CONFIRM_FRAMES = 3
CONFIRM_WINDOW_SECONDS = 12.0
MIN_TRACK_OBSERVATIONS = 2
MIN_FACE_SIZE = 48
REID_MARGIN = 0.08
IDENTITY_GRACE_SECONDS = 8.0
WARNING_MAX_MATCH_AGE = 1.5
TRACK_MISSING_SECONDS = 5.0
REID_MAX_SECONDS = 5.0
REID_MIN_SIMILARITY = 0.70
TRACK_GALLERY_SIZE = 8
TRACK_MAX_DISTANCE = 145.0
TRACK_MIN_IOU = 0.05
TRACK_EMBEDDING_MIN_SIM = 0.48
DISPLAY_UNCERTAIN_MIN_SCORE = 0.76
DISPLAY_UNCERTAIN_EXTENSION = 5.0
MIN_FACE_OVERLAP = 0.35
MIN_MATCH_SCORE = 0.45
MIN_MATCH_MARGIN = 0.15
MAX_NORMALIZED_DISTANCE = 1.8
DUPLICATE_IOU = 0.45
DUPLICATE_CONTAINMENT = 0.80


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-8 else 0.0


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    overlap = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return overlap / max(1, area_a + area_b - overlap)


class StableFaceTracker:
    """One-to-one matching with a short appearance gallery and re-identification.

    A recently lost track can be recovered only when facial appearance is
    sufficiently similar. If candidates are ambiguous, a new ID is safer.
    """

    def __init__(self):
        self.next_id = 1
        self.records = {}

    def update(self, faces, now):
        for track_id in list(self.records):
            if now - self.records[track_id]["last_seen"] > TRACK_MISSING_SECONDS:
                del self.records[track_id]
                logger.info("Track %d expired", track_id)

        candidates = []
        for track_id, record in self.records.items():
            old_box = record["bbox"]
            ox, oy = center(old_box)
            missing = now - record["last_seen"]
            for face_index, face in enumerate(faces):
                new_box = face["bbox"]
                nx, ny = center(new_box)
                distance = math.hypot(ox - nx, oy - ny)
                overlap = box_iou(old_box, new_box)
                sims = [cosine_similarity(emb, face["embedding"])
                        for emb in record["gallery"]]
                similarity = max(sims, default=0.0)
                if missing <= 1.5:
                    # Near-term matching uses both position and face appearance.
                    if distance > TRACK_MAX_DISTANCE and overlap < TRACK_MIN_IOU:
                        continue
                    if similarity < TRACK_EMBEDDING_MIN_SIM:
                        continue
                    cost = distance / TRACK_MAX_DISTANCE - 0.65 * overlap + (1.0 - similarity)
                else:
                    # Re-identification after a short disappearance.
                    if missing > REID_MAX_SECONDS or similarity < REID_MIN_SIMILARITY:
                        continue
                    cost = 2.0 + (1.0 - similarity) * 2.0 + 0.10 * missing
                candidates.append((cost, track_id, face_index, similarity, missing))

        used_tracks, used_faces = set(), set()
        assignments = {}
        for cost, track_id, face_index, similarity, missing in sorted(candidates):
            if track_id in used_tracks or face_index in used_faces:
                continue
            # Do not re-identify when another track looks nearly as similar.
            if missing > 1.5:
                competing = [c for c in candidates if c[2] == face_index and c[1] != track_id
                             and c[4] > 1.5]
                if competing and max(c[3] for c in competing) >= similarity - REID_MARGIN:
                    continue
            used_tracks.add(track_id)
            used_faces.add(face_index)
            face = faces[face_index]
            record = self.records[track_id]
            if missing > 1.5:
                logger.info("TRACK REIDENTIFIED | Track=%d | Missing=%.1fs | Similarity=%.3f",
                            track_id, missing, similarity)
            record["bbox"] = face["bbox"]
            record["gallery"].append(np.asarray(face["embedding"], dtype=np.float32).copy())
            record["gallery"] = record["gallery"][-TRACK_GALLERY_SIZE:]
            record["last_seen"] = now
            record["observations"] += 1
            assignments[track_id] = face

        for face_index, face in enumerate(faces):
            if face_index in used_faces:
                continue
            track_id = self.next_id
            self.next_id += 1
            embedding = np.asarray(face["embedding"], dtype=np.float32).copy()
            self.records[track_id] = {
                "bbox": face["bbox"], "gallery": [embedding], "last_seen": now,
                "observations": 1, "created_at": now,
            }
            assignments[track_id] = face
            logger.info("Tentative Track ID %d", track_id)

        confirmed = {tid: face for tid, face in assignments.items()
                     if self.records[tid]["observations"] >= MIN_TRACK_OBSERVATIONS}
        return {tid: face["bbox"] for tid, face in confirmed.items()}, confirmed


class StudentIdentityManager:
    def __init__(self):
        self.records = {}

    def update(self, track_id, result, now):
        record = self.records.setdefault(track_id, {
            "student_id": None, "candidate": None, "hits": 0,
            "last_match": None, "last_seen": now, "last_support": None,
            "match_history": deque(),
        })
        previous = record["student_id"]
        record["last_seen"] = now
        status = result.get("status")
        candidate = result.get("student_id")

        # Non-consecutive positive evidence within a rolling time window.
        # UNCERTAIN does not erase valid MATCH evidence, but UNKNOWN cannot
        # independently confirm a student or refresh warning authorization.
        history = record["match_history"]
        while history and now - history[0][0] > CONFIRM_WINDOW_SECONDS:
            history.popleft()
        if status == "MATCH" and candidate is not None:
            history.append((now, candidate))
            votes = {}
            for _, student in history:
                votes[student] = votes.get(student, 0) + 1
            leader, count = max(votes.items(), key=lambda item: item[1])
            runner_up = max((v for k, v in votes.items() if k != leader), default=0)
            record["candidate"] = leader
            record["hits"] = count
            if count >= CONFIRM_FRAMES and count >= runner_up + 2:
                record["student_id"] = leader
                if candidate == leader:
                    record["last_match"] = now
                    record["last_support"] = now
            elif record["student_id"] == candidate:
                record["last_match"] = now
                record["last_support"] = now
        else:
            if (status == "UNCERTAIN" and record["student_id"] is not None
                    and float(result.get("similarity") or 0.0) >= DISPLAY_UNCERTAIN_MIN_SCORE
                    and record["last_match"] is not None
                    and now - record["last_match"] <= IDENTITY_GRACE_SECONDS):
                record["last_support"] = now

        last_support = record["last_support"]
        if (record["last_match"] is None
                or now - record["last_match"] > IDENTITY_GRACE_SECONDS + DISPLAY_UNCERTAIN_EXTENSION
                or last_support is None or now - last_support > IDENTITY_GRACE_SECONDS):
            record["student_id"] = None

        if previous != record["student_id"]:
            logger.info("IDENTITY CHANGE | Track=%s | Student=%s",
                        track_id, record["student_id"] or "Unknown")
        return record

    def cleanup(self, active_track_ids):
        active = set(active_track_ids)
        for track_id in list(self.records):
            if track_id not in active:
                del self.records[track_id]

    def get(self, track_id, now, for_warning=False):
        record = self.records.get(track_id)
        if not record or not record["student_id"] or record["last_match"] is None:
            return None
        age = now - record["last_match"]
        if for_warning:
            return record["student_id"] if age <= WARNING_MAX_MATCH_AGE else None
        support = record.get("last_support")
        if support is None or now - support > IDENTITY_GRACE_SECONDS:
            return None
        return (record["student_id"] if age <=
                IDENTITY_GRACE_SECONDS + DISPLAY_UNCERTAIN_EXTENSION else None)


def ambiguous_student_ids(identities, visible_track_ids, now):
    """Find enrolled IDs claimed by multiple visible tracks; never choose a winner."""
    owners = {}
    for track_id in visible_track_ids:
        student_id = identities.get(track_id, now)
        if student_id is not None:
            owners.setdefault(student_id, []).append(track_id)
    return {student_id: ids for student_id, ids in owners.items() if len(ids) > 1}


def safe_student_id(identities, track_id, now, ambiguous_ids, for_warning=False):
    student_id = identities.get(track_id, now, for_warning=for_warning)
    return None if student_id in ambiguous_ids else student_id


def detect_faces(detector, frame):
    height, width = frame.shape[:2]
    faces = []
    for face in detector.get(frame):
        if float(face.det_score) < FACE_CONFIDENCE:
            continue
        embedding = getattr(face, "embedding", None)
        if embedding is None:
            continue
        x1, y1, x2, y2 = map(int, face.bbox)
        bbox = (max(0, min(x1, width - 1)), max(0, min(y1, height - 1)),
                max(0, min(x2, width)), max(0, min(y2, height)))
        if bbox[2] - bbox[0] >= MIN_FACE_SIZE and bbox[3] - bbox[1] >= MIN_FACE_SIZE:
            faces.append({"bbox": bbox, "embedding": embedding})
    return faces


def area(box):
    x1, y1, x2, y2 = box
    return max(
        1.0,
        float(max(0, x2 - x1) * max(0, y2 - y1))
    )


def intersection(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    width = max(0, min(ax2, bx2) - max(ax1, bx1))
    height = max(0, min(ay2, by2) - max(ay1, by1))

    return float(width * height)


def iou(a, b):
    inter = intersection(a, b)
    union = area(a) + area(b) - inter
    return inter / max(1.0, union)


def containment(a, b):
    return intersection(a, b) / min(area(a), area(b))


def face_overlap(face_box, behavior_box):
    return (
        intersection(face_box, behavior_box)
        / area(face_box)
    )


def face_center_inside(face_box, behavior_box):
    cx, cy = center(face_box)
    x1, y1, x2, y2 = behavior_box

    return x1 <= cx <= x2 and y1 <= cy <= y2


def normalized_distance(face_box, behavior_box):
    fx, fy = center(face_box)
    bx, by = center(behavior_box)

    x1, y1, x2, y2 = behavior_box

    diagonal = math.hypot(
        max(1, x2 - x1),
        max(1, y2 - y1)
    )

    return (
        math.hypot(fx - bx, fy - by)
        / max(1.0, diagonal)
    )


def remove_duplicates(detections):

    ordered = sorted(
        detections,
        key=lambda d: d["confidence"],
        reverse=True
    )

    filtered = []

    for detection in ordered:

        duplicate = False

        for kept in filtered:

            if detection["label"] != kept["label"]:
                continue

            if (
                iou(
                    detection["bbox"],
                    kept["bbox"]
                ) >= DUPLICATE_IOU
                or containment(
                    detection["bbox"],
                    kept["bbox"]
                ) >= DUPLICATE_CONTAINMENT
            ):
                duplicate = True
                break

        if not duplicate:
            filtered.append(detection)

    return filtered


def match_score(face_box, behavior_box):

    overlap = face_overlap(
        face_box,
        behavior_box
    )

    inside = face_center_inside(
        face_box,
        behavior_box
    )

    distance = normalized_distance(
        face_box,
        behavior_box
    )

    if overlap < MIN_FACE_OVERLAP and not inside:
        return 0.0

    if distance > MAX_NORMALIZED_DISTANCE:
        return 0.0

    distance_score = max(
        0.0,
        1.0 - distance / MAX_NORMALIZED_DISTANCE
    )

    return min(
        1.0,
        0.65 * overlap
        + 0.25 * distance_score
        + 0.10 * float(inside)
    )


def associate_behaviors(tracks, behaviors):

    candidates = []

    for index, behavior in enumerate(behaviors):

        for track_id, face_box in tracks.items():

            score = match_score(
                face_box,
                behavior["bbox"]
            )

            if score >= MIN_MATCH_SCORE:
                candidates.append(
                    (score, track_id, index)
                )

    candidates.sort(reverse=True)

    matches = {}
    used_tracks = set()
    used_behaviors = set()

    for score, track_id, index in candidates:

        if (
            track_id in used_tracks
            or index in used_behaviors
        ):
            continue

        competing_tracks = [
            s for s, t, b in candidates
            if b == index and t != track_id
        ]

        if competing_tracks and (
            score - max(competing_tracks)
            < MIN_MATCH_MARGIN
        ):
            continue

        competing_behaviors = [
            s for s, t, b in candidates
            if t == track_id and b != index
        ]

        if competing_behaviors and (
            score - max(competing_behaviors)
            < MIN_MATCH_MARGIN
        ):
            continue

        matches[track_id] = behaviors[index]

        used_tracks.add(track_id)
        used_behaviors.add(index)

    unmatched = [
        behavior
        for index, behavior in enumerate(behaviors)
        if index not in used_behaviors
    ]

    return matches, unmatched
