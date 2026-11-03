"""Experimental behavior + pose + phone-object video diagnostics.

This produces rule dispositions for model candidates, not ground-truth labels,
behavior events, or validated accuracy. Track IDs are supplied by ByteTrack;
when unavailable they remain blank and temporal smoothing is not fabricated.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
VENDORED = ROOT / "ai/behavior_detection/vendor"
if VENDORED.is_dir():
    sys.path.insert(0, str(VENDORED))

from ultralytics import YOLO  # noqa: E402


DEFAULT_BEHAVIOR = ROOT / "ai/behavior_detection/models/experimental/three_behavior_v2_best.pt"
V1_BEHAVIOR = ROOT / "ai/behavior_detection/models/experimental/three_behavior_v1_last.pt"
DEFAULT_PERSON_PHONE = ROOT / "ai/behavior_detection/models/pretrained_person/yolo11n_coco.pt"
DEFAULT_POSE = ROOT / "ai/behavior_detection/models/pose/yolo11s-pose.pt"
OUT_DIR = ROOT / "reports/experimental_v2_hybrid"
CLASSES = ("sleeping", "using_phone", "raising_hand")
PERSON_CLASS = 0
PHONE_CLASS = 67
BOOK_CLASS = 73
KP_CONF = 0.35
STATE_COLORS = {"confirmed": (45, 190, 65), "rejected": (60, 70, 230), "uncertain": (0, 190, 240)}


def box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-9, area_a + area_b - inter)


def as_xyxy(box) -> list[float]:
    return [float(x) for x in box.xyxy[0].cpu().tolist()]


def class_ids_for_names(model, wanted: set[str]) -> dict[str, int]:
    names = model.names
    values = names.items() if isinstance(names, dict) else enumerate(names)
    by_name = {str(name).strip().lower(): int(idx) for idx, name in values}
    missing = wanted - by_name.keys()
    if missing:
        raise RuntimeError(f"Checkpoint is missing classes {sorted(missing)}; names={by_name}")
    return {name: by_name[name] for name in wanted}


def associate_to_person(box: list[float], people: list[dict], min_iou=0.05):
    """Associate a smaller behavior/phone box by containment or meaningful IoU."""
    if not people:
        return None, False
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    ranked = []
    for idx, person in enumerate(people):
        pb = person["box"]
        inside = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
        overlap = box_iou(box, pb)
        score = overlap + (0.20 if inside else 0.0)
        ranked.append((score, idx, inside, overlap))
    ranked.sort(reverse=True)
    best = ranked[0]
    if not best[2] and best[3] < min_iou:
        return None, False
    ambiguous = len(ranked) > 1 and ranked[1][0] > 0 and best[0] - ranked[1][0] < 0.06
    return (None if ambiguous else best[1]), ambiguous


def pose_for_person(pose_result, person_box: list[float]):
    if pose_result.boxes is None or pose_result.keypoints is None or len(pose_result.boxes) == 0:
        return None, False
    boxes = pose_result.boxes.xyxy.cpu().numpy()
    points = pose_result.keypoints.data.cpu().numpy()
    ranked = sorted(((box_iou(person_box, b.tolist()), i) for i, b in enumerate(boxes)), reverse=True)
    overlap, idx = ranked[0]
    if overlap < 0.12:
        return None, False
    ambiguous = len(ranked) > 1 and ranked[1][0] >= 0.12 and overlap - ranked[1][0] < 0.04
    return (None if ambiguous else points[idx]), ambiguous


def point(kp, idx):
    if kp is None or idx >= len(kp) or float(kp[idx][2]) < KP_CONF:
        return None
    return np.asarray(kp[idx][:2], dtype=float)


def sleeping_evidence(person_box, kp):
    """Conservative posture cues. Head-down alone is never a confirmation."""
    x1, y1, x2, y2 = person_box
    h = max(1.0, y2 - y1)
    nose = point(kp, 0)
    shoulders = [p for i in (5, 6) if (p := point(kp, i)) is not None]
    hips = [p for i in (11, 12) if (p := point(kp, i)) is not None]
    elbows = [p for i in (7, 8) if (p := point(kp, i)) is not None]
    wrists = [p for i in (9, 10) if (p := point(kp, i)) is not None]
    required = nose is not None and bool(shoulders) and bool(hips)
    if not required:
        return {"assessable": False, "head_down": None, "trunk_slumped": None,
                "head_supported_by_hand": None, "detail": "nose/shoulder/hip keypoints insufficient"}
    shoulder_mid = np.mean(shoulders, axis=0)
    hip_mid = np.mean(hips, axis=0)
    head_down = bool(nose[1] >= shoulder_mid[1] + 0.06 * h)
    dx, dy = abs(float(shoulder_mid[0] - hip_mid[0])), abs(float(shoulder_mid[1] - hip_mid[1]))
    axis_len = max(1.0, math.hypot(dx, dy))
    vertical_fraction = dy / axis_len
    if vertical_fraction < 0.65:
        return {"assessable": False, "head_down": None, "trunk_slumped": None,
                "head_supported_by_hand": None, "torso_axis_vertical_fraction": round(vertical_fraction, 4),
                "detail": "torso axis is mostly horizontal in the source frame; camera orientation or lean makes sleeping posture comparison unreliable"}
    trunk_slumped = bool(dx >= 0.70 * max(dy, 1.0))
    hand_near_head = any(float(np.linalg.norm(w - nose)) <= 0.16 * h for w in wrists)
    elbow_near_head = any(float(np.linalg.norm(e - nose)) <= 0.30 * h for e in elbows)
    hand_support = bool(hand_near_head and elbow_near_head)
    upright = bool(dy >= 0.50 * max(dx, 1.0))
    hands_below_head = bool(wrists) and all(float(w[1]) >= float(nose[1]) + 0.08 * h for w in wrists)
    if head_down and trunk_slumped and hand_support:
        state = "confirmed"
        detail = "low head + slumped torso + wrist/elbow support near head"
    elif not head_down and upright and hands_below_head:
        state = "rejected"
        detail = "clear upright torso with head above shoulder line and hands below face"
    else:
        state = "uncertain"
        detail = "posture cues do not separate sleeping from writing/other head-down activity"
    return {"assessable": True, "head_down": head_down, "trunk_slumped": trunk_slumped,
            "torso_axis_vertical_fraction": round(vertical_fraction, 4),
            "head_supported_by_hand": hand_support, "state": state, "detail": detail}


def raised_hand_evidence(person_box, kp):
    h = max(1.0, person_box[3] - person_box[1])
    shoulders = [p for i in (5, 6) if (p := point(kp, i)) is not None]
    hips = [p for i in (11, 12) if (p := point(kp, i)) is not None]
    if not shoulders or not hips:
        return None, [{"side": "unknown", "assessable": False, "raised": None}]
    shoulder_mid = np.mean(shoulders, axis=0)
    gravity_down = np.mean(hips, axis=0) - shoulder_mid
    gravity_length = float(np.linalg.norm(gravity_down))
    if gravity_length < 1:
        return None, [{"side": "unknown", "assessable": False, "raised": None}]
    up_direction = -gravity_down / gravity_length
    arms = []
    for side, (shoulder_i, elbow_i, wrist_i) in {
        "left": (5, 7, 9), "right": (6, 8, 10)
    }.items():
        shoulder, elbow, wrist = point(kp, shoulder_i), point(kp, elbow_i), point(kp, wrist_i)
        if shoulder is None or elbow is None or wrist is None:
            arms.append({"side": side, "assessable": False, "raised": None})
            continue
        delta = float(np.dot(wrist - shoulder, up_direction)) / h
        arms.append({"side": side, "assessable": True, "raised": delta >= 0.08,
                     "wrist_above_shoulder_fraction": round(delta, 4),
                     "elbow_visible": True,
                     "wrist_xy": [round(float(v), 1) for v in wrist]})
    known = [arm for arm in arms if arm["assessable"]]
    if not known:
        return None, arms
    return any(arm["raised"] for arm in known), arms


class TrackTemporal:
    def __init__(self, window: int, min_positive: int):
        self.window = window
        self.min_positive = min_positive
        self.values = defaultdict(lambda: deque(maxlen=window))

    def update(self, track_id, value):
        if track_id is None:
            return "uncertain", {"known_frames": 0, "positive_frames": 0, "window": self.window}
        q = self.values[int(track_id)]
        q.append(value)
        known = [v for v in q if v is not None]
        positive = sum(bool(v) for v in known)
        if len(known) >= self.min_positive and positive >= self.min_positive:
            state = "confirmed"
        elif len(known) >= self.min_positive and positive == 0:
            state = "rejected"
        else:
            state = "uncertain"
        return state, {"known_frames": len(known), "positive_frames": positive,
                       "window": self.window}


def draw_candidate(frame, box, class_name, state, confidence=None, track_id=None):
    color = STATE_COLORS[state]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    parts = [class_name, state]
    if confidence is not None:
        parts.append(f"{confidence:.2f}")
    if track_id is not None:
        parts.append(f"id={track_id}")
    label = " | ".join(parts)
    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 2)
    top = max(0, y1 - th - base - 6)
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 6), color, -1)
    cv2.putText(frame, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                0.48, (10, 10, 10), 2, cv2.LINE_AA)


def _load_track_people(track_result):
    people, phones, books = [], [], []
    if track_result.boxes is None:
        return people, phones, books
    ids = track_result.boxes.id
    ids = ids.cpu().numpy().astype(int).tolist() if ids is not None else [None] * len(track_result.boxes)
    classes = track_result.boxes.cls.cpu().numpy().astype(int).tolist()
    for i, (class_id, track_id) in enumerate(zip(classes, ids)):
        box = as_xyxy(track_result.boxes[i])
        confidence = float(track_result.boxes.conf[i])
        if class_id == PERSON_CLASS:
            people.append({"box": box, "track_id": track_id, "confidence": confidence})
        elif class_id == PHONE_CLASS:
            phones.append({"box": box, "track_id": track_id,
                           "confidence": confidence})
        elif class_id == BOOK_CLASS:
            books.append({"box": box, "track_id": track_id, "confidence": confidence})
    # Trackers can keep overlapping person hypotheses for the same body.
    people.sort(key=lambda p: p["confidence"], reverse=True)
    deduplicated = []
    for person in people:
        if any(box_iou(person["box"], kept["box"]) >= 0.74 for kept in deduplicated):
            continue
        deduplicated.append(person)
    return deduplicated, phones, books


def _behavior_candidates(result, class_ids):
    out = []
    if result.boxes is None:
        return out
    for i, box in enumerate(result.boxes):
        class_id = int(box.cls[0])
        out.append({"class_name": class_ids[class_id], "box": as_xyxy(box),
                    "confidence": float(box.conf[0])})
    return out


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(source, behavior_path=DEFAULT_BEHAVIOR, person_phone_path=DEFAULT_PERSON_PHONE,
        pose_path=DEFAULT_POSE, conf=0.25, imgsz=640, device="cpu", window=5,
        min_positive=3, tracker="bytetrack.yaml", output=None, csv_output=None,
        summary_output=None):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output = Path(output) if output else OUT_DIR / "hybrid_video.mp4"
    csv_output = Path(csv_output) if csv_output else OUT_DIR / "hybrid_detections.csv"
    summary_output = Path(summary_output) if summary_output else OUT_DIR / "hybrid_summary.json"
    for path in (output, csv_output, summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing file: {path}")
    for path in (behavior_path, person_phone_path, pose_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"Required local weights are missing: {path}")
    try:
        import lap  # noqa: F401
    except ImportError as exc:
        raise RuntimeError("ByteTrack requires lap>=0.5.12. Install it in the project vendor directory.") from exc

    behavior_model = YOLO(str(behavior_path))
    phone_model = YOLO(str(person_phone_path))
    pose_model = YOLO(str(pose_path))
    behavior_names = behavior_model.names
    name_items = behavior_names.items() if isinstance(behavior_names, dict) else enumerate(behavior_names)
    behavior_by_id = {int(k): str(v).strip().lower() for k, v in name_items}
    behavior_ids = class_ids_for_names(behavior_model, set(CLASSES))
    if phone_model.names.get(PERSON_CLASS) != "person" or not {"cell phone", "book"}.issubset(set(phone_model.names.values())):
        raise RuntimeError("Person/phone checkpoint must expose COCO 'person', 'cell phone', and 'book' classes")

    source_arg = int(source) if str(source).isdecimal() else str(source)
    cap = cv2.VideoCapture(source_arg)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video source: {source}")
    source_fps = float(cap.get(cv2.CAP_PROP_FPS))
    fps_fallback = not math.isfinite(source_fps) or source_fps <= 0
    output_fps = 30.0 if fps_fallback else source_fps
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Source reported invalid video dimensions")
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), output_fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open MP4 writer: {output}")

    smoothers = {"raising_hand": TrackTemporal(window, min_positive)}
    baseline_counts = Counter({name: 0 for name in CLASSES})
    states = {name: Counter({"confirmed": 0, "rejected": 0, "uncertain": 0}) for name in CLASSES}
    phone_obj_count = 0
    book_obj_count = 0
    person_track_observations = 0
    untracked_person_observations = 0
    track_frames = Counter()
    track_last_seen = {}
    track_gap_reappearances = 0
    frame_no = 0
    start = time.perf_counter()
    infer_time = Counter()
    columns = ["timestamp_seconds", "frame_number", "class_name", "status", "candidate_source",
               "baseline_confidence", "track_id", "person_x1", "person_y1", "person_x2", "person_y2",
               "candidate_x1", "candidate_y1", "candidate_x2", "candidate_y2",
               "pose_assessable", "evidence_detail"]

    try:
        with Path(csv_output).open("w", newline="", encoding="utf-8") as csv_file:
            log = csv.DictWriter(csv_file, fieldnames=columns)
            log.writeheader()
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_no += 1
                pos_ms = float(cap.get(cv2.CAP_PROP_POS_MSEC))
                timestamp = max(0.0, pos_ms / 1000.0) if pos_ms > 0 else (frame_no - 1) / output_fps

                t = time.perf_counter()
                behavior_result = behavior_model.predict(source=frame, conf=conf, imgsz=imgsz,
                    classes=list(behavior_ids.values()), device=device, verbose=False)[0]
                infer_time["behavior_seconds"] += time.perf_counter() - t

                t = time.perf_counter()
                tracked_result = phone_model.track(source=frame, persist=True, tracker=tracker,
                    classes=[PERSON_CLASS, PHONE_CLASS, BOOK_CLASS], conf=conf, imgsz=imgsz,
                    device=device, verbose=False)[0]
                infer_time["person_phone_seconds"] += time.perf_counter() - t

                t = time.perf_counter()
                pose_result = pose_model.predict(source=frame, conf=conf, imgsz=imgsz,
                    device=device, verbose=False)[0]
                infer_time["pose_seconds"] += time.perf_counter() - t

                people, phones, books = _load_track_people(tracked_result)
                phone_obj_count += len(phones)
                book_obj_count += len(books)
                for person in people:
                    tid = person["track_id"]
                    if tid is None:
                        untracked_person_observations += 1
                    else:
                        person_track_observations += 1
                        track_frames[tid] += 1
                        if tid in track_last_seen and track_last_seen[tid] < frame_no - 1:
                            track_gap_reappearances += 1
                        track_last_seen[tid] = frame_no
                    kp, pose_ambiguous = pose_for_person(pose_result, person["box"])
                    person["keypoints"] = kp
                    person["pose_ambiguous"] = pose_ambiguous
                    person["sleep"] = sleeping_evidence(person["box"], kp)
                    person["raise_raw"], person["arms"] = raised_hand_evidence(person["box"], kp)
                    person["phone_boxes"] = []
                    person["book_boxes"] = []

                phone_to_person = {}
                for phone in phones:
                    idx, ambiguous = associate_to_person(phone["box"], people)
                    if idx is not None:
                        people[idx]["phone_boxes"].append(phone)
                        phone_to_person[id(phone)] = (idx, ambiguous)

                candidates = _behavior_candidates(behavior_result, behavior_by_id)
                for item in candidates:
                    baseline_counts[item["class_name"]] += 1
                    item["person_index"], item["association_ambiguous"] = associate_to_person(item["box"], people)

                per_person = defaultdict(dict)
                detached = []
                for item in candidates:
                    idx = item["person_index"]
                    if idx is None:
                        detached.append(item)
                    else:
                        current = per_person[idx].get(item["class_name"])
                        if current is None or item["confidence"] > current["confidence"]:
                            per_person[idx][item["class_name"]] = item

                # A COCO phone near a reliable wrist creates a candidate even when
                # the experimental behavior detector did not produce one.
                detached_phone_candidates = []
                for phone in phones:
                    idx, ambiguous = associate_to_person(phone["box"], people)
                    if idx is None:
                        if ambiguous:
                            detached_phone_candidates.append(phone)
                        continue
                    current = per_person[idx].get("using_phone")
                    synthetic = {"class_name": "using_phone", "box": phone["box"],
                                 "confidence": phone["confidence"], "source": "phone_object_detector"}
                    if current is None or ("baseline" in current.get("source", "baseline")) is False:
                        per_person[idx]["using_phone"] = synthetic
                    if ambiguous:
                        people[idx]["phone_association_ambiguous"] = True

                for book in books:
                    idx, _ambiguous = associate_to_person(book["box"], people)
                    if idx is not None:
                        people[idx]["book_boxes"].append(book)

                frame_rows = []
                for idx, person in enumerate(people):
                    tid, kp = person["track_id"], person["keypoints"]
                    by_class = per_person[idx]
                    pose_ok = kp is not None and not person["pose_ambiguous"]
                    raw_raise = person["raise_raw"] if pose_ok else None
                    raise_state, raise_temporal = smoothers["raising_hand"].update(tid, raw_raise)
                    raised_arms = [arm for arm in person["arms"] if arm.get("assessable") and arm.get("raised")]
                    person_h = max(1.0, person["box"][3] - person["box"][1])
                    book_contact_arms = [arm for arm in raised_arms if any(
                        book["box"][0] - 0.06 * person_h <= arm["wrist_xy"][0] <= book["box"][2] + 0.06 * person_h
                        and book["box"][1] - 0.06 * person_h <= arm["wrist_xy"][1] <= book["box"][3] + 0.06 * person_h
                        for book in person["book_boxes"])]

                    if "sleeping" in by_class:
                        item = by_class["sleeping"]
                        evidence = person["sleep"] if pose_ok else {"assessable": False, "state": "uncertain",
                                                                  "detail": "pose absent or person-pose association ambiguous"}
                        state = evidence.get("state", "uncertain") if evidence.get("assessable") else "uncertain"
                        frame_rows.append(("sleeping", state, item.get("source", "baseline_behavior"),
                            item["confidence"], tid, person["box"], item["box"], evidence["assessable"], evidence["detail"]))

                    if "using_phone" in by_class:
                        item = by_class["using_phone"]
                        wrist_points = [p for i in (9, 10) if (p := point(kp, i)) is not None] if pose_ok else []
                        near_phone = False
                        relation_detail = ""
                        for phone in person["phone_boxes"]:
                            center = np.asarray([(phone["box"][0]+phone["box"][2])/2,
                                                 (phone["box"][1]+phone["box"][3])/2])
                            if wrist_points and min(np.linalg.norm(w - center) for w in wrist_points) <= 0.22 * math.hypot(
                                max(1, person["box"][2]-person["box"][0]), max(1, person["box"][3]-person["box"][1])):
                                near_phone = True
                                break
                        if near_phone and not person.get("phone_association_ambiguous", False):
                            state, relation_detail = "confirmed", "COCO phone box associated to person and near visible wrist"
                        elif person.get("phone_association_ambiguous", False):
                            state, relation_detail = "uncertain", "phone-person association ambiguous"
                        elif not pose_ok or not wrist_points:
                            state, relation_detail = "uncertain", "phone/wrist evidence insufficient; absence is not evidence of no phone"
                        else:
                            state, relation_detail = "uncertain", "phone detector or wrist relation did not provide sufficient confirmation"
                        frame_rows.append(("using_phone", state, item.get("source", "baseline_or_phone_object"),
                            item["confidence"], tid, person["box"], item["box"], bool(wrist_points), relation_detail))

                    raise_from_behavior = "raising_hand" in by_class
                    if raise_from_behavior or raw_raise is True:
                        state, temporal = raise_state, raise_temporal
                        if tid is None:
                            state = "uncertain"
                        if book_contact_arms:
                            state = "rejected"
                            raise_detail = "raised wrist overlaps a person-associated book; object interaction conflicts with a participation gesture"
                        elif raise_from_behavior and raw_raise is True and state == "confirmed":
                            raise_detail = "behavior-model detection agrees with pose elevation in at least 3 of 5 tracked frames"
                        elif state == "rejected" and raw_raise is True:
                            state = "uncertain"
                            raise_detail = "pose-only arm elevation lacks behavior-model corroboration"
                        elif not raise_from_behavior:
                            state = "uncertain"
                            raise_detail = "pose-only arm elevation is a posture cue, not confirmed raising-hand behavior"
                        elif raw_raise is None:
                            state = "uncertain"
                            raise_detail = "pose/keypoint association is insufficient"
                        else:
                            raise_detail = "behavior detection lacks persistent pose corroboration"
                        source_name = "baseline_and_pose" if raise_from_behavior and raw_raise is not None else (
                            "baseline_behavior" if raise_from_behavior else "pose_geometry")
                        detail = {"arms": person["arms"], "temporal": temporal,
                                  "pose_association_ambiguous": person["pose_ambiguous"],
                                  "book_contact_arms": [arm["side"] for arm in book_contact_arms],
                                  "rule": raise_detail}
                        frame_rows.append(("raising_hand", state, source_name,
                            by_class.get("raising_hand", {}).get("confidence"), tid, person["box"],
                            by_class.get("raising_hand", {}).get("box", person["box"]), raw_raise is not None,
                            json.dumps(detail, separators=(",", ":"))))

                # Candidates that could not be linked to a tracked person remain uncertain.
                for item in detached:
                    frame_rows.append((item["class_name"], "uncertain", "baseline_behavior_unassociated",
                        item["confidence"], None, None, item["box"], False,
                        "no unambiguous tracked-person association"))
                for phone in detached_phone_candidates:
                    frame_rows.append(("using_phone", "uncertain", "phone_object_association_ambiguous",
                        phone["confidence"], None, None, phone["box"], False,
                        "phone detector box overlaps competing person tracks"))

                overlay = frame.copy()
                for row in frame_rows:
                    cls_name, state, _, confidence, tid, person_box, candidate_box, _, _ = row
                    draw_box = candidate_box or person_box
                    if draw_box is not None:
                        draw_candidate(overlay, draw_box, cls_name, state, confidence, tid)
                    states[cls_name][state] += 1
                writer.write(overlay)

                for row in frame_rows:
                    cls_name, state, source_name, confidence, tid, person_box, candidate_box, assessable, detail = row
                    pb = person_box or [None] * 4
                    cb = candidate_box or [None] * 4
                    log.writerow({"timestamp_seconds": round(timestamp, 3), "frame_number": frame_no,
                        "class_name": cls_name, "status": state, "candidate_source": source_name,
                        "baseline_confidence": "" if confidence is None else round(confidence, 6),
                        "track_id": "" if tid is None else tid,
                        **{f"person_{axis}": "" if value is None else round(value, 2)
                           for axis, value in zip(("x1", "y1", "x2", "y2"), pb)},
                        **{f"candidate_{axis}": "" if value is None else round(value, 2)
                           for axis, value in zip(("x1", "y1", "x2", "y2"), cb)},
                        "pose_assessable": assessable, "evidence_detail": detail})
                if frame_no % max(1, int(output_fps * 10)) == 0:
                    print(f"Hybrid processed {frame_no} frames ({timestamp:.1f}s)", flush=True)
    finally:
        cap.release()
        writer.release()

    elapsed = time.perf_counter() - start
    tracks = list(track_frames.values())
    track_summary = {
        "tracker": tracker,
        "unique_person_track_ids": len(track_frames),
        "person_track_observations": person_track_observations,
        "untracked_person_observations": untracked_person_observations,
        "reappearing_tracks_after_gap_count": track_gap_reappearances,
        "average_frames_per_track_id": round(float(np.mean(tracks)), 2) if tracks else None,
        "median_frames_per_track_id": float(np.median(tracks)) if tracks else None,
        "single_observation_track_ids": sum(n == 1 for n in tracks),
        "identity_stability_validated": False,
        "interpretation": "tracker consistency proxies only; no person identity ground truth",
    }
    baseline_summary_path = ROOT / "reports/experimental_v1_inference/summary.json"
    baseline_reference = None
    if baseline_summary_path.is_file():
        try:
            old = json.loads(baseline_summary_path.read_text(encoding="utf-8"))
            if Path(str(old.get("source", ""))).name == Path(str(source)).name:
                baseline_reference = {
                    "summary_path": str(baseline_summary_path.resolve()),
                    "frame_count": old.get("frame_count"),
                    "fps": old.get("source_fps"),
                    "per_class_frame_detections": old.get("per_class_frame_detections"),
                    "warning": "Counts come from detector boxes and are not GT behavior events.",
                }
        except (ValueError, OSError):
            baseline_reference = None
    summary = {
        "status": "COMPLETED",
        "model_status": "EXPERIMENTAL_UNVERIFIED",
        "annotation_completeness": "UNVERIFIED",
        "evaluation": "UNVERIFIED_EVALUATION",
        "source": str(source),
        "frame_count": frame_no,
        "source_fps": source_fps if not fps_fallback else None,
        "output_fps": output_fps,
        "fps_used_fallback": fps_fallback,
        "duration_seconds": round(frame_no / output_fps, 3),
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frame_no / elapsed, 3) if elapsed else None,
        "inference_seconds_by_branch": {k: round(v, 3) for k, v in infer_time.items()},
        "confidence": conf,
        "imgsz": imgsz,
        "temporal_window_frames": window,
        "raising_hand_min_positive_frames": min_positive,
        "weights": {"behavior": str(Path(behavior_path).resolve()),
                    "person_phone": str(Path(person_phone_path).resolve()),
                    "pose": str(Path(pose_path).resolve())},
        "weight_sha256": {"behavior": _sha256(behavior_path),
                           "person_phone": _sha256(person_phone_path),
                           "pose": _sha256(pose_path)},
        "source_sha256": _sha256(source) if Path(str(source)).is_file() else None,
        "class_names": behavior_by_id,
        "baseline_frame_detection_counts": dict(sorted(baseline_counts.items())),
        "baseline_reference": baseline_reference,
        "hybrid_candidate_dispositions_by_class": {name: dict(states[name]) for name in CLASSES},
        "hybrid_candidate_total_by_class": {name: sum(states[name].values()) for name in CLASSES},
        "phone_object_detections_all_frames": phone_obj_count,
        "book_object_detections_all_frames": book_obj_count,
        "phone_absence_warning": "A zero detector count does not establish that the video contains no phone.",
        "track_metrics": track_summary,
        "disposition_semantics": "heuristic evidence states only; not ground-truth behavior labels or accuracy",
        "outputs": {"video": str(Path(output).resolve()), "csv": str(Path(csv_output).resolve()),
                    "summary": str(Path(summary_output).resolve())},
    }
    Path(summary_output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path or webcam index such as 0")
    parser.add_argument("--behavior-version", choices=("v2", "v1"), default="v2",
                        help="Experimental behavior checkpoint version (default: v2)")
    parser.add_argument("--behavior", type=Path, default=None,
                        help="Optional explicit behavior checkpoint path; overrides --behavior-version")
    parser.add_argument("--person-phone", type=Path, default=DEFAULT_PERSON_PHONE)
    parser.add_argument("--pose", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min-positive", type=int, default=3)
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--output", type=Path, default=OUT_DIR / "hybrid_video.mp4")
    parser.add_argument("--csv", type=Path, default=OUT_DIR / "hybrid_detections.csv")
    parser.add_argument("--summary", type=Path, default=OUT_DIR / "hybrid_summary.json")
    args = parser.parse_args(argv)
    if not 0 < args.min_positive <= args.window:
        parser.error("--min-positive must be between 1 and --window")
    behavior_path = args.behavior or (V1_BEHAVIOR if args.behavior_version == "v1" else DEFAULT_BEHAVIOR)
    run(args.source, behavior_path, args.person_phone, args.pose, args.conf, args.imgsz,
        args.device, args.window, args.min_positive, args.tracker,
        args.output, args.csv, args.summary)


if __name__ == "__main__":
    main()
