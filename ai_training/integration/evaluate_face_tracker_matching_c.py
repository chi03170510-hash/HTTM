"""Diagnostic C: offline geometry matching, never imported by Camera Live.

Recognition metadata is preserved for exact baseline replay, not used to choose
matches or label physical actors. Only separate synthetic/manual actor labels
enable identity metrics. All variants share the recorded TTL and input order.
"""

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from replay_face_tracker_diagnostics import replay_records  # Recognition import path.
from face_tracker import FaceTracker, calculate_iou


DEFAULT_SESSION = Path(__file__).resolve().parents[1] / "runs/person_tracker_debug/session_37712fb9ffff45f5a9219897b4785309/frames.jsonl"
METHODS = ("iou_only", "iou_relaxed_control", "geometry", "geometry_conservative")


@dataclass(frozen=True)
class FaceObservation:
    bbox: tuple
    student_id: str
    score: float
    actor: str | None = None  # Only external manual/synthetic ground truth.


@dataclass(frozen=True)
class FrameObservation:
    frame_id: int
    captured_at: float
    updated_at: float
    faces: tuple[FaceObservation, ...]


@dataclass(frozen=True)
class GeometryConfig:
    min_iou: float = .05
    max_center_distance: float = .6
    min_size_similarity: float = .5
    min_score: float = .35
    ambiguity_margin: float = .15

    def __post_init__(self):
        values = asdict(self)
        if any(not finite(v) or not 0 < v <= 1 for v in values.values()):
            raise ValueError("geometry settings must be finite and in (0, 1]")


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def valid_box(box):
    return (isinstance(box, (tuple, list)) and len(box) == 4 and all(finite(v) for v in box)
            and finite(box[2] - box[0]) and finite(box[3] - box[1])
            and box[2] > box[0] and box[3] > box[1])


def validate_frames(frames):
    if not frames:
        raise ValueError("at least one frame required")
    previous_time = previous_id = None
    for frame in frames:
        if (type(frame.frame_id) is not int or frame.frame_id <= 0
                or (previous_id is not None and frame.frame_id <= previous_id)
                or not finite(frame.updated_at) or not finite(frame.captured_at)
                or (previous_time is not None and frame.updated_at <= previous_time)):
            raise ValueError("frame IDs and update timestamps must increase; timestamps must be finite")
        actors = [f.actor for f in frame.faces if f.actor is not None]
        if any(not isinstance(a, str) or not a.strip() for a in actors) or len(set(actors)) != len(actors):
            raise ValueError("actor labels must be nonempty and unique within each frame")
        if any(not valid_box(f.bbox) or not finite(f.score) for f in frame.faces):
            raise ValueError("finite recognition scores and valid positive-area face boxes required")
        previous_id, previous_time = frame.frame_id, frame.updated_at


def input_digest(frames):
    payload = json.dumps([asdict(f) for f in frames], sort_keys=True, allow_nan=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def load_session(path):
    data = Path(path).read_bytes()
    records = tuple(json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip())
    verification = replay_records(records)
    if not all(r["consistent"] for r in verification):
        raise ValueError("real FaceTracker baseline does not reproduce the source session")
    frames = []
    for record in records:
        block = record["face_tracker"]
        if record.get("details_truncated") or block["face_detection_count"] != len(block["detections"]):
            raise ValueError("complete, untruncated detections required")
        faces = []
        for index, detection in enumerate(block["detections"]):
            if detection["detection_index"] != index:
                raise ValueError("original detection order/index mismatch")
            faces.append(FaceObservation(tuple(detection["face_bbox"]), detection["current_student_id"],
                                         detection["recognition_score"]))
        frames.append(FrameObservation(record["frame_id"], record["captured_at"],
                                       block["face_tracker_updated_at"], tuple(faces)))
    frames = tuple(frames)
    validate_frames(frames)
    parameters = records[0]["face_tracker"]["parameters"]
    if parameters["iou_threshold"] != .3 or not finite(parameters["max_age"]) or parameters["max_age"] <= 0:
        raise ValueError("recorded IoU=.3 and a finite positive max_age required")
    return frames, records, hashlib.sha256(data).hexdigest(), parameters["max_age"]


def apply_annotations(frames, rows):
    """Rows: frame_id, detection_index, actor (opaque label from human review).

    No Student ID, Person ID or recorded face-track lineage is converted to truth.
    Partial annotations produce explicitly incomplete identity metrics.
    """
    if not isinstance(rows, list):
        raise ValueError("annotations must be a list")
    valid = {(f.frame_id, i) for f in frames for i in range(len(f.faces))}
    labels = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"frame_id", "detection_index", "actor"}:
            raise ValueError("annotation fields: frame_id, detection_index, actor only")
        key = (row["frame_id"], row["detection_index"])
        if any(type(v) is not int for v in key) or key not in valid or key in labels:
            raise ValueError("unknown or duplicate annotated observation")
        actor = row["actor"]
        if not isinstance(actor, str) or not actor.strip():
            raise ValueError("manual actor label must be nonempty")
        labels[key] = actor
    annotated = tuple(replace(f, faces=tuple(replace(d, actor=labels.get((f.frame_id, i)))
                                             for i, d in enumerate(f.faces))) for f in frames)
    validate_frames(annotated)
    return annotated


def geometry_features(a, b):
    aw, ah, bw, bh = a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1]
    center = math.hypot((a[0] + a[2] - b[0] - b[2]) / 2, (a[1] + a[3] - b[1] - b[3]) / 2)
    distance = center / max(math.hypot(aw, ah), math.hypot(bw, bh))
    size = min(min(aw, bw) / max(aw, bw), min(ah, bh) / max(ah, bh))
    return {"iou": calculate_iou(a, b), "center_distance": distance, "size_similarity": size}


def candidate_rows(before, frame, max_age, method, config):
    rows = []
    for tid, (box, seen) in before.items():
        for index, detection in enumerate(frame.faces):
            features = geometry_features(box, detection.bbox)
            iou, center, size = features["iou"], features["center_distance"], features["size_similarity"]
            proximity = max(0., 1. - center / config.max_center_distance)
            score = .5 * iou + .3 * proximity + .2 * size
            reasons = []
            if frame.updated_at - seen > max_age:
                reasons.append("expired_before_matching")
            if method in ("iou_only", "iou_relaxed_control"):
                threshold = .3 if method == "iou_only" else config.min_iou
                if iou < threshold:
                    reasons.append("iou_below_gate")
                score = iou
            else:
                for failed, reason in ((iou < config.min_iou, "iou_below_gate"),
                                       (center > config.max_center_distance, "center_too_far"),
                                       (size < config.min_size_similarity, "size_inconsistent"),
                                       (score < config.min_score, "score_too_low")):
                    if failed:
                        reasons.append(reason)
            rows.append({"track_id": tid, "detection_index": index, **features, "score": score,
                         "eligible": not reasons, "reasons": reasons, "selected": False})
    return rows


def select_geometry_pairs(rows, conservative, margin):
    eligible = [r for r in rows if r["eligible"]]
    by_track, by_detection = defaultdict(list), defaultdict(list)
    rank = lambda r: (r["score"], r["track_id"], r["detection_index"])
    for row in eligible:
        by_track[row["track_id"]].append(row)
        by_detection[row["detection_index"]].append(row)
    for group in (*by_track.values(), *by_detection.values()):
        group.sort(key=rank, reverse=True)
    ambiguous_tracks = {k for k, v in by_track.items() if len(v) > 1 and v[0]["score"] - v[1]["score"] < margin}
    ambiguous_detections = {k for k, v in by_detection.items() if len(v) > 1 and v[0]["score"] - v[1]["score"] < margin}
    selected, used_tracks, used_detections = [], set(), set()
    for row in sorted(eligible, key=rank, reverse=True):
        tid, index = row["track_id"], row["detection_index"]
        if conservative:
            if tid in ambiguous_tracks or index in ambiguous_detections:
                row["reasons"].append("ambiguous_competition")
                continue
            # Compare against the ORIGINAL full candidate matrix. Never fall
            # back to a second choice after an ambiguous candidate is rejected.
            if row is not by_track[tid][0] or row is not by_detection[index][0]:
                row["reasons"].append("not_mutual_best")
                continue
        if tid in used_tracks or index in used_detections:
            row["reasons"].append("greedy_conflict")
            continue
        row["selected"] = True
        selected.append((tid, index))
        used_tracks.add(tid)
        used_detections.add(index)
    rejected = sorted({r["detection_index"] for r in eligible if "ambiguous_competition" in r["reasons"]}
                      - used_detections) if conservative else []
    return selected, rejected


class RealIoUMatcher:
    """Use the unchanged real tracker; audit indices by bbox object identity.

    Each input bbox is a distinct tuple object, even for equal coordinates. The
    current real tracker stores that exact object; auditing observes the consumed
    detection, without reading recognition labels/scores to infer assignments.
    """
    def __init__(self, max_age, threshold):
        self.tracker = FaceTracker(iou_threshold=threshold, max_age=max_age)

    @property
    def tracks(self):
        return self.tracker.tracks

    @property
    def next_id(self):
        return self.tracker.next_id

    def update(self, frame, method, config):
        before = {tid: (t.bbox, t.last_seen) for tid, t in self.tracks.items()}
        rows = candidate_rows(before, frame, self.tracker.max_age, method, config)
        boxes = [tuple(v for v in f.bbox) for f in frame.faces]
        indices = {id(box): i for i, box in enumerate(boxes)}
        returned = self.tracker.update([(box, f.student_id, f.score) for box, f in zip(boxes, frame.faces)], frame.updated_at)
        assignment = {indices[id(box)]: tid for tid, box, _, _ in returned}
        if len(assignment) != len(frame.faces):
            raise ValueError("baseline index audit failed; real tracker bbox storage contract changed")
        for row in rows:
            row["selected"] = assignment[row["detection_index"]] == row["track_id"]
        return result_row(frame, before, self.tracks, assignment, rows, [], self.tracker.max_age)


@dataclass
class GeometryTrack:
    bbox: tuple
    last_seen: float


class GeometryMatcher:
    def __init__(self, max_age):
        self.max_age, self.next_id, self.tracks = max_age, 1, {}

    def update(self, frame, method, config):
        before = {tid: (t.bbox, t.last_seen) for tid, t in self.tracks.items()}
        rows = candidate_rows(before, frame, self.max_age, method, config)
        for tid, (_, seen) in before.items():
            if frame.updated_at - seen > self.max_age:
                del self.tracks[tid]
        selected, rejected = select_geometry_pairs(rows, method == "geometry_conservative", config.ambiguity_margin)
        assignment = {}
        for tid, index in selected:
            self.tracks[tid] = GeometryTrack(frame.faces[index].bbox, frame.updated_at)
            assignment[index] = tid
        for index, detection in enumerate(frame.faces):
            if index not in assignment:
                assignment[index] = self.next_id
                self.tracks[self.next_id] = GeometryTrack(detection.bbox, frame.updated_at)
                self.next_id += 1
        return result_row(frame, before, self.tracks, assignment, rows, rejected, self.max_age)


def result_row(frame, before, tracks, assignment, pairs, rejected, max_age):
    return {"frame_id": frame.frame_id, "captured_at": frame.captured_at, "updated_at": frame.updated_at,
            "input_boxes": [list(d.bbox) for d in frame.faces],
            "actor_labels": [d.actor for d in frame.faces],
            "detection_track_ids": [assignment[i] for i in range(len(frame.faces))],
            "expired_ids": [tid for tid, (_, seen) in before.items() if frame.updated_at - seen > max_age],
            "new_ids": [assignment[i] for i in range(len(frame.faces)) if assignment[i] not in before],
            "unmatched_retained_ids": [tid for tid, track in tracks.items() if track.last_seen != frame.updated_at],
            "ambiguity_rejected_detections": rejected, "pairs": pairs}


def run_matching_trial(frames, method, max_age, config=None):
    frames, config = tuple(frames), config or GeometryConfig()
    validate_frames(frames)
    if method not in METHODS or not finite(max_age) or max_age <= 0:
        raise ValueError("known method and finite positive max_age required")
    tracker = (RealIoUMatcher(max_age, .3 if method == "iou_only" else config.min_iou)
               if method in ("iou_only", "iou_relaxed_control") else GeometryMatcher(max_age))
    prior_actor_track, prior_track_actor, trace = {}, {}, []
    switches = false_links = labeled = total = 0
    for frame in frames:
        row = tracker.update(frame, method, config)
        frame_switches = frame_false = frame_labels = 0
        for face, tid in zip(frame.faces, row["detection_track_ids"]):
            total += 1
            if face.actor is None:
                continue
            frame_labels += 1
            if face.actor in prior_actor_track and prior_actor_track[face.actor] != tid:
                frame_switches += 1
            if tid in prior_track_actor and prior_track_actor[tid] != face.actor:
                frame_false += 1
            prior_actor_track[face.actor], prior_track_actor[tid] = tid, face.actor
        labeled += frame_labels
        switches += frame_switches
        false_links += frame_false
        row.update(id_changes=frame_switches if frame_labels else None,
                   false_continuations=frame_false if frame_labels else None)
        trace.append(row)
    return {"method": method, "max_age": max_age, "geometry_config": asdict(config),
            "iou_threshold": .3 if method == "iou_only" else config.min_iou,
            "input_sha256": input_digest(frames), "frames": len(frames), "detections": total,
            "ids_created": tracker.next_id - 1,
            "new_ids_after_first_frame": tracker.next_id - 1 - len(frames[0].faces),
            "expiry_count": sum(len(r["expired_ids"]) for r in trace),
            "matched_links": sum(p["selected"] for r in trace for p in r["pairs"]),
            "ambiguity_rejections": sum(len(r["ambiguity_rejected_detections"]) for r in trace),
            "id_changes": switches if labeled else None, "false_continuations": false_links if labeled else None,
            "labeled_observations": labeled, "identity_metrics_complete": total > 0 and labeled == total,
            "trace": trace}


def risk_scenarios_c():
    from evaluate_face_tracker_parameters import risk_scenarios
    cases = {name: tuple(FrameObservation(f.frame_id, f.captured_at, f.updated_at,
                                          tuple(FaceObservation(d.bbox, d.student_id, d.score, d.actor) for d in f.faces))
                         for f in frames) for name, frames in risk_scenarios().items()}
    a = FaceObservation((20, 20, 120, 140), "Unknown", .8, "actor:A")
    b = replace(a, actor="actor:B")
    cases["replacement_with_observed_absence"] = (
        FrameObservation(1, 99.5, 100., (a,)), FrameObservation(2, 101., 101.5, ()),
        FrameObservation(3, 102.5, 103., (b,)))
    cases["size_jump_same_actor"] = (FrameObservation(1, 99.5, 100., (a,)),
                                    FrameObservation(2, 101., 101.5, (replace(a, bbox=(20, 20, 120, 380)),)))
    thin = replace(a, bbox=(0, 0, 100, 20))
    cases["center_jump_same_actor"] = (FrameObservation(1, 99.5, 100., (thin,)),
                                      FrameObservation(2, 101., 101.5, (replace(thin, bbox=(80, 0, 180, 20)),)))
    return cases


def evaluate_session(path=DEFAULT_SESSION, annotations=None):
    frames, records, source_hash, age = load_session(path)
    annotation_hash = None
    if annotations is not None:
        data = Path(annotations).read_bytes()
        frames = apply_annotations(frames, json.loads(data.decode("utf-8-sig")))
        annotation_hash = hashlib.sha256(data).hexdigest()
    digest = input_digest(frames)
    trials = [run_matching_trial(frames, method, age) for method in METHODS]
    if any(t["input_sha256"] != digest for t in trials) or input_digest(frames) != digest:
        raise ValueError("trial inputs changed")
    known_mappings = 0
    for record, actual in zip(records, trials[0]["trace"]):
        block = record["face_tracker"]
        mappings = block["mapping"]
        if any(m["selection_ambiguous"] or len(m["detection_indices"]) != 1 for m in mappings):
            continue
        expected = {m["detection_indices"][0]: m["face_track_id"] for m in mappings}
        if [expected[i] for i in range(len(block["detections"]))] != actual["detection_track_ids"]:
            raise ValueError("baseline detection assignment differs from source")
        known_mappings += 1
    risks = {name: [run_matching_trial(sequence, method, age) for method in METHODS]
             for name, sequence in risk_scenarios_c().items()}
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != source_hash:
        raise ValueError("source session changed during evaluation")
    if annotations is not None and hashlib.sha256(Path(annotations).read_bytes()).hexdigest() != annotation_hash:
        raise ValueError("annotations changed during evaluation")
    return {"schema": "face_tracker_matching_diagnostic_c_v1", "source_file": str(Path(path).resolve()),
            "source_sha256": source_hash, "annotations_file": str(Path(annotations).resolve()) if annotations else None,
            "annotations_sha256": annotation_hash, "baseline_consistent_frames": len(records),
            "baseline_verified_mapping_frames": known_mappings,
            "comparison_notes": {
                "iou_only": "unchanged real FaceTracker, IoU=.3; recorded max_age for all methods",
                "iou_relaxed_control": "unchanged real FaceTracker with offline IoU=.05 control, not production",
                "geometry": "offline greedy score=.5*IoU+.3*max(0,1-center/.6)+.2*size; all gates must pass",
                "geometry_conservative": "geometry plus mutual best and top-two margin >=.15 on both sides; no second-choice fallback",
                "center": "Euclidean center distance / max(old bbox diagonal, new bbox diagonal)",
                "size": "min(min(widths)/max(widths), min(heights)/max(heights))",
                "experimental_only": "exploratory gates/weights, not calibrated, no appearance or motion prediction",
            },
            "metric_definitions": {
                "id_changes": "actor assigned a different track than its previous labeled observation, including after a miss",
                "false_continuations": "track reused by a different labeled actor; zero is not correctness proof",
                "ground_truth": "external manual/synthetic actor labels only; Student/Person/Face IDs never supply truth",
                "unlabeled": "identity metrics null when no actors; partial labels yield incomplete lower-bound counts",
                "new_ids_after_first_frame": "mechanical births after initial frame; includes new arrivals, NOT identity switches",
                "expiry": "observed age > shared max_age before matching; no artificial final flush",
            }, "camera_trials": trials, "risk_trials": risks}


def write_new_report(path, report):
    output = Path(path)
    sources = [Path(report["source_file"])]
    if report["annotations_file"]:
        sources.append(Path(report["annotations_file"]))
    if any(output.resolve() == p.resolve() for p in sources):
        raise ValueError("output must not overwrite source session or annotations")
    payload = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with output.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--annotations", type=Path, help="optional MANUAL frame_id/detection_index/actor labels; never recognition IDs")
    parser.add_argument("--output", type=Path, help="new JSON file; existing files are never overwritten")
    args = parser.parse_args()
    try:
        report = evaluate_session(args.session, args.annotations)
        if args.output:
            write_new_report(args.output, report)
        print(json.dumps({"camera": [{k: v for k, v in r.items() if k != "trace"} for r in report["camera_trials"]],
                          "risk_scenarios": len(report["risk_trials"]), "output": str(args.output) if args.output else None}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Diagnostic C rejected: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
