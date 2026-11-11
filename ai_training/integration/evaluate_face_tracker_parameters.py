"""Offline-only evaluation of 2/4/5s FaceTracker instances; no Camera Live import.

ID changes of a reference and reuse of a track by different ground-truth actors
are separate metrics. Camera person bindings are observational references, not
physical identity ground truth. No model, image, DB or network access.
"""

import argparse
import hashlib
import json
import math
import statistics
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from face_tracker_diagnostics import capture_tracks  # Sets recognition import path.
from face_tracker import FaceTracker, calculate_iou
from person_face_association_v4 import associate_v4
from replay_face_tracker_diagnostics import replay_records


MAX_AGES = (2.0, 4.0, 5.0)
DEFAULT_SESSION = Path(__file__).resolve().parents[1] / "runs/person_tracker_debug/session_9cbaa96797d7444a8b0fcc77f849eafd/frames.jsonl"


@dataclass(frozen=True)
class FaceInput:
    bbox: tuple
    student_id: str
    score: float
    reference: str | None = None
    actor: str | None = None


@dataclass(frozen=True)
class PersonInput:
    person_ref: str
    person_track_id: str | None
    bbox: tuple


@dataclass(frozen=True)
class FrameInput:
    frame_id: int
    updated_at: float
    captured_at: float
    faces: tuple[FaceInput, ...]
    people: tuple[PersonInput, ...] | None = None


def input_digest(frames):
    # Include all observation metadata and synthetic audit references. All ages
    # receive exactly these same immutable objects; no labels are tracker gates.
    payload = json.dumps([asdict(f) for f in frames], sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _association_signature(matches):
    return sorted((m["track_id"], m["person_ref"], m.get("person_track_id"), m["student_id"], m["association_status"])
                  for m in matches)


def load_camera_session(path):
    data = Path(path).read_bytes()
    records = tuple(json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip())
    verification = replay_records(records)
    if not all(row["consistent"] for row in verification):
        raise ValueError("baseline FaceTracker replay does not match the recorded session")
    frames = []
    for record in records:
        block = record["face_tracker"]
        if record.get("details_truncated") or block["parameters"] != {"iou_threshold": .3, "max_age": 2.0}:
            raise ValueError("complete source session at baseline 2.0 / IoU 0.3 required")
        faces = []
        for index, detection in enumerate(block["detections"]):
            if detection["detection_index"] != index:
                raise ValueError("detection order/index mismatch")
            references = set()
            for mapping in block["mapping"]:
                if not mapping["selection_ambiguous"] and mapping["detection_indices"] == [index]:
                    references.update(a["person_track_id"] for a in mapping["associations"] if a.get("person_track_id"))
            reference = "observed_person:" + next(iter(references)) if len(references) == 1 else None
            faces.append(FaceInput(tuple(detection["face_bbox"]), detection["current_student_id"],
                                   detection["recognition_score"], reference=reference))
        tracks_by_ref = {t["person_ref"]: t["track_id"] for t in record["tracks_after"]}
        people = tuple(PersonInput(d["person_ref"], tracks_by_ref.get(d["person_ref"]), tuple(d["bbox"]))
                       for d in record["detections"])
        frames.append(FrameInput(record["frame_id"], block["face_tracker_updated_at"], record["captured_at"], tuple(faces), people))
    return tuple(frames), records, hashlib.sha256(data).hexdigest()


def _audit_assignments(tracker, returned, faces):
    """Identify consumed inputs from post-state, without influencing matching.

    Bbox + LAST raw label/score are used only to audit the output. Fully
    indistinguishable inputs are marked ambiguous, never arbitrarily assigned.
    """
    candidates = {}
    for index, face in enumerate(faces):
        candidates[index] = [tid for tid, bbox, _, score in returned
                             if tuple(bbox) == face.bbox and score == face.score
                             and tracker.tracks[tid].labels[-1] == face.student_id]
    claimed = Counter(tid for ids in candidates.values() for tid in ids)
    assignments = {i: ids[0] for i, ids in candidates.items() if len(ids) == 1 and claimed[ids[0]] == 1}
    return assignments, len(faces) - len(assignments)


def _associate(frame, returned):
    if frame.people is None:
        return None
    counts = Counter(face.bbox for face in frame.faces)
    current = {face.bbox: face.student_id for face in frame.faces}
    face_tracks = [(tid, bbox, current[tuple(bbox)] if counts[tuple(bbox)] == 1 else "Unknown", score)
                   for tid, bbox, _, score in returned]
    people = [{"person_ref": p.person_ref, "person_bbox_xyxy_clipped": list(p.bbox), "behaviors": {}} for p in frame.people]
    matches = associate_v4(face_tracks, people)
    person_ids = {p.person_ref: p.person_track_id for p in frame.people}
    for match in matches:
        match["person_track_id"] = person_ids.get(match["person_ref"])
    return matches


def run_trial(frames, max_age):
    frames = tuple(frames)
    if not frames:
        raise ValueError("evaluation requires at least one frame")
    tracker = FaceTracker(iou_threshold=.3, max_age=max_age)
    prior_reference, prior_actor, last_time = {}, {}, None
    id_changes = retained = transitions = false_continuations = ambiguous = expiry_count = pair_count = 0
    peak_tracks = 0
    matched_ious, trace = [], []
    reference_observations = actor_observations = 0
    for frame in frames:
        if not math.isfinite(frame.updated_at) or (last_time is not None and frame.updated_at <= last_time):
            raise ValueError("evaluation requires finite, increasing update timestamps")
        before = {tid: (tuple(t.bbox), t.last_seen) for tid, t in tracker.tracks.items()}
        expired = [tid for tid, (_, seen) in before.items() if frame.updated_at - seen > max_age]
        pair_count += (len(before) - len(expired)) * len(frame.faces)
        detections = [(f.bbox, f.student_id, f.score) for f in frame.faces]
        returned = tracker.update(detections, frame.updated_at)
        matched = [calculate_iou(before[tid][0], bbox) for tid, bbox, _, _ in returned if tid in before and tid not in expired]
        matched_ious.extend(matched)
        expiry_count += len(expired)
        assignments, unresolved = _audit_assignments(tracker, returned, frame.faces)
        ambiguous += unresolved
        frame_changes, frame_false = 0, 0
        for index, tid in assignments.items():
            face = frame.faces[index]
            if face.reference is not None:
                reference_observations += 1
                if face.reference in prior_reference:
                    transitions += 1
                    if prior_reference[face.reference] != tid:
                        id_changes += 1
                        frame_changes += 1
                    else:
                        retained += 1
                prior_reference[face.reference] = tid
            if face.actor is not None:
                actor_observations += 1
                if tid in prior_actor and prior_actor[tid] != face.actor:
                    false_continuations += 1
                    frame_false += 1
                prior_actor[tid] = face.actor
        matches = _associate(frame, returned)
        reverse = {tid: index for index, tid in assignments.items()}
        bindings = None if matches is None or any(m["track_id"] not in reverse for m in matches) else sorted(
            (reverse.get(m["track_id"]), m["person_ref"], m.get("person_track_id"), m["student_id"], m["association_status"])
            for m in matches)
        trace.append({"frame_id": frame.frame_id, "updated_at": frame.updated_at, "captured_at": frame.captured_at,
                      "update_gap": None if last_time is None else frame.updated_at - last_time,
                      "face_track_ids": [row[0] for row in returned],
                      "detection_track_ids": [assignments.get(i) for i in range(len(frame.faces))],
                      "stable_labels": [row[2] for row in returned], "expired_ids": expired,
                      "matched_ious": matched, "reference_id_changes": frame_changes,
                      "wrong_actor_continuations": frame_false,
                      "association_signature": None if matches is None else _association_signature(matches),
                      "association_bindings": bindings})
        peak_tracks = max(peak_tracks, len(tracker.tracks))
        last_time = frame.updated_at
    return {"max_age": max_age, "iou_threshold": .3, "input_sha256": input_digest(frames),
            "frames": len(frames), "ids_created": tracker.next_id - 1,
            "reference_id_changes": id_changes, "reference_transitions": transitions,
            "retained_reference_observations": retained, "reference_observations": reference_observations,
            "expiry_count": expiry_count, "matched_observations": len(matched_ious),
            "matched_iou_min": min(matched_ious) if matched_ious else None,
            "matched_iou_mean": statistics.mean(matched_ious) if matched_ious else None,
            "iou_pairs_considered": pair_count, "peak_active_tracks": peak_tracks,
            "wrong_actor_continuations": false_continuations if actor_observations else None,
            "identity_metrics_complete": actor_observations > 0 and actor_observations == sum(len(f.faces) for f in frames) and ambiguous == 0,
            "ambiguous_assignment_count": ambiguous, "trace": trace}


def benchmark_updates(frames, max_age, repeats=200):
    samples = []
    prepared = [(f.updated_at, [(d.bbox, d.student_id, d.score) for d in f.faces]) for f in frames]
    for _ in range(repeats):
        tracker = FaceTracker(iou_threshold=.3, max_age=max_age)
        start = time.perf_counter_ns()
        for timestamp, detections in prepared:
            tracker.update(detections, timestamp)
        samples.append((time.perf_counter_ns() - start) / 1000 / len(prepared))
    return {"repeats": repeats, "median_update_us": statistics.median(samples),
            "minimum_update_us": min(samples), "scope": "core tracker only, existing tiny offline input; not camera FPS"}


def _f(actor, x=20, student=None, score=.8):
    return FaceInput((x, 20, x + 100, 140), student if student is not None else "SV00" + actor,
                     score, reference="actor:" + actor, actor=actor)


def risk_scenarios():
    def seq(rows, times=None, step=1.5, people=None):
        times = times if times is not None else [i * step for i in range(len(rows))]
        return tuple(FrameInput(i + 1, 100. + t, 99.5 + t, tuple(row), people)
                     for i, (t, row) in enumerate(zip(times, rows)))
    a, b = _f("1"), _f("2", 100)
    far = _f("2", 240)
    cases = {
        "stationary": seq([[a]] * 10),
        "head_turn_bbox_jump": seq([[a], [_f("1", 80)], [a]]),
        "two_near": seq([[a, b]] * 6, step=3.),
        "swap_positions": seq([[a, far], [_f("1", 240), _f("2", 20)]], step=3.),
        "crossing": seq([[_f("1", x), _f("2", y)] for x, y in ((0, 140), (30, 110), (60, 80), (100, 40), (130, 10))]),
        "replacement_1_5": seq([[a], [_f("2")]], step=1.5),
        "replacement_3_0": seq([[a], [_f("2")]], step=3.),
        "replacement_4_5": seq([[a], [_f("2")]], step=4.5),
        "missing_one": seq([[a], [], [a]]),
        "missing_many": seq([[a], [], [], [a]]),
        "temporary_unknown": seq([[a], [_f("1", student="Unknown", score=.4)], [a]]),
        "reordered_detections": seq([[a, far], [far, a], [a, far]]),
        "overlapping_reordered": seq([[a, _f("2")], [_f("2"), a]]),
        "person_stable_face_changes": seq([[a]] * 4, step=3., people=(PersonInput("person_0", "P1", (0, 0, 200, 400)),)),
        # Same physical actor with a recognition-label change: not a replacement.
        "student_label_change": seq([[a], [_f("1", student="SV002")], [a]]),
        "close_recognition_scores": seq([[_f("1", score=.800001), _f("2", 240, score=.800002)]] * 5),
    }
    for gap in (2., 2.001, 4., 4.001, 5., 5.001):
        cases[f"gap_{gap:.3f}"] = seq([[a], [a]], step=gap)
    return cases


def evaluate_session(path, benchmark_repeats=200):
    frames, records, digest = load_camera_session(path)
    trials = [run_trial(frames, age) for age in MAX_AGES]
    expected = [_association_signature([{"track_id": m["face_track_id"], **m} for m in r["associations"]]) for r in records]
    baseline_association = [row["association_signature"] == signature for row, signature in zip(trials[0]["trace"], expected)]
    if not all(baseline_association):
        raise ValueError("baseline association projection differs from recorded session")
    baseline_bindings = [row["association_bindings"] for row in trials[0]["trace"]]
    for trial in trials:
        trial["association_binding_frames_equal_baseline"] = sum(binding is not None and row["association_bindings"] == binding
                                                               for row, binding in zip(trial["trace"], baseline_bindings))
        if benchmark_repeats:
            trial["benchmark"] = benchmark_updates(frames, trial["max_age"], benchmark_repeats)
    gaps = [b.updated_at - a.updated_at for a, b in zip(frames, frames[1:])]
    capture_gaps = [b.captured_at - a.captured_at for a, b in zip(frames, frames[1:])]
    return {"schema": "face_tracker_offline_evaluation_v1", "source_file": str(Path(path)), "source_sha256": digest,
            "method": "independent real FaceTracker instances, fixed IoU=.3, original detections/order/timestamps/recognition",
            "metric_definition": {"reference_id_changes": "different track for a reference's next observed detection, including after misses",
                                  "wrong_actor_continuations": "same track assigned to a different synthetic ground-truth actor; camera has no actor ground truth",
                                  "association_scope": "identity/geometry/status projection only; recorded person boxes/IDs fixed; original behavior payload unavailable"},
            "camera_gap_stats": {"count": len(gaps), "mean": statistics.mean(gaps), "minimum": min(gaps), "maximum": max(gaps)},
            "camera_capture_gap_stats": {"count": len(capture_gaps), "mean": statistics.mean(capture_gaps),
                                         "minimum": min(capture_gaps), "maximum": max(capture_gaps)},
            "camera_baseline_replay_consistent_frames": len(records), "camera_baseline_association_equal_frames": sum(baseline_association),
            "camera_trials": trials,
            "risk_trials": {name: [run_trial(sequence, age) for age in MAX_AGES] for name, sequence in risk_scenarios().items()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", nargs="?", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--output", type=Path, help="new JSON evaluation file; never overwrite input or an existing file")
    parser.add_argument("--benchmark-repeats", type=int, default=200)
    args = parser.parse_args()
    if not 0 <= args.benchmark_repeats <= 10000:
        parser.error("benchmark repeats must be 0..10000")
    if args.output and args.output.resolve() == args.session.resolve():
        parser.error("output must not overwrite source session")
    try:
        report = evaluate_session(args.session, args.benchmark_repeats)
        if hashlib.sha256(args.session.read_bytes()).hexdigest() != report["source_sha256"]:
            raise ValueError("source session changed during evaluation")
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
                stream.write("\n")
        print(json.dumps({"camera": [{k: v for k, v in trial.items() if k != "trace"} for trial in report["camera_trials"]],
                          "risk_scenarios": len(report["risk_trials"]), "output": str(args.output) if args.output else None}, indent=2))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.exit(2, f"Evaluation rejected: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
