import queue
import json
import os
import tempfile
import threading
import time
from itertools import count
from pathlib import Path

import cv2

from camera_coordinator import (
    ROOT,
    INTEGRATION,
    CAMERA_INDEX,
    WORKER_TIMEOUT,
)
from camera_live_runtime import (LiveWorker as Worker, WorkerStopped, RESULT_QUEUE_SIZE,
                                 SHUTDOWN_TIMEOUT, observation_limit_from_environment, publish_latest)
from person_face_association_v4 import associate_v4
from person_tracker import PersonTracker
from behavior_duration import BEHAVIORS, BehaviorDurationTracker, behavior_label, duration_text
from behavior_overlay import AISnapshot
from behavior_warning import DEFAULT_WARNING_CONFIG, BehaviorWarningObserver
from person_tracker_diagnostics import DiagnosticConfig, PersonTrackerDiagnostics
from identity_continuity_diagnostics import shadow_from_environment
from person_snapshot_overlay import overlay_from_environment
from bbox_visual_metadata import enabled as visual_enabled, attach_to_result
import sys

sys.path.insert(0, str(ROOT / "ai_training" / "recognition"))
from face_tracker import FaceTracker


MAX_AI_FRAMES = None  # No default session limit; tests may inject a bounded run.
_FROM_ENVIRONMENT = object()
PERSON_TRACK_MAX_AGE = 5.0  # Observed AI interval up to 3.7s plus bounded margin.
RESULT_TTL = 0.6
WINDOW_NAME = "HTTM Camera Live"
VIEW_WIDTH = 1280
LIVE_VIEW_WIDTH = 896  # 70%; the monitoring panel uses the remaining 30%.
LIVE_HEADER_HEIGHT = 48
# No CLI/environment switch: production alerts stay disabled for all behaviors.
WARNING_CONFIG = DEFAULT_WARNING_CONFIG


def face_tracker_max_age_from_environment():
    """Camera Live only: unset keeps baseline; only literal '4.0' opts in."""
    value = os.environ.get("HTTM_FACE_TRACKER_TRIAL_MAX_AGE")
    if value is None:
        return 2.0
    if value == "4.0":
        return 4.0
    raise ValueError(
        f"Invalid HTTM_FACE_TRACKER_TRIAL_MAX_AGE={value!r}: "
        "only '4.0' is approved for the camera trial; "
        "unset the variable for baseline max_age=2.0s."
    )


def ai_loop(frame_queue, result_queue, stop_event, max_observations=_FROM_ENVIRONMENT):
    face_worker = None
    hybrid_worker = None
    identity_shadow = None
    person_tracker = PersonTracker(
        iou_threshold=0.3,
        max_age=PERSON_TRACK_MAX_AGE,
        keep_unmatched=False,
    )
    duration_tracker = BehaviorDurationTracker(max_gap=5.0)
    warning_observer = BehaviorWarningObserver(WARNING_CONFIG)
    visualization_enabled = visual_enabled()

    try:
        if max_observations is _FROM_ENVIRONMENT:
            max_observations = observation_limit_from_environment(MAX_AI_FRAMES)
        tracker = FaceTracker(
            iou_threshold=0.3, max_age=face_tracker_max_age_from_environment()
        )
        diagnostics = PersonTrackerDiagnostics(DiagnosticConfig.from_environment(ROOT))
        # Diagnostic limits remain bounded; they do not terminate continuous AI.
        shadow_budget = 10 if max_observations is None else max_observations
        identity_shadow = shadow_from_environment(ROOT, person_tracker, shadow_budget)
        # Choose once: disabled debug calls the original bound method directly.
        update_faces = tracker.update
        face_debug_before = None
        if diagnostics.active and diagnostics.config.face_enabled:
            def update_faces(detections, timestamp):
                nonlocal face_debug_before
                face_debug_before = diagnostics.before_face_update(tracker, timestamp)
                return tracker.update(detections, timestamp)

        face_worker = Worker(
            ROOT / ".venv/Scripts/python.exe",
            INTEGRATION / "face_id_stream_worker.py",
        )
        face_worker.stop_event = stop_event

        hybrid_worker = Worker(
            ROOT / ".venv-hybrid/Scripts/python.exe",
            INTEGRATION / "hybrid_v7_stream_worker.py",
        )
        hybrid_worker.stop_event = stop_event

        with tempfile.TemporaryDirectory(
            prefix="httm_live_"
        ) as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"

            frame_ids = count(1) if max_observations is None else iter(range(1, max_observations + 1))
            completed = 0
            while not stop_event.is_set() and (max_observations is None or completed < max_observations):
                try:
                    captured_at, frame = frame_queue.get(
                        timeout=1
                    )
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if stop_event.is_set():
                    break
                # Empty polling consumes no ID and no observation budget.
                frame_id = next(frame_ids)

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot save temporary frame")

                started = time.perf_counter()
                # Both workers read this JPEG. Retain its decoded pixels before
                # the next iteration overwrites the path; never reload it in UI.
                snapshot_image = cv2.imread(str(image_path))
                if snapshot_image is None:
                    raise RuntimeError("Cannot decode AI snapshot")

                face_started = time.perf_counter()
                face_result = face_worker.process_frame(
                    image_path, frame_id
                )
                face_time = time.perf_counter() - face_started

                if stop_event.is_set():
                    break

                hybrid_started = time.perf_counter()
                hybrid_result = hybrid_worker.process_frame(
                    image_path, frame_id
                )
                hybrid_time = time.perf_counter() - hybrid_started

                detections = [
                    (
                        tuple(face["face_bbox"]),
                        face["student_id"],
                        face["score"],
                    )
                    for face in face_result["faces"]
                ]

                tracked_faces = update_faces(
                    detections,
                    time.perf_counter(),
                )

                # Preserve current recognition separately from stable identity.
                # Match tracking output to detections by bbox only when unique.
                current_faces = {}
                bbox_counts = {}

                for face in face_result["faces"]:
                    key = tuple(face["face_bbox"])
                    bbox_counts[key] = bbox_counts.get(key, 0) + 1
                    current_faces[key] = face

                face_tracks = []
                identity_details = {}

                for track_id, bbox, stable_label, score in tracked_faces:
                    key = tuple(bbox)

                    if bbox_counts.get(key, 0) == 1:
                        current_face = current_faces[key]
                        current_id = current_face["student_id"]
                        current_score = current_face["score"]
                    else:
                        current_id = "Unknown"
                        current_score = None

                    identity_details[track_id] = {
                        "track_id": track_id,
                        "stable_identity": stable_label,
                        "current_face_id": current_id,
                        "current_face_score": current_score,
                    }

                    face_tracks.append(
                        (
                            track_id,
                            bbox,
                            current_id,
                            score,
                        )
                    )
                matches = associate_v4(
                    face_tracks,
                    hybrid_result["people"],
                )

                person_updated_at = time.perf_counter()
                debug_before = (diagnostics.before_update(person_tracker, person_updated_at)
                                if diagnostics.active else None)
                shadow_before = identity_shadow.before_update(person_tracker) if identity_shadow is not None else None
                tracked_people = person_tracker.update(
                    hybrid_result["people"],
                    person_updated_at,
                )

                print(
                    "[PERSON TRACKING] " + json.dumps({
                        "frame_id": frame_id,
                        "updated_at": person_updated_at,
                        "tracked_people": tracked_people,
                    }),
                    flush=True,
                )

                person_track_map = {
                    person["person_ref"]: person["track_id"]
                    for person in tracked_people
                }

                for match in matches:
                    match["person_track_id"] = person_track_map.get(
                        match["person_ref"]
                    )

                shadow_frame = (identity_shadow.analyze(
                    frame_id, captured_at, person_updated_at, shadow_before,
                    tracked_people, face_result["faces"], matches,
                ) if identity_shadow is not None else None)

                behavior_frame = duration_tracker.update(
                    frame_id, captured_at, hybrid_result["people"],
                    tracked_people, matches,
                )
                snapshot = AISnapshot(frame_id, captured_at, behavior_frame, snapshot_image)
                warning_frame = warning_observer.update(
                    behavior_frame, matches, observed_at=time.perf_counter(),
                )
                elapsed = time.perf_counter() - started

                result = {
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "completed_at": time.perf_counter(),
                    "face_count": face_result["face_count"],
                    "person_count": hybrid_result["person_count"],
                    "matches": matches,
                    "tracked_people": tracked_people,
                    "tracked_faces": tracked_faces,
                    "identity_details": identity_details,
                    "behavior_frame": behavior_frame,
                    "warning_frame": warning_frame,
                    "snapshot": snapshot,
                    "face_time": face_time,
                    "hybrid_time": hybrid_time,
                    "elapsed": elapsed,
                }

                if visualization_enabled:
                    try:
                        attach_to_result(result, image_path, face_result, tracked_faces)
                    except Exception as exc:
                        print(f"[PERSON SNAPSHOT OVERLAY] metadata disabled after error; AI continues; "
                              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
                        visualization_enabled = False
                publish_latest(result_queue, result)
                completed += 1
                if identity_shadow is not None:
                    identity_shadow.record(shadow_frame, result=result, raw_people=hybrid_result["people"],
                                           duration_max_gap=duration_tracker.max_gap)
                if debug_before is not None:
                    diagnostics.record(
                        debug_before, person_tracker, hybrid_result["people"], result, image_path,
                        face_context=(face_debug_before, tracker, face_result["faces"])
                        if face_debug_before is not None else None,
                    )

    except WorkerStopped:
        pass
    except Exception as exc:
        publish_latest(result_queue, {"error": str(exc)})

    finally:
        stop_event.set()
        cleanup_errors = []
        for resource, method in ((face_worker, "close"), (hybrid_worker, "close"), (identity_shadow, "finish")):
            if resource is not None:
                try:
                    getattr(resource, method)()
                except Exception as exc:
                    cleanup_errors.append(f"{method}: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            publish_latest(result_queue, {"error": "AI cleanup failed: " + "; ".join(cleanup_errors)})
        publish_latest(result_queue, {"finished": True})



def _wrap_panel_text(text, width, scale):
    """Wrap by actual font width, retaining long IDs instead of truncating them."""
    lines = []
    text = str(text)
    while text:
        low, high = 1, len(text)
        while low < high:
            middle = (low + high + 1) // 2
            if cv2.getTextSize(text[:middle], cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] <= width:
                low = middle
            else:
                high = middle - 1
        split = low
        if split < len(text):
            space = text.rfind(" ", 0, split + 1)
            if space > 0:
                split = space
        lines.append(text[:split])
        text = text[split:].lstrip()
    return lines or [""]


def draw_ai_panel(frame, result, page=0, now=None):
    """Compose an aspect-correct live view (70%) and historical AI panel (30%)."""
    import numpy as np

    now = time.perf_counter() if now is None else now
    panel_width = VIEW_WIDTH - LIVE_VIEW_WIDTH
    normal, muted, accent = (230, 230, 230), (180, 180, 180), (0, 220, 255)
    status = "Waiting for AI..."
    if result is not None:
        status = ("STALE - no new AI result" if now - result.get("completed_at", result["captured_at"])
                  > PERSON_TRACK_MAX_AGE else "Latest completed AI observation")
    observation = result.get("behavior_frame") if result is not None else None
    people = observation.people if observation is not None else ()
    person = people[page % len(people)] if people else None
    face_id = person.face_track_id if person is not None else None
    if isinstance(face_id, int):
        face_id = f"T{face_id}"
    rows = [
        ("AI MONITORING", accent, .62),
        ("Historical AI result - not live", muted, .44),
        (status, accent, .46),
        (f"Frame: {result['frame_id']}" if result else "Frame: --", normal, .48),
        (f"Result age: {max(0., now - result['captured_at']):.1f}s" if result else "Result age: --", normal, .48),
        (f"Processing: {result['elapsed']:.2f}s" if result else "Processing: --", normal, .48),
        (f"Faces: {result['face_count']} | People: {result['person_count']}" if result else "Faces: -- | People: --", normal, .48),
        (f"Tracks: {len(people)} | Page {page % len(people) + 1 if people else 1}/{max(1, len(people))}", muted, .46),
        (f"Student: {person.student_id or 'Unknown'}" if person else "Student: Unknown", accent, .48),
        (f"Face Track: {face_id if face_id is not None else 'Unknown'}", normal, .48),
        (f"Person Track: {person.person_track_id}" if person else "Person Track: Unknown", accent, .48),
        (f"Association: {person.association_status}" if person else "Association: unavailable", normal, .46),
        (f"Identity: {person.identity_status}" if person else "Identity: unavailable", muted, .46),
        ("BEHAVIORS (~ estimated duration)", accent, .46),
    ]
    decisions = {behavior.name: behavior for behavior in person.behaviors} if person else {}
    for name in BEHAVIORS:
        behavior = decisions.get(name)
        value = f"{behavior.state} {duration_text(behavior)}" if behavior else "unknown --"
        rows.append((f"{behavior_label(name)}: {value}", normal, .46))
    if observation is not None and not observation.timestamp_valid:
        rows.append(("Capture time invalid; duration --", accent, .44))

    # Frozen observer output only; repainting cannot advance an episode or emit.
    alerts = result.get("warning_frame") if result is not None else None
    if alerts is None or not alerts.enabled_behaviors:
        rows.append(("Experimental alerts: disabled", muted, .44))
    else:
        rows.append(("Experimental alerts: test configuration", muted, .40))
        fresh = alerts.is_fresh(now, result["frame_id"], result["captured_at"])
        for alert in alerts.views:
            if person is None or (alert.person_track_id, alert.student_id) != (person.person_track_id, person.student_id):
                continue
            if alert.behavior not in alerts.enabled_behaviors:
                continue
            if alert.state == "WARNING_EXPERIMENTAL":
                rows.append(("WARNING (EXPERIMENTAL)" if fresh else "STALE experimental alert (historical)",
                             (0, 100, 255) if fresh else muted, .46))
                rows.append((f"{alert.student_id} | {alert.person_track_id} | {behavior_label(alert.behavior)}", accent, .42))
                rows.append((f"Qualified ~{alert.qualified_seconds:.1f}s | Frame {alerts.frame_id}", normal, .42))
                rows.append(("Not an official violation", muted, .44))
            elif fresh:
                rows.append((f"{behavior_label(alert.behavior)}: {alert.state} (~{alert.qualified_seconds:.1f}s)", muted, .40))
            else:
                rows.append(("Experimental observation: historical/stale", muted, .40))

    wrapped = [(text, color, scale) for text, color, scale in rows
               for text in _wrap_panel_text(text, panel_width - 36, scale)]
    source_height, source_width = frame.shape[:2]
    live_height = max(1, round(source_height * LIVE_VIEW_WIDTH / source_width))
    height = max(540, LIVE_HEADER_HEIGHT + live_height, 28 + len(wrapped) * 24 + 40)
    panel = np.full((height, panel_width, 3), 35, dtype=np.uint8)
    for index, (text, color, scale) in enumerate(wrapped):
        cv2.putText(panel, text, (18, 28 + index * 24), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, 1, cv2.LINE_AA)
    cv2.putText(panel, "N/P: students | Q: Quit", (18, height - 16),
                cv2.FONT_HERSHEY_SIMPLEX, .44, muted, 1, cv2.LINE_AA)

    output = np.full((height, VIEW_WIDTH, 3), 24, dtype=np.uint8)
    preview = cv2.resize(frame, (LIVE_VIEW_WIDTH, live_height),
                         interpolation=cv2.INTER_AREA if LIVE_VIEW_WIDTH < source_width else cv2.INTER_LINEAR)
    top = LIVE_HEADER_HEIGHT + (height - LIVE_HEADER_HEIGHT - live_height) // 2
    output[top:top + live_height, :LIVE_VIEW_WIDTH] = preview
    output[:, LIVE_VIEW_WIDTH:] = panel
    cv2.putText(output, "LIVE CAMERA", (16, 30), cv2.FONT_HERSHEY_SIMPLEX,
                .65, (230, 230, 230), 1, cv2.LINE_AA)
    return output

def main():
    max_observations = observation_limit_from_environment(MAX_AI_FRAMES)
    # Reject invalid trial settings before camera/model/GUI resources are opened.
    face_max_age = face_tracker_max_age_from_environment()
    if face_max_age == 4.0:
        print("[FACE TRACKER TRIAL] max_age=4.0s enabled for this Camera Live session; "
              "production default remains 2.0s.", flush=True)

    for path in (
        ROOT / ".venv/Scripts/python.exe",
        ROOT / ".venv-hybrid/Scripts/python.exe",
        INTEGRATION / "face_id_stream_worker.py",
        INTEGRATION / "hybrid_v7_stream_worker.py",
    ):
        if not path.is_file():
            raise FileNotFoundError(path)

    try:
        person_overlay = overlay_from_environment(stale_after=PERSON_TRACK_MAX_AGE)
    except Exception as exc:
        print(f"[PERSON SNAPSHOT OVERLAY] initialization failed; disabled; AI continues; "
              f"{type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        person_overlay = None
    mode = "continuous; Q to stop" if max_observations is None else f"limited to {max_observations} AI observations"
    print(f"[CAMERA LIVE] {mode}", flush=True)

    for backend_name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            print(f"[CAMERA] Opened camera index {CAMERA_INDEX} with backend {backend_name}",
                  flush=True)
            break
        cap.release()
    else:
        raise RuntimeError(f"Cannot open camera index {CAMERA_INDEX}: DSHOW and MSMF both failed")

    frame_queue = queue.Queue(maxsize=1)
    result_queue = queue.Queue(maxsize=RESULT_QUEUE_SIZE)
    stop_event = threading.Event()

    worker = threading.Thread(
        target=ai_loop,
        args=(frame_queue, result_queue, stop_event, max_observations),
        daemon=True,
    )

    latest_result = None
    window_sized = False
    person_page = 0
    completed = 0
    error = None
    finished = False

    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        worker.start()

        while True:
            ok, frame = cap.read()

            if not ok:
                raise RuntimeError("Camera read failed")

            captured_at = time.perf_counter()

            if not stop_event.is_set():
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    frame_queue.put_nowait(
                        (captured_at, frame.copy())
                    )
                except queue.Full:
                    pass

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break

                if "error" in result:
                    error = result["error"]
                    stop_event.set()

                elif "finished" in result:
                    finished = True

                else:
                    latest_result = result
                    completed = result["frame_id"]

                    print(
                        f"Frame {result['frame_id']}: "
                        f"faces={result['face_count']}, "
                        f"people={result['person_count']}, "
                        f"matched={len(result['matches'])}, "
                        f"face={result['face_time']:.2f}s, "
                        f"hybrid={result['hybrid_time']:.2f}s, "
                        f"total={result['elapsed']:.2f}s, "
                        f"result_age="
                        f"{result['completed_at'] - result['captured_at']:.2f}s",
                        flush=True,
                    )

            frame = draw_ai_panel(frame, latest_result, page=person_page)
            if not window_sized:
                window_scale = min(1., 720 / frame.shape[0])
                cv2.resizeWindow(WINDOW_NAME, round(frame.shape[1] * window_scale),
                                 round(frame.shape[0] * window_scale))
                window_sized = True

            cv2.putText(
                frame,
                f"AI results: {completed}" + (" | continuous" if max_observations is None else f"/{max_observations}"),
                (LIVE_VIEW_WIDTH - 200, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow(WINDOW_NAME, frame)

            if person_overlay is not None:
                try:
                    person_overlay.show(latest_result, now=time.perf_counter())
                except Exception as exc:
                    # Last-resort boundary for the optional visualization only.
                    print(f"[PERSON SNAPSHOT OVERLAY] disabled after display error; live continues; "
                          f"{type(exc).__name__}: {exc}", flush=True)
                    person_overlay = None

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break
            if key == ord("n"):
                person_page += 1
            elif key == ord("p"):
                person_page -= 1

            if error is not None or finished:
                break

    finally:
        stop_event.set()
        try:
            cap.release()
        finally:
            try:
                cv2.destroyAllWindows()
            finally:
                if worker.is_alive():
                    print("Waiting for AI worker to close...")
                    worker.join(timeout=SHUTDOWN_TIMEOUT)
                if worker.is_alive():
                    raise RuntimeError("AI worker did not finish cleanup within shutdown timeout")
                # Q can arrive before the AI thread reports a cleanup error.
                # Inspect terminal controls after joining; do not render new data.
                while True:
                    try:
                        pending = result_queue.get_nowait()
                    except queue.Empty:
                        break
                    if "error" in pending:
                        error = pending["error"]

    if error is not None:
        raise RuntimeError(error)

    print(f"Completed AI results: {completed}")


if __name__ == "__main__":
    main()







