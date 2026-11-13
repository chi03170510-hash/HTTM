"""Camera Live runner preserving HTTM camera connection protocol with AI-Smart core."""

import queue
import sys
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
from camera_live_runtime import (
    LiveWorker as Worker,
    WorkerStopped,
    RESULT_QUEUE_SIZE,
    SHUTDOWN_TIMEOUT,
    observation_limit_from_environment,
    publish_latest,
)
from live_bbox_overlay import AI_TARGET_FPS, FrameRateMeter, InferencePacer, draw_live_bboxes


MAX_AI_FRAMES = None  # No default session limit; tests may inject a bounded run.
_FROM_ENVIRONMENT = object()
PERSON_TRACK_MAX_AGE = 5.0
RESULT_TTL = 0.6
WINDOW_NAME = "HTTM Camera Live"
VIEW_WIDTH = 1280
LIVE_VIEW_WIDTH = 896  # 70%; the monitoring panel uses the remaining 30%.
LIVE_HEADER_HEIGHT = 48

BEHAVIOR_NAMES = (
    "Using Phone",
    "Eating",
    "Sleeping",
)


def _worker_python():
    hybrid_python = ROOT / ".venv-hybrid/Scripts/python.exe"
    return hybrid_python if hybrid_python.is_file() else Path(sys.executable)


def ai_loop(frame_queue, result_queue, stop_event, max_observations=_FROM_ENVIRONMENT):
    ai_worker = None

    try:
        if max_observations is _FROM_ENVIRONMENT:
            max_observations = observation_limit_from_environment(MAX_AI_FRAMES)

        ai_worker = Worker(
            _worker_python(),
            INTEGRATION / "smart_classroom_stream_worker.py",
        )
        ai_worker.stop_event = stop_event

        with tempfile.TemporaryDirectory(
            prefix="httm_live_"
        ) as temp_dir:
            image_path = Path(temp_dir) / "frame.jpg"

            frame_ids = count(1) if max_observations is None else iter(range(1, max_observations + 1))
            completed = 0
            pacer = InferencePacer(AI_TARGET_FPS)
            while not stop_event.is_set() and (max_observations is None or completed < max_observations):
                if pacer.wait(stop_event, time.perf_counter()):
                    break
                try:
                    captured_at, frame = frame_queue.get(timeout=1)
                except queue.Empty:
                    if stop_event.is_set():
                        break
                    continue

                if stop_event.is_set():
                    break
                pacer.started(time.perf_counter())
                frame_id = next(frame_ids)

                if not cv2.imwrite(str(image_path), frame):
                    raise RuntimeError("Cannot save temporary frame")

                started = time.perf_counter()
                ai_result = ai_worker.process_frame(image_path, frame_id)
                if stop_event.is_set():
                    break

                completed_at = time.perf_counter()
                elapsed = completed_at - started
                objects = list(ai_result.get("objects") or ())
                behaviors = list(ai_result.get("behaviors") or ())
                occupancy = int(ai_result.get("occupancy", ai_result.get("person_count", 0)))

                result = {
                    "frame_id": frame_id,
                    "captured_at": captured_at,
                    "completed_at": completed_at,
                    "occupancy": occupancy,
                    "person_count": occupancy,
                    "objects": objects,
                    "behaviors": behaviors,
                    "inference_runtime": ai_result.get("inference_runtime"),
                    "source_size": (frame.shape[1], frame.shape[0]),
                    "object_time": float(ai_result.get("object_inference_time_ms", 0.0)) / 1000.0,
                    "behavior_time": float(ai_result.get("behavior_inference_time_ms", 0.0)) / 1000.0,
                    "inference_time_ms": float(ai_result.get("inference_time_ms", elapsed * 1000.0)),
                    "elapsed": elapsed,
                }

                publish_latest(result_queue, result)
                completed += 1

    except WorkerStopped:
        pass
    except Exception as exc:
        publish_latest(result_queue, {"error": str(exc)})

    finally:
        stop_event.set()
        cleanup_errors = []
        if ai_worker is not None:
            try:
                ai_worker.close()
            except Exception as exc:
                cleanup_errors.append(f"close: {type(exc).__name__}: {exc}")
        if cleanup_errors:
            publish_latest(result_queue, {"error": "AI cleanup failed: " + "; ".join(cleanup_errors)})
        publish_latest(result_queue, {"finished": True})


def _wrap_panel_text(text, width, scale):
    """Wrap by actual font width, retaining long labels instead of truncating them."""
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
    """Compose an aspect-correct live view (70%) and AI-Smart monitoring panel (30%)."""
    import numpy as np

    now = time.perf_counter() if now is None else now
    panel_width = VIEW_WIDTH - LIVE_VIEW_WIDTH
    normal, muted, accent = (230, 230, 230), (180, 180, 180), (0, 220, 255)

    stale = True
    status = "Waiting for AI..."
    if result is not None:
        stale = now - result.get("completed_at", result["captured_at"]) > PERSON_TRACK_MAX_AGE
        status = "STALE - no new AI result" if stale else "Latest completed AI observation"

    objects = list(result.get("objects") or ()) if result is not None else []
    behaviors = list(result.get("behaviors") or ()) if result is not None else []
    occupancy = result.get("occupancy", result.get("person_count", 0)) if result is not None else None

    behavior_counts = {name: 0 for name in BEHAVIOR_NAMES}
    for beh in behaviors:
        name = beh.get("class_name")
        if name in behavior_counts:
            behavior_counts[name] += 1

    items = [("Behavior", b) for b in behaviors] + [("Object", o) for o in objects]
    selected_item = items[page % len(items)] if items else None

    rows = [
        ("AI-SMART MONITORING", accent, .58),
        ("Core: yolov8n + behavior_yolo (3 cls)", muted, .43),
        (status, accent, .45),
        (f"Frame: {result['frame_id']}" if result else "Frame: --", normal, .47),
        (f"Result age: {max(0., now - result['captured_at']):.1f}s" if result else "Result age: --", normal, .47),
        (
            f"Processing: {result['elapsed']:.2f}s ({result.get('inference_time_ms', 0.0):.0f}ms)"
            if result else "Processing: --",
            normal,
            .46,
        ),
        (f"Occupancy (Persons): {occupancy}" if occupancy is not None else "Occupancy (Persons): --", accent, .48),
        (f"Objects Detected: {len(objects)}" if result else "Objects Detected: --", normal, .47),
        (f"Behaviors Detected: {len(behaviors)}" if result else "Behaviors Detected: --", normal, .47),
        ("BEHAVIOR COUNTS (3 CLASSES)", accent, .46),
    ]

    for name in BEHAVIOR_NAMES:
        count_val = behavior_counts[name] if result is not None else "--"
        rows.append((f"{name}: {count_val}", normal, .45))

    rows.append(("DETECTION INSPECTOR", accent, .46))
    if selected_item is not None:
        kind, item = selected_item
        idx = page % len(items)
        bbox = [round(float(v)) for v in item.get("bbox", (0, 0, 0, 0))]
        rows.append((f"Item {idx + 1}/{len(items)} [{kind}]", muted, .44))
        rows.append((f"Class: {item.get('class_name', 'Unknown')} ({item.get('confidence', 0.0):.2f})", normal, .45))
        rows.append((f"BBox: {bbox}", muted, .42))
    else:
        rows.append(("Item 0/0: No detections", muted, .44))

    wrapped = [
        (text, color, scale)
        for text, color, scale in rows
        for text in _wrap_panel_text(text, panel_width - 36, scale)
    ]
    source_height, source_width = frame.shape[:2]
    live_height = max(1, round(source_height * LIVE_VIEW_WIDTH / source_width))
    height = max(540, LIVE_HEADER_HEIGHT + live_height, 28 + len(wrapped) * 23 + 40)
    panel = np.full((height, panel_width, 3), 35, dtype=np.uint8)
    for index, (text, color, scale) in enumerate(wrapped):
        cv2.putText(
            panel,
            text,
            (18, 28 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        "N/P: items | I: live | Q: Quit",
        (18, height - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        .44,
        muted,
        1,
        cv2.LINE_AA,
    )

    preview_source = frame.copy()
    if result is not None and not stale:
        for obj in objects:
            x1, y1, x2, y2 = [int(v) for v in obj["bbox"]]
            color = (0, 255, 0) if str(obj["class_name"]).lower() == "person" else (255, 165, 0)
            cv2.rectangle(preview_source, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview_source,
                f"{obj['class_name']} {obj['confidence']:.2f}",
                (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )
        for beh in behaviors:
            x1, y1, x2, y2 = [int(v) for v in beh["bbox"]]
            color = (255, 0, 255)
            cv2.rectangle(preview_source, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                preview_source,
                f"BEH: {beh['class_name']} {beh['confidence']:.2f}",
                (x1, max(25, y1 - 22)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
            )

    output = np.full((height, VIEW_WIDTH, 3), 24, dtype=np.uint8)
    preview = cv2.resize(
        preview_source,
        (LIVE_VIEW_WIDTH, live_height),
        interpolation=cv2.INTER_AREA if LIVE_VIEW_WIDTH < source_width else cv2.INTER_LINEAR,
    )
    top = LIVE_HEADER_HEIGHT + (height - LIVE_HEADER_HEIGHT - live_height) // 2
    output[top:top + live_height, :LIVE_VIEW_WIDTH] = preview
    output[:, LIVE_VIEW_WIDTH:] = panel
    cv2.putText(
        output,
        "LIVE CAMERA",
        (16, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        .65,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return output


def main(*, show_diagnostics=False):
    max_observations = observation_limit_from_environment(MAX_AI_FRAMES)

    worker_python = _worker_python()
    worker_script = INTEGRATION / "smart_classroom_stream_worker.py"
    for path in (worker_python, worker_script):
        if not path.is_file():
            raise FileNotFoundError(path)

    mode = "continuous; Q to stop" if max_observations is None else f"limited to {max_observations} AI observations"
    print(f"[CAMERA LIVE] {mode}", flush=True)

    for backend_name, backend in (("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)):
        cap = cv2.VideoCapture(CAMERA_INDEX, backend)
        if cap.isOpened():
            print(
                f"[CAMERA] Opened camera index {CAMERA_INDEX} with backend {backend_name}",
                flush=True,
            )
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
    camera_rate, ai_rate = FrameRateMeter(), FrameRateMeter()
    item_page = 0
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
            camera_rate.mark(captured_at)

            if not stop_event.is_set():
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass

                try:
                    frame_queue.put_nowait((captured_at, frame.copy()))
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
                    ai_rate.mark(result["completed_at"])

                    print(
                        f"Frame {result['frame_id']}: "
                        f"occupancy={result['occupancy']}, "
                        f"objects={len(result['objects'])}, "
                        f"behaviors={len(result['behaviors'])}, "
                        f"inference={result['inference_time_ms']:.1f}ms, "
                        f"total={result['elapsed']:.2f}s, "
                        f"result_age={result['completed_at'] - result['captured_at']:.2f}s",
                        flush=True,
                    )

            if show_diagnostics:
                frame = draw_ai_panel(frame, latest_result, page=item_page)
            else:
                display_now = time.perf_counter()
                frame = draw_live_bboxes(
                    frame,
                    latest_result,
                    now=display_now,
                    camera_fps=camera_rate.fps(display_now),
                    ai_fps=ai_rate.fps(display_now),
                    view_width=VIEW_WIDTH,
                )
            if not window_sized:
                window_scale = min(1., 720 / frame.shape[0])
                cv2.resizeWindow(
                    WINDOW_NAME,
                    round(frame.shape[1] * window_scale),
                    round(frame.shape[0] * window_scale),
                )
                window_sized = True

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stop_event.set()
                break
            if key == ord("i"):
                show_diagnostics = not show_diagnostics
                window_sized = False
            if key == ord("n"):
                item_page += 1
            elif key == ord("p"):
                item_page -= 1

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
