
"""Play video on the main thread with independent latest-frame AI inference."""

import argparse
from contextlib import ExitStack
import json
import math
import time
from pathlib import Path

from .detector_adapter import DEFAULT_CHECKPOINT, DetectorAdapter
from .debug_frames import DEFAULT_DEBUG_FRAMES_DIR, DebugFrameRecorder
from .duration_limit import DurationLimit
from .overlay import draw_detections
from .realtime import FramePacer, LatestFrameWorker, RateMeter
from .unified_overlay import draw_unified_result
from .video_source import VideoSource, VideoSourceError


WINDOW_NAME = "HTTM - Classroom Monitoring"


def _quit_requested(delay_ms=1):
    import cv2
    return cv2.waitKey(delay_ms) & 0xFF in (ord("q"), ord("Q"), 27)


def _wait_for_playback(pacer, frame_index, display_enabled, worker):
    """Keep Windows GUI events responsive while waiting for a file frame."""
    while True:
        remaining = pacer.remaining(frame_index)
        if remaining <= 0:
            return True
        worker.raise_if_failed()
        if display_enabled:
            if _quit_requested(max(1, min(10, int(remaining * 1000)))):
                return False
        else:
            time.sleep(min(remaining, 0.01))


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="HTTM Round 3 - independent video playback and CPU AI inference"
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Video path or webcam index (e.g. 0)",
    )
    parser.add_argument("--source-id")
    parser.add_argument("--source-type", choices=("video", "webcam", "droidcam"))
    parser.add_argument("--sample-fps", type=float, default=1.0,
                        help="AI target FPS; video playback is independent (default: 1)")
    parser.add_argument("--detection-ttl", type=float, default=1.5,
                        help="Maximum overlay age from input capture, including AI latency, in seconds")
    parser.add_argument("--max-duration", type=float)
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ch20", action="store_true",
                        help="CH-20: InsightFace/student identity, per-track behavior and duration overlay")
    parser.add_argument("--student-db", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "students_faces.db")
    parser.add_argument("--face-model-root", type=Path, default=Path.home() / ".insightface",
                        help="Existing InsightFace root containing models/buffalo_l (no downloads)")
    parser.add_argument("--debug-detections", action="store_true",
                        help="Report YOLO candidates >= 0.10, filter decisions, and timing to stderr")
    parser.add_argument("--save-debug-frames", action="store_true",
                        help="With --debug-detections, save sampled originals, all-candidate PNGs, and summary")
    parser.add_argument("--debug-frames-dir", type=Path, default=DEFAULT_DEBUG_FRAMES_DIR,
                        help="Parent directory for unique debug sessions (default: results/debug_frames)")
    parser.add_argument("--warmup", action="store_true",
                        help="Warm up Round 3 on a 640x480 BGR dummy image before opening capture")
    parser.add_argument(
        "--pipeline",
        choices=("legacy", "unified"),
        default="legacy",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--events", type=Path)
    parser.add_argument("--display-width", type=int, default=960,
                        help="Initial preview window width in pixels (default: 960)")
    parser.add_argument("--display-height", type=int, default=720,
                        help="Initial preview window height in pixels (default: 720)")
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run without OpenCV preview window",
    )

    args = parser.parse_args(argv)

    if args.display_width <= 0 or args.display_height <= 0:
        parser.error("--display-width and --display-height must be greater than 0")
    if args.save_debug_frames and not args.debug_detections:
        parser.error("--save-debug-frames requires --debug-detections")
    if args.pipeline != "legacy" and (args.debug_detections or args.warmup or args.ch20):
        parser.error("--debug-detections, --warmup and --ch20 require the Round 3 --pipeline legacy")
    if args.ch20 and args.sample_fps != 1.0:
        parser.error("CH-20 uses --sample-fps 1 to preserve the requested AI target")

    if not math.isfinite(args.sample_fps) or args.sample_fps <= 0:
        parser.error("--sample-fps must be finite and greater than 0")
    if not math.isfinite(args.detection_ttl) or args.detection_ttl <= 0:
        parser.error("--detection-ttl must be finite and greater than 0")
    if args.max_duration is not None and (
        not math.isfinite(args.max_duration) or args.max_duration <= 0
    ):
        parser.error("--max-duration must be finite and greater than 0")

    source_arg = (
        int(args.source)
        if args.source.isdecimal()
        else args.source
    )

    for path in (args.output, args.events):
        if path and path.exists():
            parser.error(f"Output already exists: {path}")

    import cv2

    if args.pipeline == "legacy":
        detector_options = {"device": "cpu"}
        if args.debug_detections:
            detector_options["debug_detections"] = True
        adapter = DetectorAdapter.from_default_checkpoint(args.model, **detector_options)
        unified = None
    else:
        from ..pipeline.unified_pipeline import (
            UnifiedClassroomBehaviorPipeline,
        )
        adapter = None
        unified = UnifiedClassroomBehaviorPipeline()

    # Opt-in: move predictor setup/internal warm-up out of the live capture timeline.
    # 640x480 matches the laptop camera dimensions recorded in the existing log.
    warmup_ms = adapter.warmup() if args.warmup else None

    student_processor = None
    student_controller = None
    student_views = ()
    if args.ch20:
        from .student_pipeline import StudentInferenceProcessor, StudentOverlayController
        from .student_overlay import draw_student_overlays
        student_processor = StudentInferenceProcessor.from_local(
            adapter, face_root=args.face_model_root, student_db=args.student_db)
        student_controller = StudentOverlayController(evidence_ttl=args.detection_ttl)
        print("[HTTM] CH-20 enabled: per-track state/timer overlay; SQLite warning writes are not enabled")

    frame_recorder = None
    if args.save_debug_frames:
        frame_recorder = DebugFrameRecorder(
            args.debug_frames_dir, model=adapter.detector.model_path,
            thresholds=adapter.detector.thresholds, classes=adapter.detector.class_names,
        )
        adapter.debug_frame_recorder = frame_recorder
        print(f"[HTTM] Debug frames/report: {frame_recorder.run_dir}")

    duration_limit = DurationLimit(args.max_duration)

    source = VideoSource(
        source_arg,
        source_id=args.source_id,
        cv2_module=cv2,
    )

    writer = None
    event_file = None
    run_status = "completed"
    display_enabled = not args.no_display
    source_type = args.source_type or (
        ("droidcam" if source_arg == 1 else "webcam")
        if isinstance(source_arg, int) else ("droidcam" if source.is_live else "video")
    )
    video_rate = RateMeter()
    video_frames = 0
    first_display_at = None
    last_display_at = None

    def process(frame, metadata):
        if student_processor is not None:
            return student_processor.process(frame, metadata)
        return adapter.process(frame, metadata) if unified is None else unified.predict(frame)

    def handle_result(result, metadata):
        # Only the AI thread touches the event stream; GUI never waits for disk/stdout.
        nonlocal event_file
        payload = (result.to_dict() if unified is None else {
            "metadata": metadata.__dict__, "inference": result,
        })
        if args.events:
            if event_file is None:
                args.events.parent.mkdir(parents=True, exist_ok=True)
                event_file = args.events.open("x", encoding="utf-8")
            event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
            event_file.flush()
        if unified is None:
            print(json.dumps(payload, ensure_ascii=False))
        else:
            print(json.dumps({
                "metadata": metadata.__dict__,
                "runtime_ms": result.get("runtime_ms", {}),
                "raising_hand": result.get("raising_hand", {}),
                "sleeping": {"status": result.get("sleeping", {}).get("sleeping_evidence_status")},
                "using_phone": {"status": result.get("using_phone", {}).get("using_phone_status")},
                "warnings": result.get("warnings", []),
            }, ensure_ascii=False))

    worker = LatestFrameWorker(process, target_fps=args.sample_fps, on_result=handle_result,
                               debug_detections=args.debug_detections)

    try:
        with source:
            if display_enabled:
                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                # HighGUI scales the complete annotated image, including CH-20
                # labels. Only resize the window; AI/output frames stay native.
                cv2.resizeWindow(WINDOW_NAME, args.display_width, args.display_height)

            worker.start()
            output_fps = source.fps
            pacer = None if source.is_live else FramePacer(output_fps)

            while True:
                worker.raise_if_failed()
                try:
                    packet = source.read()
                except VideoSourceError as exc:
                    run_status = "read_error"
                    print(f"Camera/video error: {exc}")
                    break

                if packet is None:
                    break

                if not duration_limit.allows(
                    packet.metadata.timestamp_ms
                ):
                    break

                if pacer is not None and not _wait_for_playback(
                    pacer, packet.metadata.frame_index, display_enabled, worker
                ):
                    run_status = "stopped_by_user"
                    break

                captured_at = time.monotonic()
                worker.publish(packet, captured_at=captured_at)
                frame = packet.frame
                annotated = frame.copy()
                snapshot = worker.latest()
                if snapshot is not None and snapshot.is_fresh(
                    packet.metadata, time.monotonic(), args.detection_ttl
                ):
                    if unified is None:
                        annotated = draw_detections(
                            annotated, snapshot.result, cv2
                        )
                    else:
                        annotated = draw_unified_result(
                            annotated, snapshot.result, cv2
                        )

                stats = worker.stats()
                hud_lines = [
                    "HTTM - Classroom Monitoring",
                    f"VIDEO FPS: {video_rate.fps(time.monotonic()):.1f}",
                    f"AI FPS: {stats['ai_fps']:.1f}",
                    f"AI TARGET: {args.sample_fps:.1f}",
                    "MODEL: Round 3" if unified is None else "MODEL: Unified (experimental)",
                    f"SOURCE: {source_type.capitalize() if source_type != 'droidcam' else 'DroidCam'}",
                    f"AI latency: {snapshot.latency_ms:.0f} ms" if snapshot else "AI: waiting for result",
                ]
                cv2.rectangle(annotated, (0, 0), (380, 180), (20, 20, 20), -1)
                for index, line in enumerate(hud_lines):
                    cv2.putText(annotated, line, (10, 23 + index * 24),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

                if student_controller is not None:
                    student_views = student_controller.advance(snapshot, packet.metadata, time.monotonic())
                    annotated = draw_student_overlays(annotated, student_views, cv2)

                if args.output:
                    if writer is None:
                        args.output.parent.mkdir(
                            parents=True, exist_ok=True
                        )

                        height, width = annotated.shape[:2]

                        writer = cv2.VideoWriter(
                            str(args.output),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            output_fps,
                            (width, height),
                        )

                        if not writer.isOpened():
                            raise RuntimeError(
                                "Could not create output video"
                            )

                    writer.write(annotated)

                if display_enabled:
                    cv2.imshow(WINDOW_NAME, annotated)

                displayed_at = time.monotonic()
                first_display_at = displayed_at if first_display_at is None else first_display_at
                last_display_at = displayed_at
                video_frames += 1
                video_rate.record(displayed_at)
                if display_enabled and _quit_requested():
                    run_status = "stopped_by_user"
                    break

    except KeyboardInterrupt:
        run_status = "stopped_by_user"

    except Exception:
        run_status = "inference_or_output_error"
        raise

    finally:
        # Release GUI/capture before waiting for an in-flight native YOLO call.
        # ExitStack still runs all cleanup callbacks if a device release fails.
        with ExitStack() as cleanup:
            if unified is not None:
                cleanup.callback(unified.close)
            if student_processor is not None:
                cleanup.callback(student_processor.close)
            cleanup.callback(lambda: event_file.close() if event_file is not None else None)
            if frame_recorder is not None:
                cleanup.callback(frame_recorder.close)
            cleanup.callback(worker.stop)
            if display_enabled:
                cleanup.callback(cv2.destroyAllWindows)
            if writer is not None:
                cleanup.callback(writer.release)
            cleanup.callback(source.release)

        stats = worker.stats()
        try:
            worker.raise_if_failed()
        except RuntimeError:
            run_status = "inference_or_output_error"
        video_elapsed = ((last_display_at - first_display_at)
                         if first_display_at is not None and last_display_at is not None else 0)
        debug_report_path = None
        if frame_recorder is not None:
            debug_report_path = frame_recorder.finish(run_status, {
                "video_frames": video_frames,
                "video_fps": (video_frames - 1) / video_elapsed if video_elapsed > 0 else 0.0,
                "ai_target_fps": args.sample_fps, "ai_fps": stats["mean_ai_fps"],
                "processed_frames": stats["processed_frames"], "warmup_ms": warmup_ms,
            })
        print(json.dumps({
            "status": run_status,
            "pipeline": args.pipeline,
            "processed_frames": stats["processed_frames"],
            "mean_inference_latency_ms": (
                stats["inference_seconds"] / stats["processed_frames"] * 1000
                if stats["processed_frames"]
                else None
            ),
            "inference_fps": (
                stats["mean_ai_fps"]
            ),
            "ai_target_fps": args.sample_fps,
            "video_frames": video_frames,
            "video_fps": (video_frames - 1) / video_elapsed if video_elapsed > 0 else 0.0,
            "warmup_ms": warmup_ms,
            "first_inference_latency_ms": stats["first_inference_ms"],
            "debug_report": str(debug_report_path) if debug_report_path is not None else None,
            "ch20": args.ch20,
            "student_tracks_visible": len(student_views),
        }, ensure_ascii=False))

    worker.raise_if_failed()


if __name__ == "__main__":
    main()
