"""CLI for an explicitly requested video run; importing this module does no inference."""

import argparse
import json
from pathlib import Path
import time

from .detector_adapter import DEFAULT_CHECKPOINT, DetectorAdapter
from .duration_limit import DurationLimit
from .frame_sampler import FrameSampler
from .overlay import draw_detections
from .unified_overlay import draw_unified_result
from .video_source import VideoSource, VideoSourceError


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sample a video/webcam with HTTM behavior inference")
    parser.add_argument("--source", required=True, help="Video path or webcam index (for example 0)")
    parser.add_argument("--source-id", help="Stable caller supplied source name")
    parser.add_argument("--sample-fps", type=float, required=True)
    parser.add_argument(
        "--max-duration", type=float,
        help="Optional maximum source timeline duration in seconds",
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument(
        "--pipeline", choices=("legacy", "unified"), default="legacy",
        help="legacy runs BehaviorDetector (default V2 three-class checkpoint); unified runs all three AI evidence branches",
    )
    parser.add_argument("--output", type=Path, help="Optional annotated MP4 output")
    parser.add_argument("--events", type=Path, help="Optional JSONL event output")
    args = parser.parse_args(argv)

    source_arg = int(args.source) if args.source.isdecimal() else args.source
    for output_path in (args.output, args.events):
        if output_path and output_path.exists():
            parser.error(f"Refusing to overwrite existing output: {output_path}")

    import cv2

    adapter = None
    unified = None
    if args.pipeline == "legacy":
        adapter = DetectorAdapter.from_default_checkpoint(args.model)
    else:
        from ..pipeline.unified_pipeline import UnifiedClassroomBehaviorPipeline
        unified = UnifiedClassroomBehaviorPipeline()
    sampler = FrameSampler(args.sample_fps)
    duration_limit = DurationLimit(args.max_duration)
    source = VideoSource(source_arg, source_id=args.source_id, cv2_module=cv2)
    writer = None
    event_file = None
    processed_frames = 0
    inference_seconds = 0.0
    run_status = "completed"
    try:
        with source:
            while True:
                try:
                    packet = source.read()
                except VideoSourceError as exc:
                    run_status = "read_error"
                    print(json.dumps({"status": "read_error", "error": str(exc)}))
                    break
                if packet is None:
                    break
                if not duration_limit.allows(packet.metadata.timestamp_ms):
                    break
                if not sampler.should_sample(packet.metadata.timestamp_ms):
                    continue
                inference_started = time.perf_counter()
                if unified is None:
                    result = adapter.process(packet.frame, packet.metadata)
                else:
                    result = unified.predict(packet.frame)
                inference_elapsed = time.perf_counter() - inference_started
                processed_frames += 1
                inference_seconds += inference_elapsed
                if args.events:
                    if event_file is None:
                        args.events.parent.mkdir(parents=True, exist_ok=True)
                        event_file = args.events.open("x", encoding="utf-8")
                    payload = (result.to_dict() if unified is None else {
                        "metadata": packet.metadata.__dict__, "inference": result,
                    })
                    event_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
                    event_file.flush()
                if args.output:
                    if writer is None:
                        args.output.parent.mkdir(parents=True, exist_ok=True)
                        height, width = packet.frame.shape[:2]
                        writer = cv2.VideoWriter(
                            str(args.output), cv2.VideoWriter_fourcc(*"mp4v"),
                            args.sample_fps, (int(width), int(height)),
                        )
                        if not writer.isOpened():
                            raise RuntimeError(f"Could not open output video: {args.output}")
                    annotated = (draw_detections(packet.frame.copy(), result, cv2)
                                 if unified is None else draw_unified_result(packet.frame.copy(), result, cv2))
                    writer.write(annotated)
                if unified is None:
                    print(json.dumps(result.to_dict(), ensure_ascii=False))
                else:
                    print(json.dumps({
                        "metadata": packet.metadata.__dict__,
                        "runtime_ms": result.get("runtime_ms", {}),
                        "raising_hand": result.get("raising_hand", {}),
                        "sleeping": {"status": result.get("sleeping", {}).get("sleeping_evidence_status")},
                        "using_phone": {"status": result.get("using_phone", {}).get("using_phone_status")},
                        "warnings": result.get("warnings", []),
                    }, ensure_ascii=False))
    except Exception:
        run_status = "inference_or_output_error"
        raise
    finally:
        if writer is not None:
            writer.release()
        if event_file is not None:
            event_file.close()
        if unified is not None:
            unified.close()
        print(json.dumps({
            "status": run_status, "pipeline": args.pipeline,
            "processed_frames": processed_frames,
            "mean_inference_latency_ms": (inference_seconds / processed_frames * 1000) if processed_frames else None,
            "inference_fps": (processed_frames / inference_seconds) if inference_seconds else None,
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
