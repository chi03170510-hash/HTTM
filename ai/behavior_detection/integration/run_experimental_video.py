"""Run experimental three-class YOLO inference on video or webcam.

The model emits frame detections only. It does not determine behavior events,
violations, or identity tracks. CSV output is sampled once per source second;
the annotated video is written at the source frame rate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CHECKPOINT = ROOT / "ai/behavior_detection/models/experimental/three_behavior_v2_best.pt"
REPORT_DIR = ROOT / "reports/experimental_v2_inference"
EXPECTED = {"sleeping", "using_phone", "raising_hand"}
COLORS = {
    "sleeping": (60, 80, 240),
    "using_phone": (30, 180, 255),
    "raising_hand": (60, 210, 80),
}


def _source_value(value: str):
    return int(value) if value.isdecimal() else value


def _class_mapping(names) -> dict[int, str]:
    items = names.items() if isinstance(names, dict) else enumerate(names)
    result = {int(class_id): str(name).strip().lower() for class_id, name in items}
    missing = EXPECTED - set(result.values())
    extra = set(result.values()) - EXPECTED
    if missing or extra or len(result) != 3:
        raise RuntimeError(
            f"Checkpoint class names do not match the required three classes; "
            f"missing={sorted(missing)}, unexpected={sorted(extra)}, names={result}"
        )
    return result


def _draw(frame, boxes, class_names):
    for box in boxes:
        x1, y1, x2, y2 = [int(round(float(v))) for v in box.xyxy[0].tolist()]
        class_id = int(box.cls[0])
        name = class_names[class_id]
        confidence = float(box.conf[0])
        color = COLORS[name]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{name} {confidence:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top = max(0, y1 - th - baseline - 6)
        cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + baseline + 6), color, -1)
        cv2.putText(frame, label, (x1 + 3, top + th + 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (15, 15, 15), 2, cv2.LINE_AA)
    return frame


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video path or webcam index, e.g. 0")
    parser.add_argument("--mode", choices=("baseline", "hybrid"), default="baseline")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. cpu, 0")
    parser.add_argument("--output", type=Path, help="Annotated MP4 output path")
    parser.add_argument("--csv", type=Path, help="Detection CSV (rows sampled once per second)")
    parser.add_argument("--summary", type=Path, help="JSON summary output")
    parser.add_argument("--log-interval", type=float, default=1.0,
                        help="CSV logging interval in seconds (default: 1)")
    args = parser.parse_args(argv)

    if not 0.0 <= args.conf <= 1.0:
        parser.error("--conf must be between 0 and 1")
    if args.imgsz <= 0 or args.log_interval <= 0:
        parser.error("--imgsz and --log-interval must be positive")

    source = _source_value(args.source)
    is_webcam = isinstance(source, int)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_out = args.output or (REPORT_DIR / (f"annotated_webcam_{run_stamp}.mp4" if is_webcam else "annotated_video.mp4"))
    csv_out = args.csv or (REPORT_DIR / "detections.csv")
    summary_out = args.summary or (REPORT_DIR / "summary.json")
    for path in (video_out, csv_out, summary_out):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            parser.error(f"Refusing to overwrite existing output: {path}")

    hybrid_status = "NOT_REQUESTED"
    if args.mode == "hybrid":
        models_dir = ROOT / "ai/behavior_detection/models"
        required = [models_dir / "pretrained_person/yolo11n_coco.pt",
                    models_dir / "pose/yolo11s-pose.pt"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            hybrid_status = "HYBRID_NOT_READY"
            print("HYBRID_NOT_READY: required person/phone and pose weights are missing; "
                  "running YOLO detection baseline. Missing: " + ", ".join(missing), file=sys.stderr)
        else:
            hybrid_status = "HYBRID_NOT_READY"
            print("HYBRID_NOT_READY: auxiliary weights exist, but this experimental runner "
                  "has no validated person/phone association gate; running YOLO baseline.", file=sys.stderr)

    if not args.checkpoint.is_file():
        parser.error(f"Experimental checkpoint not found: {args.checkpoint}")

    model = YOLO(str(args.checkpoint))
    id_to_name = _class_mapping(model.names)
    name_to_id = {name: class_id for class_id, name in id_to_name.items()}
    model_classes = [name_to_id[name] for name in ("sleeping", "using_phone", "raising_hand")]
    checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        parser.error(f"Could not open video/webcam source: {args.source}")

    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    fps_was_fallback = not math.isfinite(source_fps) or source_fps <= 0
    output_fps = 30.0 if fps_was_fallback else source_fps
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if width <= 0 or height <= 0:
        capture.release()
        parser.error("Video source reported invalid frame dimensions")
    writer = cv2.VideoWriter(str(video_out), cv2.VideoWriter_fourcc(*"mp4v"), output_fps,
                             (width, height))
    if not writer.isOpened():
        capture.release()
        parser.error(f"Could not open MP4 output writer: {video_out}")

    frame_counts = Counter({name: 0 for name in EXPECTED})
    log_counts = Counter({name: 0 for name in EXPECTED})
    frame_no = 0
    last_logged_bucket = None
    started = time.perf_counter()
    error = None
    csv_columns = ["timestamp_seconds", "frame_number", "class_name", "class_id",
                   "confidence", "x1", "y1", "x2", "y2", "track_id"]

    try:
        with csv_out.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=csv_columns)
            csv_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_no += 1
                source_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
                timestamp_seconds = max(0.0, source_ms / 1000.0) if source_ms > 0 else (frame_no - 1) / output_fps
                bucket = int(timestamp_seconds / args.log_interval)
                log_this_frame = last_logged_bucket is None or bucket > last_logged_bucket

                predict_kwargs = {"source": frame, "conf": args.conf, "imgsz": args.imgsz,
                                  "classes": model_classes, "verbose": False}
                if args.device is not None:
                    predict_kwargs["device"] = args.device
                result = model.predict(**predict_kwargs)[0]
                boxes = result.boxes
                _draw(frame, boxes if boxes is not None else [], id_to_name)
                writer.write(frame)

                if boxes is not None:
                    for box in boxes:
                        class_id = int(box.cls[0])
                        name = id_to_name[class_id]
                        confidence = float(box.conf[0])
                        coords = [float(v) for v in box.xyxy[0].tolist()]
                        frame_counts[name] += 1
                        if log_this_frame:
                            csv_writer.writerow({
                                "timestamp_seconds": round(timestamp_seconds, 3),
                                "frame_number": frame_no,
                                "class_name": name,
                                "class_id": class_id,
                                "confidence": round(confidence, 6),
                                "x1": round(coords[0], 2), "y1": round(coords[1], 2),
                                "x2": round(coords[2], 2), "y2": round(coords[3], 2),
                                "track_id": "",
                            })
                            log_counts[name] += 1
                if log_this_frame:
                    last_logged_bucket = bucket
                if frame_no % max(1, int(output_fps * 10)) == 0:
                    print(f"Processed {frame_no} frames ({timestamp_seconds:.1f}s)")
    except KeyboardInterrupt:
        error = "interrupted_by_user"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        capture.release()
        writer.release()

    elapsed = time.perf_counter() - started
    summary = {
        "status": "COMPLETED" if error is None else "INTERRUPTED",
        "model_status": "EXPERIMENTAL_UNVERIFIED",
        "annotation_completeness": "UNVERIFIED",
        "evaluation": "UNVERIFIED_EVALUATION",
        "mode_requested": args.mode,
        "mode_executed": "baseline",
        "hybrid_status": hybrid_status,
        "source": str(args.source),
        "source_fps": source_fps if not fps_was_fallback else None,
        "output_fps": output_fps,
        "fps_used_fallback": fps_was_fallback,
        "frame_count": frame_no,
        "processed_every_frame": True,
        "duration_seconds": round(frame_no / output_fps, 3),
        "processing_seconds": round(elapsed, 3),
        "processing_fps": round(frame_no / elapsed, 3) if elapsed > 0 else None,
        "confidence_threshold": args.conf,
        "imgsz": args.imgsz,
        "checkpoint": str(args.checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_hash,
        "class_names": {str(k): v for k, v in id_to_name.items()},
        "per_class_frame_detections": dict(sorted(frame_counts.items())),
        "per_class_csv_logged_detections": dict(sorted(log_counts.items())),
        "csv_logging_interval_seconds": args.log_interval,
        "track_id": "blank; this baseline does not track identities",
        "behavior_events_or_violations": "not inferred; detections are not events",
        "metrics_claimed": False,
        "error": error,
        "outputs": {"video": str(video_out.resolve()), "csv": str(csv_out.resolve()),
                    "summary": str(summary_out.resolve())},
    }
    summary_out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if error is None else 130


if __name__ == "__main__":
    raise SystemExit(main())
