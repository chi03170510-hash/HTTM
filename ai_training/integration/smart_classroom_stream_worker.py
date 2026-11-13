"""Subprocess inference worker running the AI-Smart-Classroom-Monitoring-System core."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2

from inference_device import configure_hybrid_threads, inference_runtime, select_hybrid_device

AI = Path(__file__).resolve().parents[1]
if str(AI) not in sys.path:
    sys.path.insert(0, str(AI))

from src.behavior_checkpoint import behavior_checkpoint_path  # noqa: E402
from src.smart_classroom_core import UnifiedPipeline  # noqa: E402


class SmartClassroomStream:
    """Loads AI-Smart ObjectDetector + BehaviorDetector once and serves frame JSON requests."""

    def __init__(self, device=None):
        if device is None:
            device = select_hybrid_device()
        self.device = device
        self.pipeline = None
        try:
            behavior_ckpt = behavior_checkpoint_path()
            self.pipeline = UnifiedPipeline(
                behavior_model_path=behavior_ckpt,
                device=device,
            )
            configure_hybrid_threads(device)
            print(
                f"[AI-SMART CORE] object={self.pipeline.object_detector.model_path}; "
                f"behavior={self.pipeline.behavior_detector.model_path}; "
                f"classes={self.pipeline.behavior_detector.behavior_classes}",
                file=sys.stderr,
                flush=True,
            )
        except Exception:
            self.close()
            raise

    def process(self, image_path, frame_id):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        output = self.pipeline.process_frame(frame, annotate=False)
        objects = output["objects"]
        behaviors = output["behaviors"]
        occupancy = output["occupancy"]

        return {
            "frame_id": frame_id,
            "image": str(Path(image_path).resolve()),
            "mode": "AI_SMART_UNIFIED",
            "inference_runtime": inference_runtime(
                (self.pipeline.object_detector.model, self.pipeline.behavior_detector.model)
            ),
            "occupancy": occupancy,
            "person_count": occupancy,
            "objects": objects,
            "behaviors": behaviors,
            "object_inference_time_ms": output["object_inference_time_ms"],
            "behavior_inference_time_ms": output["behavior_inference_time_ms"],
            "inference_time_ms": output["inference_time_ms"],
        }

    def close(self):
        self.pipeline = None


def main():
    protocol_stdout = sys.stdout
    sys.stdout = sys.stderr

    worker = None
    try:
        print("Loading AI-Smart UnifiedPipeline...", file=sys.stderr, flush=True)
        worker = SmartClassroomStream(device=select_hybrid_device())
        print("AI-Smart UnifiedPipeline loaded once.", file=sys.stderr, flush=True)

        for line in sys.stdin:
            if not line.strip():
                continue

            frame_id = None
            try:
                request = json.loads(line)
                frame_id = request["frame_id"]
                image_path = Path(request["image"])

                if not image_path.is_file():
                    raise FileNotFoundError(image_path)

                result = worker.process(image_path, frame_id)
            except Exception as exc:
                result = {
                    "frame_id": frame_id,
                    "error": str(exc),
                }

            protocol_stdout.write(json.dumps(result, ensure_ascii=False) + "\n")
            protocol_stdout.flush()
    finally:
        if worker is not None:
            worker.close()
        sys.stdout = protocol_stdout


if __name__ == "__main__":
    main()
