"""Opt-in lossless frame evidence and counts, written only by the AI worker.

The main thread constructs this before capture and finalizes it after join.
There is no image queue or history of numpy frames retained in memory.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
import math
from pathlib import Path
import time
from uuid import uuid4

import cv2


DEFAULT_DEBUG_FRAMES_DIR = Path(__file__).resolve().parents[3] / "results" / "debug_frames"
ACCEPTED_COLOR = (0, 220, 0)       # OpenCV BGR: green
FILTERED_COLOR = (0, 165, 255)     # OpenCV BGR: orange


class DebugFrameRecorder:
    def __init__(self, directory=DEFAULT_DEBUG_FRAMES_DIR, *, model, thresholds, classes):
        run_name = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid4().hex[:8]
        self.run_dir = Path(directory) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "raw").mkdir()
        (self.run_dir / "candidates").mkdir()
        self._manifest = (self.run_dir / "frames.jsonl").open("x", encoding="utf-8")
        self.model = str(model)
        self.thresholds = dict(thresholds)
        self.classes = dict(classes)
        self._counts = Counter()
        self._classes = {name: Counter() for name in self.classes.values()}
        self._widths = {name: [] for name in self.classes.values()}
        self._image_write_ms = 0.0
        self._status = "running"
        self._runtime = None
        try:
            self._write_summary()
        except Exception:
            self._manifest.close()
            raise

    @staticmethod
    def _save_png(path, frame):
        # imencode + Path.open also supports Unicode Windows paths. No overwrite.
        ok, encoded = cv2.imencode(".png", frame, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        if not ok:
            raise OSError(f"Could not encode debug image: {path}")
        with path.open("xb") as stream:
            stream.write(encoded.tobytes())

    @staticmethod
    def _annotate(frame, metadata, candidates, result):
        rows = [
            f"frame_index={metadata.frame_index}  timestamp_ms={metadata.timestamp_ms}  source={metadata.source_id}",
            "GREEN: accepted (>= official threshold) | ORANGE: filtered | YOLO floor: 0.10",
        ]
        canvas = frame.copy()
        for index, candidate in enumerate(candidates or (), 1):
            bbox = candidate["bbox_xyxy"]
            name = candidate["class_name"] or f"class_{candidate['class_id']}"
            accepted = candidate["decision"] == "accepted"
            color = ACCEPTED_COLOR if accepted else FILTERED_COLOR
            width, height = bbox[2] - bbox[0], bbox[3] - bbox[1]
            state = "ACCEPTED" if accepted else candidate["decision"]
            rows.append(f"#{index} {name} conf={candidate['confidence']:.4f} {state} "
                        f"box={width:.1f}x{height:.1f}px "
                        f"xyxy=({', '.join(f'{v:.1f}' for v in bbox)})")
            if all(math.isfinite(v) for v in bbox):
                x1, y1, x2, y2 = (int(round(v)) for v in bbox)
                x1, x2 = (max(0, min(v, frame.shape[1] - 1)) for v in (x1, x2))
                y1, y2 = (max(0, min(v, frame.shape[0] - 1)) for v in (y1, y2))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
                cv2.putText(canvas, f"#{index}", (x1, max(12, y1 - 4)),
                            cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        if result.status != "ok":
            rows.append(f"INFERENCE ERROR: {result.error}")
        elif candidates is None:
            rows.append("YOLO candidates unavailable; not counted as an empty YOLO result")
        elif not candidates:
            rows.append("No YOLO candidates at confidence floor 0.10 (after NMS)")
        else:
            rows.append(f"Accepted event detections: {len(result.detections)}")

        # A footer, with no source resize/shift, preserves even tiny phone regions.
        # Extra width is blank margin for readable metadata, not scaled image data.
        font, scale = cv2.FONT_HERSHEY_SIMPLEX, .42
        width = max(frame.shape[1], max(cv2.getTextSize(row, font, scale, 1)[0][0]
                                       for row in rows) + 20)
        canvas = cv2.copyMakeBorder(canvas, 0, len(rows) * 22 + 12, 0,
                                   width - frame.shape[1], cv2.BORDER_CONSTANT, value=(20, 20, 20))
        for index, row in enumerate(rows):
            color = (230, 230, 230) if index < 2 else (
                ACCEPTED_COLOR if " ACCEPTED " in row else FILTERED_COLOR)
            cv2.putText(canvas, row, (10, frame.shape[0] + 22 * (index + 1)),
                        font, scale, color, 1, cv2.LINE_AA)
        return canvas

    def record(self, frame, metadata, candidates, result):
        """Save the actual sampled input, even when the accepted list is empty."""
        sequence = self._counts["processed_frames"] + 1
        filename = f"sample_{sequence:06d}_frame_{metadata.frame_index:010d}_t{metadata.timestamp_ms:012d}ms.png"
        raw_path = Path("raw") / filename
        overlay_path = Path("candidates") / filename
        candidates = deepcopy(candidates)
        started = time.perf_counter()
        self._save_png(self.run_dir / raw_path, frame)
        annotated = self._annotate(frame, metadata, candidates, result)
        self._save_png(self.run_dir / overlay_path, annotated)
        image_write_ms = (time.perf_counter() - started) * 1000

        entry = {
            "sample_index": sequence, "metadata": metadata.__dict__,
            "raw_image": raw_path.as_posix(), "candidate_image": overlay_path.as_posix(),
            "original_shape_hwc": list(frame.shape), "image_write_ms": image_write_ms,
            "candidates": candidates, "inference": result.to_dict(),
        }
        self._manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._manifest.flush()
        self._accumulate(candidates, result)
        self._image_write_ms += image_write_ms
        self._write_summary()

    def _accumulate(self, candidates, result):
        counts = self._counts
        counts["processed_frames"] += 1
        if result.status != "ok":
            counts["inference_error_frames"] += 1
            return
        counts["successful_frames"] += 1
        if candidates is None:
            counts["candidate_data_missing_frames"] += 1
        elif not candidates:
            counts["frames_without_yolo_candidates"] += 1
        counts["frames_without_accepted_detections" if not result.detections
               else "frames_with_accepted_detections"] += 1
        low = [c for c in candidates or () if c["decision"] == "below_class_threshold"]
        counts["low_confidence_candidates"] += len(low)
        counts["frames_with_low_confidence_candidates"] += bool(low)
        counts["accepted_detections"] += len(result.detections)
        counts["yolo_candidates"] += len(candidates or ())
        counts["other_filtered_candidates"] += sum(
            c["decision"] not in ("accepted", "below_class_threshold") for c in candidates or ())

        for name, class_counts in self._classes.items():
            items = [c for c in candidates or () if c["class_name"] == name]
            class_low = [c for c in items if c["decision"] == "below_class_threshold"]
            accepted = [d for d in result.detections if d.behavior == name]
            class_counts["candidate_count"] += len(items)
            class_counts["frames_with_candidates"] += bool(items)
            class_counts["frames_with_low_confidence_candidates"] += bool(class_low)
            class_counts["low_confidence_candidates"] += len(class_low)
            class_counts["accepted_detections"] += len(accepted)
            class_counts["frames_with_accepted_detections"] += bool(accepted)
            self._widths[name].extend(c["bbox_xyxy"][2] - c["bbox_xyxy"][0] for c in items
                                      if all(math.isfinite(v) for v in c["bbox_xyxy"]))

    def summary(self):
        keys = ("processed_frames", "successful_frames", "inference_error_frames",
                "candidate_data_missing_frames", "frames_without_yolo_candidates",
                "frames_without_accepted_detections", "frames_with_accepted_detections",
                "frames_with_low_confidence_candidates", "low_confidence_candidates",
                "accepted_detections", "yolo_candidates", "other_filtered_candidates")
        per_class = {}
        for name, counts in self._classes.items():
            widths = self._widths[name]
            per_class[name] = {key: counts[key] for key in (
                "candidate_count", "frames_with_candidates", "frames_with_low_confidence_candidates",
                "low_confidence_candidates", "accepted_detections", "frames_with_accepted_detections")}
            per_class[name]["bbox_width_px"] = (
                {"min": min(widths), "max": max(widths), "mean": sum(widths) / len(widths)}
                if widths else None)
        return {
            "status": self._status, "run_directory": str(self.run_dir),
            "model": self.model, "classes": self.classes, "thresholds": self.thresholds,
            "candidate_confidence_floor": .10,
            "counts": {key: self._counts[key] for key in keys}, "per_class": per_class,
            "mean_image_write_ms": (self._image_write_ms / self._counts["processed_frames"]
                                    if self._counts["processed_frames"] else 0.0),
            "runtime": self._runtime,
            "notes": [
                "Frame counts and box counts differ; one frame may contain multiple candidates/classes.",
                "Low-confidence and accepted-frame categories may overlap in the same frame.",
                "Inference errors are separate from successful frames with no candidates/detections.",
                "Candidate boxes are post-NMS at conf=0.10, in original input-pixel coordinates.",
                "Model boxes are not ground truth; visual review is required before dataset decisions.",
            ],
        }

    def _write_summary(self):
        temporary = self.run_dir / "summary.tmp"
        temporary.write_text(json.dumps(self.summary(), ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(self.run_dir / "summary.json")

    def close(self):
        self._manifest.close()

    def finish(self, status, runtime):
        self.close()
        self._status, self._runtime = status, runtime
        self._write_summary()
        return self.run_dir / "summary.json"
