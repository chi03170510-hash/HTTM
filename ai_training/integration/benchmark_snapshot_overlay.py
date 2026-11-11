"""Synthetic render benchmark/preview: no camera, inference or GUI windows."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
import time

import cv2
import numpy as np

from behavior_duration import BehaviorDuration, BehaviorFrame, PersonBehaviorView
from behavior_overlay import AISnapshot, render_snapshot
from bbox_visual_metadata import VisualBox, VisualMetadata
from person_snapshot_overlay import PersonSnapshotOverlay
from snapshot_detail_overlay import render_details


def synthetic_scene(count, *, source_size=(1280, 720), overlap=False, long_sid=False):
    """Hand-authored display inputs, not camera ground truth/model predictions."""
    if not 1 <= count <= 20:
        raise ValueError("synthetic benchmark supports 1..20 people")
    width, height = source_size
    pixels = np.full((height, width, 3), 48, dtype=np.uint8)
    columns = max(1, math.ceil(math.sqrt(count * width / height)))
    rows = max(1, math.ceil(count / columns))
    people, boxes = [], []
    for i in range(count):
        x, y = (i % columns) * width / columns, (i // columns) * height / rows
        w, h = width / columns, height / rows
        if overlap:
            x, y, w, h = width * .25, height * .15, width * .5, height * .75
        bbox = (x, y, min(width - 1., x + w - 2), min(height - 1., y + h - 2))
        sid = f"SV{i + 1:03d}" + ("_LONG_IDENTITY" * 15 if long_sid else "")
        identity = "matched"
        if i % 7 == 5:
            sid, identity = None, "unknown"
        elif i % 7 == 6:
            sid, identity = None, "ambiguous_identity"
        shown_sid = "Ambiguous" if identity.startswith("ambiguous") else sid or "Unknown"
        pid, ref = f"P{i + 1}", f"synthetic_{i + 1}"
        sleeping, phone = ("detected", "suspected") if i % 2 == 0 else ("suspected", "detected")
        behaviors = (BehaviorDuration("sleeping", sleeping, 3. if sleeping == "detected" and sid else None),
                     BehaviorDuration("using_phone", phone, 5. if phone == "detected" and sid else None))
        people.append(PersonBehaviorView(pid, ref, bbox, sid, i + 1, "matched" if sid else "unmatched", identity, behaviors))
        face = (x + w * .36, y + h * .06, x + w * .60, y + h * .30)
        sleep = (x + w * .12, y + h * .25, x + w * .88, y + h * .65)
        using = (x + w * .10, y + h * .40, x + w * .90, y + h * .91)
        obj = (x + w * .55, y + h * .61, x + w * .68, y + h * .78)
        boxes.extend((VisualBox("face", face, .72, pid, shown_sid, i + 1, "recognized" if sid else "Unknown"),
                      VisualBox("sleeping_candidate", sleep, .93, pid, shown_sid, state=sleeping,
                                duration=behaviors[0].estimated_seconds, selected=sleeping == "detected"),
                      VisualBox("using_phone_candidate", using, .81, pid, shown_sid, state=phone,
                                duration=behaviors[1].estimated_seconds),
                      VisualBox("phone_object", obj, .27, pid, shown_sid, state=phone, selected=phone == "detected")))
    frame = BehaviorFrame(7, 10., tuple(people))
    metadata = VisualMetadata(7, 10., source_size, hashlib.sha256(pixels.tobytes()).hexdigest(),
                              "synthetic-no-camera-jpeg", tuple(boxes))
    return {"frame_id": 7, "captured_at": 10., "completed_at": 11.,
            "snapshot": AISnapshot(7, 10., frame, pixels, metadata), "behavior_frame": frame}


def measure(operation, warmup, iterations):
    for _ in range(warmup):
        operation()
    times = []
    for _ in range(iterations):
        start = time.perf_counter_ns()
        operation()
        times.append((time.perf_counter_ns() - start) / 1e6)
    return {"median_ms": statistics.median(times), "p95_ms": float(np.percentile(times, 95)),
            "iterations": iterations, "warmup": warmup}


def benchmark(*, iterations=100, warmup=10, viewport=(640, 480)):
    rows = []
    for count in (1, 5, 10, 20):
        result = synthetic_scene(count)
        snap = result["snapshot"]
        def person():
            return render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., max_size=viewport, person_only=True)
        base = person()
        cached = PersonSnapshotOverlay(stale_after=5., max_size=viewport)
        cached.render(result, now=12.)
        operations = {"cold_render": lambda: PersonSnapshotOverlay(stale_after=5., max_size=viewport).render(result, now=12.),
                      "cached_repaint": lambda: cached.render(result, now=12.),
                      "person_layer": person, "detail_layer": lambda: render_details(snap, base, max_size=viewport)}
        row = {"people": count, "source_size": [1280, 720], "viewport": list(viewport), "detail_boxes": len(snap.render_metadata.boxes)}
        row.update({name: measure(fn, warmup, iterations) for name, fn in operations.items()})
        rows.append(row)
    return {"synthetic_only": True, "scope": "rendering only; no inference, camera, GUI, JPEG sidecar IO or end-to-end FPS",
            "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                            "processor": platform.processor(), "opencv": cv2.__version__, "numpy": np.__version__,
                            "opencv_threads": cv2.getNumThreads(), "timer": "perf_counter_ns"}, "results": rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new directory; existing output refused")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    args.output.mkdir(parents=True, exist_ok=False)
    data = benchmark(iterations=args.iterations, warmup=args.warmup)
    (args.output / "benchmark.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
    for count in (1, 2, 5, 10, 20):
        value = synthetic_scene(count)
        cv2.imwrite(str(args.output / f"synthetic_{count}_people.png"), PersonSnapshotOverlay(stale_after=5.).render(value, now=12.))
    for name, count, kwargs, viewport in (("overlap_20", 20, {"overlap": True}, (640, 480)),
                                         ("long_sid_small", 5, {"long_sid": True}, (320, 240)),
                                         ("portrait", 10, {"source_size": (720, 1280)}, (480, 360))):
        value = synthetic_scene(count, **kwargs)
        cv2.imwrite(str(args.output / f"synthetic_{name}.png"),
                    PersonSnapshotOverlay(stale_after=5., max_size=viewport).render(value, now=30.))
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
