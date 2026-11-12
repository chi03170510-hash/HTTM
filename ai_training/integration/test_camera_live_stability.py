"""AI3 streaming resource tests. Synthetic capture/workers, real AI pipeline.

No real camera/model is started. RSS is this mock process only, not inference
memory. Optional --report writes a NEW JSON result; no benchmark framework.
"""

import argparse
import copy
import ctypes
import gc
import hashlib
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch
import weakref

import cv2
import numpy as np

import bbox_visual_metadata as visual
import camera_coordinator_live as live
import camera_live_runtime as runtime
from behavior_duration import BehaviorDurationTracker
from behavior_overlay import AISnapshot
from person_snapshot_overlay import PersonSnapshotOverlay


MEASUREMENTS = []


def rss_bytes():
    if os.name != "nt":
        return None
    from ctypes import wintypes
    class Counters(ctypes.Structure):
        _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
            "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
    counters = Counters(); counters.cb = ctypes.sizeof(counters)
    get_process = ctypes.windll.kernel32.GetCurrentProcess
    get_process.restype = wintypes.HANDLE
    get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
    get_memory.argtypes = (wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD)
    if get_memory(get_process(), ctypes.byref(counters), counters.cb):
        return int(counters.WorkingSetSize)
    return None


class _DiscardLog:
    """Do not turn test stdout or mock call history into a false memory leak."""
    def write(self, text):
        return len(text)
    def flush(self):
        pass


def run_stress(count, overlay, shadow, *, fault=None, fault_at=40, slow=False):
    metrics = {"requested": count, "overlay": overlay, "shadow": shadow, "fault": fault,
               "slow_mock": slow, "completed": 0, "duration_updates": 0, "duplicate_ids": False,
               "input_queue_peak": 0, "result_queue_peak": 0, "snapshot_peak": 0,
               "cache_entries_peak": 0, "temp_files_peak": 0, "rss_samples": [], "errors": [],
               "warning_events": 0, "camera_opened": False, "inference_executed": False}
    clock = [100.]
    snapshots = {}
    digest = hashlib.sha256()
    frame = np.full((240, 320, 3), 45, dtype=np.uint8)
    faces = [{"face_bbox": [35, 10, 80, 60], "student_id": "SV001", "score": .85},
             {"face_bbox": [210, 10, 255, 60], "student_id": "SV002", "score": .83}]
    people = [{"person_ref": "person_0", "person_bbox_xyxy_clipped": [5, 5, 150, 235],
               "behaviors": {"using_phone": {"state": "detected", "model_confidence": .91, "bounding_box": [80, 120, 100, 150]},
                             "sleeping": {"state": "not_detected"}}},
              {"person_ref": "person_1", "person_bbox_xyxy_clipped": [170, 5, 315, 235],
               "behaviors": {"sleeping": {"state": "detected", "model_confidence": .92, "bounding_box": [180, 70, 300, 200]},
                             "using_phone": {"state": "not_detected"}}}]
    candidates = [{"class_name": "using_phone", "bbox_xyxy": [15, 65, 140, 215], "confidence": .91,
                   "person_ref": "person_0", "association_status": "associated"},
                  {"class_name": "sleeping", "bbox_xyxy": [180, 70, 300, 200], "confidence": .92,
                   "person_ref": "person_1", "association_status": "associated"}]
    objects = {"phone_detections": [{"detection_id": "phone_0", "bbox_xyxy": [80, 120, 100, 150], "confidence": .3}],
               "v6_phone_person_associations": [{"phone_detection_id": "phone_0", "person_detection_id": "person_0", "status": "ASSOCIATED_GEOMETRICALLY"}]}
    stop = threading.Event()
    class Capture:
        # No per-read mock call history: it would distort long-session RSS.
        failed = False
        releases = 0
        def read(self):
            return (False, None) if self.failed else (True, frame)
        def release(self):
            self.releases += 1
    capture = Capture()

    class Input(queue.Queue):
        index = 0
        def get(self, timeout):
            if self.index >= count or (fault == "stop" and self.index >= fault_at):
                stop.set(); raise queue.Empty()
            if fault == "camera" and self.index + 1 == fault_at:
                capture.failed = True
            ok, pixels = capture.read()
            if not ok:
                raise RuntimeError("synthetic camera read failure")
            self.index += 1
            clock[0] = 100. + (self.index - 1) * 1.5
            self.put_nowait((clock[0], pixels))
            metrics["input_queue_peak"] = max(metrics["input_queue_peak"], self.qsize())
            return super().get(timeout=timeout)

    inputs, results = Input(maxsize=1), queue.Queue(maxsize=runtime.RESULT_QUEUE_SIZE)
    workers = []
    jpeg_directory = [None]

    class Worker:
        def __init__(self, python, script):
            self.kind = "face" if "face_id" in str(script) else "hybrid"
            self.closed, self.last_id = False, 0
            workers.append(self)
        def process_frame(self, path, fid):
            if fid != self.last_id + 1:
                metrics["duplicate_ids"] = True
            self.last_id = fid
            if self.kind == "hybrid" and fid == fault_at:
                if fault == "timeout": raise TimeoutError("synthetic worker timeout")
                if fault == "exit": raise RuntimeError("Worker exited (synthetic)")
                if fault == "exception": raise ValueError("synthetic mid-session exception")
            if slow:
                time.sleep(.001)  # Mock scheduling only; no source pacing change.
            jpeg_directory[0] = Path(path).parent
            clock[0] += .1
            if self.kind == "face":
                reply = {"frame_id": fid, "image": str(Path(path).resolve()), "face_count": 2, "faces": copy.deepcopy(faces)}
                if overlay:
                    visual.write_source(path, fid, frame.shape, "face", faces=reply["faces"])
            else:
                reply = {"frame_id": fid, "image": str(Path(path).resolve()), "person_count": 2, "people": copy.deepcopy(people)}
                if overlay:
                    visual.write_source(path, fid, frame.shape, "hybrid", detections=candidates, objects=objects, people=reply["people"])
            metrics["temp_files_peak"] = max(metrics["temp_files_peak"], sum(p.is_file() for p in Path(path).parent.iterdir()))
            return reply
        def close(self):
            self.closed = True

    display = PersonSnapshotOverlay(stale_after=5., max_size=(320, 240)) if overlay else None
    original_post, original_duration = AISnapshot.__post_init__, BehaviorDurationTracker.update
    original_publish = live.publish_latest

    def post(snapshot):
        original_post(snapshot)
        key = id(snapshot)
        snapshots[key] = weakref.ref(snapshot, lambda ref, key=key: snapshots.pop(key, None))
        metrics["snapshot_peak"] = max(metrics["snapshot_peak"], len(snapshots))

    def duration(tracker, *args, **kwargs):
        metrics["duration_updates"] += 1
        digest.update(json.dumps([args, kwargs], sort_keys=True, allow_nan=False).encode())
        return original_duration(tracker, *args, **kwargs)

    def publish(target, item):
        original_publish(target, item)
        metrics["result_queue_peak"] = max(metrics["result_queue_peak"], target.qsize())
        if "frame_id" in item:
            if item["frame_id"] != metrics["completed"] + 1:
                metrics["duplicate_ids"] = True
            metrics["completed"] += 1
            metrics["warning_events"] += len(item["warning_frame"].events)
            if display:
                display.render(item, now=item["completed_at"])
                metrics["cache_entries_peak"] = max(metrics["cache_entries_peak"], int(display._cached is not None))
            if item["frame_id"] in {1, 25, 100, 250, 500, 750, 1000}:
                metrics["rss_samples"].append({"frame_id": item["frame_id"], "rss_bytes": rss_bytes()})

    metrics["rss_start_bytes"] = rss_bytes()
    with tempfile.TemporaryDirectory(prefix="httm_ai3_diagnostic_") as output:
        env = {"HTTM_PERSON_SNAPSHOT_OVERLAY": "1" if overlay else "0", "HTTM_IDENTITY_SHADOW": "1" if shadow else "0",
               "HTTM_IDENTITY_SHADOW_DIR": output}
        clean = {k: v for k, v in os.environ.items() if not k.startswith("HTTM_")}
        with (patch.dict(os.environ, {**clean, **env}, clear=True), patch.object(live, "Worker", Worker),
              patch.object(live.time, "perf_counter", new=lambda: clock[0]),
              patch.object(AISnapshot, "__post_init__", new=post),
              patch.object(BehaviorDurationTracker, "update", new=duration),
              patch.object(live, "publish_latest", new=publish),
              patch.object(sys, "stdout", _DiscardLog()), patch.object(sys, "stderr", _DiscardLog())):
            try:
                live.ai_loop(inputs, results, stop, None)
            finally:
                capture.release()
        metrics["shadow_rows"] = sum(len(p.read_text().splitlines()) for p in Path(output).glob("session_*/frames.jsonl"))
        metrics["shadow_bytes"] = sum(p.stat().st_size for p in Path(output).glob("session_*/frames.jsonl"))
    metrics["retained_snapshots_after_loop"] = len(snapshots)
    finished = False
    while not results.empty():
        message = results.get_nowait()
        if "error" in message:
            metrics["errors"].append(message["error"])
        finished |= message.get("finished", False)
    del message
    display = None
    gc.collect()
    metrics.update(finished=finished, workers_closed=all(w.closed for w in workers), worker_count=len(workers),
                   camera_released=capture.releases == 1, stop_set=stop.is_set(),
                   temp_directory_removed=jpeg_directory[0] is not None and not jpeg_directory[0].exists(),
                   temp_files_after=0 if jpeg_directory[0] is None or not jpeg_directory[0].exists() else len(list(jpeg_directory[0].iterdir())),
                   snapshots_after_release=len(snapshots), result_items_after_drain=results.qsize(),
                   duration_input_sha256=digest.hexdigest(), rss_end_bytes=rss_bytes(),
                   rss_scope="Current mock-process working set; no model/driver memory; allocator/GUI warm-up may remain")
    MEASUREMENTS.append(metrics)
    return metrics


class TestStreamingStability(unittest.TestCase):
    def assert_resources(self, metrics):
        self.assertLessEqual(metrics["input_queue_peak"], 1)
        self.assertLessEqual(metrics["result_queue_peak"], 3)
        self.assertLessEqual(metrics["snapshot_peak"], 8)
        self.assertLessEqual(metrics["cache_entries_peak"], 1)
        self.assertEqual(metrics["temp_files_after"], 0)
        self.assertEqual(metrics["snapshots_after_release"], 0)
        self.assertTrue(metrics["temp_directory_removed"])
        self.assertTrue(metrics["workers_closed"] and metrics["camera_released"] and metrics["finished"])
        self.assertFalse(metrics["duplicate_ids"])
        self.assertEqual(metrics["warning_events"], 0)
        self.assertLessEqual(metrics["shadow_rows"], 10)

    def test_100_500_1000_overlay_shadow_matrix(self):
        for count in (100, 500, 1000):
            baseline = None
            for overlay, shadow in ((False, False), (True, False), (False, True), (True, True)):
                with self.subTest(count=count, overlay=overlay, shadow=shadow):
                    metrics = run_stress(count, overlay, shadow)
                    self.assert_resources(metrics)
                    self.assertEqual(metrics["completed"], count)
                    self.assertEqual(metrics["duration_updates"], count)
                    self.assertFalse(metrics["errors"])
                    if baseline is None:
                        baseline = metrics["duration_input_sha256"]
                    self.assertEqual(metrics["duration_input_sha256"], baseline)
                    print(f"AI3 stress: {count} overlay={overlay} shadow={shadow} snapshots_peak={metrics['snapshot_peak']} RSS_end={metrics['rss_end_bytes']}", flush=True)

    def test_mid_session_failure_timeout_camera_exit_and_stop_cleanup(self):
        for fault in ("timeout", "exit", "exception", "camera", "stop"):
            with self.subTest(fault=fault):
                metrics = run_stress(500, True, True, fault=fault)
                self.assert_resources(metrics)
                self.assertEqual(metrics["completed"], 40 if fault == "stop" else 39)
                self.assertEqual(bool(metrics["errors"]), fault != "stop")

    def test_slow_mock_worker_does_not_build_queue_backlog(self):
        metrics = run_stress(100, True, True, slow=True)
        self.assert_resources(metrics)
        self.assertEqual(metrics["completed"], 100)


class TestLiveWorkerResourceFailures(unittest.TestCase):
    def process(self):
        process = Mock(); state = {"code": None}
        process.poll.side_effect = lambda: state["code"]
        process.terminate.side_effect = lambda: state.update(code=0)
        process.wait.return_value = 0
        return process

    def test_reader_start_failure_releases_process_created_inside_constructor(self):
        process = self.process()
        reader = Mock(); reader.ident = None
        reader.start.side_effect = RuntimeError("synthetic start failure")
        with patch.object(runtime.subprocess, "Popen", return_value=process), patch.object(runtime.threading, "Thread", return_value=reader):
            with self.assertRaisesRegex(RuntimeError, "synthetic start failure"):
                runtime.LiveWorker("fixture_python", "fixture_worker")
        process.terminate.assert_called_once()
        process.stdin.close.assert_called_once(); process.stdout.close.assert_called_once()
        reader.join.assert_not_called()

    def test_stdout_flood_is_bounded_and_rejected_without_silent_dropping(self):
        worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
        worker.responses = queue.Queue(maxsize=3)
        worker.process = Mock(stdout=io.StringIO('{"frame_id":1}\n' * 5000))
        worker._closing, worker._ipc_failure = False, None
        worker._read_output()
        self.assertEqual(worker.responses.qsize(), 3)
        self.assertIn("backlog", worker._ipc_failure)
        with self.assertRaisesRegex(RuntimeError, "backlog"):
            worker.process_frame(Path("fixture.jpg"), 1)

    def test_startup_logs_do_not_fill_reply_queue_or_replace_ai_result(self):
        process = self.process()
        process.stdout = io.StringIO(
            'Loading library...\nGPU initialized\n{"status":"loading"}\n'
            '{"frame_id":1,"image":"fixture.jpg"}\n')
        with (patch.object(runtime.subprocess, "Popen", return_value=process),
              patch.object(runtime.sys, "stderr", new_callable=io.StringIO) as logs):
            worker = runtime.LiveWorker("fixture_python", "fixture_worker")
            worker.reader.join(timeout=2)
            try:
                self.assertIsNone(worker._ipc_failure)
                self.assertEqual(worker.responses.qsize(), 2)  # Reply + EOF only.
                self.assertEqual(worker.process_frame(Path("fixture.jpg"), 1)["frame_id"], 1)
                self.assertIn("Loading library", logs.getvalue())
            finally:
                worker.close()

    def test_non_protocol_log_flood_remains_bounded(self):
        worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
        worker.responses = queue.Queue(maxsize=3)
        worker.process = Mock(stdout=io.StringIO("unexpected stdout\n" * 5000))
        worker._closing, worker._ipc_failure = False, None
        with patch.object(runtime.sys, "stderr", new_callable=io.StringIO):
            worker._read_output()
        self.assertEqual(worker.responses.qsize(), 1)  # EOF, no logs retained.
        self.assertIn("log flood", worker._ipc_failure)

    def test_reader_starting_after_response_wait_uses_original_bounded_queue(self):
        process = self.process(); process.stdout = io.StringIO('{"frame_id":1,"image":"fixture.jpg"}\n')
        delayed_reader = Mock(); delayed_reader.ident = None
        with patch.object(runtime.subprocess, "Popen", return_value=process), patch.object(runtime.threading, "Thread", return_value=delayed_reader):
            worker = runtime.LiveWorker("fixture_python", "fixture_worker")
        original = worker.responses
        worker.responses = runtime._StopAwareResponses(original, threading.Event())
        try:
            # Thread.start may return before the reader target starts. Exercise
            # that schedule while process_frame has already installed its proxy.
            worker._read_output()
        finally:
            worker.responses = original
            worker.close()
        self.assertIsNone(worker._ipc_failure)
        self.assertEqual(original.get_nowait()[0], "line")
        self.assertEqual(original.get_nowait(), ("eof", None))

    def test_unexpected_eof_and_worker_exit_do_not_wait_forever(self):
        for code in (None, 2):
            worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
            worker.responses = queue.Queue(maxsize=3); worker.process = Mock()
            worker.process.poll.return_value = code
            worker.responses.put(("eof", None))
            with self.assertRaisesRegex(RuntimeError, "Worker closed output|Worker exited"):
                worker.process_frame(Path("fixture.jpg"), 1)

    def test_normal_constructor_reader_protocol_and_close_leave_no_reader_thread(self):
        process = self.process(); process.stdout = io.StringIO('{"frame_id":1,"image":"fixture.jpg"}\n')
        with patch.object(runtime.subprocess, "Popen", return_value=process):
            worker = runtime.LiveWorker("fixture_python", "fixture_worker")
        worker.reader.join(timeout=2)
        try:
            self.assertEqual(worker.process_frame(Path("fixture.jpg"), 1)["frame_id"], 1)
            self.assertLessEqual(worker.responses.qsize(), 3)
        finally:
            worker.close()
        self.assertFalse(worker.reader.is_alive())
        self.assertEqual(process.poll(), 0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    args, remaining = parser.parse_known_args()
    result = unittest.main(argv=[sys.argv[0], *remaining], exit=False).result
    if args.report:
        with args.report.open("x", encoding="utf-8") as stream:
            json.dump({"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
                       "measurements": MEASUREMENTS, "physical_camera_opened": False, "inference_executed": False}, stream, indent=2)
    raise SystemExit(0 if result.wasSuccessful() else 1)
