"""AI1: actual AI loop/main/protocol with mock camera and model workers only."""

import copy
import gc
import io
import json
import os
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time
import unittest
import weakref
from contextlib import redirect_stderr, redirect_stdout
from unittest.mock import Mock, patch

import cv2
import numpy as np

import camera_coordinator_live as live
import camera_live_runtime as runtime
import person_snapshot_overlay as overlay
import test_ch20_duration_pipeline as pipeline
import test_identity_continuity_shadow as shadow_tests
from behavior_duration import BehaviorDurationTracker
from behavior_warning import DEFAULT_WARNING_CONFIG


class _Input:
    def __init__(self, items, stop, *, empty_polls=0):
        self.items, self.stop, self.polls = iter(items), stop, empty_polls
        self.paths = []

    def get(self, timeout):
        if self.polls:
            self.polls -= 1
            raise queue.Empty()
        try:
            return next(self.items)
        except StopIteration:
            self.stop.set()
            raise queue.Empty()


class TestContinuousAILoop(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.dict(os.environ))
        for key in tuple(os.environ):
            if key.startswith(("HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG")) or key in (
                    runtime.LIMIT_ENV, "HTTM_IDENTITY_SHADOW", overlay.OVERLAY_ENV, "HTTM_FACE_TRACKER_TRIAL_MAX_AGE"):
                os.environ.pop(key, None)
        os.environ["HTTM_IDENTITY_SHADOW_DIR"] = self.temp.name

    def run_stream(self, count, *, limit=None, empty_polls=0, queue_size=0, failure=None, close_failure=False,
                   stop_after_face=False, overlay_on=False, shadow=False):
        stop = threading.Event()
        sources = [pipeline.observation(i, 100. + (i - 1) * 1.5, second_student="SV002") for i in range(1, count + 1)]
        clock = pipeline._Clock()
        workers = [pipeline._ReplayWorker(kind, sources, clock) for kind in ("face", "hybrid")]
        original_face = workers[0].process_frame
        def face(path, fid):
            if failure:
                raise failure
            reply = original_face(path, fid)
            if stop_after_face:
                stop.set()
            return reply
        workers[0].process_frame = face
        if close_failure:
            workers[0].close = Mock(side_effect=OSError("synthetic close failure"))
        pixels = np.full((120, 160, 3), 70, dtype=np.uint8)
        inputs = _Input([(o.captured_at, pixels) for o in sources], stop, empty_polls=empty_polls)
        results = queue.Queue(maxsize=queue_size)
        arguments = []
        update = BehaviorDurationTracker.update
        def duration(tracker, *args, **kwargs):
            arguments.append(copy.deepcopy((args, kwargs)))
            return update(tracker, *args, **kwargs)
        if overlay_on:
            os.environ[overlay.OVERLAY_ENV] = "1"
        if shadow:
            os.environ["HTTM_IDENTITY_SHADOW"] = "1"
        log = io.StringIO()
        with (patch.object(live, "Worker", side_effect=workers),
              patch.object(live.time, "perf_counter", clock),
              patch.object(BehaviorDurationTracker, "update", duration),
              redirect_stdout(log), redirect_stderr(log)):
            live.ai_loop(inputs, results, stop, limit)
        messages = []
        while not results.empty():
            messages.append(results.get_nowait())
        self.assertTrue(stop.is_set())
        self.assertEqual(messages[-1], {"finished": True})
        self.assertTrue(workers[1].closed)
        if not close_failure:
            self.assertTrue(workers[0].closed)
        self.assertTrue(all(not path.exists() for w in workers for _, path, _ in w.requests))
        return messages, arguments, workers, log.getvalue()

    def test_default_is_continuous_and_processes_beyond_twenty_observations(self):
        self.assertIsNone(live.MAX_AI_FRAMES)
        self.assertIsNone(runtime.observation_limit_from_environment())
        messages, args, workers, _ = self.run_stream(25)
        results = [m for m in messages if "frame_id" in m]
        self.assertEqual([m["frame_id"] for m in results], list(range(1, 26)))
        self.assertEqual(len(args), 25)
        self.assertEqual(len(workers[1].requests), 25)
        self.assertTrue(all(not r["warning_frame"].events for r in results))

    def test_limits_one_ten_twenty_are_counts_of_completed_ai_observations(self):
        for limit in (1, 10, 20):
            with self.subTest(limit=limit):
                messages, args, workers, _ = self.run_stream(25, limit=limit)
                results = [m for m in messages if "frame_id" in m]
                self.assertEqual([m["frame_id"] for m in results], list(range(1, limit + 1)))
                self.assertEqual(len(args), limit)
                self.assertEqual(len(workers[1].requests), limit)

    def test_empty_polling_consumes_no_frame_id_or_limit(self):
        messages, _, _, _ = self.run_stream(10, limit=10, empty_polls=3)
        self.assertEqual([m["frame_id"] for m in messages if "frame_id" in m], list(range(1, 11)))

    def test_worker_error_and_timeout_finish_without_retry_loop(self):
        for failure in (RuntimeError("synthetic worker error"), TimeoutError("synthetic timeout")):
            with self.subTest(failure=failure):
                messages, args, workers, _ = self.run_stream(20, failure=failure)
                self.assertEqual(messages[0], {"error": str(failure)})
                self.assertFalse(args)
                self.assertFalse(workers[1].requests)

    def test_q_between_worker_requests_does_not_start_hybrid(self):
        messages, args, workers, _ = self.run_stream(20, stop_after_face=True)
        self.assertEqual(messages, [{"finished": True}])
        self.assertFalse(args)
        self.assertEqual(len(workers[0].requests), 1)
        self.assertFalse(workers[1].requests)

    def test_cleanup_exception_still_closes_second_worker_and_emits_finished(self):
        messages, _, workers, _ = self.run_stream(1, limit=1, close_failure=True)
        self.assertTrue(workers[1].closed)
        self.assertIn("synthetic close failure", messages[-2]["error"])

    def test_bounded_mailbox_drops_ui_backlog_without_skipping_duration_inputs(self):
        messages, args, _, _ = self.run_stream(25, queue_size=runtime.RESULT_QUEUE_SIZE)
        self.assertLessEqual(len(messages), runtime.RESULT_QUEUE_SIZE)
        self.assertEqual(len(args), 25)
        latest = [m for m in messages if "frame_id" in m][-1]
        self.assertEqual(latest["frame_id"], 25)
        self.assertEqual(pipeline.phone_value(latest), 36.)

    def test_limited_and_unlimited_have_identical_duration_inputs_and_views(self):
        baseline = self.run_stream(20, limit=20)
        actual = self.run_stream(20)
        self.assertEqual(baseline[1], actual[1])
        a = [m["behavior_frame"] for m in baseline[0] if "frame_id" in m]
        b = [m["behavior_frame"] for m in actual[0] if "frame_id" in m]
        self.assertEqual(a, b)

    def test_continuous_overlay_shadow_matrix_keeps_duration_identity_and_warning_unchanged(self):
        baseline = self.run_stream(12)
        for enabled, shadow in ((True, False), (False, True), (True, True)):
            os.environ.pop(overlay.OVERLAY_ENV, None); os.environ.pop("HTTM_IDENTITY_SHADOW", None)
            actual = self.run_stream(12, overlay_on=enabled, shadow=shadow)
            self.assertEqual(baseline[1], actual[1])
            self.assertEqual([m["behavior_frame"] for m in baseline[0] if "frame_id" in m],
                             [m["behavior_frame"] for m in actual[0] if "frame_id" in m])
            self.assertTrue(all(not m["warning_frame"].events for m in actual[0] if "frame_id" in m))
        logs = list(Path(self.temp.name).glob("session_*/frames.jsonl"))
        self.assertEqual(len(logs), 2)
        for path in logs:
            rows = [json.loads(s) for s in path.read_text().splitlines()]
            self.assertEqual(len(rows), 10, "Unlimited AI must not make Shadow diagnostic unbounded")
            self.assertTrue(all(r["parameters"]["T_hold"] is None and not r["production_applied"] for r in rows))

    def test_unexpected_overlay_metadata_exception_disables_only_projection(self):
        baseline = self.run_stream(3)
        with patch.object(live, "attach_to_result", side_effect=RuntimeError("synthetic overlay failure")) as attach:
            actual = self.run_stream(3, overlay_on=True)
        self.assertEqual(baseline[1], actual[1])
        self.assertIn("metadata disabled", actual[3])
        attach.assert_called_once()

    def test_second_worker_initialization_failure_closes_first_worker(self):
        stop, results = threading.Event(), queue.Queue()
        first = Mock()
        with patch.object(live, "Worker", side_effect=[first, RuntimeError("synthetic initialization")]):
            live.ai_loop(queue.Queue(), results, stop, None)
        first.close.assert_called_once()
        self.assertIn("initialization", results.get_nowait()["error"])
        self.assertEqual(results.get_nowait(), {"finished": True})


class TestRuntimePrimitives(unittest.TestCase):
    def test_environment_exact_positive_limits_and_unset(self):
        with patch.dict(os.environ):
            os.environ.pop(runtime.LIMIT_ENV, None)
            self.assertIsNone(runtime.observation_limit_from_environment())
            for value in ("1", "10", "20"):
                os.environ[runtime.LIMIT_ENV] = value
                self.assertEqual(runtime.observation_limit_from_environment(), int(value))

    def test_invalid_configuration_fails_before_camera_or_workers(self):
        for value in ("", "0", "-1", "1.0", " 10", "10 ", "nan", "unlimited", "true", "１２"):
            with (self.subTest(value=value), patch.dict(os.environ, {runtime.LIMIT_ENV: value}),
                  patch.object(live.cv2, "VideoCapture") as camera, patch.object(live, "Worker") as worker):
                with self.assertRaisesRegex(ValueError, runtime.LIMIT_ENV):
                    live.main()
                camera.assert_not_called(); worker.assert_not_called()

    def test_terminal_controls_survive_full_mailbox(self):
        results = queue.Queue(maxsize=runtime.RESULT_QUEUE_SIZE)
        for i in range(100):
            runtime.publish_latest(results, {"frame_id": i})
        for item in ({"error": "worker"}, {"error": "cleanup"}, {"finished": True}):
            runtime.publish_latest(results, item)
        self.assertEqual([results.get_nowait() for _ in range(3)],
                         [{"error": "worker"}, {"error": "cleanup"}, {"finished": True}])

    def test_stop_aware_response_queue_reuses_pending_reply_and_protocol(self):
        stop = threading.Event(); original = queue.Queue()
        proxy = runtime._StopAwareResponses(original, stop)
        original.put(("line", "already queued"))
        self.assertEqual(proxy.get(timeout=1), ("line", "already queued"))
        proxy.put(("eof", None))
        self.assertEqual(original.get_nowait(), ("eof", None))
        stop.set()
        with self.assertRaises(runtime.WorkerStopped):
            proxy.get(timeout=90)

    def test_response_wait_times_out_and_can_be_cancelled_while_waiting(self):
        stop = threading.Event(); proxy = runtime._StopAwareResponses(queue.Queue(), stop)
        with self.assertRaises(queue.Empty):
            proxy.get(timeout=.01)
        output = []
        def waiting():
            try:
                proxy.get(timeout=90)
            except runtime.WorkerStopped:
                output.append("cancelled")
        thread = threading.Thread(target=waiting)
        thread.start(); stop.set(); thread.join(timeout=2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(output, ["cancelled"])

    def test_live_worker_protocol_validates_frame_and_image_and_keeps_timeout(self):
        for response, expected in (({"frame_id": 7, "image": "frame.jpg", "faces": []}, None),
                                   ({"frame_id": 8, "image": "frame.jpg"}, "Frame ID mismatch"),
                                   ({"frame_id": 7, "image": "different.jpg"}, "Image mismatch")):
            worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
            worker.stop_event = threading.Event(); worker.process = Mock(); worker.process.poll.return_value = None
            worker.responses = queue.Queue(); original = worker.responses
            worker.responses.put(("line", json.dumps(response)))
            if expected:
                with self.assertRaisesRegex(RuntimeError, expected):
                    worker.process_frame(Path("frame.jpg"), 7)
            else:
                self.assertEqual(worker.process_frame(Path("frame.jpg"), 7), response)
            self.assertIs(worker.responses, original)
            self.assertEqual(json.loads(worker.process.stdin.write.call_args.args[0]), {"frame_id": 7, "image": "frame.jpg"})

    def test_live_worker_close_terminates_then_kills_and_closes_pipes(self):
        worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
        worker.process = Mock(); worker.reader = Mock()
        worker.process.poll.side_effect = [None, 0]
        worker.process.wait.side_effect = [subprocess.TimeoutExpired("synthetic", 3), 0]
        worker.close()
        worker.process.terminate.assert_called_once(); worker.process.kill.assert_called_once()
        self.assertEqual(worker.process.wait.call_count, 2)
        worker.process.stdin.close.assert_called_once(); worker.process.stdout.close.assert_called_once()
        worker.reader.join.assert_called_once_with(timeout=1)

    def test_q_cancels_actual_protocol_wait_without_waiting_ninety_seconds(self):
        worker = runtime.LiveWorker.__new__(runtime.LiveWorker)
        worker.stop_event = threading.Event(); worker.responses = queue.Queue()
        worker.process = Mock(); worker.process.poll.return_value = None
        output = []
        entered = threading.Event()
        worker.process.stdin.flush.side_effect = entered.set
        def waiting():
            try:
                worker.process_frame(Path("synthetic.jpg"), 1)
            except runtime.WorkerStopped:
                output.append("stopped")
        thread = threading.Thread(target=waiting)
        thread.start()
        try:
            self.assertTrue(entered.wait(timeout=2))
            worker.stop_event.set(); thread.join(timeout=2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(output, ["stopped"])
        finally:
            worker.stop_event.set(); thread.join(timeout=2)

    def test_display_mailbox_and_snapshot_cache_release_historical_pixels(self):
        from benchmark_snapshot_overlay import synthetic_scene
        results = queue.Queue(maxsize=runtime.RESULT_QUEUE_SIZE)
        display = overlay.PersonSnapshotOverlay(stale_after=5.)
        snapshots = []
        for _ in range(30):
            value = synthetic_scene(1, source_size=(160, 120))
            snapshots.append(weakref.ref(value["snapshot"]))
            runtime.publish_latest(results, value)
            display.render(value, now=12.)
        del value
        gc.collect()
        self.assertLessEqual(sum(ref() is not None for ref in snapshots), runtime.RESULT_QUEUE_SIZE)


class TestMainLifecycle(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for key in (runtime.LIMIT_ENV, "HTTM_FACE_TRACKER_TRIAL_MAX_AGE", overlay.OVERLAY_ENV):
            os.environ.pop(key, None)

    def run_main(self, *, read_failure=False, gui_failure=False, worker_error=None, overlay_error=False, cleanup_error=None):
        state = {}
        class FakeThread:
            def __init__(self, target, args, daemon):
                state.update(target=target, args=args, started=False, joined=False)
            def start(self):
                state["started"] = True
                if worker_error:
                    state["args"][1].put({"error": worker_error})
            def is_alive(self):
                return state["started"] and not state["joined"]
            def join(self, timeout):
                state.update(joined=True, join_timeout=timeout)
                if cleanup_error:
                    state["args"][1].put({"error": cleanup_error})
        cap = Mock(); cap.isOpened.return_value = True
        cap.read.return_value = (not read_failure, np.zeros((120, 160, 3), dtype=np.uint8))
        log = io.StringIO()
        with (patch.object(live.Path, "is_file", return_value=True),
              patch.object(live.cv2, "VideoCapture", return_value=cap),
              patch.object(live.threading, "Thread", FakeThread),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "imshow", side_effect=RuntimeError("synthetic GUI error") if gui_failure else None),
              patch.object(live.cv2, "waitKey", return_value=ord("q")),
              patch.object(live.cv2, "destroyAllWindows") as destroy,
              patch.object(live, "overlay_from_environment", side_effect=RuntimeError("synthetic optional overlay") if overlay_error else None,
                           return_value=None), redirect_stdout(log), redirect_stderr(log)):
            expected = "Camera read failed" if read_failure else "synthetic GUI error" if gui_failure else worker_error or cleanup_error
            if expected:
                with self.assertRaisesRegex(RuntimeError, expected):
                    live.main()
            else:
                live.main()
        cap.release.assert_called_once(); destroy.assert_called_once()
        self.assertTrue(state["args"][2].is_set())
        self.assertTrue(state["joined"])
        self.assertEqual(state["join_timeout"], runtime.SHUTDOWN_TIMEOUT)
        self.assertEqual(state["args"][1].maxsize, runtime.RESULT_QUEUE_SIZE)
        return state, log.getvalue()

    def test_q_stops_continuous_main_and_joins_worker(self):
        state, output = self.run_main()
        self.assertIsNone(state["args"][3])
        self.assertIn("continuous; Q to stop", output)

    def test_limited_main_passes_exact_budget_to_ai_loop(self):
        for limit in (1, 10, 20):
            with patch.dict(os.environ, {runtime.LIMIT_ENV: str(limit)}):
                state, output = self.run_main()
            self.assertEqual(state["args"][3], limit)
            self.assertIn(f"limited to {limit}", output)

    def test_camera_read_failure_stops_and_cleans_up(self):
        self.run_main(read_failure=True)

    def test_worker_error_stops_and_cleans_up(self):
        self.run_main(worker_error="synthetic worker timeout")

    def test_gui_exception_cleans_up_ai_worker(self):
        self.run_main(gui_failure=True)

    def test_optional_overlay_initialization_failure_does_not_stop_main(self):
        _, output = self.run_main(overlay_error=True)
        self.assertIn("AI continues", output)

    def test_cleanup_error_arriving_after_q_is_reported(self):
        self.run_main(cleanup_error="synthetic late cleanup error")


class TestThreadedCameraLive(unittest.TestCase):
    """Run real main + AI thread with mock capture/worker/GUI boundaries."""
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.dict(os.environ))
        for name in (runtime.LIMIT_ENV, "HTTM_IDENTITY_SHADOW", overlay.OVERLAY_ENV,
                     "HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG", "HTTM_FACE_TRACKER_TRIAL_MAX_AGE"):
            os.environ.pop(name, None)
        os.environ["HTTM_IDENTITY_SHADOW_DIR"] = self.temp.name

    def run_threaded(self, *, limit=None, q_after=None, failure=None, overlay_on=False, shadow=False):
        workers, paths, ids = [], [], {"face": [], "hybrid": []}
        state = {"last_hybrid": 0, "reads": 0}
        observation = pipeline.observation(1, 100., second_student="SV002")
        class FakeWorker:
            def __init__(self, python_path, script_path):
                self.kind = "face" if "face_id" in str(script_path) else "hybrid"
                self.closed = False; workers.append(self)
            def process_frame(self, path, fid):
                paths.append(Path(path)); ids[self.kind].append(fid)
                if failure and self.kind == "hybrid" and fid == 2:
                    raise TimeoutError("synthetic protocol timeout")
                result = {"frame_id": fid, "image": str(Path(path).resolve())}
                if self.kind == "face":
                    result.update(face_count=len(observation.faces), faces=copy.deepcopy(list(observation.faces)))
                else:
                    result.update(person_count=len(observation.people), people=copy.deepcopy(list(observation.people)))
                    state["last_hybrid"] = fid
                return result
            def close(self):
                self.closed = True
        if limit is not None:
            os.environ[runtime.LIMIT_ENV] = str(limit)
        if overlay_on:
            os.environ[overlay.OVERLAY_ENV] = "1"
        if shadow:
            os.environ["HTTM_IDENTITY_SHADOW"] = "1"
        cap = Mock(); cap.isOpened.return_value = True
        def read():
            state["reads"] += 1
            if state["reads"] > 1000:
                raise AssertionError("Camera loop failed to stop")
            time.sleep(.002)  # Test scheduling only; no production inference pacing.
            return True, np.full((480, 640, 3), 70, dtype=np.uint8)
        cap.read.side_effect = read
        def key(_):
            return ord("q") if q_after and state["last_hybrid"] >= q_after else -1
        log = io.StringIO()
        with (patch.object(live, "Worker", FakeWorker), patch.object(live.Path, "is_file", return_value=True),
              patch.object(live.cv2, "VideoCapture", return_value=cap),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "imshow"), patch.object(live.cv2, "destroyAllWindows"),
              patch.object(live.cv2, "waitKey", side_effect=key), redirect_stdout(log), redirect_stderr(log)):
            if failure:
                with self.assertRaisesRegex(RuntimeError, "synthetic protocol timeout"):
                    live.main()
            else:
                live.main()
        self.assertEqual(len(workers), 2)
        self.assertTrue(all(w.closed for w in workers))
        self.assertTrue(all(not p.exists() for p in paths))
        cap.release.assert_called_once()
        for channel in ids.values():
            self.assertEqual(channel, list(range(1, len(channel) + 1)))
        return ids, log.getvalue()

    def test_real_main_and_ai_thread_stop_automatically_at_one_ten_twenty(self):
        for limit in (1, 10, 20):
            with self.subTest(limit=limit):
                ids, _ = self.run_threaded(limit=limit)
                self.assertEqual(len(ids["face"]), limit)
                self.assertEqual(len(ids["hybrid"]), limit)

    def test_real_main_continuous_q_overlay_shadow_matrix_cleans_up(self):
        for enabled, shadow in ((False, False), (True, False), (False, True), (True, True)):
            with self.subTest(overlay=enabled, shadow=shadow):
                for name in (runtime.LIMIT_ENV, overlay.OVERLAY_ENV, "HTTM_IDENTITY_SHADOW"):
                    os.environ.pop(name, None)
                ids, output = self.run_threaded(q_after=22, overlay_on=enabled, shadow=shadow)
                self.assertGreaterEqual(len(ids["hybrid"]), 22)
                self.assertNotIn("AI thread did not exit", output)
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)
        self.assertFalse(live.WARNING_CONFIG.enabled_behaviors)

    def test_real_ai_thread_timeout_propagates_to_main_and_releases_workers(self):
        ids, _ = self.run_threaded(failure=True)
        self.assertEqual(ids["hybrid"], [1, 2])


if __name__ == "__main__":
    unittest.main()
