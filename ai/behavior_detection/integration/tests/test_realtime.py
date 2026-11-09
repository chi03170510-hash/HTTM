"""Offline regressions for concurrent playback; no camera, GUI, or GPU required."""

from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ai.behavior_detection.integration import run_video
from ai.behavior_detection.integration.event_schema import DetectionEvent, FrameMetadata, FrameResult
from ai.behavior_detection.integration.realtime import FramePacer, InferenceSnapshot, LatestFrameWorker, RateMeter
from ai.behavior_detection.integration.video_source import FramePacket, VideoSource, VideoSourceError
from ai.behavior_detection import run_httm


FRAME = np.zeros((240, 400, 3), dtype=np.uint8)


def metadata(index=0, timestamp_ms=0):
    return FrameMetadata('fixture', index, timestamp_ms, 400, 240)


def packet(index=0, timestamp_ms=0, value=0):
    return FramePacket(np.full_like(FRAME, value), metadata(index, timestamp_ms))


def result_for(meta):
    return FrameResult('ok', meta, (DetectionEvent(
        'sleeping', 0, .8, (100, 190, 160, 230), meta.frame_index, meta.timestamp_ms),))


def wait_until(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('Timed out waiting for AI worker')
        time.sleep(.002)


class TimingTests(unittest.TestCase):
    def test_actual_fps_and_idle_decay(self):
        meter = RateMeter()
        meter.record(10)
        self.assertEqual(meter.fps(10), 0)
        meter.record(11)
        self.assertEqual(meter.fps(11), 1)
        self.assertEqual(meter.fps(12), .5)
        self.assertEqual(meter.fps(20), 0)

    def test_pacing_preserves_native_frame_intervals(self):
        now = [100.0]
        pacer = FramePacer(25, clock=lambda: now[0])
        self.assertEqual(pacer.remaining(0), 0)
        self.assertAlmostEqual(pacer.remaining(1), .04)
        now[0] += .02
        self.assertAlmostEqual(pacer.remaining(1), .02)
        now[0] += .10
        self.assertEqual(pacer.remaining(2), 0)
        self.assertAlmostEqual(pacer.remaining(4), .04)

    def test_detection_age_includes_inference_latency(self):
        snapshot = InferenceSnapshot(result_for(metadata()), metadata(), 10, 10.1, 12)
        self.assertFalse(snapshot.is_fresh(metadata(60, 2000), 12, 1.5))
        self.assertTrue(snapshot.is_fresh(metadata(30, 1000), 11, 1.5))
        self.assertFalse(snapshot.is_fresh(metadata(30, 2000), 11, 1.5))
        self.assertFalse(snapshot.is_fresh(replace(metadata(), source_id='other'), 11, 1.5))
        self.assertFalse(snapshot.is_fresh(replace(metadata(), frame_width=800), 11, 1.5))
        self.assertFalse(snapshot.is_fresh(metadata(-1), 11, 1.5))

    def test_invalid_worker_and_pacing_rates(self):
        for rate in [0, -1, float('nan'), float('inf')]:
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                LatestFrameWorker(lambda *_: None, target_fps=rate)
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                FramePacer(rate)


class WorkerTests(unittest.TestCase):
    def test_latest_frame_replaces_backlog_and_input_is_copied(self):
        entered = threading.Event()
        release = threading.Event()
        calls = []
        def process(frame, meta):
            calls.append((meta.frame_index, int(frame[0, 0, 0])))
            if len(calls) == 1:
                entered.set()
                self.assertTrue(release.wait(3))
            return result_for(meta)
        worker = LatestFrameWorker(process, target_fps=100).start()
        try:
            original = packet(value=7)
            worker.publish(original)
            self.assertTrue(entered.wait(2))
            original.frame[:] = 99
            for index in range(1, 51):
                worker.publish(packet(index, index * 10, value=index))
            # publish/latest must stay usable while inference is waiting.
            self.assertIsNone(worker.latest())
            release.set()
            wait_until(lambda: len(calls) == 2)
            time.sleep(.03)
            self.assertEqual(calls, [(0, 7), (50, 50)])
        finally:
            release.set()
            worker.stop()
        self.assertFalse(worker._thread.is_alive())
        worker.raise_if_failed()

    def test_default_one_fps_has_no_catch_up_when_inference_is_slow(self):
        starts = []
        done = threading.Event()
        def process(frame, meta):
            starts.append((time.monotonic(), meta.frame_index))
            time.sleep(1.08)
            done.set()
            return result_for(meta)
        worker = LatestFrameWorker(process).start()
        try:
            worker.publish(packet())
            wait_until(lambda: len(starts) == 1)
            for index in range(1, 11):
                worker.publish(packet(index, index * 100))
            wait_until(lambda: len(starts) == 2)
            self.assertGreaterEqual(starts[1][0] - starts[0][0], 1.05)
            self.assertEqual(starts[1][1], 10)
        finally:
            worker.stop()
        self.assertEqual(len(starts), 2)

    def test_target_limits_fast_inference_and_does_not_repeat_a_frame(self):
        starts = []
        def process(frame, meta):
            starts.append(time.monotonic())
            return result_for(meta)
        worker = LatestFrameWorker(process).start()
        try:
            worker.publish(packet())
            wait_until(lambda: len(starts) == 1)
            worker.publish(packet(1, 100))
            wait_until(lambda: len(starts) == 2)
            self.assertGreaterEqual(starts[1] - starts[0], .98)
            time.sleep(.04)
            self.assertEqual(len(starts), 2)
        finally:
            worker.stop()

    def test_result_snapshots_are_not_shared_mutable_dictionaries(self):
        worker = LatestFrameWorker(lambda *_: {'bbox': [1, 2, 3, 4]}).start()
        try:
            worker.publish(packet())
            wait_until(lambda: worker.latest() is not None)
            snapshot = worker.latest()
            snapshot.result['bbox'][0] = 999
            self.assertEqual(worker.latest().result['bbox'][0], 1)
        finally:
            worker.stop()

    def test_worker_and_event_sink_failures_are_reported_and_joined(self):
        for sink_fails in [False, True]:
            def fail(*_): raise ValueError('fixture failure')
            worker = LatestFrameWorker(
                (lambda _, meta: result_for(meta)) if sink_fails else fail,
                on_result=fail if sink_fails else None).start()
            try:
                worker.publish(packet())
                wait_until(lambda: not worker._thread.is_alive())
                with self.assertRaisesRegex(RuntimeError, 'fixture failure'):
                    worker.raise_if_failed()
            finally:
                worker.stop()

    def test_stop_wakes_worker_with_no_input(self):
        worker = LatestFrameWorker(lambda *_: None).start()
        worker.stop()
        worker.stop()
        self.assertFalse(worker._thread.is_alive())


class CaptureTests(unittest.TestCase):
    class Capture:
        def __init__(self, fps=30, position_ms=0):
            self.index = 0
            self.native_fps = fps
            self.position_ms = position_ms
            self.released = False
        def isOpened(self): return True
        def read(self):
            self.index += 1
            return (True, FRAME.copy()) if self.index <= 2 else (False, None)
        def release(self): self.released = True
        def get(self, key):
            return {cv2.CAP_PROP_FPS: self.native_fps, cv2.CAP_PROP_POS_MSEC: self.position_ms,
                    cv2.CAP_PROP_FRAME_COUNT: 2, cv2.CAP_PROP_POS_FRAMES: self.index}.get(key, 0)

    def test_camera_zero_one_and_droidcam_url_use_live_timestamps(self):
        for value in [0, 1, 'http://fixture.invalid:4747/video']:
            capture = self.Capture()
            clock = iter([1000, 1005, 1035])
            with VideoSource(value, cv2_module=cv2, capture_factory=lambda _: capture,
                             clock_ms=lambda: next(clock)) as source:
                self.assertTrue(source.is_live)
                self.assertEqual(source.read().metadata.timestamp_ms, 5)
                self.assertEqual(source.read().metadata.timestamp_ms, 35)
                with self.assertRaises(VideoSourceError): source.read()
            self.assertTrue(capture.released)

    def test_missing_file_fps_and_broken_position_use_fallback(self):
        for bad_fps in [0, float('nan'), float('inf')]:
            capture = self.Capture(fps=bad_fps, position_ms=float('nan'))
            with VideoSource('fixture.avi', cv2_module=cv2, capture_factory=lambda _: capture) as source:
                self.assertFalse(source.is_live)
                self.assertEqual(source.fps, 30)
                self.assertEqual(source.read().metadata.timestamp_ms, 0)
                self.assertEqual(source.read().metadata.timestamp_ms, 33)
                self.assertIsNone(source.read())


class PlaybackTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='httm_playback_test_')
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.video = self.root / 'input.avi'
        writer = cv2.VideoWriter(str(self.video), cv2.VideoWriter_fourcc(*'MJPG'), 30, (400, 240))
        self.assertTrue(writer.isOpened())
        for _ in range(18): writer.write(FRAME)
        writer.release()

    def test_slow_inference_does_not_block_display_and_output_keeps_all_frames(self):
        main_thread = threading.get_ident()
        inference_threads, display_threads, display_times = [], [], []
        ended = []
        class SlowAdapter:
            def process(self, frame, meta):
                inference_threads.append(threading.get_ident())
                time.sleep(.85)
                ended.append(time.monotonic())
                return result_for(meta)
        def display(*_):
            display_threads.append(threading.get_ident())
            display_times.append(time.monotonic())
        output, events = self.root / 'output.mp4', self.root / 'events.jsonl'
        stdout = io.StringIO()
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=SlowAdapter()) as load, \
                patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow') as resize, \
                patch.object(cv2, 'imshow', side_effect=display), \
                patch.object(cv2, 'waitKey', return_value=-1), patch.object(cv2, 'destroyAllWindows') as destroy, \
                redirect_stdout(stdout):
            run_video.main(['--source', str(self.video), '--output', str(output), '--events', str(events)])
        load.assert_called_once_with(run_video.DEFAULT_CHECKPOINT, device='cpu')
        resize.assert_called_once_with(run_video.WINDOW_NAME, 960, 720)
        self.assertEqual(len(display_times), 18)
        self.assertGreater(display_times[-1] - display_times[0], .50)
        self.assertLess(display_times[-1] - display_times[0], .80)
        self.assertLess(display_times[-1], ended[0])
        self.assertEqual(set(display_threads), {main_thread})
        self.assertNotIn(main_thread, inference_threads)
        destroy.assert_called_once()
        cap = cv2.VideoCapture(str(output))
        try:
            self.assertEqual(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 18)
            self.assertAlmostEqual(cap.get(cv2.CAP_PROP_FPS), 30, places=1)
        finally: cap.release()
        rows = [json.loads(line) for line in events.read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(rows), 1)
        self.assertEqual(set(rows[0]), {'status', 'metadata', 'detections', 'error'})
        self.assertEqual(rows[0]['detections'][0]['frame_index'], rows[0]['metadata']['frame_index'])
        summary = json.loads(stdout.getvalue().splitlines()[-1])
        self.assertGreater(summary['video_fps'], 24)
        self.assertEqual(summary['ai_target_fps'], 1)
        self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))

    def test_q_closes_preview_and_joins_worker(self):
        adapter = Mock()
        adapter.process.side_effect = lambda _, meta: result_for(meta)
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter), \
                patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow') as resize, \
                patch.object(cv2, 'imshow'), \
                patch.object(cv2, 'waitKey', return_value=ord('q')), \
                patch.object(cv2, 'destroyAllWindows') as destroy, redirect_stdout(io.StringIO()) as out:
            run_video.main(['--source', str(self.video), '--display-width', '1280', '--display-height', '960'])
        resize.assert_called_once_with(run_video.WINDOW_NAME, 1280, 960)
        summary = json.loads(out.getvalue().splitlines()[-1])
        self.assertEqual(summary['status'], 'stopped_by_user')
        self.assertEqual(summary['video_frames'], 1)
        destroy.assert_called_once()
        self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))

    def test_headless_does_not_use_gui_and_reports_inference_failure(self):
        adapter = Mock()
        adapter.process.side_effect = ValueError('inference fixture failure')
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter), \
                patch.object(cv2, 'namedWindow') as window, patch.object(cv2, 'resizeWindow') as resize, \
                patch.object(cv2, 'imshow') as display, \
                patch.object(cv2, 'waitKey') as keys, redirect_stdout(io.StringIO()) as out:
            with self.assertRaisesRegex(RuntimeError, 'inference fixture failure'):
                run_video.main(['--source', str(self.video), '--no-display'])
        window.assert_not_called()
        resize.assert_not_called()
        display.assert_not_called()
        keys.assert_not_called()
        self.assertEqual(json.loads(out.getvalue().splitlines()[-1])['status'], 'inference_or_output_error')
        self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))

    def test_existing_output_is_never_overwritten(self):
        with redirect_stdout(io.StringIO()), patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_video.main(['--source', str(self.video), '--output', str(self.video)])

    def test_writer_failure_releases_writer_gui_and_ai(self):
        adapter = Mock()
        adapter.process.side_effect = lambda _, meta: result_for(meta)
        writer = Mock()
        writer.isOpened.return_value = False
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter), \
                patch.object(cv2, 'VideoWriter', return_value=writer), \
                patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
                patch.object(cv2, 'destroyAllWindows') as destroy, \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, 'Could not create output video'):
                run_video.main(['--source', str(self.video), '--output', str(self.root / 'bad.mp4')])
        writer.release.assert_called_once()
        destroy.assert_called_once()
        self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))

    def test_camera_open_failure_releases_capture_without_starting_ai(self):
        capture = Mock()
        capture.isOpened.return_value = False
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint'), \
                patch.object(cv2, 'VideoCapture', return_value=capture), redirect_stdout(io.StringIO()):
            with self.assertRaises(VideoSourceError):
                run_video.main(['--source', '0', '--no-display'])
        capture.release.assert_called_once()
        self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))


class LauncherTests(unittest.TestCase):
    def test_three_menu_sources_and_file_picker_are_preserved(self):
        for choice, expected in [('1', ('fixture.avi', 'video')), ('2', ('0', 'webcam')),
                                 ('3', ('1', 'droidcam')), ('0', (None, None))]:
            with patch('builtins.input', return_value=choice), \
                    patch.object(run_httm, 'choose_video_file', return_value='fixture.avi'), \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(run_httm.select_source(), expected)
        with tempfile.TemporaryDirectory(prefix='httm_picker_test_') as folder:
            video = Path(folder) / 'fixture.avi'
            video.touch()
            with patch.object(run_httm.tk, 'Tk') as tk_root, \
                    patch.object(run_httm.filedialog, 'askopenfilename', return_value=str(video)) as picker, \
                    redirect_stdout(io.StringIO()):
                self.assertEqual(run_httm.choose_video_file(), str(video.resolve()))
            picker.assert_called_once()
            tk_root.return_value.destroy.assert_called_once()

    def test_menu_passes_one_fps_and_source_type_to_cli(self):
        with patch.object(run_httm, 'create_output_paths', return_value=(Path('fixture.mp4'), Path('fixture.jsonl'))), \
                patch.object(run_httm.subprocess, 'run') as run, redirect_stdout(io.StringIO()):
            run_httm.run_pipeline('1', 'droidcam')
        args = run.call_args.args[0]
        self.assertEqual(args[args.index('--sample-fps') + 1], '1')
        self.assertEqual(args[args.index('--source-type') + 1], 'droidcam')


if __name__ == '__main__':
    unittest.main()
