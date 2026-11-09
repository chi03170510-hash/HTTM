"""Lossless visual evidence, honest summary counts, and independent playback."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ai.behavior_detection.behavior_detector import BehaviorDetector, DEFAULT_MODEL_PATH
from ai.behavior_detection.integration.debug_frames import (
    ACCEPTED_COLOR, FILTERED_COLOR, DebugFrameRecorder,
)
from ai.behavior_detection.integration.detector_adapter import DetectorAdapter
from ai.behavior_detection.integration.event_schema import DetectionEvent, FrameMetadata, FrameResult
from ai.behavior_detection.integration import run_video


CLASSES = {0: 'sleeping', 1: 'using_phone'}
THRESHOLDS = {'sleeping': .4, 'using_phone': .4}
FRAME = np.full((240, 400, 3), (12, 34, 56), dtype=np.uint8)


def meta(index=12):
    return FrameMetadata('webcam-fixture', index, 359 + index * 1000, 400, 240)


def candidate(confidence=.2, name='using_phone', bbox=(80., 20., 110., 60.)):
    return {'class_id': 1 if name == 'using_phone' else 0, 'class_name': name,
            'confidence': confidence, 'bbox_xyxy': list(bbox), 'official_threshold': .4,
            'decision': 'accepted' if confidence >= .4 else 'below_class_threshold',
            'bbox_in_source_bounds': True}


def result(index, accepted=()):
    metadata = meta(index)
    return FrameResult('ok', metadata, tuple(DetectionEvent(
        c['class_name'], c['class_id'], c['confidence'], tuple(map(int, c['bbox_xyxy'])),
        metadata.frame_index, metadata.timestamp_ms) for c in accepted))


def read_png(path):
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


class RecorderTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory(prefix='httm_debug_images_test_')
        self.addCleanup(self.folder.cleanup)
        self.parent = Path(self.folder.name) / 'ảnh điện thoại'

    def recorder(self):
        recorder = DebugFrameRecorder(self.parent, model=DEFAULT_MODEL_PATH,
                                      thresholds=THRESHOLDS, classes=CLASSES)
        self.addCleanup(recorder.close)
        return recorder

    def test_raw_png_is_exact_and_candidate_colors_labels_and_coordinates_are_preserved(self):
        recorder = self.recorder()
        passed = candidate(.8, bbox=(10., 10., 40., 40.))
        filtered = candidate(.2)
        source = FRAME.copy()
        with patch.object(cv2, 'putText', wraps=cv2.putText) as text:
            recorder.record(source, meta(), [passed, filtered], result(12, [passed]))
        entry = json.loads((recorder.run_dir / 'frames.jsonl').read_text(encoding='utf-8'))
        original = read_png(recorder.run_dir / entry['raw_image'])
        annotated = read_png(recorder.run_dir / entry['candidate_image'])
        np.testing.assert_array_equal(original, FRAME)
        np.testing.assert_array_equal(source, FRAME)
        np.testing.assert_array_equal(annotated[10, 10], ACCEPTED_COLOR)
        np.testing.assert_array_equal(annotated[20, 80], FILTERED_COLOR)
        np.testing.assert_array_equal(annotated[200, 300], FRAME[200, 300])
        self.assertGreater(annotated.shape[0], FRAME.shape[0])
        strings = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('frame_index=12' in line and 'timestamp_ms=12359' in line for line in strings))
        self.assertTrue(any('using_phone conf=0.8000 ACCEPTED' in line for line in strings))
        self.assertTrue(any('using_phone conf=0.2000 below_class_threshold' in line for line in strings))
        self.assertTrue(any('30.0x40.0px' in line for line in strings))
        self.assertEqual(entry['candidates'], [passed, filtered])
        self.assertEqual(entry['inference'], result(12, [passed]).to_dict())

    def test_summary_distinguishes_frames_candidates_classes_and_errors(self):
        recorder = self.recorder()
        samples = [([], result(0)),
                   ([candidate(.2), candidate(.3)], result(1)),
                   ([candidate(.2), candidate(.8)], result(2, [candidate(.8)])),
                   ([candidate(.8, 'sleeping')], result(3, [candidate(.8, 'sleeping')])),
                   (None, FrameResult('inference_error', meta(4), error='fixture failure'))]
        for index, (candidates, inference) in enumerate(samples):
            recorder.record(FRAME, meta(index), candidates, inference)
        report_path = recorder.finish('completed', {'ai_target_fps': 1, 'video_fps': 30})
        summary = json.loads(report_path.read_text(encoding='utf-8'))
        counts = summary['counts']
        self.assertEqual(counts['processed_frames'], 5)
        self.assertEqual(counts['successful_frames'], 4)
        self.assertEqual(counts['inference_error_frames'], 1)
        self.assertEqual(counts['frames_without_yolo_candidates'], 1)
        self.assertEqual(counts['frames_without_accepted_detections'], 2)
        self.assertEqual(counts['frames_with_accepted_detections'], 2)
        self.assertEqual(counts['frames_with_low_confidence_candidates'], 2)
        self.assertEqual(counts['low_confidence_candidates'], 3)
        self.assertEqual(counts['accepted_detections'], 2)
        self.assertEqual(counts['yolo_candidates'], 5)
        phone = summary['per_class']['using_phone']
        self.assertEqual(phone['low_confidence_candidates'], 3)
        self.assertEqual(phone['accepted_detections'], 1)
        self.assertEqual(phone['bbox_width_px'], {'min': 30., 'max': 30., 'mean': 30.})
        self.assertEqual(summary['thresholds'], THRESHOLDS)
        self.assertEqual(summary['status'], 'completed')
        self.assertEqual(len(list((recorder.run_dir / 'raw').glob('*.png'))), 5)
        self.assertEqual(len(list((recorder.run_dir / 'candidates').glob('*.png'))), 5)

    def test_missing_candidates_are_not_reported_as_an_empty_yolo_result(self):
        recorder = self.recorder()
        recorder.record(FRAME, meta(), None, result(12))
        self.assertEqual(recorder.summary()['counts']['candidate_data_missing_frames'], 1)
        self.assertEqual(recorder.summary()['counts']['frames_without_yolo_candidates'], 0)

    def test_unique_sessions_and_sample_names_do_not_overwrite_prior_images(self):
        first, second = self.recorder(), self.recorder()
        self.assertNotEqual(first.run_dir, second.run_dir)
        for _ in range(2): first.record(FRAME, meta(), [], result(12))
        self.assertEqual(len(list((first.run_dir / 'raw').glob('*.png'))), 2)
        self.assertEqual(second.summary()['counts']['processed_frames'], 0)

    def test_image_encode_failures_are_visible(self):
        recorder = self.recorder()
        with patch.object(cv2, 'imencode', return_value=(False, None)):
            with self.assertRaisesRegex(OSError, 'Could not encode debug image'):
                recorder.record(FRAME, meta(), [], result(12))
        self.assertEqual(recorder.summary()['counts']['processed_frames'], 0)

    def test_adapter_records_all_candidates_without_changing_events_and_resets_failed_calls(self):
        box = SimpleNamespace(cls=np.array([1]), conf=np.array([.2]),
                              xyxy=np.array([[80., 20., 110., 60.]]))
        model = SimpleNamespace(names=CLASSES, predictor=None, predict=Mock(side_effect=[
            [SimpleNamespace(boxes=[box], orig_shape=(240, 400), speed={})],
            ValueError('fixture inference failed')]))
        recorder = self.recorder()
        with patch('ai.behavior_detection.behavior_detector.YOLO', return_value=model):
            detector = BehaviorDetector(DEFAULT_MODEL_PATH, debug_detections=True)
        adapter = DetectorAdapter(detector, debug_detections=True, debug_frame_recorder=recorder)
        with redirect_stderr(io.StringIO()):
            first = adapter.process(FRAME, meta())
            second = adapter.process(FRAME, meta(13))
        entries = [json.loads(line) for line in (recorder.run_dir / 'frames.jsonl').read_text().splitlines()]
        self.assertEqual(first.status, 'ok')
        self.assertEqual(first.detections, ())
        self.assertEqual(entries[0]['candidates'][0]['confidence'], .2)
        self.assertEqual(second.status, 'inference_error')
        self.assertIsNone(entries[1]['candidates'])
        self.assertIsNone(detector.last_debug_candidates)
        self.assertEqual(detector.thresholds, THRESHOLDS)


class SaveModeTests(unittest.TestCase):
    def test_save_flag_requires_debug_before_loading_a_model_or_creating_files(self):
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint') as load, \
                patch.object(run_video, 'DebugFrameRecorder') as recorder, \
                patch('sys.stderr', new_callable=io.StringIO):
            with self.assertRaises(SystemExit):
                run_video.main(['--source', '0', '--save-debug-frames'])
        load.assert_not_called()
        recorder.assert_not_called()

    def test_text_debug_alone_does_not_create_image_files(self):
        adapter = Mock()
        source = Mock()
        source.is_live = True
        source.fps = 30
        source.__enter__ = Mock(return_value=source)
        source.__exit__ = Mock(return_value=False)
        source.read.return_value = None
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter), \
                patch.object(run_video, 'VideoSource', return_value=source), \
                patch.object(run_video, 'DebugFrameRecorder') as recorder, redirect_stdout(io.StringIO()):
            run_video.main(['--source', '0', '--debug-detections', '--no-display'])
        recorder.assert_not_called()

    def test_slow_image_writes_run_in_ai_thread_and_do_not_stall_video(self):
        with tempfile.TemporaryDirectory(prefix='httm_image_io_test_') as folder:
            root = Path(folder)
            source = root / 'fixture.avi'
            writer = cv2.VideoWriter(str(source), cv2.VideoWriter_fourcc(*'MJPG'), 30, (400, 240))
            self.assertTrue(writer.isOpened())
            for _ in range(18): writer.write(FRAME)
            writer.release()
            detector = SimpleNamespace(
                model_path=DEFAULT_MODEL_PATH, thresholds=THRESHOLDS, class_names=CLASSES,
                last_debug_candidates=[candidate(.2)], detect=lambda _: [])
            adapter = DetectorAdapter(detector, debug_detections=True)
            original_save = DebugFrameRecorder._save_png
            save_threads, display_times = [], []
            def slow_save(path, frame):
                save_threads.append(threading.get_ident())
                time.sleep(.45)
                original_save(path, frame)
            with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter), \
                    patch.object(DebugFrameRecorder, '_save_png', side_effect=slow_save), \
                    patch.object(cv2, 'namedWindow'), patch.object(cv2, 'resizeWindow'), \
                    patch.object(cv2, 'destroyAllWindows'), \
                    patch.object(cv2, 'imshow', side_effect=lambda *_: display_times.append(time.monotonic())), \
                    patch.object(cv2, 'waitKey', return_value=-1), \
                    redirect_stdout(io.StringIO()) as stdout, redirect_stderr(io.StringIO()):
                run_video.main(['--source', str(source), '--debug-detections', '--save-debug-frames',
                                '--debug-frames-dir', str(root / 'debug')])
            runtime = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(len(display_times), 18)
            self.assertGreater(runtime['video_fps'], 24)
            self.assertLess(display_times[-1] - display_times[0], .80)
            self.assertEqual(len(save_threads), 2)
            self.assertNotIn(threading.get_ident(), save_threads)
            summary = json.loads(Path(runtime['debug_report']).read_text(encoding='utf-8'))
            self.assertEqual(summary['counts']['processed_frames'], 1)
            self.assertEqual(summary['counts']['low_confidence_candidates'], 1)
            self.assertEqual(summary['runtime']['ai_target_fps'], 1)
            self.assertFalse(any(t.name == 'HTTM-AI' for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
