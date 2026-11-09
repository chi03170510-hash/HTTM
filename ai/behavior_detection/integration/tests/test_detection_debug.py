"""Debug reports must explain filtering without changing official detections."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from ultralytics.engine.predictor import BasePredictor
from ultralytics.models.yolo.detect.predict import DetectionPredictor

from ai.behavior_detection.behavior_detector import BehaviorDetector, DEFAULT_MODEL_PATH, EXPECTED_CLASSES
from ai.behavior_detection.integration.detector_adapter import DetectorAdapter
from ai.behavior_detection.integration.event_schema import FrameMetadata, FrameResult
from ai.behavior_detection.integration.realtime import LatestFrameWorker
from ai.behavior_detection.integration.video_source import FramePacket
from ai.behavior_detection.integration import run_video


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
META = FrameMetadata('webcam-fixture', 12, 359, 640, 480)


def box(class_id, confidence, bbox=(10.5, 20.5, 110.5, 220.5)):
    return SimpleNamespace(cls=np.array([class_id]), conf=np.array([confidence]), xyxy=np.array([bbox]))


def create_detector(boxes, debug=True):
    model = SimpleNamespace(names=EXPECTED_CLASSES.copy(), predictor=None)
    result = SimpleNamespace(boxes=boxes, orig_shape=(480, 640),
                             speed={'preprocess': .1, 'inference': .2, 'postprocess': .3})
    model.predict = Mock(return_value=[result])
    with patch('ai.behavior_detection.behavior_detector.YOLO', return_value=model):
        detector = BehaviorDetector(DEFAULT_MODEL_PATH, debug_detections=debug, device='cpu')
    return detector


def records(stream):
    return [json.loads(line) for line in stream.getvalue().splitlines()]


def await_worker(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError('worker timed out')
        time.sleep(.002)


class DetectorDebugTests(unittest.TestCase):
    def test_raw_phone_candidates_and_all_filter_decisions_are_reported(self):
        detector = create_detector([box(1, .15), box(1, .399), box(1, .40), box(1, .75), box(7, .8)])
        adapter = DetectorAdapter(detector, debug_detections=True)
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = adapter.process(FRAME, META)
        self.assertEqual([d.confidence for d in result.detections], [.4, .75])
        raw = records(stderr)[0]
        self.assertEqual(raw['event'], 'yolo_candidates')
        self.assertEqual(raw['metadata'], META.__dict__)
        self.assertEqual([c['decision'] for c in raw['candidates']],
                         ['below_class_threshold', 'below_class_threshold', 'accepted',
                          'accepted', 'unsupported_class_id'])
        self.assertEqual(raw['candidates'][0]['class_name'], 'using_phone')
        self.assertEqual(raw['candidates'][0]['confidence'], .15)
        self.assertEqual(raw['candidates'][0]['bbox_xyxy'], [10.5, 20.5, 110.5, 220.5])
        self.assertTrue(raw['candidates'][0]['bbox_in_source_bounds'])
        self.assertEqual(detector.thresholds, {'sleeping': .4, 'using_phone': .4})
        self.assertEqual(detector.model.predict.call_args.kwargs['conf'], .10)

    def test_no_candidates_is_distinct_from_candidates_below_threshold(self):
        for boxes, reason in [(None, 'no_yolo_candidates_at_conf_0.10'),
                              ([], 'no_yolo_candidates_at_conf_0.10'),
                              ([box(1, .25)], 'all_candidates_filtered')]:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                output = create_detector(boxes).detect(FRAME)
            self.assertEqual(output, [])
            self.assertEqual(records(stderr)[0]['empty_reason'], reason)

    def test_debug_does_not_change_events_or_legacy_detection_schema(self):
        outputs = []
        for debug in [False, True]:
            detector = create_detector([box(0, .8), box(1, .3), box(1, .9)], debug)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                result = DetectorAdapter(detector, debug_detections=debug).process(FRAME, META)
            outputs.append(result.to_dict())
            self.assertEqual(bool(stderr.getvalue()), debug)
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(set(outputs[0]), {'status', 'metadata', 'detections', 'error'})

    def test_worker_and_adapter_preserve_original_unannotated_bgr_pixels(self):
        original = FRAME.copy()
        original[:, :, 0] = 255  # OpenCV blue, not RGB red.
        detector = create_detector([], debug=True)
        seen = []
        detector.model.predict.side_effect = lambda **kwargs: (
            seen.append(kwargs['source'].copy()) or
            [SimpleNamespace(boxes=[], orig_shape=original.shape[:2], speed={})])
        worker = LatestFrameWorker(DetectorAdapter(detector, debug_detections=True).process)
        with redirect_stderr(io.StringIO()):
            worker.start()
            try:
                worker.publish(FramePacket(original, META))
                # Mutating the display/source after publish must not alter YOLO input.
                original[:] = (0, 255, 0)
                await_worker(lambda: worker.latest() is not None)
            finally:
                worker.stop()
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0].shape, (480, 640, 3))
        self.assertEqual(seen[0].dtype, np.uint8)
        np.testing.assert_array_equal(seen[0][0, 0], [255, 0, 0])

    def test_real_ultralytics_preprocessing_converts_bgr_once_and_normalizes(self):
        original = FRAME.copy()
        original[:, :, 0] = 255
        predictor = SimpleNamespace(
            device=torch.device('cpu'), model=SimpleNamespace(fp16=False),
            pre_transform=lambda images: images,
        )
        tensor = BasePredictor.preprocess(predictor, [original])
        self.assertEqual(tuple(tensor.shape), (1, 3, 480, 640))
        torch.testing.assert_close(tensor[0, :, 0, 0], torch.tensor([0., 0., 1.]))
        self.assertTrue(tensor.is_contiguous())

    def test_real_ultralytics_letterbox_preserves_aspect_and_restores_boxes(self):
        original = np.zeros((240, 400, 3), dtype=np.uint8)
        predictor = SimpleNamespace(imgsz=[640, 640], args=SimpleNamespace(rect=True),
                                    model=SimpleNamespace(format='pt', stride=32))
        transformed = BasePredictor.pre_transform(predictor, [original])[0]
        self.assertEqual(transformed.shape, (384, 640, 3))
        image_tensor = torch.zeros((1, 3, 384, 640))
        predictions = torch.tensor([[16., 32., 160., 96., .8, 1.]])
        result = DetectionPredictor.construct_result(
            SimpleNamespace(model=SimpleNamespace(names=EXPECTED_CLASSES)),
            predictions, image_tensor, original, 'fixture')
        torch.testing.assert_close(result.boxes.xyxy[0], torch.tensor([10., 20., 100., 60.]))

    def test_invalid_dtype_is_reported_as_inference_error(self):
        detector = create_detector([])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = DetectorAdapter(detector, debug_detections=True).process(FRAME.astype(np.float32), META)
        self.assertEqual(result.status, 'inference_error')
        self.assertIn('uint8', result.error)
        detector.model.predict.assert_not_called()

    def test_adapter_explains_class_id_and_name_rejections(self):
        detector = SimpleNamespace(detect=lambda _: [
            {'class_id': 3, 'class': 'other', 'confidence': .8, 'bbox': [1, 2, 3, 4]},
            {'class_id': 1, 'class': 'sleeping', 'confidence': .8, 'bbox': [1, 2, 3, 4]},
        ])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = DetectorAdapter(detector, debug_detections=True).process(FRAME, META)
        self.assertEqual(result.detections, ())
        self.assertEqual([d['reason'] for d in records(stderr)[0]['filtered']],
                         ['unsupported_class_id', 'class_name_id_mismatch'])

    def test_warmup_uses_same_yolo_settings_but_emits_no_detection_events(self):
        detector = create_detector([box(1, .8)])
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            elapsed = DetectorAdapter(detector).warmup()
        self.assertGreaterEqual(elapsed, 0)
        self.assertEqual([r['event'] for r in records(stderr)], ['yolo_warmup_start', 'yolo_warmup'])
        options = detector.model.predict.call_args.kwargs
        self.assertEqual(options['source'].shape, (480, 640, 3))
        self.assertEqual(options['source'].dtype, np.uint8)
        self.assertEqual((options['imgsz'], options['conf'], options['iou'], options['device']),
                         (640, .1, .7, 'cpu'))
        self.assertEqual(detector.thresholds['using_phone'], .4)

    def test_cli_warms_up_before_capture_and_passes_debug_flag(self):
        adapter = Mock()
        adapter.warmup.return_value = 123
        operations = []
        adapter.warmup.side_effect = lambda: operations.append('warmup') or 123
        source = Mock()
        source.is_live = True
        source.fps = 30
        source.__enter__ = Mock(side_effect=lambda: operations.append('capture') or source)
        source.__exit__ = Mock(return_value=False)
        source.read.return_value = None
        with patch.object(run_video.DetectorAdapter, 'from_default_checkpoint', return_value=adapter) as load, \
                patch.object(run_video, 'VideoSource', return_value=source), redirect_stdout(io.StringIO()) as stdout:
            run_video.main(['--source', '0', '--debug-detections', '--warmup', '--no-display'])
        self.assertEqual(operations, ['warmup', 'capture'])
        load.assert_called_once_with(run_video.DEFAULT_CHECKPOINT, device='cpu', debug_detections=True)
        summary = json.loads(stdout.getvalue().splitlines()[-1])
        self.assertEqual(summary['processed_frames'], 0)
        self.assertEqual(summary['warmup_ms'], 123)


class SamplingDebugTests(unittest.TestCase):
    def test_brief_phone_pose_between_one_fps_samples_can_be_missed(self):
        observed = []
        def process(frame, meta):
            observed.append(int(frame[0, 0, 0]))
            return FrameResult('ok', meta)
        worker = LatestFrameWorker(process).start()
        try:
            worker.publish(FramePacket(FRAME, META))
            await_worker(lambda: len(observed) == 1)
            worker.publish(FramePacket(np.full_like(FRAME, 255),
                                       FrameMetadata('webcam-fixture', 13, 600, 640, 480)))
            worker.publish(FramePacket(FRAME, FrameMetadata('webcam-fixture', 14, 900, 640, 480)))
            await_worker(lambda: len(observed) == 2)
            self.assertEqual(observed, [0, 0])
        finally:
            worker.stop()

    def test_timing_separates_processing_from_output_and_tracks_sampling_gaps(self):
        stderr = io.StringIO()
        def process(_, meta):
            time.sleep(.02)
            return FrameResult('ok', meta)
        worker = LatestFrameWorker(process, target_fps=100, debug_detections=True)
        with redirect_stderr(stderr):
            worker.start()
            try:
                worker.publish(FramePacket(FRAME, META))
                await_worker(lambda: worker.latest() is not None)
                worker.publish(FramePacket(FRAME, FrameMetadata('webcam-fixture', 42, 1359, 640, 480)))
                await_worker(lambda: worker.stats()['processed_frames'] == 2)
            finally:
                worker.stop()
        reports = records(stderr)
        self.assertEqual(len(reports), 2)
        self.assertGreater(reports[0]['processing_ms'], 15)
        self.assertEqual(reports[1]['source_gap_ms'], 1000)
        self.assertEqual(reports[1]['skipped_source_frames'], 29)
        self.assertGreater(worker.stats()['first_inference_ms'], 15)


if __name__ == '__main__':
    unittest.main()
