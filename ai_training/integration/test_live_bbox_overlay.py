"""Synthetic live-overlay and pacing tests; no camera, models or network."""

import io
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import cv2
import numpy as np

from behavior_duration import BehaviorDurationTracker
from live_bbox_overlay import (BBOX_MAX_CAPTURE_AGE, HEADER_HEIGHT, FrameRateMeter,
                               InferencePacer, draw_live_bboxes)


def result_at(captured_at=10.0, sid="SYNTHETIC_001"):
    people = [{"person_ref": "person_0", "behaviors": {
        "sleeping": {"state": "detected"}, "using_phone": {"state": "suspected"}}}]
    tracks = [{"person_ref": "person_0", "track_id": "P1", "bbox": [20, 30, 120, 200]}]
    matches = [{"person_ref": "person_0", "person_track_id": "P1", "track_id": 1,
                "student_id": sid, "association_status": "matched_v4_heuristic"}]
    view = BehaviorDurationTracker().update(1, captured_at, people, tracks, matches)
    return {"captured_at": captured_at, "completed_at": captured_at + .1,
            "source_size": (320, 240), "behavior_frame": view}


class TestLiveBBoxes(unittest.TestCase):
    def test_fresh_result_draws_boxes_on_live_pixels_without_mutating_inputs(self):
        frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        result = result_at()
        observation = result["behavior_frame"]
        with (patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle,
              patch.object(cv2, "putText", wraps=cv2.putText) as text):
            output = draw_live_bboxes(frame, result, now=10.2, view_width=320)
        rectangle.assert_called_once()
        self.assertEqual(rectangle.call_args.args[1:3], ((20, 30), (120, 200)))
        labels = [call.args[1] for call in text.call_args_list]
        self.assertIn("P1", labels)
        self.assertFalse(any("SYNTHETIC_001" in label for label in labels))
        self.assertIn("Sleeping: detected ~0.0s", labels)
        self.assertIn("Using Phone: suspected", labels)
        self.assertFalse(any("not_detected" in label or ": unknown" in label for label in labels))
        self.assertEqual(output.shape, (240 + HEADER_HEIGHT, 320, 3))
        np.testing.assert_array_equal(frame, np.full_like(frame, 90))
        np.testing.assert_array_equal(output[HEADER_HEIGHT + 220:, :], frame[220:, :])
        self.assertIs(result["behavior_frame"], observation)

    def test_device_header_distinguishes_verified_gpu_cpu_and_loading(self):
        for backend, expected in (("cuda", "YOLO: GPU / CUDA"), ("cpu", "YOLO: CPU"),
                                  ("unknown", "YOLO: checking actual backend")):
            result = result_at()
            result["face_runtime"] = {"backend": "cpu"}
            result["inference_runtime"] = {"backend": backend, "device_name": "Synthetic GPU",
                                            "models": [{"precision": "float32"}] * 2}
            with self.subTest(backend=backend), patch.object(cv2, "putText", wraps=cv2.putText) as text:
                draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.2)
            labels = [call.args[1] for call in text.call_args_list]
            self.assertTrue(any(expected in label and "Face: CPU | MediaPipe: CPU" in label for label in labels))

    def test_face_gpu_status_is_reported_separately_from_mediapipe_cpu(self):
        result = result_at()
        result["face_runtime"] = {"backend": "cuda"}
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.2)
        self.assertTrue(any("Face: GPU / CUDA | MediaPipe: CPU" in c.args[1] for c in text.call_args_list))

    def test_old_capture_is_hidden_even_when_just_completed(self):
        result = result_at()
        result["completed_at"] = 21.0
        frame = np.full((240, 320, 3), 90, dtype=np.uint8)
        with patch.object(cv2, "rectangle") as rectangle:
            output = draw_live_bboxes(frame, result, now=21.01, view_width=320)
        rectangle.assert_not_called()
        np.testing.assert_array_equal(output[HEADER_HEIGHT:], frame)

    def test_slow_cpu_result_is_visible_but_explicitly_delayed(self):
        result = result_at()
        result["completed_at"] = 12.44
        with (patch.object(cv2, "rectangle", wraps=cv2.rectangle) as boxes,
              patch.object(cv2, "putText", wraps=cv2.putText) as text):
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=12.5)
        boxes.assert_called_once()
        labels = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any("DELAYED AI BBoxes" in label for label in labels))
        self.assertIn("P1", labels)
        self.assertFalse(any("DELAYED 2.5s" in label for label in labels))

    def test_stalled_result_hidden_before_hard_capture_limit(self):
        result = result_at()
        with patch.object(cv2, "rectangle") as boxes:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=16.)
        boxes.assert_not_called()

    def test_invalid_completion_timestamps_are_hidden(self):
        for completed in (float("nan"), 9., 11., None):
            result = {**result_at(), "completed_at": completed}
            with self.subTest(completed=completed), patch.object(cv2, "rectangle") as boxes:
                draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.5)
            boxes.assert_not_called()

    def test_no_person_status_is_distinct_from_stale(self):
        result = result_at()
        result["behavior_frame"] = BehaviorDurationTracker().update(2, 10.1, [], [], [])
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.2)
        self.assertTrue(any("No valid person boxes" in call.args[1] for call in text.call_args_list))

    def test_waiting_future_invalid_and_expired_results_have_no_boxes(self):
        for result, now in ((None, 10), (result_at(), 9),
                            (result_at(), 10 + BBOX_MAX_CAPTURE_AGE + .01),
                            ({**result_at(), "captured_at": float("nan")}, 10)):
            with self.subTest(now=now), patch.object(cv2, "rectangle") as rectangle:
                draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=now)
            rectangle.assert_not_called()

    def test_current_empty_observation_clears_previous_boxes(self):
        result = result_at()
        result["behavior_frame"] = BehaviorDurationTracker().update(2, 10.1, [], [], [])
        with patch.object(cv2, "rectangle") as rectangle:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.2)
        rectangle.assert_not_called()

    def test_proportional_resolution_change_scales_coordinates(self):
        with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle:
            draw_live_bboxes(np.zeros((480, 640, 3), dtype=np.uint8), result_at(), now=10.2)
        self.assertEqual(rectangle.call_args.args[1:3], ((40, 60), (240, 400)))

    def test_aspect_ratio_change_invalidates_old_geometry(self):
        with patch.object(cv2, "rectangle") as rectangle:
            draw_live_bboxes(np.zeros((240, 640, 3), dtype=np.uint8), result_at(), now=10.2)
        rectangle.assert_not_called()

    def test_unknown_identity_does_not_borrow_a_student_id(self):
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result_at(sid="Unknown"),
                             now=10.2, view_width=640)
        labels = [call.args[1] for call in text.call_args_list]
        self.assertIn("P1", labels)
        self.assertFalse(any("Unknown" in label for label in labels))
        self.assertIn("Sleeping: detected", labels)

    def test_no_signs_shows_only_pid_and_clears_previous_behavior_labels(self):
        from dataclasses import replace
        result = result_at()
        active = result["behavior_frame"].people[0]
        quiet = replace(active, behaviors=tuple(replace(b, state="not_detected" if i == 0 else "unknown")
                                                for i, b in enumerate(active.behaviors)))
        result["behavior_frame"] = replace(result["behavior_frame"], people=(quiet,))
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=10.2)
        labels = [c.args[1] for c in text.call_args_list if c.args[4] == .28]
        self.assertEqual(set(labels), {"P1"})

    def test_redrawing_does_not_increment_duration(self):
        result = result_at()
        original = result["behavior_frame"]
        for now in (10.1, 10.2, 10.4, 11.0):
            draw_live_bboxes(np.zeros((240, 320, 3), dtype=np.uint8), result, now=now)
        self.assertIs(result["behavior_frame"], original)
        self.assertEqual(original.people[0].behaviors[0].estimated_seconds, 0.)


class TestLiveDisplayIntegration(unittest.TestCase):
    def test_main_keeps_live_pixels_changing_between_ai_results_and_hides_old_boxes(self):
        import camera_coordinator_live as live

        result, clock = result_at(), [10.2]
        result.update(frame_id=1, face_count=1, person_count=1, matches=[],
                      face_time=.04, hybrid_time=.06, elapsed=.1)
        pixels = [np.full((240, 320, 3), value, dtype=np.uint8) for value in (40, 80, 120)]
        camera = Mock()
        camera.isOpened.return_value = True
        tick = iter(zip((10.2, 10.4, 16.0), pixels))

        def read():
            clock[0], frame = next(tick)
            return True, frame

        camera.read.side_effect = read

        class FakeThread:
            def __init__(self, target, args, daemon):
                self.results = args[1]

            def start(self):
                self.results.put(result)  # Only ONE AI result for three live frames.

            def is_alive(self):
                return False

        with (patch.object(live.Path, "is_file", return_value=True),
              patch.object(live, "overlay_from_environment", return_value=None),
              patch.object(live, "observation_limit_from_environment", return_value=None),
              patch.object(live, "face_tracker_max_age_from_environment", return_value=2.),
              patch.object(live.cv2, "VideoCapture", return_value=camera),
              patch.object(live.threading, "Thread", FakeThread),
              patch.object(live.time, "perf_counter", side_effect=lambda: clock[0]),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "destroyAllWindows"),
              patch.object(live.cv2, "imshow") as show,
              patch.object(live.cv2, "rectangle", wraps=cv2.rectangle) as boxes,
              patch.object(live.cv2, "waitKey", side_effect=[-1, -1, ord("q")]),
              redirect_stdout(io.StringIO())):
            live.main()
        self.assertEqual(show.call_count, 3)
        self.assertEqual(boxes.call_count, 2, "Third live frame must hide the expired result")
        for call, value in zip(show.call_args_list, (40, 80, 120)):
            image = call.args[1]
            np.testing.assert_array_equal(image[-10:, :], np.full_like(image[-10:, :], value))
        camera.release.assert_called_once()


class TestPacingAndRates(unittest.TestCase):
    def test_fast_worker_waits_until_next_slot(self):
        pacer, stop = InferencePacer(5), Mock()
        stop.is_set.return_value = False
        stop.wait.return_value = False
        self.assertFalse(pacer.wait(stop, 10))
        pacer.started(10)
        self.assertFalse(pacer.wait(stop, 10.05))
        self.assertAlmostEqual(stop.wait.call_args.args[0], .15)

    def test_slow_worker_does_not_wait_or_catch_up(self):
        pacer, stop = InferencePacer(5), Mock()
        stop.is_set.return_value = False
        pacer.started(10)
        self.assertFalse(pacer.wait(stop, 11))
        stop.wait.assert_not_called()
        pacer.started(11)
        self.assertAlmostEqual(pacer.next_start, 11.2)

    def test_stop_interrupts_pacing(self):
        pacer, stop = InferencePacer(), threading.Event()
        pacer.started(10)
        stop.set()
        self.assertTrue(pacer.wait(stop, 10.01))

    def test_rate_counts_actual_completions_and_decays_when_idle(self):
        rate = FrameRateMeter()
        for timestamp in (10., 10.2, 10.4, 10.6):
            rate.mark(timestamp)
        self.assertAlmostEqual(rate.fps(10.6), 5.)
        self.assertLess(rate.fps(11), 5.)
        self.assertEqual(rate.fps(14), 0.)
        for index in range(1000):
            rate.mark(20. + index * .001)
        self.assertLessEqual(len(rate.timestamps), 120)


if __name__ == "__main__":
    unittest.main()
