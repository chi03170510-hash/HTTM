"""Two-pane UI regressions; synthetic frames, no real camera, GUI, or models."""

import io
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import cv2
import numpy as np

import camera_coordinator_live as live
from behavior_duration import BehaviorDurationTracker
from behavior_overlay import AISnapshot


def ui_result(student="SYNTHETIC_001", count=1):
    people, tracks, matches = [], [], []
    for index in range(count):
        ref, tid = f"person_{index}", f"P{index + 1}"
        people.append({"person_ref": ref, "behaviors": {
            "using_phone": {"state": "detected"}, "sleeping": {"state": "suspected"},
            "turning_back": {"state": "unknown"}, "eating": {"state": "not_detected"},
        }})
        tracks.append({"track_id": tid, "person_ref": ref, "bbox": [20, 10, 120, 220]})
        matches.append({"person_track_id": tid, "person_ref": ref, "student_id": f"{student}_{index + 1}",
                        "track_id": index + 2, "association_status": "matched_v4_heuristic"})
    timer = BehaviorDurationTracker()
    timer.update(1, 10.0, people, tracks, matches)
    observations = timer.update(2, 13.8, people, tracks, matches)
    return {
        "frame_id": 2, "captured_at": 13.8, "completed_at": 17.5, "elapsed": 3.7,
        "face_count": count, "person_count": count, "behavior_frame": observations,
        "tracked_faces": [(index + 2, (), "Unknown", .82) for index in range(count)],
        "identity_details": {}, "matches": matches, "face_time": .5, "hybrid_time": 3.2,
        "snapshot": AISnapshot(2, 13.8, observations, np.full((480, 640, 3), (200, 0, 200), dtype=np.uint8)),
    }


class TestCameraLiveLayout(unittest.TestCase):
    def panel_text(self, frame, result, **kwargs):
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text:
            output = live.draw_ai_panel(frame, result, now=20.0, **kwargs)
        panel_calls = [call for call in put_text.call_args_list if call.args[0].shape[1] == 384]
        return output, [call.args[1] for call in panel_calls], panel_calls

    def test_two_panes_are_seventy_thirty_and_four_three_video_fills_its_area(self):
        source = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
        before = source.copy()
        output = live.draw_ai_panel(source, ui_result(), now=20.0)
        self.assertEqual(output.shape, (720, 1280, 3))
        expected = cv2.resize(source, (896, 672), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(output[48:720, :896], expected)
        np.testing.assert_array_equal(source, before)

    def test_widescreen_video_has_no_old_850_pixel_padding(self):
        source = np.full((720, 1280, 3), 95, dtype=np.uint8)
        output = live.draw_ai_panel(source, None)
        self.assertEqual(output.shape, (552, 1280, 3))
        np.testing.assert_array_equal(output[48:, :896], np.full((504, 896, 3), 95, dtype=np.uint8))

    def test_portrait_video_is_resized_proportionally(self):
        source = np.arange(640 * 480 * 3, dtype=np.uint8).reshape(640, 480, 3)
        output = live.draw_ai_panel(source, ui_result(), now=20.0)
        self.assertEqual(output.shape, (1243, 1280, 3))
        expected = cv2.resize(source, (896, 1195), interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(output[48:, :896], expected)

    def test_panel_contains_required_fields_and_all_four_states_without_clipping(self):
        output, labels, calls = self.panel_text(np.zeros((480, 640, 3), dtype=np.uint8), ui_result())
        for expected in ("Frame: 2", "Result age: 6.2s", "Processing: 3.70s",
                         "Face Track: T2", "Student: SYNTHETIC_001_1", "Person Track: P1",
                         "Association: matched_v4_heuristic", "Using Phone: detected ~3.8s",
                         "Sleeping: suspected --", "Turning Back: unknown --", "Eating: not_detected --"):
            self.assertIn(expected, labels)
        for call in calls:
            text_size, baseline = cv2.getTextSize(call.args[1], call.args[3], call.args[4], call.args[6])
            x, y = call.args[2]
            self.assertGreaterEqual(x, 0)
            self.assertLessEqual(x + text_size[0], 384)
            self.assertLessEqual(y + baseline, output.shape[0])
        self.assertFalse(any("SNAPSHOT" in label for label in labels))

    def test_long_identity_wraps_without_losing_content(self):
        identity = "SYNTHETIC_" + "A" * 160
        _, labels, calls = self.panel_text(np.zeros((360, 640, 3), dtype=np.uint8), ui_result(identity))
        self.assertIn(identity + "_1", "".join(labels))
        for call in calls:
            text_width = cv2.getTextSize(call.args[1], call.args[3], call.args[4], call.args[6])[0][0]
            self.assertLessEqual(call.args[2][0] + text_width, 384)

    def test_waiting_and_stale_are_explicit(self):
        source = np.zeros((480, 640, 3), dtype=np.uint8)
        _, waiting, _ = self.panel_text(source, None)
        self.assertIn("Waiting for AI...", waiting)
        result = ui_result()
        result["completed_at"] = 10.0
        _, stale, _ = self.panel_text(source, result)
        self.assertTrue(any("STALE" in label for label in stale))
        self.assertTrue(any("Historical" in label for label in stale))

    def test_rendering_has_no_boxes_snapshot_work_or_duration_side_effects(self):
        result = ui_result()
        observation = result["behavior_frame"]
        pixels = result["snapshot"].image.copy()
        with (patch.object(live.cv2, "rectangle", side_effect=AssertionError("no historical boxes")),
              patch.object(live, "render_snapshot", side_effect=AssertionError("snapshot hidden"), create=True),
              patch("sqlite3.connect", side_effect=AssertionError("database forbidden")),
              patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))):
            for now in (20.0, 30.0, 50.0):
                with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text:
                    output = live.draw_ai_panel(np.zeros((480, 640, 3), dtype=np.uint8), result, now=now)
                labels = [call.args[1] for call in put_text.call_args_list]
                self.assertIn("Using Phone: detected ~3.8s", labels)
                self.assertFalse(any("WARNING" in label for label in labels))
                self.assertEqual(output.shape[1], 1280)
        self.assertIs(result["behavior_frame"], observation)
        np.testing.assert_array_equal(result["snapshot"].image, pixels)

    def test_main_keeps_n_p_q_and_does_not_render_snapshot(self):
        result = ui_result(count=3)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, np.full((480, 640, 3), 100, dtype=np.uint8)) for _ in range(4)]
        stops = []

        class FakeThread:
            def __init__(self, target, args, daemon):
                self.results = args[1]
                stops.append(args[2])

            def start(self):
                self.results.put(result)

            def is_alive(self):
                return False

        with (patch.object(live.cv2, "VideoCapture", return_value=cap),
              patch.object(live.threading, "Thread", FakeThread),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "imshow") as show,
              patch.object(live.cv2, "waitKey", side_effect=[ord("i"), ord("n"), ord("p"), ord("q")]),
              patch.object(live.cv2, "destroyAllWindows"),
              patch.object(live.Path, "is_file", return_value=True),
              patch.object(live, "draw_live_bboxes", wraps=live.draw_live_bboxes) as boxes,
              patch.object(live, "draw_ai_panel", wraps=live.draw_ai_panel) as draw,
              patch.object(live, "render_snapshot", side_effect=AssertionError("snapshot hidden"), create=True),
              redirect_stdout(io.StringIO())):
            live.main()
        self.assertEqual([call.kwargs["page"] for call in draw.call_args_list], [0, 1, 0])
        self.assertEqual(show.call_count, 4)
        boxes.assert_called_once()
        self.assertTrue(stops[0].is_set())
        cap.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
