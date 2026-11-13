"""Camera-free regressions using real trackers/Association V4 and fake AI workers.

No camera, GUI, model startup, real student data, or external services are used.
"""

import copy
import io
import json
import queue
import threading
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, patch

import numpy as np

import camera_coordinator_live as live
from behavior_duration import BehaviorDurationTracker
from behavior_overlay import render_snapshot


class FakeClock:
    def __init__(self, now=100.0):
        self.now = now

    def __call__(self):
        return self.now


class TestCameraPersonTracking(unittest.TestCase):
    def run_frames(self, frames, gap=1.0, updated_at=None, frame_ids=None, source_images=None,
                   captured_at=None):
        """Each frame contains synthetic (people, faces) worker responses."""
        frame_queue, result_queue = queue.Queue(), queue.Queue()
        if updated_at is None:
            updated_at = [100.0 + (index + 1) * gap for index in range(len(frames))]
        if frame_ids is None:
            frame_ids = list(range(1, len(frames) + 1))
        if captured_at is None:
            captured_at = [value - gap for value in updated_at]
        clock = FakeClock(updated_at[0] - gap)
        index_by_frame = {frame_id: index for index, frame_id in enumerate(frame_ids)}
        inferred_images = {"face": {}, "hybrid": {}}
        for index in range(len(frames)):
            image = (source_images[index] if source_images is not None
                     else np.full((480, 640, 3), index * 10, dtype=np.uint8))
            frame_queue.put((captured_at[index], image))

        def response(kind, image_path, frame_id):
            index = index_by_frame[frame_id]
            inferred_images[kind][frame_id] = live.cv2.imread(str(image_path))
            people, faces = copy.deepcopy(frames[index])
            if kind == "face":
                clock.now = max(clock.now, updated_at[index] - gap / 2)
                return {"face_count": len(faces), "faces": faces}
            clock.now = updated_at[index]
            return {"person_count": len(people), "people": people}

        face_worker, hybrid_worker = Mock(), Mock()
        face_worker.process_frame.side_effect = lambda path, fid: response("face", path, fid)
        hybrid_worker.process_frame.side_effect = lambda path, fid: response("hybrid", path, fid)
        output = io.StringIO()
        with (
            patch.object(live, "MAX_AI_FRAMES", len(frames)),
            patch.object(live, "range", return_value=iter(frame_ids), create=True),
            patch.object(live, "Worker", side_effect=[face_worker, hybrid_worker]),
            patch.object(live.time, "perf_counter", clock),
            redirect_stdout(output),
        ):
            live.ai_loop(frame_queue, result_queue, threading.Event())

        results = []
        while not result_queue.empty():
            results.append(result_queue.get_nowait())
        self.assertEqual(results[-1], {"finished": True})
        results.pop()
        self.assertFalse(any("error" in result for result in results), results)
        self.assertEqual(len(results), len(frames))
        for worker in (face_worker, hybrid_worker):
            worker.close.assert_called_once_with()
            self.assertEqual(worker.process_frame.call_count, len(frames))

        log_lines = output.getvalue().splitlines()
        self.assertEqual(len(log_lines), len(frames))
        logs = []
        for index, (line, result) in enumerate(zip(log_lines, results)):
            self.assertTrue(line.startswith("[PERSON TRACKING] "), line)
            log = json.loads(line.removeprefix("[PERSON TRACKING] "))
            self.assertEqual(log["frame_id"], frame_ids[index])
            self.assertAlmostEqual(log["updated_at"], updated_at[index])
            # The logged timestamp is the timestamp used for tracker expiry.
            self.assertEqual(log["tracked_people"], json.loads(json.dumps(result["tracked_people"])))
            logs.append(log)
            snapshot = result["snapshot"]
            self.assertEqual((snapshot.frame_id, snapshot.captured_at),
                             (result["frame_id"], result["captured_at"]))
            self.assertIs(snapshot.observations, result["behavior_frame"])
            for kind in inferred_images:
                np.testing.assert_array_equal(snapshot.image, inferred_images[kind][frame_ids[index]])
            self.assertFalse(snapshot.image.flags.writeable)
        # The temporary file is already gone; all later UI rendering uses memory.
        self.assertTrue(all(not call.args[0].exists() for call in face_worker.process_frame.call_args_list))
        return results, logs

    def test_ten_frames_propagate_person_ids_through_association(self):
        frames = []
        for index in range(10):
            ref = f"person_{index}"
            frames.append(([{
                "person_ref": ref,
                "person_bbox_xyxy_clipped": [20, 10, 120, 220],
                "behaviors": {"using_phone": {"state": "suspected"}},
            }], [{
                "face_bbox": [40, 20, 80, 70],
                "student_id": "SYNTHETIC_001", "score": 0.82,
            }]))
        results, _ = self.run_frames(frames)
        for index, result in enumerate(results):
            with self.subTest(frame=index + 1):
                self.assertEqual(result["tracked_people"][0]["track_id"], "P1")
                self.assertEqual(len(result["matches"]), 1)
                match = result["matches"][0]
                self.assertEqual(match["person_track_id"], "P1")
                self.assertEqual(match["person_ref"], f"person_{index}")
                self.assertEqual(match["student_id"], "SYNTHETIC_001")
                self.assertEqual(match["track_id"], result["tracked_faces"][0][0])
                self.assertEqual(match["behaviors"]["using_phone"]["state"], "suspected")
        self.assertEqual(results[-1]["tracked_faces"][0][2], "SYNTHETIC_001")

    def test_two_people_reordered_keep_correct_student_and_person_ids(self):
        def people(left_ref, right_ref):
            return [
                {"person_ref": left_ref, "bbox_xyxy": [20, 10, 120, 220], "behaviors": {}},
                {"person_ref": right_ref, "bbox_xyxy": [220, 10, 320, 220], "behaviors": {}},
            ]

        faces = [
            {"face_bbox": [40, 20, 80, 70], "student_id": "SYNTHETIC_001", "score": 0.82},
            {"face_bbox": [240, 20, 280, 70], "student_id": "SYNTHETIC_002", "score": 0.78},
        ]
        frames = [(people("person_0", "person_1"), faces),
                  (list(reversed(people("person_1", "person_0"))), list(reversed(faces)))]
        for gap in (1.0, 3.74):
            with self.subTest(gap=gap):
                results, _ = self.run_frames(frames, gap=gap)
                for result in results:
                    self.assertEqual({m["student_id"]: m["person_track_id"] for m in result["matches"]},
                                     {"SYNTHETIC_001": "P1", "SYNTHETIC_002": "P2"})

    def test_no_people_and_person_without_face_still_log_each_frame(self):
        results, logs = self.run_frames([
            ([], []), ([{"person_ref": "person_0", "bbox_xyxy": [0, 0, 100, 200]}], []),
            ([], []),
        ])
        self.assertEqual(logs[0]["tracked_people"], [])
        self.assertEqual(logs[1]["tracked_people"][0]["track_id"], "P1")
        self.assertEqual(logs[2]["tracked_people"], [])
        self.assertTrue(all(result["matches"] == [] for result in results))

    def test_slow_ai_processing_keeps_id_until_bounded_limit(self):
        frame = ([{"person_ref": "person_0", "bbox_xyxy": [0, 0, 100, 200]}], [])
        for gap, expected in [(3.74, "P1"), (5.0, "P1"), (5.001, "P2"), (30.0, "P2")]:
            with self.subTest(gap=gap):
                results, _ = self.run_frames([frame, frame], gap=gap)
                self.assertEqual([r["tracked_people"][0]["track_id"] for r in results], ["P1", expected])

    def test_exact_camera_frame8_to9_keeps_p2_in_live_pipeline(self):
        box8 = [1.3, 92.81, 405.04, 415.77]
        box9 = [8.61, 75.21, 377.65, 373.33]
        times = [245650.0, 245676.420614, 245680.0724245]
        boxes = [[1000, 1000, 1100, 1200], box8, box9]
        frames = [([{"person_ref": "person_0", "person_bbox_xyxy_clipped": box,
                     "behaviors": {}}], []) for box in boxes]
        results, logs = self.run_frames(frames, updated_at=times, frame_ids=[1, 8, 9])
        self.assertEqual([r["tracked_people"][0]["track_id"] for r in results], ["P1", "P2", "P2"])
        self.assertEqual(logs[1]["frame_id"], 8)
        self.assertEqual(logs[2]["frame_id"], 9)
        self.assertAlmostEqual(logs[2]["updated_at"] - logs[1]["updated_at"], 3.6518105)

    def test_person_returns_after_brief_observed_absence_keeps_id(self):
        frame = ([{"person_ref": "person_0", "bbox_xyxy": [0, 0, 100, 200]}], [])
        results, _ = self.run_frames([frame, ([], []), frame])
        self.assertEqual(results[0]["tracked_people"][0]["track_id"], "P1")
        self.assertEqual(results[1]["tracked_people"], [])
        self.assertEqual(results[2]["tracked_people"][0]["track_id"], "P1")

    def test_live_crossing_keeps_anonymous_ids_without_any_faces(self):
        frames = []
        for index, (a, b) in enumerate(((0, 140), (30, 110), (60, 80), (100, 40), (130, 10))):
            people = [
                {"person_ref": f"a_{index}", "bbox_xyxy": [a, 0, a + 100, 200]},
                {"person_ref": f"b_{index}", "bbox_xyxy": [b, 0, b + 100, 200]},
            ]
            if index % 2:
                people.reverse()
            frames.append((people, []))
        results, _ = self.run_frames(frames)
        for index, result in enumerate(results):
            self.assertEqual({p["person_ref"]: p["track_id"] for p in result["tracked_people"]},
                             {f"a_{index}": "P1", f"b_{index}": "P2"})
            self.assertEqual(result["matches"], [])

    def test_live_occlusion_emits_no_ghost_and_ends_behavior_episode(self):
        face = [{"face_bbox": [20, 20, 60, 60], "student_id": "SYNTHETIC_001", "score": .9}]
        detection = [{"person_ref": "a", "bbox_xyxy": [0, 0, 100, 200],
                      "behaviors": {"using_phone": {"state": "detected"}}}]
        results, _ = self.run_frames([(detection, face), (detection, face),
                                     ([], []), (detection, face)])
        self.assertEqual(results[2]["tracked_people"], [])
        self.assertEqual(results[2]["behavior_frame"].people, ())
        self.assertEqual(results[3]["tracked_people"][0]["track_id"], "P1")
        self.assertEqual(results[1]["behavior_frame"].people[0].behaviors[1].estimated_seconds, 1.)
        self.assertEqual(results[3]["behavior_frame"].people[0].behaviors[1].estimated_seconds, 0.)

    def test_two_people_slow_pipeline_expires_missing_track(self):
        left = {"person_ref": "left", "bbox_xyxy": [0, 0, 100, 200]}
        right = {"person_ref": "right", "bbox_xyxy": [200, 0, 300, 200]}
        replacement = {"person_ref": "new_left", "bbox_xyxy": [0, 0, 100, 200]}
        results, _ = self.run_frames([
            ([left, right], []), ([right], []), ([replacement, right], []),
        ], gap=3.74)
        self.assertEqual({p["person_ref"]: p["track_id"] for p in results[-1]["tracked_people"]},
                         {"right": "P2", "new_left": "P3"})

    def test_ai_panel_renders_propagated_person_track(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = {
            "frame_id": 1, "captured_at": 0.0, "elapsed": 1.0,
            "face_count": 1, "person_count": 1,
            "matches": [{"student_id": "SYNTHETIC_001", "person_ref": "person_0",
                         "person_track_id": "P7", "association_status": "matched"}],
        }
        result["behavior_frame"] = BehaviorDurationTracker().update(
            1, 0.0,
            [{"person_ref": "person_0", "behaviors": {}}],
            [{"person_ref": "person_0", "track_id": "P7", "bbox": [0, 0, 100, 200]}],
            result["matches"],
        )
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text:
            panel = live.draw_ai_panel(frame, result)
        self.assertEqual(panel.shape, (720, 1280, 3))
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Person Track: P7", labels)
        self.assertIn("Student: SYNTHETIC_001", labels)

    def test_duration_uses_capture_time_once_and_rendering_does_not_advance_it(self):
        people = [{"person_ref": "person_0", "bbox_xyxy": [20, 10, 120, 220],
                   "behaviors": {"using_phone": {"state": "detected"}}}]
        faces = [{"face_bbox": [40, 20, 80, 70], "student_id": "SYNTHETIC_001", "score": .82}]
        with (patch("sqlite3.connect", side_effect=AssertionError("database forbidden")),
              patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))):
            results, _ = self.run_frames([(people, faces)] * 3, gap=3.8)
            last = results[-1]
            self.assertAlmostEqual(last["behavior_frame"].people[0].behaviors[1].estimated_seconds, 7.6)
            with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text:
                for now in (120.0, 121.0, 150.0):
                    live.draw_ai_panel(np.zeros((480, 640, 3), dtype=np.uint8), last, now=now)
            labels = [call.args[1] for call in put_text.call_args_list]
            self.assertEqual(labels.count("Using Phone: detected ~7.6s"), 3)
            self.assertFalse(any("WARNING" in label for label in labels))
        self.assertEqual(results[0]["behavior_frame"].people[0].behaviors[1].estimated_seconds, 0.0)

    def test_capture_clock_is_used_even_when_processing_intervals_differ(self):
        people = [{"person_ref": "person_0", "bbox_xyxy": [20, 10, 120, 220],
                   "behaviors": {"using_phone": {"state": "detected"}}}]
        faces = [{"face_bbox": [40, 20, 80, 70], "student_id": "SYNTHETIC_001", "score": .82}]
        results, _ = self.run_frames([(people, faces)] * 3, gap=3.0,
                                    captured_at=[10.0, 12.0, 15.0], updated_at=[101.0, 104.0, 107.0])
        self.assertEqual(results[-1]["behavior_frame"].people[0].behaviors[1].estimated_seconds, 5.0)

    def test_duplicate_student_ids_are_suppressed_after_real_association_v4(self):
        people = [{"person_ref": f"person_{i}", "bbox_xyxy": [20 + i * 200, 10, 120 + i * 200, 220],
                   "behaviors": {"using_phone": {"state": "detected"}}} for i in range(2)]
        faces = [{"face_bbox": [40 + i * 200, 20, 80 + i * 200, 70],
                  "student_id": "SYNTHETIC_001", "score": .82} for i in range(2)]
        results, _ = self.run_frames([(people, faces)] * 2)
        self.assertEqual(len(results[-1]["matches"]), 2)
        for view in results[-1]["behavior_frame"].people:
            self.assertIsNone(view.student_id)
            self.assertEqual(view.identity_status, "ambiguous_student")
            self.assertIsNone(view.behaviors[1].estimated_seconds)

    def test_snapshot_survives_jpeg_path_reuse_and_live_preview_has_no_historical_boxes(self):
        people = [{"person_ref": "person_0", "bbox_xyxy": [20, 10, 120, 220], "behaviors": {}}]
        images = [np.full((480, 640, 3), (60, 100, 150), dtype=np.uint8),
                  np.full((480, 640, 3), (140, 80, 30), dtype=np.uint8)]
        results, _ = self.run_frames([(people, [])] * 2, source_images=images)
        self.assertFalse(np.array_equal(results[0]["snapshot"].image, results[1]["snapshot"].image))
        first = results[0]
        with patch.object(live.cv2, "imread", side_effect=AssertionError("UI must not reload frame.jpg")):
            rendered = render_snapshot(first["snapshot"], expected_frame_id=first["frame_id"],
                                       expected_captured_at=first["captured_at"])
            preview = np.full((480, 640, 3), 210, dtype=np.uint8)
            with patch.object(live.cv2, "rectangle", wraps=live.cv2.rectangle) as rectangle:
                output = live.draw_ai_panel(preview, first)
            rectangle.assert_not_called()
        np.testing.assert_array_equal(output[48:720, :896],
                                     live.cv2.resize(preview, (896, 672), interpolation=live.cv2.INTER_LINEAR))
        np.testing.assert_array_equal(preview, np.full_like(preview, 210))
        self.assertEqual(output.shape, (720, 1280, 3))
        with self.assertRaises(ValueError):
            rendered.validate(results[1]["frame_id"], results[1]["captured_at"])

    def test_main_preserves_snapshot_for_debug_without_displaying_it(self):
        results, _ = self.run_frames([([], [])])
        result = results[0]
        preview = np.full((480, 640, 3), 100, dtype=np.uint8)
        cap = Mock()
        cap.isOpened.return_value = True
        cap.read.side_effect = [(True, preview.copy()) for _ in range(3)]

        class FakeThread:
            def __init__(self, target, args, daemon):
                self.results = args[1]

            def start(self):
                self.results.put(result)

            def is_alive(self):
                return False

        with (patch.object(live.cv2, "VideoCapture", return_value=cap),
              patch.object(live.threading, "Thread", FakeThread),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "imshow") as show,
              patch.object(live.cv2, "waitKey", side_effect=[-1, -1, ord("q")]),
              patch.object(live.cv2, "destroyAllWindows"),
              patch.object(live, "render_snapshot", wraps=render_snapshot, create=True) as render,
              patch.object(live.Path, "is_file", return_value=True),
              redirect_stdout(io.StringIO())):
            live.main()
        render.assert_not_called()
        self.assertEqual(result["snapshot"].frame_id, result["frame_id"])
        self.assertIs(result["snapshot"].observations, result["behavior_frame"])
        self.assertEqual(show.call_count, 3)
        cap.release.assert_called_once_with()

    def test_panel_paging_exposes_students_beyond_the_first_two(self):
        people = [{"person_ref": f"person_{i}", "bbox_xyxy": [i * 120, 0, i * 120 + 100, 200],
                   "behaviors": {}} for i in range(3)]
        results, _ = self.run_frames([(people, [])])
        with patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text:
            live.draw_ai_panel(np.zeros((480, 640, 3), dtype=np.uint8), results[0], page=2)
        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertIn("Person Track: P3", labels)
        self.assertNotIn("Person Track: P1", labels)


if __name__ == "__main__":
    unittest.main()
