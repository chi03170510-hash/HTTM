"""Opt-in trial through the real Camera Live constructor, AI loop and panel.

Only worker replies, clock, camera and GUI are mocked. No model is executed.
Recorded camera input stays read-only; synthetic identity risk has actor truth.
"""

import hashlib
import json
import os
import queue
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import camera_coordinator as coordinator
import camera_coordinator_live as live
from behavior_warning import DEFAULT_WARNING_CONFIG
from evaluate_face_tracker_parameters import DEFAULT_SESSION, load_camera_session, risk_scenarios, run_trial
from face_tracker import FaceTracker
from face_tracker_diagnostics import FaceTrackerDiagnostics
from person_tracker_diagnostics import DiagnosticConfig, PersonTrackerDiagnostics
from replay_face_tracker_diagnostics import replay_records as replay_faces
from replay_person_tracker_diagnostics import replay_records as replay_people
from test_ch20_duration_pipeline import Observation, PipelineHarness, _ReplayWorker, phone_value
from test_ch21_warning_pipeline import events, run_pipeline, source
from test_face_tracker_diagnostic_pipeline import collect, observations


ENV = "HTTM_FACE_TRACKER_TRIAL_MAX_AGE"


class TestFaceTrackerCameraTrial(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for key in tuple(os.environ):
            if key == ENV or key.startswith(("HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG")):
                os.environ.pop(key)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def enable_trial(self):
        os.environ[ENV] = "4.0"

    def test_unset_environment_keeps_baseline_two_seconds(self):
        self.assertEqual(live.face_tracker_max_age_from_environment(), 2.0)

    def test_exact_four_seconds_enables_trial(self):
        self.enable_trial()
        self.assertEqual(live.face_tracker_max_age_from_environment(), 4.0)

    def test_invalid_values_stop_before_camera_workers_or_gui(self):
        for value in ("", "2.0", "5.0", "3.0", "4", "4.00", " 4.0", "4.0 ",
                      "nan", "inf", "-1", "true", "4,0", "4e0"):
            with (self.subTest(value=value), patch.dict(os.environ, {ENV: value}),
                  patch.object(live.cv2, "VideoCapture") as camera,
                  patch.object(live.cv2, "namedWindow") as window,
                  patch.object(live, "Worker") as worker,
                  patch.object(live.threading, "Thread") as thread):
                with self.assertRaisesRegex(ValueError, "HTTM_FACE_TRACKER_TRIAL_MAX_AGE.*only '4.0'"):
                    live.main()
                for resource in (camera, window, worker, thread):
                    resource.assert_not_called()

    def test_invalid_configuration_also_fails_cleanly_in_direct_ai_worker(self):
        os.environ[ENV] = "5.0"
        results = queue.Queue()
        with patch.object(live, "Worker") as worker:
            live.ai_loop(queue.Queue(), results, threading.Event())
        worker.assert_not_called()
        self.assertIn(ENV, results.get_nowait()["error"])
        self.assertEqual(results.get_nowait(), {"finished": True})
        self.assertTrue(results.empty())

    def test_unsetting_restores_baseline_in_same_process(self):
        self.enable_trial()
        trial = run_pipeline(observations(2), config=DEFAULT_WARNING_CONFIG)
        del os.environ[ENV]
        baseline = run_pipeline(observations(2), config=DEFAULT_WARNING_CONFIG)
        self.assertEqual([r["matches"][0]["track_id"] for r in trial.results], [1, 1])
        self.assertEqual([r["matches"][0]["track_id"] for r in baseline.results], [1, 2])
        self.assertIn("[FACE TRACKER TRIAL] max_age=4.0s", trial.output)
        self.assertNotIn("[FACE TRACKER TRIAL]", baseline.output)

    def test_trial_is_scoped_to_camera_live_not_face_tracker_class_defaults(self):
        self.enable_trial()
        self.assertIs(live.FaceTracker, FaceTracker)
        self.assertEqual(FaceTracker().max_age, 2.0)
        self.assertEqual(FaceTracker().iou_threshold, .3)
        self.assertEqual(live.PERSON_TRACK_MAX_AGE, 5.0)
        self.assertEqual(live.CAMERA_INDEX, coordinator.CAMERA_INDEX)
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)
        self.assertEqual(live.WARNING_CONFIG.enabled_behaviors, ())

    def recorded_replay(self):
        """Exact camera detections/order/scores/capture/update timestamps.

        Person boxes are recorded inputs, not a second inference. Behaviors were
        not logged, so they remain empty (unknown), never invented as evidence.
        Only face-update times are reproduced, not measured processing latency.
        """
        frames, original_records, digest = load_camera_session(DEFAULT_SESSION)
        items = [Observation(
            frame.frame_id, frame.captured_at,
            tuple({"person_ref": p.person_ref, "person_bbox_xyxy_clipped": list(p.bbox),
                   "bbox_xyxy": list(p.bbox), "behaviors": {}} for p in frame.people),
            tuple({"face_bbox": list(f.bbox), "student_id": f.student_id, "score": f.score}
                  for f in frame.faces),
        ) for frame in frames]
        original_process = _ReplayWorker.process_frame

        def recorded_completion(worker, path, frame_id):
            reply = original_process(worker, path, frame_id)
            index = len(worker.requests) - 1
            worker.clock.now = frames[index].updated_at - (.3 if worker.kind == "face" else 0.)
            return reply

        config = DiagnosticConfig(True, output_dir=self.root, face_enabled=True)
        with (patch.object(live.DiagnosticConfig, "from_environment", return_value=config),
              patch.object(_ReplayWorker, "process_frame", recorded_completion)):
            trace = PipelineHarness().run(items)
        paths = list(self.root.glob("session_*/frames.jsonl"))
        self.assertEqual(len(paths), 1)
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        self.assertEqual(len(rows), 10)
        self.assertEqual(hashlib.sha256(DEFAULT_SESSION.read_bytes()).hexdigest(), digest)
        for row, recorded in zip(rows, original_records):
            for key in ("captured_at", "face_tracker_updated_at", "face_update_gap", "detections"):
                self.assertEqual(row["face_tracker"][key], recorded["face_tracker"][key])
            self.assertEqual(row["parameters"], {"iou_threshold": .3, "max_age": 5., "keep_unmatched": False})
        self.assertTrue(all(r["consistent"] for r in replay_faces(rows)))
        self.assertTrue(all(r["consistent"] for r in replay_people(rows)))
        self.assertFalse(events(trace))
        self.assertEqual(len(trace.samples), 20)
        self.assertTrue(all(r["matches"][0]["person_track_id"] == "P1" for r in trace.results))
        self.assertTrue(all(r["matches"][0]["student_id"] == "SV001" for r in trace.results))
        self.assertTrue(all(r["matches"][0]["association_status"] == "matched" for r in trace.results))
        self.assertEqual([p.name for p in paths[0].parent.iterdir()], ["frames.jsonl"])
        return trace, rows

    def test_recorded_ten_camera_frames_baseline_still_reproduces_six_expiries(self):
        trace, rows = self.recorded_replay()
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1, 1, 2, 3, 4, 5, 6, 6, 7, 7])
        self.assertTrue(all(r["face_tracker"]["parameters"]["max_age"] == 2. for r in rows))
        self.assertEqual(sum(d["action"] == "expired" for r in rows for d in r["face_tracker"]["decisions"]), 6)
        self.assertNotIn("[FACE TRACKER TRIAL]", trace.output)

    def test_recorded_ten_camera_frames_use_real_opt_in_constructor_and_debug_four(self):
        self.enable_trial()
        trace, rows = self.recorded_replay()
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1] * 10)
        self.assertTrue(all(r["face_tracker"]["parameters"] == {"iou_threshold": .3, "max_age": 4.} for r in rows))
        self.assertEqual(sum(d["action"] == "expired" for r in rows for d in r["face_tracker"]["decisions"]), 0)
        self.assertEqual(sum(d["action"] == "matched" for r in rows for d in r["face_tracker"]["decisions"]), 9)
        self.assertIn("[FACE TRACKER TRIAL] max_age=4.0s", trace.output)
        # Recorded extra person detections have no face association. N/P must
        # show Unknown on those pages, never borrow SV001's face track.
        matched_pages = [s for s in trace.samples if "Person Track: P1" in s.labels]
        self.assertEqual({s.frame_id for s in matched_pages}, set(range(1, 11)))
        self.assertTrue(all("Face Track: T1" in s.labels and "Student: SV001" in s.labels for s in matched_pages))
        for sample in trace.samples:
            if sample not in matched_pages:
                self.assertIn("Face Track: Unknown", sample.labels)
                self.assertIn("Student: Unknown", sample.labels)

    def test_debug_disabled_never_calls_observer_or_writer_with_trial_on(self):
        self.enable_trial()
        with (patch.object(FaceTrackerDiagnostics, "before_update") as before_face,
              patch.object(FaceTrackerDiagnostics, "build") as face_build,
              patch.object(PersonTrackerDiagnostics, "before_update") as before_person,
              patch.object(PersonTrackerDiagnostics, "record") as record):
            trace = run_pipeline(observations(3), config=DEFAULT_WARNING_CONFIG)
        for operation in (before_face, face_build, before_person, record):
            operation.assert_not_called()
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1] * 3)

    def test_real_debug_environment_keeps_shared_quota_and_does_not_save_images(self):
        self.enable_trial()
        os.environ.update({"HTTM_PERSON_DEBUG": "1", "HTTM_FACE_DEBUG": "1",
                           "HTTM_PERSON_DEBUG_DIR": str(self.root)})
        config = DiagnosticConfig.from_environment(live.ROOT)
        self.assertEqual((config.max_frames, config.max_bytes, config.snapshots), (10, 20 * 1024 * 1024, False))
        trace = run_pipeline(observations(3), config=DEFAULT_WARNING_CONFIG)
        paths = list(self.root.glob("session_*/frames.jsonl"))
        self.assertEqual(len(paths), 1)
        rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r["face_tracker"]["parameters"]["max_age"] == 4. for r in rows))
        self.assertTrue(all(r["parameters"]["max_age"] == 5. for r in rows))
        self.assertEqual([p.name for p in paths[0].parent.iterdir()], ["frames.jsonl"])
        self.assertFalse(events(trace))

    def test_trial_keeps_person_tracking_duration_and_test_only_warning_episode(self):
        self.enable_trial()
        trace, rows, _ = collect(self.root, observations())  # Alerts enabled only by test harness.
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1] * 5)
        self.assertEqual([r["matches"][0]["person_track_id"] for r in trace.results], ["P1"] * 5)
        self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., 9., 12.])
        self.assertEqual([e.frame_id for e in events(trace)], [3])
        self.assertTrue(all(r["parameters"]["max_age"] == 5. for r in rows))
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)

    def test_unknown_duplicate_and_missing_association_still_break_episodes(self):
        self.enable_trial()
        for mode, kwargs in (("unknown", {"student": "Unknown"}),
                             ("duplicate", {"second_student": "SV001"}),
                             ("unassociated", {"associated": False}), ("missing_face", {})):
            with self.subTest(mode=mode):
                items = observations(7)
                items[3] = source(4, 207.5, **kwargs)
                if mode == "missing_face":
                    items[3] = replace(items[3], faces=())
                trace = run_pipeline(items)
                self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., None, 0., 3., 6.])
                self.assertEqual([e.frame_id for e in events(trace)], [3, 7])
                self.assertNotEqual(events(trace)[0].episode_id, events(trace)[1].episode_id)
                self.assertTrue(all(v.state == "CLEAR" for v in trace.results[3]["warning_frame"].views))

    def test_student_change_resets_even_when_face_and_person_ids_are_reused(self):
        self.enable_trial()
        items = observations(3) + [source(i, 198.5 + 3. * (i - 1), student="SV002") for i in range(4, 7)]
        trace = run_pipeline(items)
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1] * 6)
        self.assertEqual([r["matches"][0]["person_track_id"] for r in trace.results], ["P1"] * 6)
        self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., 0., 3., 6.])
        self.assertEqual([(e.frame_id, e.student_id) for e in events(trace)], [(3, "SV001"), (6, "SV002")])
        self.assertFalse(trace.results[3]["warning_frame"].events)

    def test_actual_trial_expiry_boundary_is_strict_and_person_stays_five_seconds(self):
        self.enable_trial()
        for gap, expected in ((4., [1, 1]), (4.001, [1, 2])):
            with self.subTest(gap=gap):
                items = [source(1, 198.5), source(2, 198.5 + gap)]
                trace = run_pipeline(items, config=DEFAULT_WARNING_CONFIG)
                self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], expected)
                self.assertEqual([r["matches"][0]["person_track_id"] for r in trace.results], ["P1", "P1"])
                self.assertAlmostEqual(phone_value(trace.results[1]), gap)

    def test_replacement_same_position_after_three_seconds_still_reuses_wrong_actor_track(self):
        self.enable_trial()
        case = risk_scenarios()["replacement_3_0"]
        trial = run_trial(case, live.face_tracker_max_age_from_environment())
        baseline = run_trial(case, 2.0)
        self.assertEqual(trial["ids_created"], 1)
        self.assertEqual(trial["wrong_actor_continuations"], 1)
        self.assertEqual(baseline["ids_created"], 2)
        self.assertEqual(baseline["wrong_actor_continuations"], 0)

    def test_crossing_can_still_switch_physical_actors_with_trial(self):
        self.enable_trial()
        trial = run_trial(risk_scenarios()["crossing"], live.face_tracker_max_age_from_environment())
        self.assertEqual(trial["reference_id_changes"], 2)
        self.assertEqual(trial["wrong_actor_continuations"], 2)

    def test_two_distinct_faces_keep_tracks_when_detection_order_reverses(self):
        self.enable_trial()
        items = [source(i, 198.5 + 3. * (i - 1), second_student="SV002") for i in range(1, 4)]
        items[1] = replace(items[1], faces=tuple(reversed(items[1].faces)))
        trace = run_pipeline(items, config=DEFAULT_WARNING_CONFIG)
        for result in trace.results:
            self.assertEqual({m["student_id"]: (m["track_id"], m["person_track_id"]) for m in result["matches"]},
                             {"SV001": (1, "P1"), "SV002": (2, "P2")})

    def test_panel_repaints_preserve_durations_and_production_alerts_stay_disabled(self):
        self.enable_trial()
        items = [source(i) for i in range(1, 5)]
        trace = run_pipeline(items, config=DEFAULT_WARNING_CONFIG)
        self.assertEqual([phone_value(r) for r in trace.results], [0., 1.5, 3., 4.5])
        self.assertEqual(len(trace.samples), 8)
        for index, duration in enumerate((0., 1.5, 3., 4.5)):
            samples = [s for s in trace.samples if s.observation_index == index]
            self.assertEqual([s.render_only for s in samples], [False, True])
            for sample in samples:
                for label in ("Student: SV001", "Face Track: T1", "Person Track: P1",
                              f"Using Phone: detected ~{duration:.1f}s", "Experimental alerts: disabled"):
                    self.assertIn(label, sample.labels)
                self.assertFalse(any("WARNING" in label for label in sample.labels))
                self.assertEqual(sample.image_shape, (720, 1280, 3))
        self.assertFalse(events(trace))
        self.assertFalse(trace.errors)


if __name__ == "__main__":
    unittest.main()
