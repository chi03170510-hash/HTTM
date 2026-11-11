"""CH-21 replay at the worker boundary through real trackers, association and UI.

Mock configuration and synthetic evidence exist only in this test process.
No production simulation flag, camera, model execution, database or API calls.
"""

import ast
import copy
import inspect
import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

import behavior_warning
import camera_coordinator_live as live
from behavior_warning import DEFAULT_WARNING_CONFIG, BehaviorWarningObserver, WarningConfig
from test_behavior_warning import detected_evidence, phone_view
from test_ch20_duration_pipeline import PipelineHarness, _ReplayWorker, observation, phone_value


def source(frame_id, captured_at=None, *, extra_behavior=None, **kwargs):
    item = observation(frame_id, 198.5 + 1.5 * (frame_id - 1) if captured_at is None else captured_at, **kwargs)
    for person in item.people:
        for name, decision in person["behaviors"].items():
            if decision["state"] == "detected":
                person["behaviors"][name] = detected_evidence(name)
        if extra_behavior:
            person["behaviors"][extra_behavior] = detected_evidence(extra_behavior)
    return item


def sequence(count=5, **kwargs):
    return [source(i, **kwargs) for i in range(1, count + 1)]


def run_pipeline(observations, config=WarningConfig(using_phone=True), **kwargs):
    original = _ReplayWorker.process_frame

    def realistic_completion(worker, path, frame_id):
        reply = original(worker, path, frame_id)
        index = len(worker.requests) - 1
        # Retain baseline virtual latency; a large capture gap must also advance
        # completion time, instead of making a future capture accidentally fail.
        completed = max(200. + 1.5 * index, worker.observations[index].captured_at + 1.5)
        worker.clock.now = completed - (.3 if worker.kind == "face" else 0.)
        return reply

    with (patch.object(live, "WARNING_CONFIG", config),
          patch.object(_ReplayWorker, "process_frame", realistic_completion)):
        return PipelineHarness().run(observations, **kwargs)


def events(trace):
    return tuple(event for result in trace.results for event in result["warning_frame"].events)


class TestCH21WarningPipeline(unittest.TestCase):
    def test_production_disabled_keeps_duration_and_all_four_states_on_panel(self):
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)
        trace = run_pipeline(sequence(), config=DEFAULT_WARNING_CONFIG)
        self.assertEqual(len(trace.results), 5)
        self.assertFalse(events(trace))
        self.assertEqual(phone_value(trace.results[-1]), 6.)
        for sample in trace.samples[-2:]:
            for label in ("Experimental alerts: disabled", "Using Phone: detected ~6.0s",
                          "Sleeping: suspected --", "Turning Back: unknown --", "Eating: not_detected --"):
                self.assertIn(label, sample.labels)
            self.assertFalse(any("WARNING" in label for label in sample.labels))
            self.assertEqual(sample.image_shape, (720, 1280, 3))
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)

    def test_exact_boundaries_use_real_ai_result_association_tracking_and_panel(self):
        times = (198.5, 200., 201.5, 203.4, 203.5, 203.51)
        trace = run_pipeline([source(i, t) for i, t in enumerate(times, 1)])
        self.assertEqual(trace.association_calls, 6)
        self.assertEqual([len(r["warning_frame"].events) for r in trace.results], [0, 0, 0, 0, 0, 1])
        for index, expected in ((3, 4.9), (4, 5.), (5, 5.01)):
            result = trace.results[index]
            self.assertAlmostEqual(phone_value(result), expected)
            self.assertAlmostEqual(phone_view(result["warning_frame"]).qualified_seconds, expected)
            self.assertEqual(result["matches"][0]["association_status"], "matched")
            self.assertEqual(result["matches"][0]["person_track_id"], "P1")
            self.assertEqual(result["matches"][0]["student_id"], "SV001")
            for sample in (s for s in trace.samples if s.observation_index == index):
                self.assertIn(f"Using Phone: detected ~{expected:.1f}s", sample.labels)
                if index == 5:
                    self.assertIn("WARNING (EXPERIMENTAL)", sample.labels)
                else:
                    self.assertNotIn("WARNING (EXPERIMENTAL)", sample.labels)
        last = trace.samples[-1]
        for label in ("Student: SV001", "Person Track: P1", "Not an official violation",
                      "SV001 | P1 | Using Phone", "Qualified ~5.0s | Frame 6"):
            self.assertIn(label, last.labels)

    def test_one_event_and_repainting_never_calls_observer_or_changes_duration(self):
        update = BehaviorWarningObserver.update
        calls = []

        def record(observer, *args, **kwargs):
            calls.append(args[0].frame_id)
            return update(observer, *args, **kwargs)

        with patch.object(BehaviorWarningObserver, "update", record):
            trace = run_pipeline(sequence(7))
        self.assertEqual(calls, list(range(1, 8)))
        self.assertEqual(len(events(trace)), 1)
        result = trace.results[-1]
        before = (result["behavior_frame"], result["warning_frame"])
        with patch.object(BehaviorWarningObserver, "update", side_effect=AssertionError("render must only read")):
            for age in (1.6, 2., 4., 5., 5.01, 20.):
                labels, _, _ = self.draw_labels(result, now=result["captured_at"] + age)
                self.assertIn("Using Phone: detected ~9.0s", labels)
        self.assertEqual((result["behavior_frame"], result["warning_frame"]), before)

    def test_stale_capture_is_not_an_active_alert_even_when_completed_result_is_recent(self):
        trace = run_pipeline(sequence())
        result = trace.results[-1]
        # A recent completion cannot refresh an old source photograph.
        changed = dict(result, completed_at=result["captured_at"] + 5.4)
        labels, _, _ = self.draw_labels(changed, now=result["captured_at"] + 5.5)
        self.assertNotIn("WARNING (EXPERIMENTAL)", labels)
        self.assertIn("STALE experimental alert (historical)", labels)
        self.assertIn("Not an official violation", labels)
        self.assertEqual(len(events(trace)), 1)

    def test_missing_or_conflicting_evidence_blocks_pipeline_warning(self):
        for defect in ("missing", "contradiction", "missing_geometry"):
            with self.subTest(defect=defect):
                observations = sequence()
                for item in observations:
                    decision = item.people[0]["behaviors"]["using_phone"]
                    if defect == "missing":
                        del decision["supporting_evidence"]
                    elif defect == "contradiction":
                        decision["contradicting_evidence"] = ["ownership_ambiguous"]
                    else:
                        del decision["evidence"]["phone_object_geometry"]
                trace = run_pipeline(observations)
                self.assertEqual(phone_value(trace.results[-1]), 6.)
                self.assertFalse(events(trace))
                self.assertFalse(any("WARNING (EXPERIMENTAL)" in s.labels for s in trace.samples))

    def test_qualified_evidence_restarts_instead_of_reusing_old_ch20_duration(self):
        observations = sequence(9)
        observations[2].people[0]["behaviors"]["using_phone"]["contradicting_evidence"] = ["conflict"]
        trace = run_pipeline(observations)
        self.assertEqual(phone_value(trace.results[4]), 6.)
        self.assertEqual(phone_view(trace.results[4]["warning_frame"]).qualified_seconds, 1.5)
        self.assertFalse(trace.results[4]["warning_frame"].events)
        self.assertEqual([e.frame_id for e in events(trace)], [8])
        self.assertEqual(events(trace)[0].qualified_seconds, 6.)
        self.assertEqual(events(trace)[0].estimated_seconds, 10.5)

    def test_negative_behavior_ends_and_later_episode_can_warn_again(self):
        for state in ("suspected", "unknown", "not_detected"):
            with self.subTest(state=state):
                observations = sequence(11)
                observations[5] = source(6, state=state)
                trace = run_pipeline(observations)
                self.assertEqual([e.frame_id for e in events(trace)], [5, 11])
                self.assertEqual(phone_view(trace.results[5]["warning_frame"]).state, "CLEAR")
                self.assertNotEqual(events(trace)[0].episode_id, events(trace)[1].episode_id)

    def test_real_person_tracker_change_does_not_inherit_warning(self):
        observations = sequence(5) + [source(i, shift=260) for i in range(6, 11)]
        trace = run_pipeline(observations)
        self.assertEqual([(e.frame_id, e.person_track_id) for e in events(trace)], [(5, "P1"), (10, "P2")])
        self.assertEqual(phone_value(trace.results[5]), 0.)

    def test_current_student_change_does_not_inherit_warning(self):
        observations = sequence(5) + [source(i, student="SV002") for i in range(6, 11)]
        trace = run_pipeline(observations)
        self.assertEqual([(e.frame_id, e.student_id) for e in events(trace)], [(5, "SV001"), (10, "SV002")])
        self.assertEqual({e.person_track_id for e in events(trace)}, {"P1"})

    def test_association_loss_and_unknown_student_break_candidate(self):
        for changes in ({"associated": False}, {"student": "Unknown"}):
            with self.subTest(changes=changes):
                observations = sequence()
                observations[2] = source(3, **changes)
                trace = run_pipeline(observations)
                self.assertFalse(events(trace))
                self.assertEqual(phone_value(trace.results[3]), 0.)
                self.assertIn("Student: Unknown", trace.samples[4].labels)

    def test_duplicate_student_claims_are_ambiguous_and_block_both_people(self):
        trace = run_pipeline(sequence(second_student="SV001"))
        self.assertFalse(events(trace))
        for result in trace.results:
            self.assertEqual(len(result["matches"]), 2)
            self.assertEqual([p.identity_status for p in result["behavior_frame"].people],
                             ["ambiguous_student", "ambiguous_student"])
        for sample in trace.samples:
            self.assertIn("Student: Unknown", sample.labels)
            self.assertIn("Using Phone: detected --", sample.labels)

    def test_real_v4_heuristic_keeps_ch20_duration_but_never_qualifies_warning(self):
        observations = []
        for item in sequence():
            other = copy.deepcopy(item.people[0])
            other.update(person_ref="overlapping_person", bbox_xyxy=[60, 40, 220, 420],
                         person_bbox_xyxy_clipped=[60, 40, 220, 420])
            # Both equal-area boxes contain the face. V1/V2 reject ambiguity;
            # real V4 can select the nearer upper-body center heuristically.
            observations.append(replace(item, people=item.people + (other,)))
        trace = run_pipeline(observations)
        self.assertFalse(events(trace))
        for result in trace.results:
            self.assertEqual([m["association_status"] for m in result["matches"]], ["matched_v4_heuristic"])
            self.assertEqual(phone_view(result["warning_frame"]).reason, "association_not_matched")
        self.assertEqual(phone_value(trace.results[-1]), 6.)

    def test_gap_over_five_is_fresh_but_starts_new_episode(self):
        times = (198.5, 200., 205.01, 206.51, 208.01)
        trace = run_pipeline([source(i, t) for i, t in enumerate(times, 1)])
        self.assertFalse(events(trace))
        result = trace.results[2]
        self.assertLessEqual(result["completed_at"] - result["captured_at"], 5.)
        self.assertEqual(phone_value(result), 0.)
        self.assertEqual(phone_view(result["warning_frame"], "P2").qualified_seconds, 0.)

    def test_all_evidence_valid_but_stale_source_cannot_warn(self):
        trace = run_pipeline([source(i, 100. + 1.5 * (i - 1)) for i in range(1, 6)])
        self.assertEqual(phone_value(trace.results[-1]), 6.)
        self.assertFalse(events(trace))
        self.assertTrue(all(phone_view(r["warning_frame"]).reason == "stale_capture" for r in trace.results))

    def test_duplicate_ai_frame_does_not_repeat_event(self):
        observations = sequence()
        observations += [copy.deepcopy(observations[-1]), source(6)]
        trace = run_pipeline(observations)
        self.assertEqual([e.frame_id for e in events(trace)], [5])
        self.assertEqual(trace.results[5]["warning_frame"].events, ())
        self.assertIs(trace.results[4]["behavior_frame"], trace.results[5]["behavior_frame"])

    def test_non_increasing_timestamp_and_duplicate_metadata_fail_closed(self):
        for timestamp in (200., 199.9):
            with self.subTest(timestamp=timestamp):
                observations = sequence()
                observations[2] = source(3, timestamp)
                trace = run_pipeline(observations)
                self.assertFalse(events(trace))
                self.assertFalse(trace.results[2]["warning_frame"].valid)
                self.assertEqual(phone_view(trace.results[3]["warning_frame"]).qualified_seconds, 0.)
        trace = run_pipeline([source(1), source(2), source(2, 201.5)], allow_error=True)
        self.assertEqual(len(trace.errors), 1)
        self.assertIn("snapshot and behavior metadata", trace.errors[0])
        self.assertFalse(events(trace))

    def test_two_behaviors_and_two_students_reach_correct_panel_pages(self):
        trace = run_pipeline(sequence(extra_behavior="sleeping", second_student="SV002"),
                             config=WarningConfig(using_phone=True, sleeping=True))
        self.assertEqual(len(events(trace)), 4)
        self.assertEqual({(e.student_id, e.behavior) for e in events(trace)},
                         {("SV001", "using_phone"), ("SV001", "sleeping"),
                          ("SV002", "using_phone"), ("SV002", "sleeping")})
        last_samples = [s for s in trace.samples if s.observation_index == 4]
        self.assertEqual([s.page for s in last_samples], [0, 1])
        for sample in last_samples:
            self.assertEqual(sample.labels.count("WARNING (EXPERIMENTAL)"), 2)
            self.assertIn("N/P: students | Q: Quit", sample.labels)

    def test_person_disappearing_and_returning_does_not_reuse_warning(self):
        observations = sequence(11)
        observations[5] = source(6, present=False)
        trace = run_pipeline(observations)
        self.assertEqual([(e.frame_id, e.person_track_id) for e in events(trace)], [(5, "P1"), (11, "P2")])
        self.assertEqual(trace.results[5]["warning_frame"].views, ())
        self.assertEqual(phone_value(trace.results[6]), 0.)

    def test_warning_panel_preserves_aspect_ratio_no_live_boxes_and_no_clipped_labels(self):
        trace = run_pipeline(sequence())
        result = trace.results[-1]
        labels, rendered, calls = self.draw_labels(result, now=result["captured_at"] + 2.)
        self.assertIn("WARNING (EXPERIMENTAL)", labels)
        preview = cv2.resize(np.full((480, 640, 3), 90, dtype=np.uint8), (896, 672))
        top = 48 + (rendered.shape[0] - 48 - 672) // 2
        np.testing.assert_array_equal(rendered[top:top + 672, :896], preview)
        for call in calls:
            size, baseline = cv2.getTextSize(call.args[1], call.args[3], call.args[4], call.args[6])
            x, y = call.args[2]
            self.assertLessEqual(x + size[0], 384)
            self.assertLessEqual(y + baseline, rendered.shape[0])

    def test_runtime_side_effect_guards_and_observer_has_no_persistence_or_api_imports(self):
        # PipelineHarness raises immediately on SQLite, socket, urllib or process
        # creation. This positive-warning replay must reach its actual UI anyway.
        with patch("http.client.HTTPConnection.request", side_effect=AssertionError("API forbidden")):
            trace = run_pipeline(sequence())
        self.assertEqual(len(events(trace)), 1)
        self.assertEqual(len(trace.results), 5)
        self.assertFalse(trace.errors)
        tree = ast.parse(inspect.getsource(behavior_warning))
        imports = {node.module.split(".")[0] for node in ast.walk(tree)
                   if isinstance(node, ast.ImportFrom) and node.module}
        imports.update(alias.name.split(".")[0] for node in ast.walk(tree)
                       if isinstance(node, ast.Import) for alias in node.names)
        self.assertTrue(imports.isdisjoint({"sqlite3", "requests", "urllib", "socket", "http", "subprocess",
                                            "behavior_timer", "behavior_state_manager"}))

    @staticmethod
    def draw_labels(result, *, now):
        with (patch.object(live.cv2, "putText", wraps=live.cv2.putText) as put_text,
              patch.object(live.cv2, "rectangle", side_effect=AssertionError("no historical bbox on live")),
              patch("sqlite3.connect", side_effect=AssertionError("SQLite forbidden")),
              patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))):
            rendered = live.draw_ai_panel(np.full((480, 640, 3), 90, dtype=np.uint8), result, now=now)
        calls = [call for call in put_text.call_args_list if call.args[0].shape[1] == 384]
        return tuple(call.args[1] for call in calls), rendered, calls


if __name__ == "__main__":
    unittest.main()
