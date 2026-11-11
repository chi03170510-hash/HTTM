"""Real ai_loop/CH20/UI with controlled worker replies; never opens a camera."""

import copy
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import camera_coordinator_live as live
import camera_live_test_toolkit as toolkit
import identity_continuity_diagnostics as diagnostic
from behavior_duration import BehaviorDurationTracker
from behavior_warning import BehaviorWarningObserver, DEFAULT_WARNING_CONFIG
from identity_continuity import IdentityContinuityResolver
from replay_identity_continuity_shadow import replay_logged_observations
from test_ch20_duration_pipeline import PipelineHarness, detected_sequence, observation, phone_value
from test_identity_continuity import Scenario


class TestIdentityContinuityShadow(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for name in (diagnostic.SHADOW_ENV, "HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG"):
            os.environ.pop(name, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / "shadow"
        os.environ[diagnostic.SHADOW_DIR_ENV] = str(self.output)

    def run_mode(self, observations, enabled):
        if enabled:
            os.environ[diagnostic.SHADOW_ENV] = "1"
        else:
            os.environ.pop(diagnostic.SHADOW_ENV, None)
        duration_args, warning_args = [], []
        duration_update, warning_update = BehaviorDurationTracker.update, BehaviorWarningObserver.update

        def duration(tracker, *args, **kwargs):
            duration_args.append(copy.deepcopy((args, kwargs)))
            return duration_update(tracker, *args, **kwargs)

        def warning(observer, *args, **kwargs):
            warning_args.append(copy.deepcopy((args, kwargs)))
            return warning_update(observer, *args, **kwargs)

        with (patch.object(BehaviorDurationTracker, "update", duration),
              patch.object(BehaviorWarningObserver, "update", warning)):
            trace = PipelineHarness().run(observations)
        return trace, duration_args, warning_args

    def assert_unchanged(self, baseline, shadow):
        a, da, wa = baseline
        b, db, wb = shadow
        self.assertEqual(da, db, "DurationTracker must receive identical arguments")
        self.assertEqual(wa, wb, "WarningObserver must receive identical arguments")
        self.assertEqual(a.samples, b.samples, "AI Panel fields and geometry must be unchanged")
        self.assertEqual(a.errors, b.errors)
        self.assertEqual(a.association_calls, b.association_calls)
        for x, y in zip(a.results, b.results):
            self.assertEqual(set(x), set(y))
            for key in set(x) - {"snapshot"}:
                self.assertEqual(x[key], y[key], key)
            self.assertEqual(x["snapshot"].observations, y["snapshot"].observations)
            np.testing.assert_array_equal(x["snapshot"].image, y["snapshot"].image)
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)
        self.assertFalse(any(r["warning_frame"].events for r in b.results))

    def read_rows(self):
        logs = list(self.output.glob("session_*/frames.jsonl"))
        self.assertEqual(len(logs), 1)
        return [json.loads(s) for s in logs[0].read_text(encoding="utf-8").splitlines()]

    def test_disabled_does_not_construct_resolver_copy_tracks_or_write(self):
        with (patch.object(diagnostic, "IdentityContinuityResolver", side_effect=AssertionError("disabled")),
              patch.object(diagnostic, "capture_prior_tracks", side_effect=AssertionError("disabled")),
              patch.object(diagnostic.IdentityContinuityShadow, "record", side_effect=AssertionError("disabled"))):
            self.run_mode(detected_sequence(), False)
        self.assertFalse(self.output.exists())

    def test_using_phone_two_students_unchanged_with_shadow_on(self):
        observations = [observation(i, 100. + 1.5 * (i - 1), second_student="SV002") for i in range(1, 5)]
        original = copy.deepcopy(observations)
        baseline = self.run_mode(observations, False)
        shadow = self.run_mode(observations, True)
        self.assert_unchanged(baseline, shadow)
        self.assertEqual(observations, original)
        self.assertEqual(phone_value(shadow[0].results[-1]), 4.5)
        self.assertEqual(phone_value(shadow[0].results[-1], 1), 4.5)
        rows = self.read_rows()
        self.assertTrue(all(r["simulation_hold_seconds"] is None and not r["production_applied"] for r in rows))
        self.assertEqual({o["proposed_student_id"] for o in rows[-1]["observations"]}, {"SV001", "SV002"})
        self.assertTrue(all(r["behavior_diagnostic_version"] == 2 for r in rows))
        for result, record in zip(shadow[0].results, rows):
            self.assertEqual(len(record["using_phone"]), 2)
            by_track = {p.person_track_id: p for p in result["behavior_frame"].people}
            for row in record["using_phone"]:
                production = by_track[row["person_track_id"]]
                value = next(b for b in production.behaviors if b.name == "using_phone")
                self.assertEqual((row["student_id"], row["using_phone_state"], row["using_phone_duration_seconds"]),
                                 (production.student_id, value.state, value.estimated_seconds))
                self.assertEqual(row["using_phone_model_confidence"], .8)

    def test_two_students_different_behaviors_keep_production_attribution(self):
        observations = [observation(i, 100. + 1.5 * (i - 1), second_student="SV002",
                                    second_state="not_detected") for i in range(1, 4)]
        for o in observations:
            for index, p in enumerate(o.people):
                p["behaviors"]["sleeping"].update(state="detected" if index else "not_detected")
                p["behaviors"]["using_phone"].update(model_confidence=.37 if index else .91,
                    decision_reason="synthetic_other_student" if index else "synthetic_phone_evidence")
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        record = self.read_rows()[-1]
        phone = {p["student_id"]: p for p in record["using_phone"]}
        sleeping = {p["student_id"]: p for p in record["sleeping"]}
        self.assertEqual(phone["SV001"]["using_phone_duration_seconds"], 3.)
        self.assertEqual(phone["SV001"]["using_phone_model_confidence"], .91)
        self.assertEqual(phone["SV001"]["using_phone_decision_reason"], "synthetic_phone_evidence")
        self.assertIsNone(phone["SV002"]["using_phone_duration_seconds"])
        self.assertEqual(phone["SV002"]["using_phone_model_confidence"], .37)
        self.assertEqual(sleeping["SV002"]["sleeping_duration_seconds"], 3.)
        self.assertIsNone(sleeping["SV001"]["sleeping_duration_seconds"])

    def test_using_phone_missing_confidence_and_behavior_fields_stay_unavailable(self):
        observations = detected_sequence()[:2]
        observations[0].people[0]["behaviors"]["using_phone"].pop("model_confidence")
        observations[1].people[0]["behaviors"].pop("using_phone")
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        rows = self.read_rows()
        self.assertIsNone(rows[0]["using_phone"][0]["using_phone_model_confidence"])
        self.assertIsNone(rows[0]["using_phone"][0]["using_phone_decision_reason"])
        phone = rows[1]["using_phone"][0]
        self.assertEqual(phone["using_phone_state"], "unknown")  # Actual production default, not logger invention.
        self.assertIsNone(phone["using_phone_duration_seconds"])
        self.assertIn("using_phone_unknown", phone["duration_blockers"])

    def test_using_phone_association_loss_blocks_then_restarts_duration(self):
        observations = [observation(1, 100.), observation(2, 101.5),
                        observation(3, 103., associated=False), observation(4, 104.5)]
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        phones = [r["using_phone"][0] for r in self.read_rows()]
        self.assertEqual([p["using_phone_duration_seconds"] for p in phones], [0., 1.5, None, 0.])
        self.assertEqual(phones[2]["association_status"], "unmatched")
        self.assertIsNone(phones[2]["student_id"])
        self.assertIn("identity_unassociated", phones[2]["duration_blockers"])
        self.assertIn("previous_identity_ineligible", phones[3]["duration_reset_reasons"])

    def test_using_phone_unknown_student_is_not_filled_from_shadow(self):
        observations = [observation(1, 100.), observation(2, 101.5, student="Unknown")]
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        phone = self.read_rows()[1]["using_phone"][0]
        self.assertIsNone(phone["student_id"])
        self.assertIsNone(phone["using_phone_duration_seconds"])
        self.assertIn("identity_unknown", phone["duration_blockers"])

    def test_using_phone_not_detected_then_restart_is_logged(self):
        observations = [observation(1, 100.), observation(2, 101.5),
                        observation(3, 103., state="not_detected"), observation(4, 104.5)]
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        phones = [r["using_phone"][0] for r in self.read_rows()]
        self.assertEqual([p["using_phone_duration_seconds"] for p in phones], [0., 1.5, None, 0.])
        self.assertIn("using_phone_not_detected", phones[2]["duration_blockers"])
        self.assertIn("previous_using_phone_not_detected", phones[3]["duration_reset_reasons"])

    def test_using_phone_invalid_timestamp_and_gap_reset_are_diagnostic_only(self):
        observations = [observation(i, t) for i, t in enumerate((100., 101.5, 101., 103., 110.), 1)]
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        rows = self.read_rows()
        self.assertIsNone(rows[2]["using_phone"][0]["using_phone_duration_seconds"])
        self.assertIn("capture_timestamp_invalid", rows[2]["using_phone"][0]["duration_blockers"])
        self.assertIn("previous_capture_timestamp_invalid", rows[3]["using_phone"][0]["duration_reset_reasons"])
        self.assertEqual(rows[4]["using_phone"][0]["using_phone_duration_seconds"], 0.)
        self.assertIn("capture_gap_exceeded", rows[4]["using_phone"][0]["duration_reset_reasons"])

    def test_using_phone_duplicate_student_never_receives_duration(self):
        observations = [observation(1, 100., second_student="SV001")]
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        for row in self.read_rows()[0]["using_phone"]:
            self.assertIsNone(row["student_id"])
            self.assertIsNone(row["using_phone_duration_seconds"])
            self.assertIn("identity_ambiguous_student", row["duration_blockers"])

    def test_writer_to_analyzer_roundtrip_both_behaviors_without_double_counting(self):
        observations = detected_sequence()
        self.run_mode(observations, True)
        data = toolkit.analyze(next(self.output.glob("session_*/frames.jsonl")), "synthetic")
        self.assertEqual(data["ai_frame_count"], 4)
        self.assertEqual(data["person_observations"], 4)
        self.assertEqual(data["behaviors"]["using_phone"]["states"], {"detected": 4})
        self.assertEqual(data["behaviors"]["sleeping"]["states"], {"suspected": 4})
        self.assertTrue(all(p["student_id"] == "SV001" for f in data["frames"] for p in f["people"]))
        self.assertEqual(data["issues"], [])
        self.assertEqual(next(c["status"] for c in toolkit.technical_checks(data, "using_phone", "duration_progress")
                              if c["check"] == "duration_progress"), "PASS")

    def test_behavior_diagnostic_failure_cannot_change_production_results(self):
        baseline = self.run_mode(detected_sequence(), False)
        with patch.object(diagnostic, "behavior_diagnostic", side_effect=RuntimeError("synthetic behavior diagnostic error")):
            shadow = self.run_mode(detected_sequence(), True)
        self.assert_unchanged(baseline, shadow)
        self.assertIn("synthetic behavior diagnostic error", shadow[0].output)
        self.assertFalse(self.output.exists())

    def test_ambiguous_raw_person_ref_does_not_borrow_behavior_confidence(self):
        observations = detected_sequence()[:1]
        result = self.run_mode(observations, False)[0].results[0]
        frame = replace(Scenario(hold=None).confirm(), frame_id=result["frame_id"], captured_at=result["captured_at"])
        original = copy.deepcopy(observations[0].people)
        output = diagnostic.behavior_diagnostic(result["behavior_frame"], original + original, None, 5., frame)
        self.assertEqual(original, observations[0].people)
        for name in ("sleeping", "using_phone"):
            self.assertIsNone(output[name][0][f"{name}_model_confidence"])
            self.assertIsNone(output[name][0][f"{name}_decision_reason"])
        legacy = diagnostic.sleeping_diagnostic(result["behavior_frame"], original, None, 5., frame)
        self.assertNotIn("using_phone", legacy)

    def test_sleep_detected_face_loss_does_not_change_duration_or_panel(self):
        observations = detected_sequence()
        for o in observations:
            o.people[0]["behaviors"]["sleeping"]["state"] = "detected"
        observations[1:] = [replace(o, faces=()) for o in observations[1:]]
        baseline = self.run_mode(observations, False)
        shadow = self.run_mode(observations, True)
        self.assert_unchanged(baseline, shadow)
        for result in shadow[0].results[1:]:
            self.assertIsNone(phone_value(result))
            sleeping = next(b for b in result["behavior_frame"].people[0].behaviors if b.name == "sleeping")
            self.assertEqual(sleeping.state, "detected")
            self.assertIsNone(sleeping.estimated_seconds)
        rows = self.read_rows()
        self.assertEqual(rows[1]["observations"][0]["reason"], "hold_period_unapproved")
        self.assertTrue(all(r["observations"][0]["proposed_student_id"] is None for r in rows[1:]))
        sleeping = rows[1]["sleeping"][0]
        self.assertEqual(sleeping["sleeping_state"], "detected")
        self.assertEqual(sleeping["sleeping_model_confidence"], .8)
        self.assertIsNone(sleeping["student_id"])
        self.assertIsNone(sleeping["sleeping_duration_seconds"])
        self.assertEqual(sleeping["duration_event"], "unavailable")
        self.assertIn("identity_unassociated", sleeping["duration_blockers"])
        self.assertEqual(rows[2]["sleeping"][0]["duration_reset_reasons"], [])

    def test_even_forced_wrong_shadow_proposal_cannot_reach_duration(self):
        baseline = self.run_mode(detected_sequence(), False)
        original = IdentityContinuityResolver.update

        def fake_proposal(resolver, *args, **kwargs):
            frame = original(resolver, *args, **kwargs)
            return replace(frame, observations=tuple(replace(r, proposed_student_id="SHADOW_ONLY_TEST",
                                                            status="track_retained") for r in frame.observations))

        with patch.object(IdentityContinuityResolver, "update", fake_proposal):
            shadow = self.run_mode(detected_sequence(), True)
        self.assert_unchanged(baseline, shadow)
        self.assertEqual(self.read_rows()[0]["observations"][0]["proposed_student_id"], "SHADOW_ONLY_TEST")

    def test_resolver_failure_stops_shadow_only(self):
        baseline = self.run_mode(detected_sequence(), False)
        with patch.object(IdentityContinuityResolver, "update", side_effect=RuntimeError("synthetic failure")):
            shadow = self.run_mode(detected_sequence(), True)
        self.assert_unchanged(baseline, shadow)
        self.assertIn("[IDENTITY SHADOW] stopped", shadow[0].output)

    def test_initialization_failure_is_reported_and_does_not_crash_pipeline(self):
        baseline = self.run_mode(detected_sequence(), False)
        with patch.object(diagnostic, "IdentityContinuityShadow", side_effect=OSError("synthetic init failure")):
            shadow = self.run_mode(detected_sequence(), True)
        self.assert_unchanged(baseline, shadow)
        self.assertIn("initialization failed", shadow[0].output)
        self.assertIn("synthetic init failure", shadow[0].output)
        self.assertIn(str(self.output), shadow[0].output)
        self.assertFalse(self.output.exists())

    def test_diagnostic_write_failure_stops_shadow_only(self):
        baseline = self.run_mode(detected_sequence(), False)
        with patch.object(Path, "mkdir", side_effect=PermissionError("synthetic write failure")):
            shadow = self.run_mode(detected_sequence(), True)
        self.assert_unchanged(baseline, shadow)
        self.assertIn("[IDENTITY SHADOW] stopped", shadow[0].output)

    def test_diagnostic_is_bounded_and_contains_no_biometric_payloads(self):
        s = Scenario(hold=None)
        frame = s.confirm()
        writer = diagnostic.IdentityContinuityShadow(self.output, iou_threshold=.3, max_update_gap=5., max_frames=1)
        self.assertTrue(writer.record(frame))
        self.assertFalse(writer.record(frame))
        payload = json.dumps(self.read_rows())
        self.assertIsNone(self.read_rows()[0]["parameters"]["T_hold"])
        self.assertEqual(self.read_rows()[0]["clock"], "perf_counter")
        for forbidden in ("embedding", "face_bbox", "image", "behaviors"):
            self.assertNotIn(forbidden, payload)
        self.assertEqual(len(list(self.output.glob("session_*/*"))), 1)
        self.assertIsNone(writer.resolver.simulation_hold_seconds)

    def test_opt_in_is_explicit_and_does_not_depend_on_person_debug(self):
        tracker = live.PersonTracker()
        for value in (None, "0", "true", ""):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(diagnostic.SHADOW_ENV, None)
                else:
                    os.environ[diagnostic.SHADOW_ENV] = value
                self.assertIsNone(diagnostic.shadow_from_environment(self.output, tracker, 10))
        os.environ[diagnostic.SHADOW_ENV] = "1"
        writer = diagnostic.shadow_from_environment(self.output, tracker, 10)
        self.assertIsNone(writer.resolver.simulation_hold_seconds)
        self.assertFalse(self.output.exists())

    def test_same_output_root_creates_new_sessions_without_overwrite(self):
        frame = Scenario(hold=None).confirm()
        writers = [diagnostic.IdentityContinuityShadow(self.output, iou_threshold=.3,
                   max_update_gap=5., max_frames=1) for _ in range(2)]
        for writer in writers:
            self.assertTrue(writer.record(frame))
        self.assertNotEqual(writers[0].session_dir, writers[1].session_dir)
        self.assertEqual(len(list(self.output.glob("session_*/frames.jsonl"))), 2)

    def test_enabled_without_ai_frames_reports_waiting_and_zero_written(self):
        trace, duration, warning = self.run_mode([], True)
        self.assertEqual(trace.results, ())
        self.assertEqual((duration, warning), ([], []))
        self.assertIn("enabled; output_dir=", trace.output)
        self.assertIn("waiting for first completed AI result", trace.output)
        self.assertIn("frames_written=0; no diagnostic frames written", trace.output)
        self.assertNotIn("first frame written", trace.output)
        self.assertFalse(self.output.exists())

    def test_sleeping_schema_reads_actual_confidence_duration_and_reason(self):
        observations = detected_sequence()
        for o in observations:
            o.people[0]["behaviors"]["sleeping"].update(state="detected", model_confidence=.91,
                decision_reason="synthetic_sleeping_evidence")
        baseline = self.run_mode(observations, False)
        shadow = self.run_mode(observations, True)
        self.assert_unchanged(baseline, shadow)
        rows = self.read_rows()
        self.assertEqual([r["sleeping"][0]["sleeping_duration_seconds"] for r in rows], [0., 1.5, 3., 4.5])
        self.assertEqual(rows[0]["sleeping"][0]["duration_event"], "started_or_restarted")
        self.assertEqual(rows[1]["sleeping"][0]["duration_event"], "increased")
        for result, record in zip(shadow[0].results, rows):
            self.assertEqual(record["schema"], "identity_continuity_shadow_v1")
            self.assertEqual(record["production_observation_status"], "available")
            self.assertEqual(record["frame_id"], result["frame_id"])
            self.assertEqual(record["captured_at"], result["captured_at"])
            value = record["sleeping"][0]
            self.assertEqual(value["student_id"], "SV001")
            self.assertEqual(value["association_status"], "matched")
            self.assertEqual(value["identity_continuity_status"], "face_confirmed")
            self.assertEqual(value["sleeping_model_confidence"], .91)
            self.assertEqual(value["sleeping_decision_reason"], "synthetic_sleeping_evidence")
            self.assertEqual(value["duration_blockers"], [])
        payload = json.dumps(rows)
        for forbidden in ("embedding", "face_bbox", "image", "behaviors"):
            self.assertNotIn(forbidden, payload)
        self.assertIn("first frame written", shadow[0].output)
        self.assertIn("frames_written=4", shadow[0].output)

    def test_sleeping_missing_confidence_is_not_borrowed_from_face(self):
        observations = detected_sequence()[:1]
        observations[0].people[0]["behaviors"]["sleeping"].pop("model_confidence")
        self.run_mode(observations, True)
        self.assertIsNone(self.read_rows()[0]["sleeping"][0]["sleeping_model_confidence"])

    def test_sleeping_not_detected_break_and_restart_are_logged(self):
        observations = detected_sequence()[:3]
        for o in observations:
            o.people[0]["behaviors"]["sleeping"]["state"] = "detected"
        observations[1].people[0]["behaviors"]["sleeping"]["state"] = "not_detected"
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        rows = self.read_rows()
        self.assertIsNone(rows[1]["sleeping"][0]["sleeping_duration_seconds"])
        self.assertIn("sleeping_not_detected", rows[1]["sleeping"][0]["duration_blockers"])
        self.assertEqual(rows[2]["sleeping"][0]["sleeping_duration_seconds"], 0.)
        self.assertIn("previous_sleeping_not_detected", rows[2]["sleeping"][0]["duration_reset_reasons"])

    def test_capture_gap_reset_is_logged_without_changing_tracker(self):
        observations = [observation(i, t) for i, t in enumerate((100., 101.5, 110.), 1)]
        for o in observations:
            o.people[0]["behaviors"]["sleeping"]["state"] = "detected"
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        record = self.read_rows()[-1]
        self.assertEqual(record["duration_max_gap"], 5.)
        self.assertEqual(record["duration_capture_gap_seconds"], 8.5)
        self.assertEqual(record["sleeping"][0]["sleeping_duration_seconds"], 0.)
        self.assertIn("capture_gap_exceeded", record["sleeping"][0]["duration_reset_reasons"])

    def test_student_change_and_missing_person_are_logged_from_production(self):
        observations = [observation(1, 100.), observation(2, 101.5, student="SV002"),
                        observation(3, 103., present=False), observation(4, 104.5, student="SV002")]
        for o in observations:
            for p in o.people:
                p["behaviors"]["sleeping"]["state"] = "detected"
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        rows = self.read_rows()
        self.assertEqual(rows[1]["sleeping"][0]["student_id"], "SV002")
        self.assertIn("reported_student_id_changed", rows[1]["sleeping"][0]["duration_reset_reasons"])
        self.assertEqual(rows[2]["sleeping"], [])
        self.assertEqual(rows[2]["duration_missing_person_track_ids"], ["P1"])
        self.assertEqual(rows[3]["sleeping"][0]["sleeping_duration_seconds"], 0.)
        self.assertIn("new_person_track", rows[3]["sleeping"][0]["duration_reset_reasons"])

    def test_duplicate_student_diagnostic_does_not_assign_a_student(self):
        observations = [observation(1, 100., second_student="SV001")]
        for p in observations[0].people:
            p["behaviors"]["sleeping"]["state"] = "detected"
        self.assert_unchanged(self.run_mode(observations, False), self.run_mode(observations, True))
        for value in self.read_rows()[0]["sleeping"]:
            self.assertIsNone(value["student_id"])
            self.assertIsNone(value["sleeping_duration_seconds"])
            self.assertIn("identity_ambiguous_student", value["duration_blockers"])

    def test_custom_relative_directory_is_reported_as_absolute_without_creating_it(self):
        os.environ[diagnostic.SHADOW_ENV] = "1"
        os.environ[diagnostic.SHADOW_DIR_ENV] = "relative_shadow_diagnostic_test"
        log = io.StringIO()
        with redirect_stderr(log):
            writer = diagnostic.shadow_from_environment(live.ROOT, live.PersonTracker(), 10)
            writer.finish()
        self.assertEqual(writer.output_dir, Path.cwd() / "relative_shadow_diagnostic_test")
        self.assertIn(str(writer.output_dir), log.getvalue())
        self.assertFalse(writer.output_dir.exists())

    def test_byte_limit_is_reported_without_claiming_a_written_frame(self):
        log = io.StringIO()
        with redirect_stderr(log):
            writer = diagnostic.IdentityContinuityShadow(self.output, iou_threshold=.3,
                                                         max_update_gap=5., max_frames=10)
            with patch.object(diagnostic, "MAX_BYTES", 1):
                self.assertFalse(writer.record(Scenario(hold=None).confirm()))
            writer.finish()
        self.assertIn("byte limit reached", log.getvalue())
        self.assertIn("frames_written=0", log.getvalue())
        self.assertFalse(self.output.exists())

    def test_metadata_mismatch_stops_only_diagnostic(self):
        observations = detected_sequence()[:1]
        baseline = self.run_mode(observations, False)
        result = baseline[0].results[0]
        frame = Scenario(hold=None).confirm()
        log = io.StringIO()
        with redirect_stderr(log):
            writer = diagnostic.IdentityContinuityShadow(self.output, iou_threshold=.3,
                                                         max_update_gap=5., max_frames=10)
            self.assertFalse(writer.record(frame, result=result, raw_people=observations[0].people,
                                           duration_max_gap=5.))
            writer.finish()
        self.assertIn("metadata mismatch", log.getvalue())
        self.assertFalse(self.output.exists())

    def test_real_session_replay_is_explicitly_partial_and_does_not_assign_p3(self):
        path = live.ROOT / "ai_training/runs/person_tracker_debug/session_808dd75fae434122b1d0a03c44fa819c/frames.jsonl"
        records = [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]
        self.assertTrue(all("face_tracker" not in r for r in records))
        report = replay_logged_observations(records)
        self.assertFalse(report["full_identity_replay"])
        self.assertFalse(report["association_geometry_recomputed"])
        self.assertEqual((report["frame_count"], report["person_tracker_consistent_frames"]), (10, 10))
        p3 = [o for frame in report["frames"] for o in frame["observations"] if o["person_track_id"] == "P3"]
        self.assertEqual(len(p3), 9)
        self.assertTrue(all(o["proposed_student_id"] is None and o["face_present"] is None for o in p3))


if __name__ == "__main__":
    unittest.main()
