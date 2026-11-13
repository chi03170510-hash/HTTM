"""Phase 2 synthetic safety checks. FAIL findings are expected counterexamples.

The sole simulated hold below is arbitrary test data (boundary probing), NOT an
estimate from camera logs, a recommended period, or production configuration.
"""

import copy
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from dataclasses import asdict, replace
from pathlib import Path
from unittest.mock import patch

import evaluate_identity_continuity_safety as evaluation
from identity_continuity_safety_fixtures import Detection, SafetyFixture, frame, safety_fixtures


UNIT_TEST_HOLD_SECONDS = 2.0  # TEST ONLY, deliberately not exported to any live configuration.


class TestIdentityContinuitySafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.baseline = evaluation.evaluate_suite()
        cls.simulation = evaluation.evaluate_suite(simulation_hold_seconds=UNIT_TEST_HOLD_SECONDS)

    def case(self, name, *, simulation=True):
        report = self.simulation if simulation else self.baseline
        return next(c for c in report["cases"] if c["case_id"] == name)

    def row(self, name, index=-1, tid="P1", *, simulation=True):
        return next(r for r in self.case(name, simulation=simulation)["trace"][index]["observations"]
                    if r["person_track_id"] == tid)

    def test_fixtures_cover_all_requested_risks_with_explicit_synthetic_provenance(self):
        self.assertEqual(len(safety_fixtures()), 22)
        self.assertEqual(len({f.name for f in safety_fixtures()}), 22)
        self.assertTrue(all(f.synthetic for f in safety_fixtures()))
        self.assertIsNone(self.baseline["production_T_hold"])
        self.assertFalse(self.simulation["production_applied"])

    def test_current_tracker_settings_are_read_from_production_source(self):
        self.assertEqual(evaluation.production_tracker_policy(),
                         {"iou_threshold": .3, "max_age": 5., "keep_unmatched": True,
                          "motion_matching": True})

    def test_ground_truth_is_not_part_of_resolver_inputs(self):
        normal = next(f for f in safety_fixtures() if f.name == "same_track_face_absent")
        replaced = next(f for f in safety_fixtures() if f.name == "same_slot_unobserved_replacement")
        for a, b in zip(normal.frames, replaced.frames):
            self.assertEqual(evaluation.observation_inputs(a), evaluation.observation_inputs(b))
        self.assertNotEqual(normal.frames[1].actor_labels, replaced.frames[1].actor_labels)
        with patch.object(evaluation, "project_current", wraps=evaluation.project_current) as project:
            evaluation.evaluate_fixture(normal)
        for call in project.call_args_list:
            self.assertNotIn("actor", json.dumps(call.args))

    def test_same_observations_can_have_different_identity_safety(self):
        correct = self.case("same_track_face_absent")
        wrong = self.case("same_slot_unobserved_replacement")
        self.assertEqual(correct["safety_status"], "PASS")
        self.assertEqual(wrong["safety_status"], "FAIL")
        self.assertEqual(correct["metrics"]["correct_retentions"], 2)
        self.assertEqual(wrong["metrics"]["false_identity_retentions"], 2)
        self.assertEqual(self.row("same_slot_unobserved_replacement")["status"], "track_retained")

    def test_default_no_hold_never_retains_sid(self):
        self.assertEqual(self.baseline["metrics"]["correct_retentions"], 0)
        self.assertEqual(self.baseline["metrics"]["false_identity_retentions"], 0)
        self.assertIsNone(self.row("same_track_face_absent", index=1, simulation=False)["proposed_student_id"])
        self.assertEqual(self.row("same_track_face_absent", index=1, simulation=False)["reason"], "hold_period_unapproved")

    def test_new_track_without_new_face_does_not_inherit(self):
        row = self.row("new_track_face_absent", tid="P2")
        self.assertIsNone(row["proposed_student_id"])
        self.assertEqual(row["reason"], "no_valid_same_track_anchor")

    def test_unobserved_swap_is_false_retention_despite_stable_pids(self):
        case = self.case("swap_positions_unobserved")
        self.assertEqual(case["safety_status"], "FAIL")
        self.assertEqual(case["metrics"]["false_identity_retentions"], 2)
        self.assertEqual({r["person_track_id"] for r in case["trace"][0]["observations"]}, {"P1", "P2"})
        self.assertEqual({r["person_track_id"] for r in case["trace"][1]["observations"]}, {"P1", "P2"})

    def test_overlap_is_rejected_and_does_not_resume_after_swap(self):
        for name in ("overlapping_people", "overlap_then_swap"):
            with self.subTest(name=name):
                self.assertEqual(self.case(name)["safety_status"], "PASS")
                self.assertTrue(all(r["proposed_student_id"] is None for f in self.case(name)["trace"][1:]
                                    for r in f["observations"]))

    def test_short_absence_may_reuse_pid_but_does_not_authorize_identity_hold(self):
        case = self.case("observed_exit_replacement")
        self.assertEqual(case["metrics"]["false_identity_retentions"], 0)
        self.assertEqual(case["trace"][-1]["observations"][0]["person_track_id"], "P1")
        self.assertIsNone(self.row("observed_exit_replacement")["proposed_student_id"])

    def test_fresh_correct_different_sid_wins_over_cached_sid(self):
        fresh = self.row("fresh_face_changed_actor", index=1)
        self.assertEqual((fresh["proposed_student_id"], fresh["status"]), ("SV002", "face_confirmed"))
        self.assertEqual(self.case("fresh_face_changed_actor")["safety_status"], "PASS")

    def test_wrong_fresh_recognition_is_not_identity_ground_truth(self):
        for simulation in (False, True):
            with self.subTest(simulation=simulation):
                case = self.case("fresh_face_misrecognition", simulation=simulation)
                self.assertEqual(case["safety_status"], "FAIL")
                self.assertEqual(case["metrics"]["false_fresh_face_proposals"], 1)
        self.assertEqual(self.case("fresh_face_misrecognition")["metrics"]["false_identity_retentions"], 1)
        self.assertEqual(self.row("fresh_face_misrecognition", index=1)["ground_truth_student_id"], "SV001")

    def test_duplicate_sid_claims_never_create_duplicate_proposals(self):
        case = self.case("duplicate_recognized_sid")
        self.assertEqual(case["safety_status"], "PASS")
        self.assertEqual(case["metrics"]["duplicate_sid_frames"], 0)
        self.assertTrue(all(r["proposed_student_id"] is None for f in case["trace"] for r in f["observations"]))

    def test_nonincreasing_capture_cancels_retention(self):
        row = self.row("capture_timestamp_not_increasing")
        self.assertIsNone(row["proposed_student_id"])
        self.assertFalse(self.case("capture_timestamp_not_increasing")["trace"][-1]["timestamp_valid"])

    def test_capture_gap_and_tracker_expiry_are_distinct(self):
        capture = self.case("capture_gap_exceeded")["trace"][-1]
        expired = self.case("track_expired_recreated")["trace"][-1]
        self.assertEqual(capture["reset_reason"], "capture_gap_exceeds_existing_limit")
        self.assertIsNone(expired["reset_reason"])
        self.assertEqual(self.row("capture_gap_exceeded")["track_continuity"], "same_track")
        self.assertEqual(self.row("track_expired_recreated")["track_continuity"], "expired")
        self.assertIsNone(self.row("track_expired_recreated", tid="P2")["proposed_student_id"])

    def test_face_unknown_absent_and_unavailable_are_not_conflated(self):
        unknown = self.row("face_unknown", index=1)
        unavailable = self.row("face_data_unavailable", index=1)
        absent = self.row("same_track_face_absent", index=1)
        self.assertEqual((unknown["face_present"], unknown["reason"]), (True, "current_face_unknown"))
        self.assertEqual((unavailable["face_present"], unavailable["reason"]), (None, "current_face_data_unavailable"))
        self.assertEqual((absent["face_present"], absent["status"]), (False, "track_retained"))
        for name in ("face_unknown", "face_data_unavailable", "face_without_association", "heuristic_cannot_seed"):
            self.assertTrue(all(r["proposed_student_id"] is None for f in self.case(name)["trace"][1:]
                                for r in f["observations"]))

    def test_fresh_face_can_recover_new_pid_without_cache_transfer(self):
        row = self.row("fresh_face_recovers_new_track", index=1, tid="P2")
        self.assertEqual((row["status"], row["proposed_student_id"]), ("face_recovered", "SV001"))
        self.assertEqual(self.simulation["metrics"]["wrong_track_retention_transfers"], 0)

    def test_bbox_does_not_extend_unit_test_hold(self):
        case = self.case("hold_not_refreshed_by_bbox")
        self.assertEqual(case["trace"][-2]["observations"][0]["status"], "track_retained")
        self.assertEqual(case["trace"][-1]["observations"][0]["reason"], "identity_hold_expired")
        self.assertAlmostEqual(case["trace"][-1]["observations"][0]["last_face_age_seconds"], 2.001)

    def test_unknown_actor_truth_is_inconclusive_even_with_retained_sid(self):
        for simulation in (False, True):
            case = self.case("actor_truth_unavailable", simulation=simulation)
            self.assertEqual(case["safety_status"], "INCONCLUSIVE")
            self.assertEqual(case["metrics"]["unscorable_observations"], 1)

    def test_face_recovered_can_be_wrong_actor_when_recognition_is_wrong(self):
        row = self.row("fresh_face_wrong_actor_new_track", index=1, tid="P2")
        self.assertEqual(row["status"], "face_recovered")
        self.assertTrue(row["wrong_proposal"])
        self.assertEqual(row["ground_truth_actor"], "actor_b")
        self.assertEqual(self.case("fresh_face_wrong_actor_new_track")["metrics"]["false_identity_retentions"], 1)

    def test_current_frame_reference_reorder_cannot_swap_actor_truth(self):
        case = self.case("detection_order_reversed")
        self.assertEqual(case["safety_status"], "PASS")
        self.assertEqual(case["metrics"]["correct_retentions"], 2)
        for row in case["trace"][-1]["observations"]:
            self.assertFalse(row["wrong_proposal"])

    def test_false_findings_are_not_hidden_by_passing_test_harness(self):
        self.assertEqual(self.baseline["counts"], {"PASS": 19, "FAIL": 2, "INCONCLUSIVE": 1})
        self.assertEqual(self.simulation["counts"], {"PASS": 17, "FAIL": 4, "INCONCLUSIVE": 1})
        self.assertEqual(self.simulation["metrics"]["false_identity_retentions"], 6)
        self.assertEqual(self.simulation["metrics"]["false_fresh_face_proposals"], 2)
        self.assertEqual(self.simulation["metrics"]["wrong_identity_proposals"], 8)

    def test_fixtures_and_actor_labels_are_immutable_and_not_modified(self):
        fixtures = safety_fixtures()
        before = copy.deepcopy(fixtures)
        for fixture in fixtures:
            evaluation.evaluate_fixture(fixture, simulation_hold_seconds=UNIT_TEST_HOLD_SECONDS)
        self.assertEqual(fixtures, before)

    def test_invalid_fixture_label_or_missing_detection_mapping_is_rejected(self):
        original = safety_fixtures()[0]
        bad = [replace(original, synthetic=False), replace(original, frames=()),
               replace(original, frames=(replace(original.frames[0], actor_labels=()),)),
               replace(original, frames=(replace(original.frames[0], actor_labels=("SV001",)),)),
               replace(original, frames=(frame(1, 0., [Detection(bbox=(0., 0., 0., 0.))], ["actor_a"]),))]
        for fixture in bad:
            with self.subTest(fixture=fixture), self.assertRaises(ValueError):
                evaluation.evaluate_fixture(fixture)

    def test_reports_preserve_failures_and_never_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            evaluation.write_report(self.simulation, output)
            report = json.loads((output / "safety.json").read_text(encoding="utf-8"))
            self.assertEqual(report["counts"]["FAIL"], 4)
            self.assertIn("same_slot_unobserved_replacement | FAIL", (output / "safety.md").read_text(encoding="utf-8"))
            self.assertIn("false_identity_retentions", (output / "safety.csv").read_text(encoding="utf-8-sig"))
            with self.assertRaises(FileExistsError):
                evaluation.write_report(self.simulation, output)

    def test_cli_no_hold_reports_known_false_face_claim_and_has_no_simulation_switch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report"
            with redirect_stdout(io.StringIO()):
                code = evaluation.main(["--output-dir", str(output)])
            self.assertEqual(code, 1, "A labelled unsafe fresh-face proposal must not be hidden by exit 0")
            data = json.loads((output / "safety.json").read_text(encoding="utf-8"))
            self.assertIsNone(data["simulation_hold_seconds"])
            self.assertIsNone(data["production_T_hold"])

    def test_cli_runs_without_camera_or_inference_dependencies(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-S", "-B", evaluation.__file__,
                                     "--output-dir", str(Path(directory) / "new")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('"FAIL": 2', result.stdout)

    def test_recorded_timing_alone_cannot_choose_hold(self):
        data = evaluation.audit_recorded_sessions()
        self.assertGreater(data["session_count"], 0)
        self.assertGreater(data["frame_count"], 0)
        self.assertEqual(data["T_hold_selection"], "INCONCLUSIVE")
        self.assertIsNone(data["production_T_hold"])
        self.assertTrue(all(not s["independent_actor_truth_supplied"] for s in data["sessions"]))


if __name__ == "__main__":
    unittest.main()
