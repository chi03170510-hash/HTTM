"""Diagnostic B: golden camera replay + explicit synthetic identity failures."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

from evaluate_face_tracker_parameters import (
    DEFAULT_SESSION, MAX_AGES, FrameInput, FaceInput, evaluate_session, input_digest,
    load_camera_session, main, risk_scenarios, run_trial,
)
import face_tracker


class TestFaceTrackerParameterEvaluation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source_hash = hashlib.sha256(DEFAULT_SESSION.read_bytes()).hexdigest()
        cls.frames, cls.original_records, _ = load_camera_session(DEFAULT_SESSION)
        cls.report = evaluate_session(DEFAULT_SESSION, benchmark_repeats=0)
        cls.cases = risk_scenarios()

    def risks(self, name):
        return self.report["risk_trials"][name]

    def test_camera_baseline_matches_all_recorded_face_states_and_associations(self):
        self.assertEqual(self.report["camera_baseline_replay_consistent_frames"], 10)
        self.assertEqual(self.report["camera_baseline_association_equal_frames"], 10)
        self.assertEqual([r["face_track_ids"] for r in self.report["camera_trials"][0]["trace"]],
                         [[1], [1], [2], [3], [4], [5], [6], [6], [7], [7]])

    def test_camera_candidates_remove_six_observed_id_changes(self):
        trials = self.report["camera_trials"]
        self.assertEqual([t["ids_created"] for t in trials], [7, 1, 1])
        self.assertEqual([t["reference_id_changes"] for t in trials], [6, 0, 0])
        self.assertEqual([t["expiry_count"] for t in trials], [6, 0, 0])
        self.assertEqual([t["retained_reference_observations"] for t in trials], [3, 9, 9])
        self.assertTrue(all(t["reference_transitions"] == 9 for t in trials))
        self.assertTrue(all(r["face_track_ids"] == [1] for t in trials[1:] for r in t["trace"]))

    def test_camera_iou_metrics_and_unchanged_identity_binding(self):
        trials = self.report["camera_trials"]
        self.assertEqual([t["matched_observations"] for t in trials], [3, 9, 9])
        self.assertAlmostEqual(trials[0]["matched_iou_mean"], .7561363984)
        self.assertAlmostEqual(trials[1]["matched_iou_mean"], .8405732413)
        self.assertTrue(all(t["association_binding_frames_equal_baseline"] == 10 for t in trials))
        self.assertTrue(all(t["matched_iou_min"] > .3 for t in trials))

    def test_camera_is_not_treated_as_ground_truth_identity_validation(self):
        for trial in self.report["camera_trials"]:
            self.assertIsNone(trial["wrong_actor_continuations"])
            self.assertFalse(trial["identity_metrics_complete"])
            self.assertEqual(trial["reference_observations"], 10)

    def test_all_three_ages_receive_identical_immutable_inputs_and_source_is_unchanged(self):
        old = copy.deepcopy(self.original_records)
        digest = input_digest(self.frames)
        self.assertEqual({t["input_sha256"] for t in self.report["camera_trials"]}, {digest})
        for age in MAX_AGES:
            run_trial(self.frames, age)
        self.assertEqual(old, self.original_records)
        self.assertEqual(hashlib.sha256(DEFAULT_SESSION.read_bytes()).hexdigest(), self.source_hash)
        with self.assertRaises(FrozenInstanceError):
            self.frames[0].faces[0].student_id = "MODIFIED"
        self.assertEqual(self.frames[0].faces[0].score, .4915679693222046)

    def test_stationary_actor_all_ages_keep_one_track(self):
        self.assertEqual([r["ids_created"] for r in self.risks("stationary")], [1, 1, 1])
        self.assertEqual([r["wrong_actor_continuations"] for r in self.risks("stationary")], [0, 0, 0])

    def test_head_turn_bbox_jump_is_not_fixed_by_more_retention(self):
        self.assertEqual([r["reference_id_changes"] for r in self.risks("head_turn_bbox_jump")], [2, 2, 2])
        self.assertEqual([r["ids_created"] for r in self.risks("head_turn_bbox_jump")], [3, 2, 2])

    def test_two_near_stationary_people_benefit_without_merging_in_this_fixture(self):
        rows = self.risks("two_near")
        self.assertEqual([r["ids_created"] for r in rows], [12, 2, 2])
        self.assertEqual([r["reference_id_changes"] for r in rows], [10, 0, 0])
        self.assertEqual([r["wrong_actor_continuations"] for r in rows], [0, 0, 0])

    def test_swapping_positions_has_more_wrong_continuations_at_longer_age(self):
        rows = self.risks("swap_positions")
        self.assertEqual([r["reference_id_changes"] for r in rows], [2, 2, 2])
        self.assertEqual([r["wrong_actor_continuations"] for r in rows], [0, 2, 2])

    def test_crossing_can_switch_actors_at_every_age(self):
        rows = self.risks("crossing")
        self.assertEqual([r["reference_id_changes"] for r in rows], [2, 2, 2])
        self.assertEqual([r["wrong_actor_continuations"] for r in rows], [2, 2, 2])

    def test_same_position_replacement_is_invisible_to_iou_even_with_correct_new_student_id(self):
        self.assertEqual([r["wrong_actor_continuations"] for r in self.risks("replacement_1_5")], [1, 1, 1])
        self.assertEqual([r["wrong_actor_continuations"] for r in self.risks("replacement_3_0")], [0, 1, 1])
        # ID-change count alone misses this failure: the two actors differ.
        self.assertEqual([r["reference_id_changes"] for r in self.risks("replacement_3_0")], [0, 0, 0])

    def test_five_seconds_adds_replacement_risk_without_camera_benefit_over_four(self):
        self.assertEqual([r["wrong_actor_continuations"] for r in self.risks("replacement_4_5")], [0, 0, 1])
        self.assertEqual(self.report["camera_trials"][1]["reference_id_changes"], self.report["camera_trials"][2]["reference_id_changes"])

    def test_one_and_multiple_missed_frames_show_recovery_tradeoff(self):
        self.assertEqual([r["reference_id_changes"] for r in self.risks("missing_one")], [1, 0, 0])
        self.assertEqual([r["reference_id_changes"] for r in self.risks("missing_many")], [1, 1, 0])

    def test_unknown_recognition_does_not_change_tracking_affinity(self):
        rows = self.risks("temporary_unknown")
        self.assertEqual([r["ids_created"] for r in rows], [1, 1, 1])
        self.assertEqual([r["wrong_actor_continuations"] for r in rows], [0, 0, 0])

    def test_reordered_distinct_boxes_preserve_tracks_but_full_overlap_does_not(self):
        self.assertEqual([r["reference_id_changes"] for r in self.risks("reordered_detections")], [0, 0, 0])
        self.assertEqual([r["wrong_actor_continuations"] for r in self.risks("overlapping_reordered")], [2, 2, 2])

    def test_expiry_boundaries_two_four_five_are_strictly_greater(self):
        expected = {"2.000": [1, 1, 1], "2.001": [2, 1, 1], "4.000": [2, 1, 1],
                    "4.001": [2, 2, 1], "5.000": [2, 2, 1], "5.001": [2, 2, 2]}
        for gap, ids in expected.items():
            with self.subTest(gap=gap):
                self.assertEqual([r["ids_created"] for r in self.risks("gap_" + gap)], ids)

    def test_person_stable_face_changes_and_student_label_is_not_actor_ground_truth(self):
        self.assertEqual([r["reference_id_changes"] for r in self.risks("person_stable_face_changes")], [3, 0, 0])
        rows = self.risks("student_label_change")
        self.assertEqual([r["ids_created"] for r in rows], [1, 1, 1])
        self.assertEqual([r["wrong_actor_continuations"] for r in rows], [0, 0, 0])

    def test_close_scores_and_reversed_score_ranking_do_not_change_geometry_matching(self):
        frames = self.cases["close_recognition_scores"]
        changed = tuple(replace(f, faces=tuple(replace(d, score=1.600003 - d.score) for d in f.faces)) for f in frames)
        for age in MAX_AGES:
            normal, other = run_trial(frames, age), run_trial(changed, age)
            self.assertEqual([r["detection_track_ids"] for r in normal["trace"]], [r["detection_track_ids"] for r in other["trace"]])
            self.assertEqual(normal["wrong_actor_continuations"], 0)

    def test_ambiguous_identical_observations_are_not_scored_as_identity_success(self):
        a = FaceInput((20, 20, 120, 140), "Unknown", .4, "actor:A", "A")
        b = replace(a, reference="actor:B", actor="B")
        frames = (FrameInput(1, 100., 99.5, (a, b)), FrameInput(2, 101.5, 101., (a, b)))
        for age in MAX_AGES:
            trial = run_trial(frames, age)
            self.assertEqual(trial["ambiguous_assignment_count"], 4)
            self.assertIsNone(trial["wrong_actor_continuations"])
            self.assertFalse(trial["identity_metrics_complete"])

    def test_iou_pair_count_matches_calls_to_actual_tracker_function(self):
        original = face_tracker.calculate_iou
        for age, expected in ((2., 3), (4., 9), (5., 9)):
            with self.subTest(age=age), patch.object(face_tracker, "calculate_iou", wraps=original) as iou:
                trial = run_trial(self.frames, age)
                self.assertEqual(iou.call_count, expected)
                self.assertEqual(trial["iou_pairs_considered"], expected)

    def test_cli_refuses_overwriting_source_or_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / "existing.json"
            existing.write_text("USER DATA")
            for output in (DEFAULT_SESSION, existing):
                argv = ["evaluate", str(DEFAULT_SESSION), "--output", str(output), "--benchmark-repeats", "0"]
                with (self.subTest(output=output), patch.object(sys, "argv", argv),
                      redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error):
                    main()
                self.assertEqual(error.exception.code, 2)
            self.assertEqual(existing.read_text(), "USER DATA")
        self.assertEqual(hashlib.sha256(DEFAULT_SESSION.read_bytes()).hexdigest(), self.source_hash)


if __name__ == "__main__":
    unittest.main()
