"""Diagnostic C ground-truth separation, geometry, ambiguity and source safety."""

import copy
import hashlib
import io
import json
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from unittest.mock import patch

import evaluate_face_tracker_matching_c as c
from evaluate_face_tracker_parameters import risk_scenarios, run_trial


class TestFaceTrackerMatchingC(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames, cls.records, cls.source_hash, cls.age = c.load_session(c.DEFAULT_SESSION)
        cls.report = c.evaluate_session()
        cls.cases = c.risk_scenarios_c()

    def risks(self, name):
        return self.report["risk_trials"][name]

    def test_source_four_second_session_replays_exactly_without_synthesizing_actor_truth(self):
        self.assertEqual((len(self.frames), self.age), (10, 4.))
        self.assertEqual(self.report["baseline_consistent_frames"], 10)
        self.assertEqual(self.report["baseline_verified_mapping_frames"], 10)
        self.assertTrue(all(d.actor is None for f in self.frames for d in f.faces))
        self.assertEqual(hashlib.sha256(c.DEFAULT_SESSION.read_bytes()).hexdigest(), self.source_hash)

    def test_metadata_order_and_both_time_domains_are_preserved_in_every_trial(self):
        digests = set()
        for trial in self.report["camera_trials"]:
            digests.add(trial["input_sha256"])
            self.assertEqual(trial["max_age"], 4.)
            for frame, row, record in zip(self.frames, trial["trace"], self.records):
                block = record["face_tracker"]
                self.assertEqual((row["frame_id"], row["captured_at"], row["updated_at"]),
                                 (record["frame_id"], record["captured_at"], block["face_tracker_updated_at"]))
                self.assertEqual([(d.bbox, d.student_id, d.score) for d in frame.faces],
                                 [(tuple(d["face_bbox"]), d["current_student_id"], d["recognition_score"])
                                  for d in block["detections"]])
                self.assertEqual(len(row["detection_track_ids"]), len(frame.faces))
                self.assertEqual(row["input_boxes"], [list(d.bbox) for d in frame.faces])
                self.assertEqual(row["actor_labels"], [None] * len(frame.faces))
        self.assertEqual(digests, {c.input_digest(self.frames)})

    def test_unlabeled_camera_identity_metrics_are_unknown_not_false_zeroes(self):
        for trial in self.report["camera_trials"]:
            self.assertIsNone(trial["id_changes"])
            self.assertIsNone(trial["false_continuations"])
            self.assertFalse(trial["identity_metrics_complete"])
            self.assertEqual(trial["labeled_observations"], 0)

    def test_camera_baseline_birth_expiry_and_motion_recovery_are_separate(self):
        trials = self.report["camera_trials"]
        self.assertEqual([t["ids_created"] for t in trials], [5, 4, 4, 4])
        self.assertEqual([t["expiry_count"] for t in trials], [4, 3, 3, 3])
        self.assertEqual([t["matched_links"] for t in trials], [8, 9, 9, 9])
        self.assertEqual([t["trace"][2]["detection_track_ids"] for t in trials], [[3, 2], [1, 2], [1, 2], [1, 2]])
        # A relaxed-IoU control achieves the same counts; no unique geometry gain.
        self.assertEqual(trials[1]["trace"][2]["detection_track_ids"], trials[2]["trace"][2]["detection_track_ids"])

    def test_frame_ten_missed_observation_expires_before_even_relaxed_matching(self):
        for trial in self.report["camera_trials"]:
            self.assertEqual(trial["trace"][8]["detection_track_ids"], [])
            last = trial["trace"][-1]
            self.assertEqual(len(last["expired_ids"]), 1)
            self.assertEqual(len(last["new_ids"]), 1)
            self.assertAlmostEqual(last["pairs"][0]["iou"], .2876245742399394)
            self.assertIn("expired_before_matching", last["pairs"][0]["reasons"])
            self.assertFalse(last["pairs"][0]["selected"])

    def test_real_iou_baseline_matches_diagnostic_b_on_all_synthetic_sequences(self):
        for name, original in risk_scenarios().items():
            with self.subTest(name=name):
                expected = run_trial(original, 4.)
                actual = self.risks(name)[0]
                self.assertEqual(actual["ids_created"], expected["ids_created"])
                self.assertEqual(actual["expiry_count"], expected["expiry_count"])
                self.assertEqual(actual["id_changes"], expected["reference_id_changes"])
                self.assertEqual(actual["false_continuations"], expected["wrong_actor_continuations"])
                self.assertEqual([r["detection_track_ids"] for r in actual["trace"]],
                                 [r["detection_track_ids"] for r in expected["trace"]])

    def test_stationary_and_distinct_reordered_faces_stay_separate(self):
        for name in ("stationary", "two_near", "reordered_detections"):
            for trial in self.risks(name):
                with self.subTest(name=name, method=trial["method"]):
                    self.assertEqual(trial["id_changes"], 0)
                    self.assertEqual(trial["false_continuations"], 0)
                    self.assertTrue(trial["identity_metrics_complete"])

    def test_crossing_and_position_swap_remain_identity_counterexamples(self):
        for name in ("crossing", "swap_positions"):
            for trial in self.risks(name):
                with self.subTest(name=name, method=trial["method"]):
                    self.assertEqual(trial["id_changes"], 2)
                    self.assertEqual(trial["false_continuations"], 2)

    def test_same_position_replacement_cannot_be_identified_from_geometry(self):
        for name in ("replacement_1_5", "replacement_3_0", "replacement_with_observed_absence"):
            for trial in self.risks(name):
                with self.subTest(name=name, method=trial["method"]):
                    self.assertEqual(trial["false_continuations"], 1)
                    self.assertEqual(trial["ids_created"], 1)
        self.assertTrue(all(t["false_continuations"] == 0 for t in self.risks("replacement_4_5")))

    def test_overlapping_reordered_is_rejected_conservatively_with_fragmentation_cost(self):
        trials = self.risks("overlapping_reordered")
        self.assertEqual([t["false_continuations"] for t in trials], [2, 2, 2, 0])
        self.assertEqual([t["ids_created"] for t in trials], [2, 2, 2, 4])
        self.assertEqual(trials[-1]["ambiguity_rejections"], 2)
        self.assertEqual(trials[-1]["id_changes"], 2)

    def test_equal_boxes_and_equal_recognition_metadata_still_have_exact_baseline_index_audit(self):
        frames = self.cases["overlapping_reordered"]
        identical = tuple(replace(f, faces=tuple(replace(d, student_id="Unknown", score=.5) for d in f.faces)) for f in frames)
        actual = c.run_matching_trial(identical, "iou_only", 4.)
        self.assertEqual(actual["trace"][1]["detection_track_ids"], [1, 2])
        self.assertEqual(actual["false_continuations"], 2)
        self.assertTrue(actual["identity_metrics_complete"])

    def test_student_scores_and_person_references_do_not_control_matching_or_actor_truth(self):
        for name, frames in self.cases.items():
            changed = tuple(replace(f, faces=tuple(replace(d, student_id="Same label for everyone", score=.01)
                                                  for d in f.faces)) for f in frames)
            for method in c.METHODS:
                with self.subTest(name=name, method=method):
                    original, alternate = c.run_matching_trial(frames, method, 4.), c.run_matching_trial(changed, method, 4.)
                    self.assertEqual(original["trace"], alternate["trace"])
                    self.assertEqual(original["false_continuations"], alternate["false_continuations"])
        # Person associations may be wrong; they are not part of Diagnostic C inputs.
        records = copy.deepcopy(self.records)
        for record in records:
            for mapping in record["face_tracker"]["mapping"]:
                for association in mapping["associations"]:
                    association.update(student_id="SV999", person_track_id="P999", person_ref="wrong")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "input.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
            loaded, _, _, _ = c.load_session(path)
        self.assertEqual(loaded, self.frames)

    def test_geometry_features_are_translation_and_scale_invariant(self):
        a, b = (0, 0, 100, 120), (20, 10, 110, 130)
        original = c.geometry_features(a, b)
        translated = c.geometry_features(tuple(v + 45 for v in a), tuple(v + 45 for v in b))
        scaled = c.geometry_features(tuple(v * 3 for v in a), tuple(v * 3 for v in b))
        for key in original:
            self.assertAlmostEqual(original[key], translated[key])
            self.assertAlmostEqual(original[key], scaled[key])

    def test_size_and_center_gates_can_fragment_a_correct_actor(self):
        size, center = self.risks("size_jump_same_actor"), self.risks("center_jump_same_actor")
        self.assertEqual([t["id_changes"] for t in size], [0, 0, 1, 1])
        self.assertEqual([t["id_changes"] for t in center], [1, 0, 1, 1])
        self.assertIn("size_inconsistent", size[2]["trace"][1]["pairs"][0]["reasons"])
        self.assertIn("center_too_far", center[2]["trace"][1]["pairs"][0]["reasons"])

    def test_ambiguity_refusal_never_falls_back_to_second_choice(self):
        rows = [{"track_id": tid, "detection_index": index, "score": score,
                 "eligible": True, "selected": False, "reasons": []}
                for tid, index, score in ((1, 0, .9), (1, 1, .85), (2, 1, .7))]
        selected, rejected = c.select_geometry_pairs(rows, True, .15)
        self.assertEqual(selected, [])
        self.assertEqual(rejected, [0, 1])
        self.assertIn("not_mutual_best", rows[2]["reasons"])

    def test_all_expiry_boundaries_share_the_original_four_second_lifetime(self):
        for gap, ids in (("2.000", 1), ("2.001", 1), ("4.000", 1), ("4.001", 2), ("5.000", 2), ("5.001", 2)):
            for trial in self.risks("gap_" + gap):
                with self.subTest(gap=gap, method=trial["method"]):
                    self.assertEqual(trial["ids_created"], ids)
                    self.assertEqual(trial["expiry_count"], ids - 1)

    def test_inputs_are_immutable_and_repeated_evaluation_is_deterministic(self):
        before = c.input_digest(self.frames)
        with self.assertRaises(FrozenInstanceError):
            self.frames[0].updated_at = 0.
        for trial in self.report["camera_trials"]:
            self.assertEqual(c.run_matching_trial(self.frames, trial["method"], 4.), trial)
        self.assertEqual(c.input_digest(self.frames), before)

    def test_manual_annotations_enable_identity_metrics_without_changing_matching(self):
        frames = self.cases["gap_5.001"]
        unlabeled = tuple(replace(f, faces=tuple(replace(d, actor=None) for d in f.faces)) for f in frames)
        annotated = c.apply_annotations(unlabeled, [{"frame_id": i, "detection_index": 0, "actor": "reviewed_actor_A"} for i in (1, 2)])
        for method in c.METHODS:
            before, after = c.run_matching_trial(unlabeled, method, 4.), c.run_matching_trial(annotated, method, 4.)
            self.assertEqual(after["id_changes"], 1)
            self.assertEqual(after["false_continuations"], 0)
            self.assertTrue(after["identity_metrics_complete"])
            self.assertEqual(after["trace"][0]["actor_labels"], ["reviewed_actor_A"])
            self.assertEqual([r["detection_track_ids"] for r in before["trace"]], [r["detection_track_ids"] for r in after["trace"]])

    def test_partial_annotations_do_not_claim_complete_identity_metrics(self):
        partial = c.apply_annotations(self.frames, [{"frame_id": 1, "detection_index": 0, "actor": "reviewed_actor_A"}])
        result = c.run_matching_trial(partial, "iou_only", 4.)
        self.assertEqual(result["labeled_observations"], 1)
        self.assertFalse(result["identity_metrics_complete"])

    def test_invalid_or_duplicate_annotations_are_rejected(self):
        valid = {"frame_id": 2, "detection_index": 0, "actor": "actor_A"}
        for rows in ({}, [valid, valid], [{**valid, "detection_index": 99}], [{**valid, "actor": ""}],
                     [{**valid, "student_id": "SV001"}], [valid, {**valid, "detection_index": 1}]):
            with self.subTest(rows=rows), self.assertRaises(ValueError):
                c.apply_annotations(self.frames, rows)

    def test_invalid_geometry_parameters_and_time_inputs_are_rejected(self):
        for kwargs in ({"ambiguity_margin": 0}, {"min_iou": float("nan")}, {"min_score": True}, {"max_center_distance": -1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                c.GeometryConfig(**kwargs)
        for frames in ((), (self.frames[0], self.frames[0]),
                       (replace(self.frames[0], updated_at=float("inf")),),
                       (replace(self.frames[0], faces=(replace(self.frames[0].faces[0], bbox=(0, 0, 0, 10)),)),)):
            with self.subTest(frames=frames), self.assertRaises(ValueError):
                c.run_matching_trial(frames, "geometry", 4.)
        with self.assertRaises(ValueError):
            c.run_matching_trial(self.frames, "unsupported", 4.)

    def test_corrupt_or_incomplete_logs_cannot_be_silently_replayed(self):
        for mode in ("truncated", "order", "count", "state"):
            records = copy.deepcopy(self.records)
            if mode == "truncated":
                records[0]["details_truncated"] = True
            elif mode == "order":
                records[1]["face_tracker"]["detections"][0]["detection_index"] = 1
            elif mode == "count":
                records[0]["face_tracker"]["face_detection_count"] = 99
            else:
                records[0]["face_tracker"]["next_id_after"] = 99
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "invalid.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
                with self.subTest(mode=mode), self.assertRaises(ValueError):
                    c.load_session(path)

    def test_evaluation_has_no_camera_model_database_or_network_side_effects(self):
        with ExitStack() as stack:
            guards = [stack.enter_context(patch(target, side_effect=AssertionError(target))) for target in
                      ("cv2.VideoCapture", "subprocess.Popen", "sqlite3.connect", "socket.create_connection", "urllib.request.urlopen")]
            report = c.evaluate_session()
        for guard in guards:
            guard.assert_not_called()
        self.assertEqual(len(report["risk_trials"]), 25)

    def test_output_is_exclusive_and_cannot_overwrite_session_annotations_or_old_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            c.write_new_report(path, self.report)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["source_sha256"], self.source_hash)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError):
                c.write_new_report(path, self.report)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                c.write_new_report(c.DEFAULT_SESSION, self.report)
            annotated_report = {**self.report, "annotations_file": str(path)}
            with self.assertRaises(ValueError):
                c.write_new_report(path, annotated_report)
        self.assertEqual(hashlib.sha256(c.DEFAULT_SESSION.read_bytes()).hexdigest(), self.source_hash)

    def test_cli_writes_only_new_output_and_reports_invalid_output_as_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "new.json"
            args = ["diagnostic-c", str(c.DEFAULT_SESSION), "--output", str(path)]
            with patch.object(sys, "argv", args), redirect_stdout(io.StringIO()):
                self.assertEqual(c.main(), 0)
            with (patch.object(sys, "argv", args), redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()),
                  self.assertRaises(SystemExit) as error):
                c.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
