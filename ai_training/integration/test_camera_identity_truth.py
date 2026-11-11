"""Offline manual-truth comparator; synthetic evidence never becomes camera PASS."""

import copy
import csv
import hashlib
import io
import json
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import camera_identity_truth as truth
from camera_identity_truth_fixtures import confirmation, write_bundle


class TestCameraIdentityTruth(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = write_bundle(self.root / "positive")
        self.data = json.loads(self.path.read_text(encoding="utf-8"))

    def evaluate(self):
        self.path.write_text(json.dumps(self.data), encoding="utf-8")
        return truth.evaluate(self.path)

    def change_source(self, role, mutation, *, rebind=True):
        descriptor = self.data["sources"][role]
        path = Path(descriptor["path"])
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        mutation(records)
        path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
        if rebind:
            descriptor.update(truth.source_descriptor(path))

    def codes(self, report):
        return {i["code"] for i in report["issues"]}

    def test_positive_independent_labels_compare_both_channels(self):
        result = self.evaluate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(result["effective_provenance"], "synthetic")
        self.assertEqual(result["metrics"]["total_observations"], 4)
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 4)
        self.assertEqual(result["metrics"]["production"]["correct"], 4)
        self.assertEqual(result["metrics"]["shadow"]["correct"], 4)
        self.assertFalse(result["production_applied"])

    def test_adversarial_report_finds_wrong_actor_retention_duplicate_and_recovery(self):
        path = write_bundle(self.root / "adversarial", adversarial=True)
        result = truth.evaluate(path)
        metrics = result["metrics"]
        self.assertEqual(result["status"], "FAIL")
        self.assertEqual(metrics["total_observations"], 12)
        self.assertEqual(metrics["observations_with_valid_actor_truth"], 12)
        self.assertEqual((metrics["production"]["correct"], metrics["production"]["wrong"], metrics["production"]["unknown"]), (5, 3, 4))
        self.assertEqual((metrics["shadow"]["correct"], metrics["shadow"]["wrong"], metrics["shadow"]["unknown"]), (6, 4, 2))
        self.assertEqual(metrics["false_identity_retention"], 3)
        self.assertEqual(metrics["production"]["duplicate_sid_frames"], 1)
        self.assertEqual(metrics["shadow"]["duplicate_sid_frames"], 1)
        self.assertEqual(metrics["shadow"]["ambiguous_status"], 2)
        events = {e["kind"] for e in result["events"]}
        self.assertTrue({"wrong_sid", "labelled_actor_person_track_changed", "labelled_actor_face_lost",
                         "labelled_actor_face_reappeared", "different_labelled_actor_on_same_pid", "person_bbox_overlap"} <= events)

    def test_init_does_not_copy_any_ai_sid_or_create_actor_truth(self):
        template = truth.make_template(self.data["sources"]["person"]["path"], shadow=self.data["sources"]["shadow"]["path"],
                                       capture_session_id="human_run")
        self.assertEqual(template["actors"], [])
        self.assertEqual(template["annotations"], [])
        self.assertFalse(template["pairing"]["confirmed"])
        self.assertNotIn("SV001", json.dumps(template))

    def test_session_ids_can_differ_when_binding_and_timestamps_are_verified(self):
        self.assertNotEqual(self.data["sources"]["person"]["session_id"], self.data["sources"]["shadow"]["session_id"])
        self.assertEqual(self.evaluate()["metrics"]["paired_frames"], 2)

    def test_equal_frame_ids_without_same_run_confirmation_are_not_joined(self):
        self.data["pairing"]["confirmed"] = False
        result = self.evaluate()
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["metrics"]["paired_frames"], 0)
        self.assertEqual(result["metrics"]["unmatched_shadow_observations"], 4)
        self.assertEqual(result["metrics"]["production"]["missing"], 4)

    def test_changed_source_hash_is_rejected(self):
        self.change_source("person", lambda records: records[0].update(extra=True), rebind=False)
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            self.evaluate()

    def test_wrong_source_session_id_is_rejected(self):
        self.data["sources"]["person"]["session_id"] = "another_session"
        with self.assertRaisesRegex(ValueError, "session/path/SHA256 mismatch"):
            self.evaluate()

    def test_same_frame_id_with_changed_capture_timestamp_is_not_joined(self):
        self.change_source("shadow", lambda records: records[0].update(captured_at=101.001))
        result = self.evaluate()
        self.assertIn("frame_timestamp_or_clock_mismatch", self.codes(result))
        self.assertEqual(result["metrics"]["paired_frames"], 1)
        self.assertEqual(result["metrics"]["production"]["missing"], 2)

    def test_same_capture_but_changed_tracker_update_is_not_joined(self):
        self.change_source("shadow", lambda records: records[0].update(tracker_updated_at=201.001))
        self.assertIn("frame_timestamp_or_clock_mismatch", self.codes(self.evaluate()))

    def test_wrong_clock_is_not_calibrated_or_guessed(self):
        self.change_source("shadow", lambda records: records[0].update(clock="unix_time"))
        result = self.evaluate()
        self.assertIn("source_timestamp_or_clock_invalid", self.codes(result))
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_different_frame_ids_do_not_match_just_by_identical_timestamps(self):
        self.change_source("shadow", lambda records: [r.update(frame_id=r["frame_id"] + 100) for r in records])
        result = self.evaluate()
        self.assertEqual(result["metrics"]["paired_frames"], 0)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_explicit_verified_id_mapping_allows_equal_clocks_and_times(self):
        self.change_source("shadow", lambda records: [r.update(frame_id=r["frame_id"] + 100) for r in records])
        self.data["pairing"]["frame_map"] = [{"person_frame_id": fid, "shadow_frame_id": fid + 100, **confirmation()}
                                               for fid in (1, 2)]
        result = self.evaluate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual({r["shadow_frame_id"] for r in result["observations"]}, {101, 102})

    def test_explicit_mapping_does_not_override_timestamp_mismatch(self):
        self.change_source("shadow", lambda records: records[0].update(captured_at=101.01))
        self.data["pairing"]["frame_map"] = [{"person_frame_id": 1, "shadow_frame_id": 1, **confirmation()}]
        self.assertIn("frame_timestamp_or_clock_mismatch", self.codes(self.evaluate()))

    def test_mapping_must_be_one_to_one_including_automatic_matches(self):
        self.data["pairing"]["frame_map"] = [{"person_frame_id": 1, "shadow_frame_id": 2, **confirmation()}]
        result = self.evaluate()
        self.assertEqual(result["metrics"]["paired_frames"], 0)
        self.assertIn("frame_mapping_not_one_to_one", self.codes(result))

    def test_unconfirmed_mapping_is_rejected(self):
        self.data["pairing"]["frame_map"] = [{"person_frame_id": 1, "shadow_frame_id": 1}]
        with self.assertRaisesRegex(ValueError, "human confirmation"):
            self.evaluate()

    def test_duplicate_source_frame_id_is_rejected(self):
        self.change_source("person", lambda records: records.append(copy.deepcopy(records[0])))
        with self.assertRaisesRegex(ValueError, "duplicate frame ID"):
            self.evaluate()

    def test_log_line_order_cannot_substitute_for_timestamp_integrity(self):
        self.change_source("shadow", lambda records: records.reverse())
        result = self.evaluate()
        self.assertNotEqual(result["status"], "PASS")
        self.assertIn("source_timestamp_or_clock_invalid", self.codes(result))

    def test_detection_and_behavior_row_reorder_uses_person_track_key(self):
        def reverse(records):
            for record in records:
                for key in ("observations", "sleeping", "using_phone"):
                    record[key].reverse()
        self.change_source("shadow", reverse)
        self.assertEqual(self.evaluate()["status"], "PASS")

    def test_person_sets_mismatch_is_not_silently_joined(self):
        def remove(records):
            for key in ("observations", "sleeping", "using_phone"):
                records[0][key].pop()
        self.change_source("shadow", remove)
        result = self.evaluate()
        self.assertIn("paired_person_sets_differ", self.codes(result))
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_no_actor_truth_is_inconclusive_even_when_predictions_are_consistent(self):
        self.data["actors"], self.data["annotations"] = [], []
        result = self.evaluate()
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 0)
        self.assertEqual(result["metrics"]["production"]["unscorable"], 4)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_roster_claim_must_be_independent_confirmed_and_reviewed(self):
        for field, value in (("independent_of_ai", False), ("confirmed", False), ("basis", "ai_prediction"), ("evidence", "")):
            with self.subTest(field=field):
                original = copy.deepcopy(self.data)
                self.data["actors"][0][field] = value
                self.assertEqual(self.evaluate()["metrics"]["observations_with_valid_actor_truth"], 2)
                self.data = original

    def test_manual_visual_label_must_not_be_based_only_on_ai_identity(self):
        self.data["annotations"][0]["basis"] = "production_sid"
        result = self.evaluate()
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 3)
        self.assertIn("annotation_not_independently_confirmed", self.codes(result))

    def test_member_requires_exact_frame_pid_capture_and_update(self):
        for field, value in (("captured_at", 101.001), ("tracker_updated_at", 201.001),
                             ("person_track_id", "P99"), ("person_frame_id", 999)):
            with self.subTest(field=field):
                original = copy.deepcopy(self.data)
                self.data["annotations"][0]["members"][0][field] = value
                result = self.evaluate()
                self.assertIn("annotation_observation_not_exactly_matched", self.codes(result))
                self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 3)
                self.data = original

    def test_segment_labels_only_explicit_members_not_every_pid_in_range(self):
        first = self.data["annotations"][0]
        first.update(kind="segment", capture_start=101., capture_end=102.)
        self.data["annotations"] = self.data["annotations"][:2]
        result = self.evaluate()
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 2)
        self.assertFalse(any(r["actor_truth_valid"] for r in result["observations"] if r["person_frame_id"] == 2))

    def test_segment_can_explicitly_bind_actor_after_person_track_change(self):
        # Two independent listed members, not automatic inheritance across PIDs.
        def new_person(records):
            records[1]["tracks_after"][0]["track_id"] = "P3"
            records[1]["associations"][0]["person_track_id"] = "P3"
        def new_shadow(records):
            for key in ("observations", "sleeping", "using_phone"):
                records[1][key][0]["person_track_id"] = "P3"
        self.change_source("person", new_person)
        self.change_source("shadow", new_shadow)
        first = self.data["annotations"][0]
        second = self.data["annotations"][2]
        second["members"][0]["person_track_id"] = "P3"
        first.update(kind="segment", capture_start=101., capture_end=102., members=first["members"] + second["members"])
        self.data["annotations"].remove(second)
        result = self.evaluate()
        self.assertEqual(result["status"], "PASS")
        self.assertIn("labelled_actor_person_track_changed", {e["kind"] for e in result["events"]})

    def test_segment_member_outside_declared_interval_is_unscorable(self):
        self.data["annotations"][0].update(kind="segment", capture_start=102., capture_end=103.)
        self.assertIn("annotation_observation_not_exactly_matched", self.codes(self.evaluate()))

    def test_conflicting_actor_labels_invalidate_observation(self):
        duplicate = copy.deepcopy(self.data["annotations"][0])
        duplicate["actor_id"] = "actor_b"
        self.data["annotations"].append(duplicate)
        result = self.evaluate()
        self.assertIn("annotation_conflict", self.codes(result))
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 3)

    def test_same_actor_cannot_label_two_simultaneous_person_tracks(self):
        self.data["annotations"][1]["actor_id"] = "actor_a"
        result = self.evaluate()
        self.assertIn("actor_assigned_to_multiple_tracks_in_frame", self.codes(result))
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 2)

    def test_roster_sid_conflict_is_not_used_as_identity_oracle(self):
        self.data["actors"][1]["expected_student_id"] = "SV001"
        result = self.evaluate()
        self.assertIn("actor_roster_sid_conflict", self.codes(result))
        self.assertEqual(result["metrics"]["production"]["correct"], 0)

    def test_known_actor_without_roster_sid_is_not_sid_correct(self):
        self.data["actors"][0]["expected_student_id"] = None
        result = self.evaluate()
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 4)
        self.assertEqual(result["metrics"]["production"]["unscorable"], 2)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_raw_association_cannot_fill_missing_production_identity(self):
        self.data["sources"].pop("shadow")
        result = self.evaluate()
        self.assertEqual(result["metrics"]["production"]["missing"], 4)
        self.assertTrue(all(r["association_sid"] is not None for r in result["observations"]))
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_legacy_shadow_without_production_rows_does_not_borrow_proposal_sid(self):
        def legacy(records):
            for record in records:
                for key in ("production_observation_status", "behavior_diagnostic_version", "sleeping", "using_phone"):
                    record.pop(key)
        self.change_source("shadow", legacy)
        result = self.evaluate()
        self.assertEqual(result["metrics"]["production"]["missing"], 4)
        self.assertEqual(result["metrics"]["shadow"]["correct"], 4)

    def test_sleeping_only_shadow_still_supplies_actual_production_identity(self):
        self.change_source("shadow", lambda records: [(r.pop("using_phone"), r.pop("behavior_diagnostic_version")) for r in records])
        self.assertEqual(self.evaluate()["status"], "PASS")

    def test_sleeping_phone_identity_conflict_prevents_join(self):
        self.change_source("shadow", lambda records: records[0]["using_phone"][0].update(student_id="SV999"))
        result = self.evaluate()
        self.assertIn("shadow_production_metadata_incomplete", self.codes(result))
        self.assertEqual(result["metrics"]["production"]["missing"], 2)

    def test_duplicate_sid_is_detectable_without_actor_truth(self):
        path = write_bundle(self.root / "bad", adversarial=True)
        data = json.loads(path.read_text())
        data["actors"], data["annotations"] = [], []
        path.write_text(json.dumps(data))
        result = truth.evaluate(path)
        self.assertEqual(result["metrics"]["false_identity_retention"], 0)
        self.assertGreater(result["metrics"]["retention_unscorable"], 0)
        self.assertEqual(result["metrics"]["shadow"]["duplicate_sid_frames"], 1)
        self.assertEqual(result["status"], "FAIL")  # Logged invariant failure, not a ground-truth accuracy claim.

    def test_synthetic_source_cannot_be_promoted_by_recorded_declaration(self):
        self.data["provenance"] = "recorded"
        self.assertEqual(self.evaluate()["effective_provenance"], "synthetic")

    def test_empty_session_does_not_pass(self):
        self.change_source("person", lambda records: records.clear())
        self.data["annotations"] = []
        result = self.evaluate()
        self.assertEqual(result["metrics"]["total_observations"], 0)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_extra_shadow_observations_remain_visible_and_inconclusive(self):
        def extra(records):
            record = copy.deepcopy(records[-1])
            record.update(frame_id=3, captured_at=103., tracker_updated_at=203.)
            records.append(record)
        self.change_source("shadow", extra)
        result = self.evaluate()
        self.assertEqual(result["metrics"]["unmatched_shadow_observations"], 2)
        self.assertEqual(result["metrics"]["production"]["correct"], 4)
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_retired_unobserved_tracks_are_not_counted_as_current_people(self):
        def extra(records):
            records[0]["tracks_after"].append({"track_id": "P99", "person_ref": "old", "bbox": [0, 0, 10, 10], "last_seen": 200.})
        self.change_source("person", extra)
        self.assertEqual(self.evaluate()["metrics"]["total_observations"], 4)

    def test_unverified_provenance_cannot_pass_recorded_camera_comparison(self):
        for role in ("person", "shadow"):
            self.change_source(role, lambda records: [r.pop("synthetic") for r in records])
        self.data["provenance"] = "unverified"
        self.assertEqual(self.evaluate()["status"], "INCONCLUSIVE")

    def test_invalid_shadow_sid_type_and_status_cannot_be_scored(self):
        for field, value in (("proposed_student_id", ["SV001"]), ("status", [])):
            with self.subTest(field=field):
                original = copy.deepcopy(self.data)
                source = Path(self.data["sources"]["shadow"]["path"])
                payload = source.read_bytes()
                self.change_source("shadow", lambda records: records[0]["observations"][0].update({field: value}))
                result = self.evaluate()
                self.assertIn("shadow_identity_metadata_invalid", self.codes(result))
                self.assertEqual(result["metrics"]["production"]["missing"], 2)
                source.write_bytes(payload)
                self.data = original

    def test_missing_shadow_timestamp_validity_is_not_assumed_valid(self):
        self.change_source("shadow", lambda records: records[0].pop("timestamp_valid"))
        self.assertEqual(self.evaluate()["metrics"]["paired_frames"], 1)

    def test_incomplete_or_invalid_person_frame_cannot_validate_actor_truth(self):
        self.change_source("person", lambda records: records[0].update(details_truncated=True))
        result = self.evaluate()
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 2)
        self.assertIn("person_details_incomplete", self.codes(result))

    def test_malformed_json_and_duplicate_keys_are_rejected(self):
        path = Path(self.data["sources"]["person"]["path"])
        path.write_text('{"frame_id":1,"frame_id":2}\n')
        self.data["sources"]["person"].update(truth.source_descriptor(path))
        with self.assertRaisesRegex(ValueError, "Duplicate JSON key"):
            self.evaluate()

    def test_nonfinite_timestamp_json_is_rejected(self):
        path = Path(self.data["sources"]["person"]["path"])
        path.write_text('{"captured_at":NaN}\n')
        self.data["sources"]["person"].update(truth.source_descriptor(path))
        with self.assertRaisesRegex(ValueError, "Non-finite"):
            self.evaluate()

    def test_report_formats_include_wrong_identity_evidence_and_preserve_sources(self):
        path = write_bundle(self.root / "bad", adversarial=True)
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in (self.root / "bad").rglob('*') if p.is_file()}
        report = truth.evaluate(path)
        output = self.root / "report"
        truth.write_report(report, output)
        self.assertEqual(json.loads((output / "identity.json").read_text())["metrics"]["false_identity_retention"], 3)
        self.assertIn("WRONG", (output / "identity.md").read_text())
        self.assertIn("labelled_actor_face_reappeared", (output / "identity.md").read_text())
        with (output / "identity.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 12)
        self.assertTrue(any(r["shadow_verdict"] == "WRONG" for r in rows))
        self.assertEqual(before, {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in before})
        with self.assertRaises(FileExistsError):
            truth.write_report(report, output)

    def test_cli_works_without_site_packages_and_reports_fail_exit_code(self):
        path = write_bundle(self.root / "bad", adversarial=True)
        command = [sys.executable, "-I", "-S", "-B", truth.__file__, "evaluate", str(path), "--output-dir", str(self.root / "cli")]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn('"status": "FAIL"', result.stdout)

    def test_cli_init_does_not_overwrite_existing_truth(self):
        args = ["init", "--person", self.data["sources"]["person"]["path"], "--capture-session-id", "run", "--output", str(self.path)]
        with redirect_stderr(io.StringIO()):
            self.assertEqual(truth.main(args), 2)

    def test_real_person_log_can_generate_blank_template_only(self):
        source = Path(truth.__file__).resolve().parents[1] / "runs/person_tracker_debug/session_808dd75fae434122b1d0a03c44fa819c"
        data = truth.make_template(source, capture_session_id="historical_session_unlabelled", provenance="recorded")
        self.assertEqual(data["actors"], [])
        self.assertEqual(data["annotations"], [])
        path = self.root / "old_truth.json"
        path.write_text(json.dumps(data))
        result = truth.evaluate(path)
        self.assertEqual(result["status"], "INCONCLUSIVE")
        self.assertEqual(result["metrics"]["observations_with_valid_actor_truth"], 0)


if __name__ == "__main__":
    unittest.main()
