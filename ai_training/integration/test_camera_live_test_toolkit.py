"""Offline toolkit tests only: no imports of models, camera or production code."""

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

import camera_live_test_toolkit as toolkit


ROOT = Path(__file__).resolve().parents[2]


def fixture(fid, capture=None, *, pid="P1", student="SV001", state="detected", duration=0.):
    return {"schema": "camera_test_fixture_v1", "synthetic": True,
            "frame_id": fid, "captured_at": float(fid) if capture is None else capture,
            "production_timestamp_valid": True, "duration_max_gap": 5.,
            "people": [{"person_track_id": pid, "student_id": student, "association_status": "matched",
                        "identity_status": "matched", "behaviors": {
                            name: {"state": state, "estimated_seconds": duration, "model_confidence": .8}
                            for name in toolkit.BEHAVIORS}}]}


def shadow_record(fid, capture, duration):
    # Synthetic construction of the REAL persisted schema, only for decision-table tests.
    return {"schema": "identity_continuity_shadow_v1", "frame_id": fid, "captured_at": capture,
            "production_timestamp_valid": True, "duration_max_gap": 5.,
            "production_observation_status": "available",
            "observations": [{"person_track_id": "P1", "status": "face_confirmed", "face_present": True}],
            "sleeping": [{"person_track_id": "P1", "student_id": "SV001", "association_status": "matched",
                          "identity_status": "matched", "sleeping_state": "detected",
                          "sleeping_duration_seconds": duration, "sleeping_model_confidence": .8}]}


def complete_shadow_record(fid=1, capture=100., duration=0.):
    """Synthetic data in the additive production schema, not a camera recording."""
    row = shadow_record(fid, capture, duration)
    row["behavior_diagnostic_version"] = 2
    row["using_phone"] = [{key.replace("sleeping_", "using_phone_", 1): value for key, value in p.items()}
                          for p in row["sleeping"]]
    row["using_phone"][0].update(using_phone_decision_reason="synthetic_unique_phone_evidence",
                                duration_event="increased" if duration else "started_or_restarted",
                                duration_blockers=[], duration_reset_reasons=[])
    return row


class TestCameraLiveTestToolkit(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def log(self, rows, name="frames.jsonl"):
        path = self.root / name
        with path.open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row) + "\n")
        return path

    def analyze(self, rows):
        return toolkit.analyze(self.log(rows), "synthetic")

    def manifest(self, *, rows=None, provenance="recorded", reviewed=False, verdict="PASS", case="S1"):
        if rows is None:
            rows = [shadow_record(1, 100., 0.), shadow_record(2, 101.5, 1.5)]
        log = self.log(rows)
        manifest = toolkit.new_manifest()
        entry = next(e for e in manifest["cases"] if e["case_id"] == case)
        entry.update(run=True, log=log.name, provenance=provenance,
                     ground_truth={"confirmed": reviewed, "verdict": verdict,
                                   "reviewer": "UNIT_TEST_REVIEWER", "evidence": "SIMULATED_REVIEW_NOT_A_REAL_CAMERA_PASS"})
        path = self.root / "manifest.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        return path

    def result(self, path, case="S1"):
        return next(r for r in toolkit.evaluate_manifest(path)["cases"] if r["case_id"] == case)

    def kinds(self, data):
        return {e["kind"] for e in data["events"]}

    def test_shadow_both_behaviors_join_one_production_person(self):
        data = self.analyze([complete_shadow_record(), complete_shadow_record(2, 101.5, 1.5)])
        self.assertEqual((data["ai_frame_count"], data["person_observations"]), (2, 2))
        self.assertEqual(data["reported_student_ids"], {"SV001": 2})
        self.assertNotIn("duplicate_production_sid", self.kinds(data))
        for name in toolkit.BEHAVIORS:
            self.assertEqual(data["behaviors"][name]["states"], {"detected": 2})
            self.assertEqual(data["behaviors"][name]["duration_rows"], 2)
        self.assertEqual(data["issues"], [])

    def test_shadow_two_students_reordered_behavior_rows_join_by_pid(self):
        row = complete_shadow_record()
        for name in toolkit.BEHAVIORS:
            other = copy.deepcopy(row[name][0])
            other.update(person_track_id="P2", student_id="SV002")
            row[name].append(other)
        row["using_phone"][1].update(using_phone_state="not_detected", using_phone_duration_seconds=None,
                                      using_phone_model_confidence=.26)
        row["using_phone"].reverse()
        data = self.analyze([row])
        people = {p["person_track_id"]: p for p in data["frames"][0]["people"]}
        self.assertEqual(data["person_observations"], 2)
        self.assertEqual(people["P1"]["student_id"], "SV001")
        self.assertEqual(people["P1"]["behaviors"]["using_phone"]["state"], "detected")
        self.assertEqual(people["P2"]["student_id"], "SV002")
        self.assertEqual(people["P2"]["behaviors"]["using_phone"]["state"], "not_detected")
        self.assertEqual(people["P2"]["behaviors"]["using_phone"]["confidence"], .26)
        self.assertEqual(data["issues"], [])

    def test_legacy_sleeping_only_stays_readable_without_fabricating_phone(self):
        data = self.analyze([shadow_record(1, 100., 0.)])
        self.assertEqual(data["behaviors"]["sleeping"]["states"], {"detected": 1})
        self.assertIsNone(data["behaviors"]["using_phone"]["states"])
        self.assertEqual(data["behaviors"]["using_phone"]["missing_duration_rows"], 1)
        self.assertEqual(data["issues"], [])

    def test_version_two_missing_phone_array_is_reported(self):
        row = complete_shadow_record()
        row.pop("using_phone")
        data = self.analyze([row])
        self.assertFalse(data["frames"][0]["people_complete"])
        self.assertIsNone(data["behaviors"]["using_phone"]["states"])
        self.assertTrue(any(i["code"] == "missing_field" and i["detail"] == "using_phone" for i in data["issues"]))
        self.assertFalse(any(i["severity"] == "error" for i in data["issues"]))
        self.assertEqual(next(c["status"] for c in toolkit.technical_checks(data, "using_phone")
                              if c["check"] == "behavior_fields_available"), "INCONCLUSIVE")

    def test_phone_array_with_missing_state_duration_confidence_reason(self):
        row = complete_shadow_record()
        for field in ("using_phone_state", "using_phone_duration_seconds", "using_phone_model_confidence",
                      "using_phone_decision_reason"):
            row["using_phone"][0].pop(field)
        value = self.analyze([row])["frames"][0]["people"][0]["behaviors"]["using_phone"]
        self.assertFalse(value["state_available"])
        self.assertFalse(value["duration_available"])
        self.assertIsNone(value["confidence"])
        self.assertIsNone(value["decision_reason"])

    def test_conflicting_behavior_identities_are_not_silently_selected(self):
        row = complete_shadow_record()
        row["using_phone"][0]["student_id"] = "SV002"
        data = self.analyze([row])
        self.assertTrue(any(i["code"] == "behavior_identity_conflict" for i in data["issues"]))
        self.assertIsNone(data["frames"][0]["people"][0]["student_id"])
        self.assertEqual(toolkit.technical_checks(data)[0]["status"], "FAIL")

    def test_missing_behavior_identity_does_not_borrow_sleeping_identity(self):
        row = complete_shadow_record()
        row["using_phone"][0].pop("student_id")
        data = self.analyze([row])
        p = data["frames"][0]["people"][0]
        self.assertFalse(p["student_id_available"])
        self.assertIsNone(p["student_id"])
        self.assertTrue(any(i["code"] == "behavior_identity_missing" for i in data["issues"]))

    def test_mismatching_behavior_person_sets_are_reported_not_guessed(self):
        row = complete_shadow_record()
        row["using_phone"][0].update(person_track_id="P2", student_id="SV002")
        data = self.analyze([row])
        self.assertEqual(data["person_observations"], 2)
        self.assertFalse(data["frames"][0]["people_complete"])
        self.assertTrue(any(i["code"] == "behavior_person_track_sets_differ" for i in data["issues"]))
        self.assertEqual(toolkit.technical_checks(data)[0]["status"], "FAIL")

    def test_duplicate_pid_within_behavior_is_not_collapsed(self):
        row = complete_shadow_record()
        row["using_phone"].append(copy.deepcopy(row["using_phone"][0]))
        data = self.analyze([row])
        self.assertIn("duplicate_person_track_id", self.kinds(data))
        self.assertTrue(any(i["code"] == "duplicate_behavior_person_track" for i in data["issues"]))

    def test_malformed_phone_array_keeps_sleeping_evidence(self):
        for invalid in (None, "bad", [7]):
            with self.subTest(invalid=invalid):
                row = complete_shadow_record()
                row["using_phone"] = invalid
                data = self.analyze([row])
                self.assertEqual(data["behaviors"]["sleeping"]["state_rows"], 1)
                self.assertIsNone(data["behaviors"]["using_phone"]["states"])
                self.assertTrue(any(i["code"] == "invalid_array" for i in data["issues"]))

    def test_using_phone_blocks_resets_reason_are_in_all_report_formats(self):
        rows = [complete_shadow_record(), complete_shadow_record(2, 101.5, 1.5),
                complete_shadow_record(3, 103., None), complete_shadow_record(4, 104.5, 0.)]
        for name in toolkit.BEHAVIORS:
            rows[2][name][0].update(student_id=None, association_status="unmatched", identity_status="unassociated")
            rows[2][name][0][f"{name}_duration_seconds"] = None
        rows[2]["using_phone"][0].update(duration_event="unavailable", duration_blockers=["identity_unassociated"])
        rows[3]["using_phone"][0].update(duration_reset_reasons=["previous_identity_ineligible"])
        data = self.analyze(rows)
        self.assertIn("association_lost", self.kinds(data))
        self.assertIn("student_id_lost", self.kinds(data))
        output = self.root / "new_reports"
        toolkit.write_analysis(data, output)
        with (output / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
            phone = [r for r in csv.DictReader(stream) if r["behavior"] == "using_phone"]
        self.assertEqual(len(phone), 4)
        self.assertEqual(phone[0]["decision_reason"], "synthetic_unique_phone_evidence")
        self.assertEqual(phone[2]["identity_status"], "unassociated")
        self.assertIn("identity_unassociated", phone[2]["logged_blockers"])
        self.assertIn("previous_identity_ineligible", phone[3]["logged_reset_reasons"])
        for filename in ("analysis.md", "analysis.json"):
            text = (output / filename).read_text(encoding="utf-8")
            for evidence in ("using_phone", "synthetic_unique_phone_evidence", "identity_unassociated",
                             "previous_identity_ineligible"):
                self.assertIn(evidence, text)

    def test_new_phone_logs_still_require_human_ground_truth_review(self):
        rows = [complete_shadow_record(), complete_shadow_record(2, 101.5, 1.5)]
        result = self.result(self.manifest(rows=rows, case="P1", reviewed=False), "P1")
        self.assertTrue(all(c["status"] == "PASS" for c in result["checks"]))
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_missing_behavior_is_missing_not_not_detected_or_zero_detections(self):
        row = fixture(1)
        row["people"][0].pop("behaviors")
        data = self.analyze([row])
        for name in toolkit.BEHAVIORS:
            self.assertIsNone(data["behaviors"][name]["states"])
            self.assertEqual(data["behaviors"][name]["state_rows"], 0)
            self.assertEqual(data["behaviors"][name]["missing_duration_rows"], 1)

    def test_missing_duration_key_differs_from_recorded_null(self):
        row = fixture(1, duration=None)
        row["people"][0]["behaviors"]["using_phone"].pop("estimated_seconds")
        data = self.analyze([row])
        self.assertEqual(data["behaviors"]["sleeping"]["null_durations"], 1)
        self.assertEqual(data["behaviors"]["using_phone"]["duration_rows"], 0)

    def test_invalid_json_line_keeps_other_evidence_and_reports_failure(self):
        path = self.log([fixture(1)])
        with path.open("a", encoding="utf-8") as stream:
            stream.write('{broken\n')
        data = toolkit.analyze(path)
        self.assertEqual(data["ai_frame_count"], 1)
        self.assertEqual(toolkit.technical_checks(data)[0]["status"], "FAIL")
        self.assertTrue(any(i["code"] == "invalid_json" and i["line"] == 2 for i in data["issues"]))

    def test_empty_session_is_inconclusive(self):
        data = self.analyze([])
        self.assertEqual(data["ai_frame_count"], 0)
        self.assertTrue(all(c["status"] == "INCONCLUSIVE" for c in toolkit.technical_checks(data)))

    def test_missing_frame_and_timestamp_fields_are_not_invented(self):
        row = fixture(1)
        row.pop("frame_id")
        row.pop("captured_at")
        data = self.analyze([row, fixture(2, duration=1.)])
        self.assertEqual(data["ai_frame_count"], 1)
        self.assertIsNone(data["frames"][0]["captured_at"])
        toolkit.technical_checks(data, "sleeping", "duration_progress")

    def test_student_and_association_loss_are_observed_on_same_track(self):
        a, b = fixture(1), fixture(2, student=None, duration=None)
        b["people"][0].update(association_status="unmatched", identity_status="unassociated")
        data = self.analyze([a, b])
        self.assertTrue({"student_id_lost", "association_lost", "duration_became_unavailable"} <= self.kinds(data))

    def test_person_track_changes_are_not_physical_identity_matches(self):
        data = self.analyze([fixture(1), fixture(2, pid="P3")])
        self.assertIn("person_track_set_changed", self.kinds(data))
        inferred = next(e for e in data["events"] if e["kind"] == "reported_student_changed_track")
        self.assertEqual(inferred["basis"], "inferred")
        self.assertFalse(json.loads(inferred["detail"])["physical_identity_verified"])

    def test_duration_increase_and_stall_are_separate_events(self):
        data = self.analyze([fixture(1), fixture(2, duration=1.), fixture(3, duration=1.)])
        self.assertTrue({"duration_increased", "duration_not_increased"} <= self.kinds(data))

    def test_duration_zero_after_positive_is_only_an_inferred_reset(self):
        data = self.analyze([fixture(1, duration=4.), fixture(2, duration=0.)])
        event = next(e for e in data["events"] if e["kind"] == "duration_reset_candidate")
        self.assertEqual(event["basis"], "inferred")

    def test_time_regression_and_declared_invalid_time_fail_technical_order(self):
        row = fixture(2, 9.)
        row["production_timestamp_valid"] = False
        data = self.analyze([fixture(1, 10.), row])
        statuses = {c["check"]: c["status"] for c in toolkit.technical_checks(data)}
        self.assertEqual(statuses["observation_timestamps"], "FAIL")
        self.assertNotIn("duration_increased", self.kinds(data))

    def test_non_finite_json_numbers_are_rejected(self):
        path = self.root / "bad.jsonl"
        path.write_text('{"schema":"camera_test_fixture_v1","captured_at":NaN}\n', encoding="utf-8")
        data = toolkit.analyze(path)
        self.assertEqual(data["issues"][0]["code"], "invalid_json")

    def test_duplicate_production_student_is_a_log_invariant_failure(self):
        row = fixture(1)
        row["people"].append({**copy.deepcopy(row["people"][0]), "person_track_id": "P2"})
        data = self.analyze([row])
        check = next(c for c in toolkit.technical_checks(data) if c["check"] == "production_sid_unique")
        self.assertEqual(check["status"], "FAIL")

    def test_raw_duplicate_association_sid_is_not_called_production_failure(self):
        row = {"schema": "person_tracker_diagnostic_v1", "frame_id": 1, "captured_at": 1., "tracker_updated_at": 2.,
               "details_truncated": False, "tracks_after": [{"track_id": "P1", "last_seen": 2.}, {"track_id": "P2", "last_seen": 2.}],
               "associations": [{"person_track_id": p, "student_id": "SV001", "association_status": "matched"} for p in ("P1", "P2")]}
        data = self.analyze([row])
        self.assertIn("duplicate_association_sid", self.kinds(data))
        check = next(c for c in toolkit.technical_checks(data) if c["check"] == "production_sid_unique")
        self.assertEqual(check["status"], "INCONCLUSIVE")

    def test_shadow_proposal_cannot_become_production_student(self):
        row = {"schema": "identity_continuity_shadow_v1", "frame_id": 1, "captured_at": 1.,
               "observations": [{"person_track_id": "P1", "status": "track_retained", "track_continuity": "same_track",
                                 "proposed_student_id": "SV001", "face_student_id": None, "face_present": False,
                                 "association_status": "unmatched"}]}
        p = self.analyze([row])["frames"][0]["people"][0]
        self.assertIsNone(p["student_id"])
        self.assertFalse(p["student_id_available"])
        self.assertEqual(p["shadow_proposed_student_id"], "SV001")

    def test_exact_duplicate_records_are_not_double_counted(self):
        row = fixture(1)
        data = self.analyze([row, row, fixture(2, duration=1.)])
        self.assertEqual(data["ai_frame_count"], 2)
        self.assertEqual(data["duplicate_records_ignored"], 1)
        self.assertEqual(data["behaviors"]["sleeping"]["state_rows"], 2)

    def test_conflicting_duplicate_frame_id_is_reported(self):
        data = self.analyze([fixture(1), fixture(1, student="SV002")])
        self.assertTrue(any(i["code"] == "conflicting_frame_id" for i in data["issues"]))

    def test_unsupported_or_malformed_field_types_do_not_crash(self):
        rows = [{"schema": []}, {**fixture(1), "people": "bad"},
                {**fixture(2), "frame_id": []}]
        bad = fixture(3)
        bad["people"][0]["person_track_id"] = {}
        bad["people"][0]["association_status"] = []
        bad["people"][0]["behaviors"]["sleeping"]["state"] = {}
        data = self.analyze(rows + [bad])
        self.assertTrue(any(i["severity"] == "error" for i in data["issues"]))

    def test_negative_duration_is_not_a_valid_measurement(self):
        data = self.analyze([fixture(1, duration=-1.)])
        self.assertTrue(any(i["code"] == "invalid_duration" for i in data["issues"]))
        self.assertEqual(data["behaviors"]["sleeping"]["null_durations"], 0)

    def test_missing_observation_break_does_not_create_duration_inference(self):
        data = self.analyze([fixture(1), fixture(3, duration=3.)])
        self.assertNotIn("duration_increased", self.kinds(data))

    def test_ineligible_duration_non_null_fails_contract(self):
        row = fixture(1, student=None, duration=5.)
        row["people"][0]["identity_status"] = "unassociated"
        checks = toolkit.technical_checks(self.analyze([row]), "sleeping")
        self.assertEqual(next(c for c in checks if c["check"] == "duration_identity_gate")["status"], "FAIL")

    def test_valid_identity_but_all_null_sleep_duration_is_inconclusive_progress(self):
        data = self.analyze([fixture(1, student=None, duration=None), fixture(2, student=None, duration=None)])
        check = next(c for c in toolkit.technical_checks(data, "sleeping", "duration_progress") if c["check"] == "duration_progress")
        self.assertEqual(check["status"], "INCONCLUSIVE")

    def test_duration_progress_does_not_invent_max_gap(self):
        rows = [fixture(1), fixture(2, duration=1.)]
        for r in rows:
            r.pop("duration_max_gap")
        check = next(c for c in toolkit.technical_checks(self.analyze(rows), "sleeping", "duration_progress") if c["check"] == "duration_progress")
        self.assertEqual(check["status"], "INCONCLUSIVE")

    def test_real_person_session_reports_missing_behaviors(self):
        path = ROOT / 'ai_training/runs/person_tracker_debug/session_808dd75fae434122b1d0a03c44fa819c'
        data = toolkit.analyze(path, "recorded")
        self.assertEqual(data["ai_frame_count"], 10)
        self.assertEqual(set(data["reported_person_tracks"]), {"P1", "P2", "P3"})
        self.assertEqual(data["expiry_events_logged"], 2)
        self.assertIsNone(data["behaviors"]["sleeping"]["states"])
        self.assertIsNone(data["behaviors"]["using_phone"]["states"])
        self.assertIn("person_track_set_changed", self.kinds(data))

    def test_saved_shadow_sample_has_sleeping_but_no_using_phone(self):
        path = ROOT / 'ai_training/runs/identity_shadow_sleep_logging_validation_e4e7670cf9f84549a1c13a409f4d748f/synthetic_sleeping_sample_verified.jsonl'
        before = hashlib.sha256(path.read_bytes()).hexdigest()
        data = toolkit.analyze(path, "synthetic")
        self.assertEqual(data["ai_frame_count"], 6)
        self.assertEqual(data["behaviors"]["sleeping"]["states"], {"not_detected": 2, "detected": 4})
        self.assertIsNone(data["behaviors"]["using_phone"]["states"])
        self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_empty_manifest_is_all_not_run(self):
        path = self.root / "manifest.json"
        path.write_text(json.dumps(toolkit.new_manifest()), encoding="utf-8")
        self.assertEqual(toolkit.evaluate_manifest(path)["counts"],
                         {"PASS": 0, "FAIL": 0, "INCONCLUSIVE": 0, "NOT RUN": 16})

    def test_any_eligible_duration_stall_prevents_progress_pass(self):
        data = self.analyze([fixture(1), fixture(2, duration=1.), fixture(3, duration=1.)])
        result = next(c for c in toolkit.technical_checks(data, "sleeping", "duration_progress")
                      if c["check"] == "duration_progress")
        self.assertEqual(result["status"], "FAIL")

    def test_duplicate_track_ids_fail_even_with_different_student_ids(self):
        row = fixture(1)
        second = copy.deepcopy(row["people"][0])
        second["student_id"] = "SV002"
        row["people"].append(second)
        result = next(c for c in toolkit.technical_checks(self.analyze([row]))
                      if c["check"] == "person_track_unique")
        self.assertEqual(result["status"], "FAIL")

    def test_missing_track_id_does_not_fabricate_a_duplicate(self):
        row = fixture(1)
        row["people"][0].pop("person_track_id")
        data = self.analyze([row])
        self.assertNotIn("duplicate_person_track_id", self.kinds(data))
        self.assertEqual(next(c for c in toolkit.technical_checks(data)
                              if c["check"] == "person_track_unique")["status"], "INCONCLUSIVE")

    def test_invalid_sid_type_is_not_silently_treated_as_unknown(self):
        data = self.analyze([fixture(1, student=123)])
        self.assertTrue(any(i["code"] == "invalid_student_id_type" for i in data["issues"]))

    def test_duplicate_json_keys_are_rejected(self):
        path = self.root / "duplicate_keys.jsonl"
        path.write_text('{"schema":"camera_test_fixture_v1","frame_id":1,"frame_id":2}\n')
        self.assertTrue(any(i["code"] == "invalid_json" for i in toolkit.analyze(path)["issues"]))

    def test_persisted_reset_reason_is_marked_logged_not_new_inference(self):
        row = fixture(1)
        row["people"][0]["behaviors"]["sleeping"]["duration_reset_reasons"] = ["reported_student_id_changed"]
        event = next(e for e in self.analyze([row])["events"] if e["kind"] == "logged_duration_start_or_reset_reasons")
        self.assertEqual(event["basis"], "logged_diagnostic")

    def test_missing_sid_duration_context_is_explicitly_inferred(self):
        data = self.analyze([fixture(1, student=None, duration=None)])
        event = next(e for e in data["events"] if e["kind"] == "duration_unavailable_context")
        self.assertEqual(event["basis"], "inferred")

    def test_truncated_person_log_does_not_infer_unmatched_association(self):
        row = {"schema": "person_tracker_diagnostic_v1", "frame_id": 1, "captured_at": 1.,
               "tracker_updated_at": 2., "details_truncated": True,
               "tracks_after": [{"track_id": "P1", "last_seen": 2.}], "associations": []}
        p = self.analyze([row])["frames"][0]["people"][0]
        self.assertFalse(p["association_available"])
        self.assertIsNone(p["association_status"])

    def test_null_association_is_missing_not_observed_loss(self):
        a, b = fixture(1), fixture(2)
        b["people"][0]["association_status"] = None
        self.assertNotIn("association_lost", self.kinds(self.analyze([a, b])))

    def test_non_boolean_face_presence_is_not_false(self):
        row = fixture(1)
        row["people"][0]["face_present"] = "false"
        data = self.analyze([row])
        self.assertIsNone(data["frames"][0]["people"][0]["face_present"])
        self.assertTrue(any(i["code"] == "invalid_field_type" for i in data["issues"]))

    def test_malformed_shadow_continuity_is_handled_without_crash(self):
        row = {"schema": "identity_continuity_shadow_v1", "frame_id": 1, "captured_at": 1.,
               "observations": [{"person_track_id": "P1", "track_continuity": {}, "status": "unknown"}]}
        self.assertEqual(self.analyze([row])["ai_frame_count"], 1)

    def test_invalid_logged_max_gap_is_not_used_as_a_policy(self):
        row = fixture(1)
        row["duration_max_gap"] = -1
        data = self.analyze([row])
        self.assertIsNone(data["frames"][0]["duration_max_gap"])
        self.assertTrue(any(i["code"] == "invalid_duration_max_gap" for i in data["issues"]))

    def test_null_logged_max_gap_is_missing_not_a_fabricated_policy(self):
        row = fixture(1)
        row["duration_max_gap"] = None
        data = self.analyze([row])
        self.assertTrue(any(i["severity"] == "missing" for i in data["issues"]))
        self.assertFalse(any(i["severity"] == "error" for i in data["issues"]))

    def test_association_count_drop_is_visible_across_new_track_without_linking_actors(self):
        a, b = fixture(1), fixture(2, pid="P3", student=None, duration=None)
        b["people"][0].update(association_status="unmatched", identity_status="unassociated")
        data = self.analyze([a, b])
        event = next(e for e in data["events"] if e["kind"] == "association_match_count_dropped")
        self.assertEqual(event["frame_id"], 2)
        self.assertFalse(json.loads(event["detail"])["actor_correspondence_inferred"])

    def test_mixed_schemas_do_not_infer_continuity(self):
        a, b = fixture(1), shadow_record(2, 2., 1.)
        data = self.analyze([a, b])
        self.assertTrue(any(i["code"] == "mixed_schemas" for i in data["issues"]))
        self.assertNotIn("duration_increased", self.kinds(data))

    def test_departure_without_arrival_is_not_a_completed_turnover_test(self):
        a, b = fixture(1), fixture(2)
        b["people"] = []
        result = next(c for c in toolkit.technical_checks(self.analyze([a, b]), special="track_turnover_seen")
                      if c["check"] == "track_turnover_seen")
        self.assertEqual(result["status"], "INCONCLUSIVE")

    def test_track_departure_then_arrival_is_reported_without_actor_truth(self):
        a, b, c = fixture(1), fixture(2), fixture(3, pid="P2")
        b["people"] = []
        result = next(x for x in toolkit.technical_checks(self.analyze([a, b, c]), special="track_turnover_seen")
                      if x["check"] == "track_turnover_seen")
        self.assertEqual(result["status"], "PASS")
        self.assertIn("not proof", result["reason"])

    def test_detected_and_increasing_duration_without_manual_review_never_passes_camera_case(self):
        r = self.result(self.manifest(reviewed=False))
        self.assertTrue(all(c["status"] == "PASS" for c in r["checks"]))
        self.assertEqual(r["status"], "INCONCLUSIVE")

    def test_aggregation_pass_requires_explicit_human_confirmation(self):
        # Tests the decision table, not an actual camera PASS.
        self.assertEqual(self.result(self.manifest(reviewed=True))["status"], "PASS")

    def test_human_failure_is_distinguished_from_technical_log_failure(self):
        r = self.result(self.manifest(reviewed=True, verdict="FAIL"))
        self.assertEqual(r["status"], "FAIL")
        self.assertIn("Human", r["reason"])

    def test_synthetic_fixture_cannot_be_promoted_by_recorded_declaration(self):
        r = self.result(self.manifest(rows=[fixture(1), fixture(2, duration=1.)], reviewed=True))
        self.assertEqual(r["status"], "INCONCLUSIVE")
        self.assertEqual(r["effective_provenance"], "synthetic")

    def test_ground_truth_missing_reviewer_or_evidence_is_inconclusive(self):
        path = self.manifest(reviewed=True)
        data = json.loads(path.read_text())
        data["cases"][0]["ground_truth"]["evidence"] = ""
        path.write_text(json.dumps(data))
        self.assertEqual(self.result(path)["status"], "INCONCLUSIVE")

    def test_nonexistent_log_in_declared_run_is_inconclusive(self):
        path = self.manifest()
        data = json.loads(path.read_text())
        data["cases"][0]["log"] = "does_not_exist.jsonl"
        path.write_text(json.dumps(data))
        self.assertEqual(self.result(path)["status"], "INCONCLUSIVE")

    def test_using_phone_missing_stays_inconclusive_even_with_manual_pass(self):
        r = self.result(self.manifest(reviewed=True, case="P1"), "P1")
        self.assertEqual(r["status"], "INCONCLUSIVE")

    def test_requested_frame_missing_is_reported(self):
        path = self.log([fixture(1)])
        data = toolkit.analyze(path, frame_ids=[99])
        self.assertEqual(data["ai_frame_count"], 0)
        self.assertTrue(any(i["code"] == "requested_frame_missing" for i in data["issues"]))

    def test_report_files_and_csv_missing_flags(self):
        row = fixture(1)
        row["people"][0].pop("behaviors")
        data = self.analyze([row])
        out = self.root / "analysis"
        toolkit.write_analysis(data, out)
        self.assertEqual({p.name for p in out.iterdir()}, {"analysis.md", "analysis.json", "frames.csv", "observations.csv", "events.csv"})
        with (out / "observations.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["state_available"], "False")
        self.assertEqual(rows[0]["state"], "")
        self.assertIn("MISSING DATA", (out / "analysis.md").read_text(encoding="utf-8"))
        with self.assertRaises(FileExistsError):
            toolkit.write_analysis(data, out)

    def test_manifest_and_result_output_never_overwrite(self):
        path = self.root / "manifest.json"
        with redirect_stdout(io.StringIO()):
            self.assertEqual(toolkit.main(["init-manifest", "--output", str(path)]), 0)
        before = path.read_bytes()
        with redirect_stderr(io.StringIO()):
            self.assertEqual(toolkit.main(["init-manifest", "--output", str(path)]), 2)
        self.assertEqual(path.read_bytes(), before)
        report = toolkit.evaluate_manifest(path)
        out = self.root / "results"
        toolkit.write_results(report, out)
        self.assertTrue((out / "test_results.md").is_file())
        self.assertTrue((out / "test_results.csv").is_file())
        with self.assertRaises(FileExistsError):
            toolkit.write_results(report, out)

    def test_ambiguous_directory_requires_explicit_log(self):
        self.log([], "a.jsonl")
        self.log([], "b.jsonl")
        with self.assertRaisesRegex(ValueError, "zero or multiple"):
            toolkit.analyze(self.root)

    def test_cli_runs_without_site_packages_or_inference(self):
        path = self.log([fixture(1)])
        result = subprocess.run([sys.executable, "-I", "-S", "-B", str(Path(toolkit.__file__).resolve()),
                                 "analyze", str(path), "--output-dir", str(self.root / "cli")],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((self.root / "cli/analysis.md").is_file())


if __name__ == "__main__":
    unittest.main()
