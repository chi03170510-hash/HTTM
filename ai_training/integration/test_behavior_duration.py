"""CH-20 synthetic duration tests: no camera, models, database, or network."""

import copy
import unittest
from dataclasses import FrozenInstanceError
from unittest.mock import patch

from behavior_duration import BehaviorDurationTracker, duration_text


def fixture(tid="P1", sid="SYNTHETIC_001", states=None):
    ref = f"ref_{tid}"
    people = [{"person_ref": ref, "behaviors": {
        name: {"state": state} for name, state in
        (states if states is not None else {"using_phone": "detected"}).items()
    }}]
    tracks = [{"track_id": tid, "person_ref": ref, "bbox": (0, 0, 100, 200)}]
    matches = [{"person_track_id": tid, "person_ref": ref, "student_id": sid,
                "track_id": tid, "association_status": "matched"}]
    return people, tracks, matches


def values(frame, index=0):
    return {b.name: b.estimated_seconds for b in frame.people[index].behaviors}


class TestBehaviorDuration(unittest.TestCase):
    def test_four_states_only_detected_counts_and_others_break_the_episode(self):
        for state in ("detected", "suspected", "unknown", "not_detected"):
            with self.subTest(state=state):
                timer = BehaviorDurationTracker()
                timer.update(1, 10.0, *fixture())
                result = timer.update(2, 13.8, *fixture(states={"using_phone": state}))
                if state == "detected":
                    self.assertAlmostEqual(values(result)["using_phone"], 3.8)
                else:
                    self.assertIsNone(values(result)["using_phone"])
                result = timer.update(3, 14.0, *fixture())
                self.assertEqual(values(result)["using_phone"], 4.0 if state == "detected" else 0.0)

    def test_behaviors_have_independent_start_times(self):
        timer = BehaviorDurationTracker()
        timer.update(1, 0.0, *fixture(states={"using_phone": "detected"}))
        timer.update(2, 2.0, *fixture(states={"using_phone": "detected", "sleeping": "detected"}))
        result = timer.update(3, 4.0, *fixture(states={"using_phone": "suspected", "sleeping": "detected"}))
        self.assertEqual(values(result), {"sleeping": 2.0, "using_phone": None,
                                         "turning_back": None, "eating": None})

    def test_all_four_detected_behaviors_can_have_independent_spans(self):
        states = {name: "detected" for name in ("sleeping", "using_phone", "turning_back", "eating")}
        timer = BehaviorDurationTracker()
        timer.update(1, 10.0, *fixture(states=states))
        result = timer.update(2, 13.5, *fixture(states=states))
        self.assertEqual(values(result), {name: 3.5 for name in states})

    def test_same_face_claiming_two_people_cannot_attribute_duration(self):
        people, tracks, matches = tuple(a + b for a, b in zip(fixture(), fixture("P2", "SYNTHETIC_002")))
        for match in matches:
            match["track_id"] = 1
        result = BehaviorDurationTracker().update(1, 0.0, people, tracks, matches)
        self.assertEqual([p.identity_status for p in result.people], ["ambiguous_face"] * 2)
        self.assertTrue(all(p.student_id is None for p in result.people))

    def test_people_and_new_person_track_have_independent_episodes(self):
        timer = BehaviorDurationTracker()
        first = fixture()
        second = fixture("P2", "SYNTHETIC_002")
        timer.update(1, 0.0, *first)
        combined = tuple(a + b for a, b in zip(first, second))
        result = timer.update(2, 3.0, *combined)
        self.assertEqual([values(result, i)["using_phone"] for i in range(2)], [3.0, 0.0])
        result = timer.update(3, 4.0, *fixture("P3", "SYNTHETIC_001"))
        self.assertEqual(values(result)["using_phone"], 0.0)

    def test_identity_change_resets_even_on_the_same_person_track(self):
        timer = BehaviorDurationTracker()
        timer.update(1, 0.0, *fixture())
        timer.update(2, 2.0, *fixture())
        result = timer.update(3, 3.0, *fixture(sid="SYNTHETIC_002"))
        self.assertEqual(values(result)["using_phone"], 0.0)

    def test_missing_association_and_unknown_identity_break_named_duration(self):
        for sid, drop_match in [("Unknown", False), (None, False), ("SYNTHETIC_001", True)]:
            with self.subTest(sid=sid, missing=drop_match):
                timer = BehaviorDurationTracker()
                timer.update(1, 0.0, *fixture())
                people, tracks, matches = fixture(sid=sid)
                result = timer.update(2, 1.0, people, tracks, [] if drop_match else matches)
                self.assertIsNone(result.people[0].student_id)
                self.assertIsNone(values(result)["using_phone"])
                self.assertEqual(values(timer.update(3, 2.0, *fixture()))["using_phone"], 0.0)

    def test_duplicate_frame_is_idempotent_and_out_of_order_is_rejected(self):
        timer = BehaviorDurationTracker()
        original = timer.update(1, 1.0, *fixture())
        self.assertIs(timer.update(1, 99.0, *fixture(sid="OTHER")), original)
        self.assertEqual(values(timer.update(2, 3.0, *fixture()))["using_phone"], 2.0)
        with self.assertRaises(ValueError):
            timer.update(1, 4.0, *fixture())

    def test_non_increasing_timestamp_clears_episode_and_does_not_move_clock_back(self):
        for bad_time in (10.0, 9.0):
            with self.subTest(timestamp=bad_time):
                timer = BehaviorDurationTracker()
                timer.update(1, 10.0, *fixture())
                result = timer.update(2, bad_time, *fixture())
                self.assertFalse(result.timestamp_valid)
                self.assertIsNone(values(result)["using_phone"])
                self.assertEqual(values(timer.update(3, 11.0, *fixture()))["using_phone"], 0.0)

    def test_gap_five_seconds_is_allowed_but_larger_gap_restarts(self):
        for gap, expected in [(3.74, 3.74), (5.0, 5.0), (5.001, 0.0), (30.0, 0.0)]:
            with self.subTest(gap=gap):
                timer = BehaviorDurationTracker()
                timer.update(1, 0.0, *fixture())
                self.assertEqual(values(timer.update(2, gap, *fixture()))["using_phone"], expected)

    def test_absence_and_return_do_not_resume_the_old_episode(self):
        timer = BehaviorDurationTracker()
        timer.update(1, 0.0, *fixture())
        self.assertEqual(timer.update(2, 1.0, [], [], []).people, ())
        self.assertEqual(values(timer.update(3, 2.0, *fixture()))["using_phone"], 0.0)

    def test_duplicate_student_claims_suppress_both_people_and_reset(self):
        timer = BehaviorDurationTracker()
        timer.update(1, 0.0, *fixture())
        combined = tuple(a + b for a, b in zip(fixture(), fixture("P2")))
        result = timer.update(2, 1.0, *combined)
        self.assertEqual([p.identity_status for p in result.people], ["ambiguous_student"] * 2)
        self.assertTrue(all(p.student_id is None for p in result.people))
        self.assertTrue(all(values(result, i)["using_phone"] is None for i in range(2)))
        self.assertEqual(values(timer.update(3, 2.0, *fixture()))["using_phone"], 0.0)

    def test_reference_and_face_track_change_do_not_reset_same_known_person(self):
        timer = BehaviorDurationTracker()
        timer.update(1, 0.0, *fixture())
        people, tracks, matches = fixture()
        for row in (people[0], tracks[0], matches[0]):
            row["person_ref"] = "changed_ref"
        matches[0]["track_id"] = "new_face_track"
        self.assertEqual(values(timer.update(2, 3.0, people, tracks, matches))["using_phone"], 3.0)

    def test_ambiguous_or_wrong_reference_associations_do_not_attribute_duration(self):
        for kind in ("duplicate_match", "wrong_reference", "unmatched_status", "duplicate_reference"):
            with self.subTest(kind=kind):
                people, tracks, matches = fixture()
                if kind == "duplicate_match":
                    matches.append(copy.deepcopy(matches[0]))
                elif kind == "wrong_reference":
                    matches[0]["person_ref"] = "other_person"
                elif kind == "unmatched_status":
                    matches[0]["association_status"] = "unknown"
                else:
                    people.append(copy.deepcopy(people[0]))
                result = BehaviorDurationTracker().update(1, 0.0, people, tracks, matches)
                self.assertIsNone(result.people[0].student_id)
                self.assertIsNone(values(result)["using_phone"])

    def test_outputs_are_frozen_and_inputs_are_not_modified(self):
        timer = BehaviorDurationTracker()
        inputs = fixture()
        original = copy.deepcopy(inputs)
        before = timer.update(1, 0.0, *inputs)
        timer.update(2, 3.0, *inputs)
        self.assertEqual(values(before)["using_phone"], 0.0)
        self.assertEqual(inputs, original)
        with self.assertRaises(FrozenInstanceError):
            before.people[0].student_id = "OTHER"

    def test_long_estimate_never_emits_warning_database_or_network_actions(self):
        with (patch("sqlite3.connect", side_effect=AssertionError("database forbidden")),
              patch("socket.socket.connect", side_effect=AssertionError("network forbidden"))):
            timer = BehaviorDurationTracker()
            result = None
            for frame_id, timestamp in enumerate((0.0, 3.8, 7.6, 11.4), 1):
                result = timer.update(frame_id, timestamp, *fixture())
        self.assertAlmostEqual(values(result)["using_phone"], 11.4)
        self.assertEqual(duration_text(result.people[0].behaviors[1]), "~11.4s")
        self.assertFalse(hasattr(result, "warnings"))
        self.assertFalse(hasattr(result.people[0].behaviors[1], "warned"))

    def test_invalid_capture_values_and_frame_ids_are_rejected(self):
        timer = BehaviorDurationTracker()
        for timestamp in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                timer.update(1, timestamp, *fixture())
        for frame_id in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                timer.update(frame_id, 0.0, *fixture())


if __name__ == "__main__":
    unittest.main()
