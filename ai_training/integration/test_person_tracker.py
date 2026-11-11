r"""PT-01..PT-08: synthetic IoU tests; no camera, models, or real identities.

Run: .\.venv\Scripts\python.exe ai_training/integration/test_person_tracker.py -v
"""

import copy
import unittest
from unittest.mock import patch

import person_tracker as tracker_module
from person_tracker import PersonTracker, calculate_iou


def person(ref, box):
    return {"person_ref": ref, "bbox_xyxy": box}


def ids_by_ref(results):
    return {item["person_ref"]: item["track_id"] for item in results}


class TestPersonTracker(unittest.TestCase):
    def test_pt01_stationary_person_for_ten_updates(self):
        tracker = PersonTracker()
        people = [person("person_0", [20, 10, 120, 220])]
        original = copy.deepcopy(people)
        for frame in range(10):
            with self.subTest(frame=frame + 1):
                result = tracker.update(people, timestamp=float(frame))
                self.assertEqual(ids_by_ref(result), {"person_0": "P1"})
                self.assertEqual(result[0]["bbox"], (20., 10., 120., 220.))
        self.assertEqual(people, original)

    def test_pt02_small_movements_preserve_id(self):
        tracker = PersonTracker()
        for frame in range(10):
            box = [20 + frame * 3, 10 + frame, 120 + frame * 3, 220 + frame]
            result = tracker.update([person("person_0", box)], float(frame))
            with self.subTest(frame=frame + 1):
                self.assertEqual(result[0]["track_id"], "P1")
                self.assertEqual(result[0]["bbox"], tuple(box))

    def test_pt03_reference_changes_and_reorders_are_frame_local(self):
        tracker = PersonTracker()
        left, right = [0, 0, 100, 200], [200, 0, 300, 200]
        self.assertEqual(ids_by_ref(tracker.update([
            person("person_0", left), person("person_1", right),
        ], 0.0)), {"person_0": "P1", "person_1": "P2"})
        # Detection order reverses and the references now describe other boxes.
        result = tracker.update([
            person("person_0", right), person("person_1", left),
        ], 1.0)
        self.assertEqual(ids_by_ref(result), {"person_1": "P1", "person_0": "P2"})
        result = tracker.update([person("new_ref", left)], 2.0)
        self.assertEqual(ids_by_ref(result), {"new_ref": "P1"})

    def test_pt04_reappearance_before_and_after_expiry(self):
        tracker = PersonTracker(max_age=3.0)
        box = [0, 0, 100, 200]
        self.assertEqual(tracker.update([person("first", box)], 0.0)[0]["track_id"], "P1")
        self.assertEqual(tracker.update([], 1.0), [])
        self.assertEqual(tracker.update([person("back", box)], 2.0)[0]["track_id"], "P1")
        self.assertEqual(tracker.update([], 5.01), [])
        self.assertEqual(tracker.tracks, {})
        self.assertEqual(tracker.update([person("back_again", box)], 5.1)[0]["track_id"], "P2")

    def test_pt05_two_people_remain_distinct(self):
        tracker = PersonTracker()
        for frame in range(10):
            people = [
                person("left", [frame, 0, 100 + frame, 200]),
                person("right", [200 - frame, 0, 300 - frame, 200]),
            ]
            if frame % 2:
                people.reverse()
            result = tracker.update(people, float(frame))
            with self.subTest(frame=frame + 1):
                self.assertEqual(ids_by_ref(result), {"left": "P1", "right": "P2"})

    def test_pt06_overlap_and_crossing_only_guarantee_unique_assignments(self):
        tracker = PersonTracker()
        trajectory = [(0, 140), (30, 110), (60, 80), (100, 40), (130, 10)]
        assignments = []
        for frame, (a_x, b_x) in enumerate(trajectory):
            result = tracker.update([
                person("actor_a", [a_x, 0, a_x + 100, 200]),
                person("actor_b", [b_x, 0, b_x + 100, 200]),
            ], float(frame))
            mapping = ids_by_ref(result)
            assignments.append(mapping)
            with self.subTest(frame=frame + 1):
                self.assertEqual(len(result), 2)
                self.assertEqual(set(mapping.values()), {"P1", "P2"})
        # Position alone cannot prove identity during crossing. This controlled
        # trajectory demonstrates a switch; it does not promise re-identification.
        self.assertNotEqual(assignments[0], assignments[-1])

        # Complete overlap must still assign at most one detection to each ID.
        tracker = PersonTracker()
        tracker.update([
            person("actor_a", [60, 0, 160, 200]),
            person("actor_b", [80, 0, 180, 200]),
        ], 0.0)
        result = tracker.update([
            person("actor_a", [70, 0, 170, 200]),
            person("actor_b", [70, 0, 170, 200]),
        ], 1.0)
        self.assertEqual(len(result), 2)
        self.assertEqual({item["track_id"] for item in result}, {"P1", "P2"})

    def test_pt07_invalid_boxes_are_skipped_without_corrupting_tracks(self):
        invalid_boxes = {
            "missing": None, "empty": [], "short": [0, 1, 2],
            "long": [0, 1, 2, 3, 4], "zero_width": [1, 0, 1, 10],
            "zero_height": [0, 1, 10, 1], "reversed": [10, 10, 0, 0],
            "nan": [0, 0, float("nan"), 10],
            "infinity": [0, 0, 10, float("inf")],
            "negative_infinity": [float("-inf"), 0, 10, 10],
            "nonnumeric": [0, 0, "bad", 10], "nested": [[0], 0, 10, 10],
            "scalar": 42, "boolean": True, "text": "1234",
            "mapping": {"0": 0, "1": 0, "2": 10, "3": 10},
            "overflow": [0, 0, 10 ** 400, 10],
        }
        for name, box in invalid_boxes.items():
            with self.subTest(box=name):
                tracker = PersonTracker()
                valid = person("valid", [0, 0, 100, 200])
                tracker.update([valid], 0.0)
                result = tracker.update([person("bad", box), valid], 1.0)
                self.assertEqual(ids_by_ref(result), {"valid": "P1"})
                self.assertEqual(len(tracker.tracks), 1)
                self.assertEqual(tracker.next_id, 2)

    def test_pt08_ai_gap_at_and_beyond_max_age(self):
        for gap, expected in [(1.53, "P1"), (3.0, "P1"), (3.001, "P2"), (3.74, "P2")]:
            with self.subTest(gap=gap):
                tracker = PersonTracker(max_age=3.0)
                people = [person("person_0", [0, 0, 100, 200])]
                tracker.update(people, 100.0)
                result = tracker.update(people, 100.0 + gap)
                self.assertEqual(result[0]["track_id"], expected)
                self.assertEqual(len(tracker.tracks), 1)

    def test_clipped_box_precedes_raw_box_and_empty_clipped_falls_back(self):
        tracker = PersonTracker()
        detection = person("person_0", [-10, -10, 120, 220])
        detection["person_bbox_xyxy_clipped"] = [0, 0, 100, 200]
        result = tracker.update([detection], 0.0)
        self.assertEqual(result[0]["bbox"], (0., 0., 100., 200.))
        detection["person_bbox_xyxy_clipped"] = []
        result = tracker.update([detection], 1.0)
        self.assertEqual(result[0]["track_id"], "P1")
        self.assertEqual(result[0]["bbox"], (-10., -10., 120., 220.))


class TestLivePersonTrackerPolicy(unittest.TestCase):
    """Bounded AI latency tolerance without keeping observed-missing tracks."""

    def make_tracker(self):
        return PersonTracker(max_age=5.0, keep_unmatched=False)

    def test_exact_camera_frame8_to9_expiry_and_fix(self):
        box8 = [1.3, 92.81, 405.04, 415.77]
        box9 = [8.61, 75.21, 377.65, 373.33]
        time8, time9 = 245676.420614, 245680.0724245
        self.assertAlmostEqual(time9 - time8, 3.6518105)
        self.assertAlmostEqual(calculate_iou(box8, box9), 0.7562669951159354)
        for max_age, expected_id, expected_iou_calls in [(3.0, "P3", 0), (5.0, "P2", 1)]:
            with self.subTest(max_age=max_age):
                tracker = PersonTracker(max_age=max_age, keep_unmatched=False)
                # Allocate P1 and let it expire; frame 8 then starts with P2.
                tracker.update([person("seed", [1000, 1000, 1100, 1200])], time8 - 20)
                before = tracker.update([person("person_0", box8)], time8)
                self.assertEqual(before[0]["track_id"], "P2")
                with patch.object(tracker_module, "calculate_iou", wraps=calculate_iou) as iou:
                    after = tracker.update([person("person_0", box9)], time9)
                self.assertEqual(after[0]["track_id"], expected_id)
                self.assertEqual(after[0]["bbox"], tuple(box9))
                self.assertEqual(iou.call_count, expected_iou_calls)

    def test_ten_slow_updates_preserve_continuous_person(self):
        tracker = self.make_tracker()
        timestamp = 100.0
        for index in range(10):
            timestamp += (1.53, 2.995747, 3.6518105, 3.74)[index % 4]
            result = tracker.update([person(f"ref_{index}", [index, 0, index + 100, 200])], timestamp)
            with self.subTest(frame=index + 1):
                self.assertEqual(result[0]["track_id"], "P1")
                self.assertEqual(tracker.next_id, 2)

    def test_slow_update_limit_is_bounded_and_applies_before_iou(self):
        for gap, expected in [(3.74, "P1"), (5.0, "P1"), (5.001, "P2"), (30.0, "P2")]:
            with self.subTest(gap=gap):
                tracker = self.make_tracker()
                people = [person("same_ref", [0, 0, 100, 200])]
                tracker.update(people, 100.0)
                self.assertEqual(tracker.update(people, 100.0 + gap)[0]["track_id"], expected)

    def test_empty_result_retires_track_before_same_position_return(self):
        tracker = self.make_tracker()
        box = [0, 0, 100, 200]
        tracker.update([person("person_0", box)], 100.0)
        self.assertEqual(tracker.update([], 101.0), [])
        self.assertEqual(tracker.tracks, {})
        # Even an identical reference and bbox must not revive the retired ID.
        self.assertEqual(tracker.update([person("person_0", box)], 102.0)[0]["track_id"], "P2")

    def test_unmatched_track_is_retired_when_other_person_remains(self):
        tracker = self.make_tracker()
        left, right = [0, 0, 100, 200], [200, 0, 300, 200]
        tracker.update([person("left", left), person("right", right)], 100.0)
        result = tracker.update([person("new_right_ref", right)], 103.74)
        self.assertEqual(ids_by_ref(result), {"new_right_ref": "P2"})
        self.assertEqual(set(tracker.tracks), {2})
        # A new person at the departed person's old position gets a fresh ID.
        result = tracker.update([person("replacement", left), person("right", right)], 107.48)
        self.assertEqual(ids_by_ref(result), {"right": "P2", "replacement": "P3"})

    def test_two_people_stay_distinct_at_slow_update_rate(self):
        tracker = self.make_tracker()
        for index in range(10):
            people = [person(f"left_{index}", [index, 0, index + 100, 200]),
                      person(f"right_{index}", [300 - index, 0, 400 - index, 200])]
            if index % 2:
                people.reverse()
            result = tracker.update(people, 100.0 + index * 3.74)
            with self.subTest(frame=index + 1):
                self.assertEqual(ids_by_ref(result), {f"left_{index}": "P1", f"right_{index}": "P2"})

    def test_new_distant_person_is_not_assigned_a_departed_id(self):
        tracker = self.make_tracker()
        tracker.update([person("reused_ref", [0, 0, 100, 200])], 100.0)
        result = tracker.update([person("reused_ref", [500, 0, 600, 200])], 103.74)
        self.assertEqual(result[0]["track_id"], "P2")
        self.assertEqual(set(tracker.tracks), {2})

    def test_invalid_detection_cannot_keep_a_missing_person_alive(self):
        tracker = self.make_tracker()
        tracker.update([person("person_0", [0, 0, 100, 200])], 100.0)
        self.assertEqual(tracker.update([person("person_0", [0, 0, 0, 200])], 101.0), [])
        self.assertEqual(tracker.tracks, {})
        self.assertEqual(tracker.update([person("person_0", [0, 0, 100, 200])], 102.0)[0]["track_id"], "P2")


if __name__ == "__main__":
    unittest.main()
