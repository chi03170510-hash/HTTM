"""Anonymous motion tracking regressions; no camera, face data or models."""

import itertools
import random
import unittest

from person_tracker import PersonTracker, _minimum_assignment
from test_person_tracker import person, ids_by_ref


class TestMotionTracker(unittest.TestCase):
    def make_tracker(self):
        return PersonTracker(max_age=5.0, keep_unmatched=True, motion_matching=True)

    def test_assignment_matches_brute_force_optimum(self):
        rng = random.Random(42)
        for n in range(1, 5):
            for _ in range(20):
                costs = [[rng.uniform(-1, 2) for _ in range(n + 2)] for _ in range(n)]
                pairs = _minimum_assignment(costs)
                actual = sum(costs[row][column] for row, column in pairs)
                expected = min(sum(costs[row][column] for row, column in enumerate(columns))
                               for columns in itertools.permutations(range(n + 2), n))
                self.assertAlmostEqual(actual, expected)
                self.assertEqual(len({column for _, column in pairs}), n)

    def test_crossing_and_detection_reordering_preserve_directional_ids(self):
        tracker = self.make_tracker()
        trajectory = [(0, 140), (30, 110), (60, 80), (100, 40), (130, 10)]
        for index, (a, b) in enumerate(trajectory):
            detections = [person('a', [a, 0, a + 100, 200]),
                          person('b', [b, 0, b + 100, 200])]
            if index % 2:
                detections.reverse()
            self.assertEqual(ids_by_ref(tracker.update(detections, float(index))),
                             {'a': 'P1', 'b': 'P2'})

    def test_brief_occlusion_keeps_id_but_does_not_emit_ghost_box(self):
        tracker = self.make_tracker()
        tracker.update([person('a', [0, 0, 100, 200])], 0.)
        tracker.update([person('a', [40, 0, 140, 200])], 1.)
        self.assertEqual(tracker.update([], 2.), [])
        result = tracker.update([person('new_ref', [120, 0, 220, 200])], 3.)
        self.assertEqual(ids_by_ref(result), {'new_ref': 'P1'})
        self.assertEqual(result[0]['bbox'], (120., 0., 220., 200.))

    def test_constant_velocity_uses_elapsed_seconds_not_frame_count(self):
        tracker = self.make_tracker()
        for timestamp in (0., .5, 1.5, 3.):
            x = timestamp * 80
            result = tracker.update([person('a', [x, 0, x + 100, 200])], timestamp)
            self.assertEqual(result[0]['track_id'], 'P1')

    def test_same_position_after_expiry_gets_new_id(self):
        tracker = self.make_tracker()
        detections = [person('a', [0, 0, 100, 200])]
        tracker.update(detections, 0.)
        self.assertEqual(tracker.update(detections, 5.001)[0]['track_id'], 'P2')

    def test_distant_arrival_does_not_take_dormant_id(self):
        tracker = self.make_tracker()
        tracker.update([person('a', [0, 0, 100, 200])], 0.)
        tracker.update([], 1.)
        result = tracker.update([person('a', [500, 0, 600, 200])], 2.)
        self.assertEqual(result[0]['track_id'], 'P2')

    def test_full_overlap_still_has_unique_assignments(self):
        tracker = self.make_tracker()
        tracker.update([person('a', [0, 0, 100, 200]),
                        person('b', [100, 0, 200, 200])], 0.)
        result = tracker.update([person('x', [50, 0, 150, 200]),
                                 person('y', [50, 0, 150, 200])], 1.)
        self.assertEqual(len(result), 2)
        self.assertEqual({item['track_id'] for item in result}, {'P1', 'P2'})

    def test_global_assignment_avoids_greedy_local_best(self):
        # A-X is strongest, but B can only take X. Global matching preserves both.
        tracker = self.make_tracker()
        tracker.update([person('a', [0, 0, 100, 200]),
                        person('b', [80, 0, 180, 200])], 0.)
        result = tracker.update([person('x', [30, 0, 130, 200]),
                                 person('y', [-50, 0, 50, 200])], 1.)
        self.assertEqual(ids_by_ref(result), {'y': 'P1', 'x': 'P2'})

    def test_invalid_or_backwards_timestamp_does_not_mutate_tracks(self):
        tracker = self.make_tracker()
        tracker.update([person('a', [0, 0, 100, 200])], 1.)
        for value in (float('nan'), float('inf'), 0.):
            with self.assertRaises(ValueError):
                tracker.update([], value)
        self.assertEqual(tracker.tracks[1].last_seen, 1.)


if __name__ == '__main__':
    unittest.main()
