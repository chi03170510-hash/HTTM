"""Test-only factory injection; production source/config never changed."""

import unittest
from unittest.mock import patch

import camera_coordinator_live as live
from behavior_warning import DEFAULT_WARNING_CONFIG, WarningConfig
from evaluate_face_tracker_parameters import MAX_AGES
from face_tracker import FaceTracker as RealFaceTracker
from person_tracker_diagnostics import DiagnosticConfig
from test_ch21_warning_pipeline import events, run_pipeline, source
from test_ch20_duration_pipeline import phone_value


def samples(count=5, student="SV001"):
    return [source(i, 198.5 + 3. * (i - 1), student=student) for i in range(1, count + 1)]


def replay_pipeline(items, age, config=WarningConfig(using_phone=True)):
    def independent_tracker(**requested):
        if requested != {"iou_threshold": .3, "max_age": 2.0}:
            raise AssertionError("Camera Live production parameters changed")
        return RealFaceTracker(iou_threshold=requested["iou_threshold"], max_age=age)
    with (patch.object(live, "FaceTracker", side_effect=independent_tracker),
          patch.object(live.DiagnosticConfig, "from_environment", return_value=DiagnosticConfig()),
          patch("http.client.HTTPConnection.request", side_effect=AssertionError("API forbidden"))):
        return run_pipeline(items, config=config)


class TestFaceTrackerParameterPipeline(unittest.TestCase):
    def test_face_id_changes_alone_do_not_reset_duration_or_warning_at_any_age(self):
        items = samples()
        for age in MAX_AGES:
            with self.subTest(age=age):
                trace = replay_pipeline(items, age)
                ids = [r["matches"][0]["track_id"] for r in trace.results]
                self.assertEqual(ids, [1, 2, 3, 4, 5] if age == 2. else [1] * 5)
                self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., 9., 12.])
                self.assertEqual([e.frame_id for e in events(trace)], [3])
                self.assertEqual({e.student_id for e in events(trace)}, {"SV001"})
                self.assertTrue(all(r["matches"][0]["person_track_id"] == "P1" for r in trace.results))
                self.assertEqual(len(trace.samples), 10)

    def test_unknown_duplicate_and_lost_association_break_episodes_for_all_ages(self):
        for age in MAX_AGES:
            for mode, kwargs in (("unknown", {"student": "Unknown"}), ("duplicate", {"second_student": "SV001"}),
                                 ("association_lost", {"associated": False})):
                with self.subTest(age=age, mode=mode):
                    items = samples(7)
                    items[3] = source(4, 207.5, **kwargs)
                    trace = replay_pipeline(items, age)
                    self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., None, 0., 3., 6.])
                    self.assertEqual([e.frame_id for e in events(trace)], [3, 7])
                    self.assertNotEqual(events(trace)[0].episode_id, events(trace)[1].episode_id)
                    self.assertTrue(all(v.state == "CLEAR" for v in trace.results[3]["warning_frame"].views))

    def test_known_student_change_resets_despite_reused_face_track_and_person(self):
        for age in MAX_AGES:
            with self.subTest(age=age):
                items = samples(3) + [source(i, 198.5 + 3. * (i - 1), student="SV002") for i in range(4, 9)]
                trace = replay_pipeline(items, age)
                self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., 0., 3., 6., 9., 12.])
                self.assertEqual([(e.frame_id, e.student_id) for e in events(trace)], [(3, "SV001"), (6, "SV002")])
                self.assertFalse(trace.results[3]["warning_frame"].events)
                self.assertNotEqual(events(trace)[0].episode_id, events(trace)[1].episode_id)
                if age != 2.:
                    self.assertEqual({r["matches"][0]["track_id"] for r in trace.results}, {1})

    def test_default_warning_stays_disabled_and_no_official_side_effects(self):
        for age in MAX_AGES:
            with self.subTest(age=age):
                trace = replay_pipeline(samples(), age, config=DEFAULT_WARNING_CONFIG)
                self.assertFalse(events(trace))
                self.assertFalse(trace.errors)
                self.assertEqual(phone_value(trace.results[-1]), 12.)
                self.assertTrue(all(r["warning_frame"].enabled_behaviors == () for r in trace.results))
                self.assertFalse(any("WARNING" in label for s in trace.samples for label in s.labels))
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)
        self.assertEqual(live.WARNING_CONFIG.enabled_behaviors, ())
        # Constructor patch is restored; normal instances still default to 2s.
        self.assertIs(live.FaceTracker, RealFaceTracker)
        self.assertEqual(live.FaceTracker().max_age, 2.)


if __name__ == "__main__":
    unittest.main()
