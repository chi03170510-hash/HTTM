"""Controlled V7 evidence fixtures; no camera, models, persistence or network."""

import copy
import unittest
from dataclasses import FrozenInstanceError, replace

from behavior_duration import BEHAVIORS, BehaviorDurationTracker
from behavior_warning import (
    BehaviorWarningObserver, DEFAULT_WARNING_CONFIG, WarningConfig, evidence_check,
)


def detected_evidence(name="using_phone"):
    """Use the existing V7 field contracts, not just a detected label."""
    evidence = {"missing_evidence": []}
    if name == "using_phone":
        support = ["behavior_yolo_using_phone_candidate", "unique_same_person_coco_phone_object"]
        evidence.update(phone_association_status="ASSOCIATED_GEOMETRICALLY",
                        phone_detection_ids=["phone_1"], phone_boxes=[[90, 120, 105, 145]],
                        phone_object_geometry={"area_ratio": .01, "height_ratio": .07,
                                               "width_ratio": .1,
                                               "limits": {"area_ratio": .08, "height_ratio": .25, "width_ratio": .4}})
    elif name == "sleeping":
        support = ["behavior_yolo_sleeping_candidate"]
        evidence.update(sleeping_yolo_threshold=.75, sleeping_yolo_confidence=.8,
                        sleeping_yolo_box=[20, 40, 180, 420])
    elif name == "eating":
        support = ["unique_same_person_supported_food_object", "food_box_near_visible_mouth"]
        evidence.update(same_person_food_association=True, mouth_source="face_landmark",
                        food_mouth_matches=[{"food_detection_id": "food_1", "class_name": "banana",
                                             "object_confidence": .6, "mouth_distance_head_height": .1,
                                             "proximity_threshold_head_height": .22}])
    else:
        support = ["head_orientation_matches_calibrated_rear_direction"]
        evidence.update(rear_direction_calibrated=True, head_yaw_deg=85.,
                        rear_yaw_target_deg=90., rear_direction_tolerance_deg=20.)
    return {"state": "detected", "status": "IMAGE_ONLY", "decision": "detected",
            "model_confidence": None if name == "eating" else .8,
            "supporting_evidence": support, "contradicting_evidence": [], "evidence": evidence}


class WarningReplay:
    """Actual CH-20 duration feeding the independent CH-21 observer."""
    def __init__(self, config=WarningConfig(using_phone=True)):
        self.duration = BehaviorDurationTracker()
        self.observer = BehaviorWarningObserver(config)
        self.frame_id = 0

    def step(self, captured_at, *, tid="P1", student="SV001", state="detected",
             association="matched", enabled_states=None, decision=None, present=True,
             second_student=None, age=.2):
        self.frame_id += 1
        people, tracks, matches = [], [], []
        if present:
            for index, sid in enumerate([student] + ([second_student] if second_student is not None else [])):
                track = tid if index == 0 else "P2"
                ref = f"person_{index}"
                states = {name: "not_detected" for name in BEHAVIORS}
                states.update(enabled_states or {"using_phone": state})
                decisions = {name: detected_evidence(name) if value == "detected" else {"state": value}
                             for name, value in states.items()}
                if decision is not None:
                    decisions["using_phone"] = copy.deepcopy(decision)
                people.append({"person_ref": ref, "behaviors": decisions})
                tracks.append({"track_id": track, "person_ref": ref, "bbox": [20, 40, 180, 420]})
                if association is not None:
                    matches.append({"person_track_id": track, "person_ref": ref, "student_id": sid,
                                    "association_status": association, "track_id": index + 1,
                                    "behaviors": copy.deepcopy(decisions)})
        frame = self.duration.update(self.frame_id, captured_at, people, tracks, matches)
        self.last_behavior, self.last_matches = frame, matches
        return self.observer.update(frame, matches, observed_at=captured_at + age)


def phone_view(frame, person="P1"):
    return next(v for v in frame.views if v.person_track_id == person and v.behavior == "using_phone")


class TestBehaviorWarning(unittest.TestCase):
    def test_production_defaults_are_all_disabled_and_immutable(self):
        self.assertEqual(DEFAULT_WARNING_CONFIG.enabled_behaviors, ())
        with self.assertRaises(FrozenInstanceError):
            DEFAULT_WARNING_CONFIG.using_phone = True
        replay = WarningReplay(DEFAULT_WARNING_CONFIG)
        for t in (100., 103., 106.):
            result = replay.step(t, enabled_states={name: "detected" for name in BEHAVIORS})
            self.assertEqual(result.events, ())
            self.assertTrue(all(view.reason == "disabled" for view in result.views))

    def test_configuration_requires_explicit_boolean_flags(self):
        for flag in ("false", "true", 1, None):
            with self.subTest(flag=flag), self.assertRaises(ValueError):
                WarningConfig(using_phone=flag)

    def test_raw_duration_4_9_5_0_5_01_and_one_event(self):
        replay = WarningReplay()
        results = [replay.step(t) for t in (100., 102., 104.9, 105., 105.01, 106.5)]
        self.assertEqual([len(r.events) for r in results], [0, 0, 0, 0, 1, 0])
        for result, seconds in zip(results[2:5], (4.9, 5., 5.01)):
            self.assertAlmostEqual(phone_view(result).qualified_seconds, seconds)
        self.assertEqual(phone_view(results[3]).state, "CANDIDATE")
        self.assertEqual(phone_view(results[4]).state, "WARNING_EXPERIMENTAL")
        self.assertEqual(f"{results[4].events[0].estimated_seconds:.1f}", "5.0")
        self.assertEqual(results[4].events[0].observation_count, 5)

    def test_duplicate_frame_returns_no_event_and_does_not_advance(self):
        replay = WarningReplay()
        for t in (100., 103., 106.):
            final = replay.step(t)
        for _ in range(10):
            duplicate = replay.observer.update(replay.last_behavior, replay.last_matches, observed_at=106.4)
            self.assertEqual(duplicate.events, ())
            self.assertEqual(duplicate.views, final.views)
        self.assertEqual(replay.step(107.).events, ())

    def test_each_non_detected_state_ends_episode(self):
        for state in ("suspected", "unknown", "not_detected"):
            with self.subTest(state=state):
                replay = WarningReplay()
                replay.step(100.)
                old = replay.step(103.)
                self.assertEqual(phone_view(replay.step(104., state=state)).state, "CLEAR")
                new = replay.step(105.)
                self.assertNotEqual(phone_view(new).episode_id, phone_view(old).episode_id)
                self.assertEqual(phone_view(new).qualified_seconds, 0.)

    def test_missing_and_contradictory_evidence_cannot_warn(self):
        missing = detected_evidence()
        del missing["evidence"]
        conflict = detected_evidence()
        conflict["contradicting_evidence"] = ["ambiguous_phone_owner"]
        absent = detected_evidence()
        absent["evidence"]["missing_evidence"] = ["phone_object"]
        inconsistent = detected_evidence()
        inconsistent["decision"] = "suspected"
        for decision in (missing, conflict, absent, inconsistent):
            with self.subTest(decision=decision):
                replay = WarningReplay()
                for t in (100., 103., 106.):
                    result = replay.step(t, decision=decision)
                    self.assertEqual(result.events, ())
                    self.assertEqual(phone_view(result).state, "CLEAR")

    def test_evidence_break_restarts_qualified_span_while_ch20_continues(self):
        replay = WarningReplay()
        replay.step(100.)
        replay.step(103.)
        bad = detected_evidence()
        bad["supporting_evidence"] = []
        replay.step(104., decision=bad)
        first = replay.step(106.)
        duration = next(b for b in replay.last_behavior.people[0].behaviors if b.name == "using_phone")
        self.assertEqual(duration.estimated_seconds, 6.)
        self.assertEqual(phone_view(first).qualified_seconds, 0.)
        self.assertFalse(first.events)
        self.assertFalse(replay.step(109.).events)
        final = replay.step(112.)
        self.assertEqual(final.events[0].qualified_seconds, 6.)
        self.assertEqual(final.events[0].estimated_seconds, 12.)

    def test_one_event_latch_survives_evidence_break_in_same_episode(self):
        replay = WarningReplay()
        for t in (100., 103., 106.):
            warned = replay.step(t)
        bad = detected_evidence()
        bad["contradicting_evidence"] = ["conflict"]
        broken = replay.step(107., decision=bad)
        for t in (108., 111., 114.):
            resumed = replay.step(t)
            self.assertEqual(resumed.events, ())
        self.assertEqual(phone_view(broken).episode_id, warned.events[0].episode_id)
        self.assertEqual(phone_view(resumed).episode_id, warned.events[0].episode_id)

    def test_heuristic_and_missing_association_do_not_warn(self):
        for association in ("matched_v4_heuristic", "unmatched", None):
            with self.subTest(association=association):
                replay = WarningReplay()
                for t in (100., 103., 106.):
                    self.assertEqual(replay.step(t, association=association).events, ())

    def test_unknown_and_duplicate_student_do_not_warn(self):
        for args in ({"student": "Unknown"}, {"student": ""}, {"second_student": "SV001"}):
            with self.subTest(args=args):
                replay = WarningReplay()
                for t in (100., 103., 106.):
                    result = replay.step(t, **args)
                    self.assertFalse(result.events)
                    self.assertTrue(all(v.state == "CLEAR" for v in result.views))

    def test_invalid_person_track_is_rejected(self):
        for tid in ("", "Unknown", "P0", "person_1", None):
            with self.subTest(tid=tid):
                replay = WarningReplay()
                for t in (100., 103., 106.):
                    self.assertFalse(replay.step(t, tid=tid).events)

    def test_person_and_student_change_start_new_episode(self):
        for changes in ({"tid": "P3"}, {"student": "SV002"}):
            with self.subTest(changes=changes):
                replay = WarningReplay()
                replay.step(100.)
                old = replay.step(103.)
                new = replay.step(106., **changes)
                self.assertFalse(new.events)
                view = next(v for v in new.views if v.behavior == "using_phone")
                self.assertEqual(view.qualified_seconds, 0.)
                self.assertNotEqual(view.episode_id, phone_view(old).episode_id)

    def test_association_loss_resets_before_recovery(self):
        replay = WarningReplay()
        replay.step(100.)
        replay.step(103.)
        self.assertFalse(replay.step(104., association=None).events)
        self.assertEqual(phone_view(replay.step(106.)).qualified_seconds, 0.)

    def test_capture_gap_above_five_restarts_at_zero(self):
        replay = WarningReplay()
        replay.step(100.)
        replay.step(103.)
        result = replay.step(108.01)
        self.assertFalse(result.events)
        self.assertEqual(phone_view(result).qualified_seconds, 0.)

    def test_exact_five_observation_gap_is_allowed(self):
        replay = WarningReplay()
        for t in (100., 103., 108.):
            result = replay.step(t)
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].qualified_seconds, 8.)

    def test_capture_age_boundary_and_stale_break(self):
        replay = WarningReplay()
        replay.step(100., age=5.)
        replay.step(103., age=5.)
        self.assertEqual(len(replay.step(106., age=5.).events), 1)
        replay = WarningReplay()
        replay.step(100.)
        replay.step(103.)
        stale = replay.step(106., age=5.01)
        self.assertEqual(phone_view(stale).reason, "stale_capture")
        self.assertFalse(stale.events)
        self.assertEqual(phone_view(replay.step(107.)).qualified_seconds, 0.)

    def test_one_event_latch_survives_a_stale_observation(self):
        replay = WarningReplay()
        for t in (100., 103., 106.):
            first = replay.step(t)
        replay.step(107., age=5.01)
        for t in (108., 111., 114.):
            result = replay.step(t)
            self.assertFalse(result.events)
        self.assertEqual(phone_view(result).state, "WARNING_EXPERIMENTAL")
        self.assertEqual(phone_view(result).episode_id, first.events[0].episode_id)

    def test_non_increasing_timestamp_rejects_and_resets(self):
        for bad_time in (103., 102.):
            with self.subTest(timestamp=bad_time):
                replay = WarningReplay()
                replay.step(100.)
                replay.step(103.)
                self.assertFalse(replay.step(bad_time).valid)
                self.assertEqual(phone_view(replay.step(106.)).qualified_seconds, 0.)

    def test_invalid_metadata_and_future_timestamp_do_not_warn(self):
        for change, now in (({"captured_at": float("nan")}, 104.),
                            ({"captured_at": float("inf")}, 104.),
                            ({"frame_id": 0}, 104.), ({"timestamp_valid": False}, 104.),
                            ({}, 99.), ({}, float("nan"))):
            with self.subTest(change=change, now=now):
                replay = WarningReplay()
                replay.step(100.)
                bad = replace(replay.last_behavior, **change)
                result = replay.observer.update(bad, replay.last_matches, observed_at=now)
                self.assertFalse(result.valid)
                self.assertFalse(result.events)

    def test_duplicate_metadata_mismatch_and_old_frame_are_rejected(self):
        replay = WarningReplay()
        replay.step(100.)
        first = replay.last_behavior
        replay.step(103.)
        for bad in (replace(replay.last_behavior, captured_at=103.1), first):
            result = replay.observer.update(bad, replay.last_matches, observed_at=104.)
            self.assertFalse(result.valid)
            self.assertFalse(result.events)

    def test_two_behaviors_are_independent(self):
        replay = WarningReplay(WarningConfig(sleeping=True, using_phone=True))
        for t in (100., 103., 106.):
            result = replay.step(t, enabled_states={"sleeping": "detected", "using_phone": "detected"})
        self.assertEqual({e.behavior for e in result.events}, {"sleeping", "using_phone"})
        self.assertEqual(len({e.episode_id for e in result.events}), 2)

    def test_two_students_are_independent(self):
        replay = WarningReplay()
        for t in (100., 103., 106.):
            result = replay.step(t, second_student="SV002")
        self.assertEqual({(e.student_id, e.person_track_id) for e in result.events},
                         {("SV001", "P1"), ("SV002", "P2")})

    def test_absence_or_behavior_end_allows_a_new_episode_warning(self):
        for changes in ({"present": False}, {"state": "not_detected"}):
            with self.subTest(changes=changes):
                replay = WarningReplay()
                for t in (100., 103., 106.):
                    first = replay.step(t)
                replay.step(107., **changes)
                for t in (108., 111., 114.):
                    second = replay.step(t)
                self.assertEqual(len(second.events), 1)
                self.assertNotEqual(first.events[0].episode_id, second.events[0].episode_id)

    def test_behavior_specific_evidence_contracts(self):
        for name in BEHAVIORS:
            with self.subTest(behavior=name):
                self.assertTrue(evidence_check(name, detected_evidence(name))[0])
                replay = WarningReplay(WarningConfig(**{name: True}))
                for t in (100., 103., 106.):
                    result = replay.step(t, enabled_states={name: "detected"})
                self.assertEqual([e.behavior for e in result.events], [name])

    def test_behavior_specific_missing_geometry_mouth_calibration_and_threshold(self):
        for name, field, value in (("using_phone", "phone_object_geometry", {}),
                                   ("using_phone", "phone_detection_ids", [None]),
                                   ("using_phone", "phone_detection_ids", ["phone_1", "phone_1"]),
                                   ("eating", "mouth_source", "pose_or_person_head_proxy"),
                                   ("eating", "food_mouth_matches", [{}]),
                                   ("turning_back", "rear_direction_calibrated", False),
                                   ("sleeping", "sleeping_yolo_confidence", .74)):
            with self.subTest(behavior=name):
                decision = detected_evidence(name)
                decision["evidence"][field] = value
                self.assertFalse(evidence_check(name, decision)[0])

    def test_raw_duration_cannot_be_substituted_by_qualified_span(self):
        replay = WarningReplay()
        for t in (100., 103.):
            replay.step(t)
        replay.step(106.)
        behavior = replay.last_behavior.people[0]
        behaviors = tuple(replace(b, estimated_seconds=5.) if b.name == "using_phone" else b for b in behavior.behaviors)
        # A malformed duration must not gain a warning just from elapsed captures.
        observer = BehaviorWarningObserver(WarningConfig(using_phone=True))
        for fid, capture, duration in ((1, 100., 1.), (2, 103., 3.), (3, 106., 5.)):
            item = replace(behavior, behaviors=tuple(replace(b, estimated_seconds=duration)
                                                     if b.name == "using_phone" else b for b in behaviors))
            frame = replace(replay.last_behavior, frame_id=fid, captured_at=capture, people=(item,))
            result = observer.update(frame, replay.last_matches, observed_at=capture + .2)
        self.assertFalse(result.events)

    def test_frozen_outputs_and_capture_based_panel_freshness(self):
        replay = WarningReplay()
        for t in (100., 103., 106.):
            result = replay.step(t)
        with self.assertRaises(FrozenInstanceError):
            result.events[0].student_id = "SV002"
        self.assertTrue(result.is_fresh(111., 3, 106.))
        self.assertFalse(result.is_fresh(111.01, 3, 106.))
        self.assertFalse(result.is_fresh(107., 4, 106.))
        self.assertFalse(result.is_fresh(107., 3, 105.))


if __name__ == "__main__":
    unittest.main()
