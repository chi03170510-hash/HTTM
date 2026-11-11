"""Synthetic actor/recognition fixtures, never camera identity ground truth."""

import copy
import unittest
from dataclasses import FrozenInstanceError

from identity_continuity import (
    CurrentAssociation, CurrentFace, CurrentPerson, IdentityContinuityResolver,
    PriorTrack, project_current,
)


PERSON = (20., 40., 180., 420.)
FACE = (65., 65., 110., 115.)
# TEST DATA ONLY: not a production policy or an approved retention period.
SIMULATED_HOLD_SECONDS = 10.


def person(tid="P1", ref="person_0", bbox=PERSON):
    return CurrentPerson(tid, ref, bbox)


def face(sid="SV001", bbox=FACE):
    return CurrentFace(bbox, sid, .82)


def match(tid="P1", ref="person_0", sid="SV001", fid=1, bbox=FACE, status="matched"):
    return CurrentAssociation(tid, ref, fid, sid, status, bbox, .82)


class Scenario:
    def __init__(self, hold=SIMULATED_HOLD_SECONDS):
        self.resolver = IdentityContinuityResolver(iou_threshold=.3, max_update_gap=5.,
                                                  simulation_hold_seconds=hold)
        self.before = ()
        self.frame_id = 0

    def step(self, t, people=None, faces=(), matches=()):
        self.frame_id += 1
        people = (person(),) if people is None else tuple(people)
        result = self.resolver.update(self.frame_id, t, 100. + t, self.before, people,
                                      None if faces is None else tuple(faces), tuple(matches))
        self.before = tuple(PriorTrack(p.person_track_id, p.bbox, 100. + t) for p in people)
        return result

    def confirm(self, t=0., sid="SV001"):
        return self.step(t, faces=[face(sid)], matches=[match(sid=sid)])


def row(frame, tid="P1"):
    return next(r for r in frame.observations if r.person_track_id == tid)


class TestIdentityContinuity(unittest.TestCase):
    def test_confirmed_then_same_track_face_absent_is_simulated_retention(self):
        s = Scenario()
        self.assertEqual(row(s.confirm()).status, "face_confirmed")
        r = row(s.step(1.))
        self.assertEqual((r.status, r.proposed_student_id, r.face_present), ("track_retained", "SV001", False))
        self.assertEqual(r.last_face_age_seconds, 1.)

    def test_no_approved_hold_means_no_face_absent_proposal(self):
        s = Scenario(hold=None)
        s.confirm()
        r = row(s.step(1.))
        self.assertEqual((r.status, r.proposed_student_id, r.reason),
                         ("unknown", None, "hold_period_unapproved"))

    def test_expired_tracks_and_new_track_do_not_inherit(self):
        s = Scenario()
        s.confirm()
        result = s.step(5.116, people=[person("P3")])
        self.assertEqual(row(result).track_continuity, "expired")
        self.assertIsNone(row(result, "P3").proposed_student_id)

    def test_new_id_without_expiry_also_cannot_inherit(self):
        s = Scenario()
        s.confirm()
        r = row(s.step(1., people=[person("P3")]), "P3")
        self.assertEqual(r.reason, "no_valid_same_track_anchor")
        self.assertIsNone(r.proposed_student_id)

    def test_fresh_face_on_new_track_is_recovered_without_geometry_inheritance(self):
        s = Scenario()
        s.confirm()
        r = row(s.step(1., people=[person("P3")], faces=[face()], matches=[match("P3")]), "P3")
        self.assertEqual((r.status, r.proposed_student_id, r.last_face_age_seconds),
                         ("face_recovered", "SV001", 0.))

    def test_duplicate_student_claims_are_rejected_together(self):
        second_box, second_face = (320., 40., 480., 420.), (365., 65., 410., 115.)
        s = Scenario()
        result = s.step(0., people=[person(), person("P2", "second", second_box)],
                        faces=[face(), face(bbox=second_face)],
                        matches=[match(), match("P2", "second", fid=2, bbox=second_face)])
        self.assertTrue(all(r.status == "ambiguous" and r.proposed_student_id is None
                            for r in result.observations))
        self.assertTrue(all(r.proposed_student_id is None for r in s.step(1., people=[
            person(), person("P2", "second", second_box)]).observations))

    def test_crossing_drops_cache_even_if_tracker_ids_survive(self):
        s = Scenario()
        s.confirm()
        result = s.step(1., people=[person(), person("P2", "second", (100., 40., 260., 420.))])
        self.assertEqual(row(result).status, "ambiguous")
        self.assertIsNone(row(s.step(2.)).proposed_student_id)

    def test_unknown_face_cannot_be_hidden_by_cached_identity(self):
        s = Scenario()
        s.confirm()
        r = row(s.step(1., faces=[face("Unknown")], matches=[match(sid="Unknown")]))
        self.assertEqual((r.status, r.face_student_id, r.reason), ("unknown", "Unknown", "current_face_unknown"))
        self.assertIsNone(row(s.step(2.)).proposed_student_id)

    def test_unassociated_or_ambiguous_face_revokes_cache(self):
        for matches in ([], [match(status="ambiguous")]):
            with self.subTest(matches=matches):
                s = Scenario()
                s.confirm()
                self.assertEqual(row(s.step(1., faces=[face()], matches=matches)).status, "ambiguous")
                self.assertIsNone(row(s.step(2.)).proposed_student_id)
        s = Scenario()
        s.confirm()
        r = row(s.step(1., faces=[face(bbox=(500., 60., 550., 110.))]))
        self.assertEqual(r.status, "ambiguous")
        self.assertIsNone(r.face_present, "An unlocalized face is not proof of absence")
        self.assertIsNone(r.proposed_student_id)

    def test_heuristic_association_does_not_seed(self):
        s = Scenario()
        r = row(s.step(0., faces=[face()], matches=[match(status="matched_v4_heuristic")]))
        self.assertEqual(r.reason, "heuristic_association_cannot_seed")
        self.assertIsNone(row(s.step(1.)).proposed_student_id)

    def test_missing_face_data_is_not_reported_as_face_absence(self):
        s = Scenario()
        r = row(s.step(0., faces=None, matches=[match()]))
        self.assertIsNone(r.face_present)
        self.assertEqual(r.reason, "current_face_data_unavailable")
        self.assertIsNone(r.proposed_student_id)

    def test_invalid_or_non_increasing_timestamp_cancels(self):
        for t in (float("nan"), float("inf"), 0., -1.):
            with self.subTest(t=t):
                s = Scenario()
                s.confirm()
                result = s.step(t)
                self.assertFalse(result.timestamp_valid)
                self.assertIsNone(row(result).proposed_student_id)
                self.assertIsNone(row(s.step(1.)).proposed_student_id)

    def test_tracker_clock_regression_cancels(self):
        s = Scenario()
        s.confirm()
        result = s.resolver.update(2, 1., 99., s.before, (person(),), (), ())
        self.assertFalse(result.timestamp_valid)
        self.assertIsNone(row(result).proposed_student_id)

    def test_future_capture_or_prior_track_timestamp_is_invalid(self):
        for captured, updated, prior in (
                (101., 100., ()), (1., 101., (PriorTrack("P1", PERSON, 102.),))):
            with self.subTest(captured=captured, prior=prior):
                result = Scenario().resolver.update(1, captured, updated, prior, (person(),), (), ())
                self.assertFalse(result.timestamp_valid)
                self.assertIsNone(row(result).proposed_student_id)

    def test_capture_gap_is_separate_from_tracker_update_gap(self):
        s = Scenario()
        s.confirm()
        result = s.resolver.update(2, 6., 101., s.before, (person(),), (), ())
        self.assertEqual(result.reset_reason, "capture_gap_exceeds_existing_limit")
        self.assertIsNone(row(result).proposed_student_id)

    def test_identity_expiry_uses_last_face_not_last_bbox(self):
        s = Scenario(hold=2.)  # Explicit synthetic test budget only.
        s.confirm()
        self.assertEqual(row(s.step(1.)).last_face_age_seconds, 1.)
        self.assertEqual(row(s.step(2.)).status, "track_retained")
        r = row(s.step(2.001))
        self.assertEqual((r.status, r.reason), ("unknown", "identity_hold_expired"))
        self.assertAlmostEqual(r.last_face_age_seconds, 2.001)

    def test_detection_loss_cancels_then_same_location_cannot_resume(self):
        s = Scenario()
        s.confirm()
        self.assertEqual(row(s.step(1., people=[])).reason, "person_detection_missing")
        self.assertIsNone(row(s.step(2., people=[person("P3")]), "P3").proposed_student_id)

    def test_retired_track_diagnostic_order_is_deterministic(self):
        s = Scenario()
        s.step(0., people=[person("P2", "second", (320., 40., 480., 420.)), person()])
        result = s.step(1., people=[])
        self.assertEqual([r.person_track_id for r in result.observations], ["P1", "P2"])

    def test_new_face_has_priority_over_previous_student(self):
        s = Scenario()
        s.confirm()
        r = row(s.confirm(1., "SV002"))
        self.assertEqual(r.proposed_student_id, "SV002")
        self.assertEqual(row(s.step(2.)).proposed_student_id, "SV002")

    def test_new_face_owner_revokes_old_track_before_retention(self):
        s = Scenario()
        s.confirm()
        second_box, second_face = (320., 40., 480., 420.), (365., 65., 410., 115.)
        result = s.step(1., people=[person(), person("P2", "second", second_box)],
                        faces=[face(bbox=second_face)],
                        matches=[match("P2", "second", bbox=second_face)])
        self.assertIsNone(row(result).proposed_student_id)
        self.assertEqual(row(result, "P2").proposed_student_id, "SV001")

    def test_unresolved_fresh_face_elsewhere_also_blocks_cached_student(self):
        s = Scenario()
        s.confirm()
        second_box, second_face = (320., 40., 480., 420.), (365., 65., 410., 115.)
        result = s.step(1., people=[person(), person("P2", "second", second_box)],
                        faces=[face(bbox=second_face)], matches=[])
        self.assertIsNone(row(result).proposed_student_id)
        self.assertEqual(row(result).reason, "student_claimed_elsewhere")

    def test_person_reference_reorder_is_not_an_identity_key(self):
        s = Scenario()
        s.confirm()
        self.assertEqual(row(s.step(1., people=[person(ref="changed")])).proposed_student_id, "SV001")

    def test_unobserved_replacement_is_indistinguishable_in_simulation(self):
        # Same boxes/no visible face cannot prove the actor is unchanged. This is
        # a documented limitation, not a successful real-identity assertion.
        s = Scenario()
        s.confirm()
        self.assertEqual(row(s.step(1.)).status, "track_retained")
        safe = Scenario(hold=None)
        safe.confirm()
        self.assertIsNone(row(safe.step(1.)).proposed_student_id)

    def test_raw_recognition_mismatch_cannot_seed(self):
        s = Scenario()
        r = row(s.step(0., faces=[face("SV002")], matches=[match()]))
        self.assertIsNone(r.proposed_student_id)
        self.assertEqual(r.reason, "current_recognition_not_verified")

    def test_projections_are_frozen_copies_and_exclude_behavior_inputs(self):
        people = [{"track_id": "P1", "person_ref": "person_0", "bbox": list(PERSON),
                   "behaviors": {"sleeping": {"state": "detected"}}}]
        faces = [{"face_bbox": list(FACE), "student_id": "SV001", "score": .82, "embedding": [123]}]
        matches = [{"person_track_id": "P1", "person_ref": "person_0", "track_id": 1,
                    "student_id": "SV001", "association_status": "matched", "face_bbox": list(FACE),
                    "face_score": .82, "behaviors": people[0]["behaviors"]}]
        original = copy.deepcopy((people, faces, matches))
        projected = project_current(people, faces, matches)
        result = IdentityContinuityResolver(iou_threshold=.3, max_update_gap=5.).update(
            1, 0., 100., (), *projected)
        self.assertEqual((people, faces, matches), original)
        with self.assertRaises(FrozenInstanceError):
            projected[0][0].person_ref = "changed"
        self.assertFalse(hasattr(projected[0][0], "behaviors"))
        people[0]["bbox"][0] = -100
        self.assertEqual(projected[0][0].bbox, PERSON)
        with self.assertRaises(FrozenInstanceError):
            result.observations[0].proposed_student_id = "changed"

    def test_invalid_simulation_period_is_rejected(self):
        for value in (0, -1, float("inf"), float("nan"), True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Scenario(hold=value)


if __name__ == "__main__":
    unittest.main()
