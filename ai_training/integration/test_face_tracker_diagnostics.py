"""Offline A-F and diagnostic integrity; no camera/model/biometric vectors."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

from face_tracker_diagnostics import FaceTrackerDiagnostics
from face_tracker import FaceTracker
from person_tracker_diagnostics import DiagnosticConfig, PersonTrackerDiagnostics
from replay_face_tracker_diagnostics import replay_records


CAMERA_GAPS = (3.715, 3.377, 2.926, 2.975, 2.847, 3.106)


def face(box=(30, 40, 80, 100), student="SV001", score=.8):
    return {"face_bbox": list(box), "student_id": student, "score": score}


class FaceReplay:
    def __init__(self):
        self.tracker = FaceTracker(iou_threshold=.3, max_age=2.)
        self.observer = FaceTrackerDiagnostics()
        self.records = []

    def step(self, timestamp, faces=None, *, captured_at=None, person_offset=.01, matches=()):
        faces = [face()] if faces is None else faces
        frame_id = len(self.records) + 1
        before = self.observer.before_update(self.tracker, timestamp)
        detections = [(tuple(f["face_bbox"]), f.get("student_id"), f.get("score")) for f in faces]
        output = self.tracker.update(detections, timestamp)
        result = {"frame_id": frame_id, "captured_at": timestamp - .5 if captured_at is None else captured_at,
                  "face_time": .2, "hybrid_time": .25, "matches": list(matches)}
        block = self.observer.build(before, self.tracker, faces, result, timestamp + person_offset)
        record = {"schema": "person_tracker_diagnostic_v1", "frame_id": frame_id,
                  "captured_at": result["captured_at"], "tracker_updated_at": timestamp + person_offset,
                  "face_tracker": block}
        self.records.append(record)
        return block, output


class TestFaceTrackerDiagnostics(unittest.TestCase):
    def test_case_a_gap_1_5_keeps_face_id(self):
        replay = FaceReplay()
        replay.step(100.)
        block, output = replay.step(101.5)
        self.assertEqual([t[0] for t in output], [1])
        self.assertEqual(block["face_update_gap"], 1.5)
        self.assertEqual(block["decisions"], [{"face_track_id": 1, "action": "matched", "reason": "selected_by_tracker"}])

    def test_case_b_exact_two_keeps_face_id(self):
        replay = FaceReplay()
        replay.step(100.)
        block, output = replay.step(102.)
        self.assertEqual([t[0] for t in output], [1])
        self.assertEqual(block["tracks_before"][0]["age_seconds"], 2.)
        self.assertTrue(block["candidate_ious"][0]["considered_by_tracker"])
        self.assertFalse(any(d["action"] == "expired" for d in block["decisions"]))

    def test_case_c_2_001_expires_before_iou_even_with_identical_bbox(self):
        replay = FaceReplay()
        replay.step(100.)
        block, output = replay.step(102.001)
        self.assertEqual([t[0] for t in output], [2])
        self.assertEqual(block["candidate_ious"][0]["iou"], 1.)
        self.assertTrue(block["candidate_ious"][0]["threshold_passed"])
        self.assertFalse(block["candidate_ious"][0]["considered_by_tracker"])
        self.assertEqual([d["action"] for d in block["decisions"]], ["expired", "new"])

    def test_case_d_six_camera_gap_values_force_expiry_on_unchanged_face(self):
        replay = FaceReplay()
        timestamp = 100.
        replay.step(timestamp)
        for index, gap in enumerate(CAMERA_GAPS, 1):
            timestamp += gap
            block, output = replay.step(timestamp, captured_at=99.5 + index * 1.5)
            self.assertAlmostEqual(block["face_update_gap"], gap)
            self.assertEqual([t[0] for t in output], [index + 1])
            self.assertEqual([d["action"] for d in block["decisions"]], ["expired", "new"])
            self.assertEqual(block["candidate_ious"][0]["iou"], 1.)
        self.assertTrue(all(r["consistent"] for r in replay_records(replay.records)))

    def test_case_e_two_stationary_faces_keep_distinct_tracks(self):
        replay = FaceReplay()
        faces = [face(), face((300, 40, 350, 100), "SV002")]
        replay.step(100., faces)
        for i in range(1, 10):
            block, output = replay.step(100. + i * 1.5, faces)
            self.assertEqual([t[0] for t in output], [1, 2])
            self.assertEqual([t["last_student_id"] for t in block["tracks_after"]], ["SV001", "SV002"])
        self.assertEqual([t["stable_label"] for t in block["tracks_after"]], ["SV001", "SV002"])

    def test_case_f_detection_order_reversal_preserves_unambiguous_geometry(self):
        replay = FaceReplay()
        faces = [face(), face((300, 40, 350, 100), "SV002")]
        replay.step(100., faces)
        block, output = replay.step(101.5, faces[::-1])
        self.assertEqual([(t[0], t[1]) for t in output], [(1, tuple(faces[0]["face_bbox"])), (2, tuple(faces[1]["face_bbox"]))])
        self.assertEqual([(m["face_track_id"], m["detection_indices"]) for m in block["mapping"]], [(1, [1]), (2, [0])])

    def test_missing_face_is_retained_until_age_expiry_and_not_returned(self):
        replay = FaceReplay()
        replay.step(100.)
        block, output = replay.step(101., [])
        self.assertEqual(output, [])
        self.assertEqual(block["decisions"][0]["action"], "unmatched_retained")
        self.assertEqual(block["mapping"], [])
        self.assertFalse(block["tracks_after"][0]["updated_in_frame"])
        _, output = replay.step(101.5)
        self.assertEqual([t[0] for t in output], [1])
        block, _ = replay.step(104., [])
        self.assertEqual(block["decisions"][0]["action"], "expired")
        self.assertEqual(block["tracks_after"], [])

    def test_identity_change_and_unknown_do_not_control_iou_matching(self):
        replay = FaceReplay()
        replay.step(100.)
        _, output = replay.step(101., [face(student="Unknown")])
        self.assertEqual(output[0][0], 1)
        block, output = replay.step(102., [face(student="SV002")])
        self.assertEqual(output[0][0], 1)
        self.assertEqual(block["detections"][0]["current_student_id"], "SV002")

    def test_unassociated_faces_and_exact_raw_metadata_are_recorded_without_embeddings(self):
        replay = FaceReplay()
        raw = [face(), face((300, 40, 350, 100), "Unknown")]
        raw[1].update(embedding=[1, 2, 3], landmarks=[4, 5], student_name="PRIVATE")
        original = copy.deepcopy(raw)
        block, _ = replay.step(100., raw, matches=[{"track_id": 1, "person_ref": "person_0", "person_track_id": "P1",
                                                  "association_status": "matched", "student_id": "SV001"}])
        self.assertEqual(block["face_detection_count"], 2)
        self.assertEqual(block["mapping"][1]["associations"], [])
        self.assertEqual(block["detections"][1]["recognition_score"], .8)
        self.assertEqual(block["tracks_after"][0]["stable_label"], "Unknown")
        self.assertEqual(raw, original)
        for secret in ("embedding", "landmarks", "PRIVATE", "student_name"):
            self.assertNotIn(secret, json.dumps(block))

    def test_absent_score_or_student_is_not_fabricated(self):
        replay = FaceReplay()
        block, _ = replay.step(100., [{"face_bbox": [30, 40, 80, 100]}])
        self.assertNotIn("current_student_id", block["detections"][0])
        self.assertNotIn("recognition_score", block["detections"][0])
        with self.assertRaisesRegex(ValueError, "original recognition metadata"):
            replay_records(replay.records)

    def test_duplicate_face_detection_selection_is_marked_ambiguous(self):
        replay = FaceReplay()
        replay.step(100.)
        block, _ = replay.step(101., [face(), face()])
        self.assertEqual([p["selected"] for p in block["candidate_ious"]], [None, None])
        self.assertTrue(all(m["selection_ambiguous"] for m in block["mapping"]))

    def test_eligible_unmatched_face_is_retained_after_greedy_competition(self):
        replay = FaceReplay()
        replay.step(100., [face(), face(student="SV002")])
        block, output = replay.step(101.)
        # Equal-IoU tie in the existing algorithm selects the higher track ID;
        # this is a diagnostic counterexample, not an identity guarantee.
        self.assertEqual([t[0] for t in output], [2])
        self.assertEqual(block["decisions"][0], {"face_track_id": 1, "action": "unmatched_retained",
                                               "reason": "not_selected_by_greedy_matching"})
        self.assertEqual(block["tracks_after"][0]["age_seconds"], 1.)

    def test_detail_limit_is_explicit_and_does_not_filter_real_tracker_output(self):
        replay = FaceReplay()
        with patch("face_tracker_diagnostics.MAX_FACE_DETAILS", 1):
            block, output = replay.step(100., [face(), face((300, 40, 350, 100), "SV002")])
        self.assertEqual(len(output), 2)
        self.assertEqual(block["face_detection_count"], 2)
        self.assertTrue(block["details_truncated"])
        with self.assertRaises(ValueError):
            replay_records(replay.records)

    def test_face_timestamp_is_distinct_from_person_and_capture_clocks(self):
        replay = FaceReplay()
        replay.step(100., captured_at=90., person_offset=.1)
        block, _ = replay.step(101.5, captured_at=95., person_offset=.3)
        self.assertAlmostEqual(block["face_update_gap"], 1.5)
        self.assertAlmostEqual(block["face_to_person_seconds"], .3)
        self.assertEqual(block["processing"], {"face": .2, "hybrid": .25})
        self.assertTrue(all(row["consistent"] for row in replay_records(replay.records)))

    def test_opt_in_requires_master_and_face_flags_and_keeps_quotas(self):
        for env, enabled in (({}, False), ({"HTTM_FACE_DEBUG": "1"}, False),
                             ({"HTTM_PERSON_DEBUG": "1"}, False),
                             ({"HTTM_PERSON_DEBUG": "1", "HTTM_FACE_DEBUG": "1"}, True)):
            with self.subTest(env=env), patch.dict(os.environ, env, clear=True):
                config = DiagnosticConfig.from_environment()
                self.assertEqual(config.enabled and config.face_enabled, enabled)
                self.assertEqual((config.max_frames, config.max_bytes), (10, 20971520))
                self.assertFalse(config.snapshots)

    def test_disabled_face_diagnostics_does_not_copy_or_create_output(self):
        with (patch.object(FaceTrackerDiagnostics, "before_update", side_effect=AssertionError("disabled copy")),
              patch.object(Path, "mkdir", side_effect=AssertionError("disabled IO"))):
            logger = PersonTrackerDiagnostics(DiagnosticConfig())
            self.assertIsNone(logger.before_face_update(FaceTracker(), 100.))
            self.assertIsNone(logger.session_dir)

    def test_before_observation_failure_stops_debug_not_tracker(self):
        logger = PersonTrackerDiagnostics(DiagnosticConfig(True, face_enabled=True))
        tracker = FaceTracker()
        with (patch.object(FaceTrackerDiagnostics, "before_update", side_effect=ValueError("diagnostic failure")),
              redirect_stderr(io.StringIO())):
            self.assertIsNone(logger.before_face_update(tracker, 100.))
        self.assertFalse(logger.active)
        self.assertEqual(tracker.update([((30, 40, 80, 100), "SV001", .8)], 100.)[0][0], 1)

    def test_replay_accepts_json_roundtrip_and_detects_state_mismatch(self):
        replay = FaceReplay()
        replay.step(100.)
        replay.step(101.5)
        records = json.loads(json.dumps(replay.records))
        self.assertTrue(all(row["consistent"] for row in replay_records(records)))
        records[1]["face_tracker"]["tracks_after"][0]["face_track_id"] = 99
        self.assertFalse(replay_records(records)[1]["consistent"])

    def test_replay_rejects_old_partial_mismatched_or_non_increasing_logs(self):
        replay = FaceReplay()
        replay.step(100.)
        replay.step(101.5)
        old = copy.deepcopy(replay.records)
        del old[0]["face_tracker"]
        bad_frame = copy.deepcopy(replay.records)
        bad_frame[1]["face_tracker"]["frame_id"] = 99
        bad_time = copy.deepcopy(replay.records)
        bad_time[1]["face_tracker"]["face_tracker_updated_at"] = 100.
        truncated = copy.deepcopy(replay.records)
        truncated[0]["face_tracker"]["details_truncated"] = True
        for records in ([], old, replay.records[1:], bad_frame, bad_time, truncated):
            with self.subTest(size=len(records)), self.assertRaises(ValueError):
                replay_records(records)


if __name__ == "__main__":
    unittest.main()
