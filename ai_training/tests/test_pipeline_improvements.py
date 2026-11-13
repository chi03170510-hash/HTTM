"""Unit tests for the 4 pipeline improvements (Preprocessing, Multimodal Sleeping, Adaptive Association, Tracking & Identity Anchoring)."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
import numpy as np

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
sys.path.insert(0, str(AI / "integration"))
sys.path.insert(0, str(AI / "recognition"))

from src.image_preprocessing import (
    apply_clahe_lab,
    auto_gamma_correction,
    denoise_and_sharpen_roi,
    preprocess_frame,
)
from src.hybrid_v7 import assess_multimodal_sleeping_v7
from person_face_association_v4 import associate_v4
from face_tracker import FaceTracker
from behavior_duration import BehaviorDurationTracker


class TestPipelineImprovements(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((AI / "config" / "hybrid_v7.json").read_text(encoding="utf-8"))

    def test_clahe_and_gamma_and_roi_sharpen_preserve_shape_and_dtype(self):
        dark_frame = np.full((120, 160, 3), 35, dtype=np.uint8)
        dark_frame[30:90, 40:120] = 70
        gamma_out = auto_gamma_correction(dark_frame)
        self.assertEqual(gamma_out.shape, dark_frame.shape)
        self.assertEqual(gamma_out.dtype, np.uint8)
        self.assertGreater(float(np.mean(gamma_out)), float(np.mean(dark_frame)))

        clahe_out = apply_clahe_lab(dark_frame)
        self.assertEqual(clahe_out.shape, dark_frame.shape)
        self.assertEqual(clahe_out.dtype, np.uint8)

        sharp_out = denoise_and_sharpen_roi(dark_frame)
        self.assertEqual(sharp_out.shape, dark_frame.shape)
        self.assertEqual(sharp_out.dtype, np.uint8)

        full_out = preprocess_frame(dark_frame, self.cfg)
        self.assertEqual(full_out.shape, dark_frame.shape)

    def test_multimodal_sleeping_combines_yolo_and_geometric_cues(self):
        box = [100, 120, 480, 450]
        # Without face or pose cues, 0.55 remains suspected
        state_no_cue, _, _ = assess_multimodal_sleeping_v7(0.55, box, {}, {}, box, self.cfg)
        self.assertEqual(state_no_cue, "suspected")

        # With closed eyes (EAR = 0.14 < 0.18), 0.55 is promoted to detected
        face_closed = {"ear_left": 0.13, "ear_right": 0.15, "eyes_valid": True}
        state_ear, reason_ear, meta_ear = assess_multimodal_sleeping_v7(
            0.55, box, face_closed, {}, box, self.cfg
        )
        self.assertEqual(state_ear, "detected")
        self.assertEqual(reason_ear, "multimodal_sleeping_yolo_plus_geometric_evidence")
        self.assertTrue(meta_ear["ear_closed"])

        # With pitched head (pitch_deg = 32 >= 25), 0.50 is promoted to detected
        face_pitched = {"head_pose": {"pitch_deg": 32.0, "yaw_deg": 2.0, "roll_deg": 3.0, "valid": True}}
        state_pitch, _, meta_pitch = assess_multimodal_sleeping_v7(
            0.50, box, face_pitched, {}, box, self.cfg
        )
        self.assertEqual(state_pitch, "detected")
        self.assertTrue(meta_pitch["head_pitched_or_tilted"])

    def test_adaptive_close_up_association_v4(self):
        # Close-up upper-body person box where top of head slightly exceeds person y1 (overlap < 0.95)
        # and face center is at ry = 0.625 (> 0.60 limit of v1 and non-adaptive v4)
        people = [{"person_ref": "person_0", "bbox_xyxy": [40.0, 255.0, 500.0, 335.0], "behaviors": {}}]
        face_tracks = [("T1", [220.0, 245.0, 320.0, 365.0], "SV001", 0.82)]

        # Standard associate_v4 rejects ry = 0.625 > 0.60
        self.assertEqual(associate_v4(face_tracks, people, adaptive_close_up=False), [])
        # Adaptive close-up associate_v4 matches it cleanly
        matched = associate_v4(face_tracks, people, adaptive_close_up=True)
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["student_id"], "SV001")
        self.assertEqual(matched[0]["person_ref"], "person_0")

    def test_face_tracker_motion_matching(self):
        tracker = FaceTracker(iou_threshold=0.3, max_age=4.0, motion_matching=True)
        tracker.update([((100, 100, 160, 160), "SV001", 0.85)], timestamp=1.0)
        tracker.update([((130, 100, 190, 160), "SV001", 0.86)], timestamp=2.0)
        # At t=3.5s, box moved further right where static IoU with t=2.0 would drop below 0.30,
        # but velocity prediction (vx = +30 px/s -> dx = +45 px) matches track_id = 1
        out = tracker.update([((175, 100, 235, 160), "SV001", 0.88)], timestamp=3.5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0][0], 1)
        self.assertEqual(out[0][2], "SV001")

    def test_conditional_identity_anchoring_during_sleeping(self):
        tracker = BehaviorDurationTracker(max_gap=5.0, anchor_sleeping_identity=True)
        people_f1 = [{
            "person_ref": "p0",
            "behaviors": {"sleeping": {"state": "detected"}, "using_phone": {"state": "not_detected"},
                          "turning_back": {"state": "unknown"}, "eating": {"state": "unknown"}},
        }]
        tracked_f1 = [{"track_id": "P1", "person_ref": "p0", "bbox": (100.0, 100.0, 400.0, 450.0)}]
        matches_f1 = [{
            "person_track_id": "P1",
            "person_ref": "p0",
            "track_id": 1,
            "student_id": "SV001",
            "association_status": "matched",
        }]
        frame1 = tracker.update(1, 10.0, people_f1, tracked_f1, matches_f1)
        self.assertEqual(frame1.people[0].student_id, "SV001")

        # Frame 2: student bows head down on desk -> face disappears (matches_f2 = []),
        # person bbox stays stationary (IoU > 0.65), sleeping is still detected.
        tracked_f2 = [{"track_id": "P1", "person_ref": "p0", "bbox": (102.0, 104.0, 398.0, 450.0)}]
        frame2 = tracker.update(2, 12.5, people_f1, tracked_f2, [])
        self.assertEqual(frame2.people[0].student_id, "SV001")
        self.assertEqual(frame2.people[0].identity_status, "anchored_sleeping")
        sleep_dur = dict((b.name, b.estimated_seconds) for b in frame2.people[0].behaviors)["sleeping"]
        self.assertAlmostEqual(sleep_dur, 2.5)


if __name__ == "__main__":
    unittest.main()
