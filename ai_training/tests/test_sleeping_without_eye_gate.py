"""Sleeping model decisions must not depend on eye closure or visibility."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))

from src.hybrid_v2 import TemporalV2Fusion
from src.hybrid_v3 import spatial_v3
from src.hybrid_v4 import TemporalV4Fusion
from src.hybrid_v7 import TemporalV6Fusion, spatial_v6

BEHAVIORS = ("sleeping", "using_phone", "turning_back", "eating")


class TestSleepingWithoutEyeGate(unittest.TestCase):
    def setUp(self):
        self.cfg = json.loads((AI / "config/hybrid_v7.json").read_text(encoding="utf-8"))

    @staticmethod
    def behaviors():
        return {b: {"state": "unknown", "decision_reason": "none", "evidence": {}}
                for b in BEHAVIORS}

    def spatial(self, fn, score, face):
        detections = [] if score is None else [{"class_name": "sleeping", "confidence": score,
                                               "bbox_xyxy": [10, 10, 80, 80]}]
        base = {"people": [{"person_ref": "p1", "bbox_xyxy": [0, 0, 100, 200],
                            "yolo_detections": detections, "evidence": {"face": face},
                            "evidence_validity": {"eyes": bool(face), "pose": False},
                            "behaviors": self.behaviors()}]}
        return fn(np.zeros((200, 100, 3), dtype=np.uint8), detections, {}, None,
                  self.cfg, lambda *args: base, None)["people"][0]["behaviors"]["sleeping"]

    def test_spatial_model_detection_with_open_or_unobservable_eyes(self):
        for fn in (spatial_v3, spatial_v6):
            for face in ({}, {"ear_left": .35, "ear_right": .35, "eyes_valid": True}):
                with self.subTest(pipeline=fn.__name__, face=face):
                    result = self.spatial(fn, .81, face)
                    self.assertEqual(result["state"], "detected")
                    self.assertNotIn("eye_state_unobservable", result.get("contradicting_evidence", []))

    def test_closed_eyes_without_model_detection_do_not_create_sleep_candidate(self):
        face = {"ear_left": .10, "ear_right": .10, "eyes_valid": True}
        for fn in (spatial_v3, spatial_v6):
            with self.subTest(pipeline=fn.__name__):
                self.assertEqual(self.spatial(fn, None, face)["state"], "not_detected")
                self.assertEqual(self.spatial(fn, .30, face)["state"], "suspected")

    def test_temporal_confirmation_without_eye_or_pose_support(self):
        for cls in (TemporalV2Fusion, TemporalV4Fusion, TemporalV6Fusion):
            for eye_valid in (False, True):
                with self.subTest(pipeline=cls.__name__, eye_valid=eye_valid):
                    fusion = cls(self.cfg)
                    behaviors = self.behaviors()
                    behaviors["sleeping"] = {
                        "state": "detected", "decision_reason": "yolo", "model_confidence": .81,
                        "evidence": {"eye_closed": False, "eye_valid": eye_valid,
                                     "eyes_open_candidate": eye_valid,
                                     "head_supported_proxy": False, "head_body_tilt": False}}
                    outputs = [fusion.update(1, i * .5, behaviors) for i in range(11)]
                    self.assertEqual(outputs[0]["sleeping"]["state"], "suspected")
                    for result in outputs[8:]:
                        self.assertEqual(result["sleeping"]["state"], "detected")
                        self.assertFalse(result["sleeping"]["evidence"].get(
                            "awake_alternative_blocks_sleep_confirmation", False))


if __name__ == "__main__":
    unittest.main()
