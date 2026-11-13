"""Final V7 rules through the live worker; no camera or real model inference.

Run with .venv-hybrid, since worker import needs its normal runtime packages.
"""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

INTEGRATION = Path(__file__).resolve().parent
AI = INTEGRATION.parent
sys.path.insert(0, str(INTEGRATION))
sys.path.insert(0, str(AI))

import hybrid_v7_stream_worker as stream
from src import hybrid_v7
from person_tracker_diagnostics import DiagnosticConfig


class TestHybridV7StreamRules(unittest.TestCase):
    def test_live_worker_keeps_final_desk_guard_and_unique_crop_association(self):
        worker = stream.HybridV7Stream.__new__(stream.HybridV7Stream)
        worker.model, worker.objects, worker.face = Mock(), Mock(), None
        worker.objects.food_classes = {"apple"}
        worker.config = json.loads((AI / "config/hybrid_v7.json").read_text(encoding="utf-8"))
        worker.config["food_mouth_crop_enabled"] = False
        worker.device, worker.spatial_config = "cpu", object()
        worker.names = {0: "sleeping", 1: "using_phone"}
        worker.diagnostic_config, worker._diagnostic_frames = DiagnosticConfig(), 0
        worker.visualization_enabled, worker.trial_min_conf = False, None
        pbox = [0, 0, 200, 400]
        detections = [{"class_name": "using_phone", "confidence": .8, "bbox_xyxy": pbox}]
        base_person = {"person_ref": "p", "bbox_xyxy": pbox,
                       "person_bbox_xyxy_clipped": pbox, "yolo_detections": detections,
                       "evidence": {"face": {}, "pose_landmarks": {}},
                       "behaviors": {b: {"state": "unknown", "evidence": {}}
                                     for b in ("sleeping", "using_phone", "eating", "turning_back")}}
        obj = {"person_detections": [{"detection_id": "p", "bbox_xyxy": pbox}],
               "phone_detections": [], "food_detections": []}
        crop = [{"detection_id": "phone_crop", "class_name": "cell phone", "confidence": .8,
                 "bbox_xyxy": [50, 300, 110, 320], "source": "person_crop_coco", "person_crop_ref": "p"}]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        with (patch.object(stream.cv2, "imread", return_value=frame),
              patch.object(hybrid_v7, "prepare_objects_v6", return_value=obj),
              patch.object(hybrid_v7, "_crop_person_object_proposals", return_value=crop),
              patch.object(stream, "infer_v3", return_value={"detections": detections}),
              patch.object(stream, "add_sleeping_highres_proposals_v7", return_value=([], {})),
              patch.object(hybrid_v7, "spatial_v3", return_value={
                  "people": [base_person], "phone_detections": crop, "food_detections": []}),
              patch.object(stream, "inference_runtime", return_value={})):
            result = worker.process(Path("synthetic_frame.jpg"), 7)
        self.assertIs(stream.prepare_objects_v7, hybrid_v7.prepare_objects_v6_1)
        self.assertIs(stream.spatial_v7, hybrid_v7.spatial_v7)
        self.assertEqual(result["person_count"], 1)
        decision = result["people"][0]["behaviors"]["using_phone"]
        self.assertEqual(decision["state"], "suspected")
        self.assertEqual(decision["evidence"]["reason_code"], "PHONE_SURFACE_CONFLICT")
        self.assertEqual(decision["evidence"]["phone_detection_ids"], ["phone_crop"])
        self.assertIsNone(decision["bounding_box"])
        self.assertEqual(result["frame_id"], 7)
        json.dumps(result, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
