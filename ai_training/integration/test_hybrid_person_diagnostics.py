"""Run in .venv-hybrid; real worker process method, mocked model boundaries."""

import json
import io
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import numpy as np

import hybrid_v7_stream_worker as worker_module
from person_tracker_diagnostics import DiagnosticConfig, sidecar_path


class TestHybridPersonDiagnostics(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))

    def run_worker(self, config, path, raw_people):
        worker = worker_module.HybridV7Stream.__new__(worker_module.HybridV7Stream)
        worker.model = worker.objects = worker.face = worker.spatial_config = object()
        worker.device = "cpu"
        worker.names = {0: "sleeping", 1: "using_phone"}
        worker.config = {"inference_conf": .05, "person_roi_enabled": True,
                         "person_roi_padding": .12, "dedupe_iou": .65}
        worker.diagnostic_config, worker._diagnostic_frames = config, 0
        people = [{"person_ref": "person_0", "bbox_xyxy": [20, 40, 180, 420],
                   "person_bbox_xyxy_clipped": [20., 40., 180., 420.], "behaviors": {}}]
        with (patch.object(worker_module.cv2, "imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)),
              patch.object(worker_module, "prepare_objects_v7", return_value={"person_detections": raw_people}),
              patch.object(worker_module, "infer_v3", return_value={"detections": []}),
              patch.object(worker_module, "add_sleeping_highres_proposals_v7", return_value=([], {})),
              patch.object(worker_module, "spatial_v7", return_value={"people": people})):
            replies = [worker.process(path, i) for i in (1, 2)]
        return replies

    def test_default_worker_schema_and_no_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            replies = self.run_worker(DiagnosticConfig(), path, [])
            self.assertFalse(sidecar_path(path).exists())
            self.assertEqual(set(replies[0]), {"frame_id", "image", "mode", "person_count", "people"})
            self.assertEqual(set(replies[0]["people"][0]), {"person_ref", "bbox_xyxy", "person_bbox_xyxy_clipped", "behaviors"})

    def test_debug_worker_exports_actual_person_metadata_without_changing_reply_schema(self):
        raw = [{"detection_id": "person_0", "class_name": "person", "confidence": .42,
                "sources": ["coco_person_640"], "person_quality": {"quality": "usable"}}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            enabled = self.run_worker(DiagnosticConfig(True), path, raw)
            data = json.loads(sidecar_path(path).read_bytes())
            disabled = self.run_worker(DiagnosticConfig(), path, raw)
        self.assertEqual(enabled, disabled)
        self.assertEqual(data["frame_id"], 2)
        self.assertEqual(data["detections"][0]["confidence"], .42)
        self.assertEqual(data["detections"][0]["sources"], ["coco_person_640"])

    def test_worker_frame_limit_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.jpg"
            self.run_worker(DiagnosticConfig(True, max_frames=1), path, [])
            data = json.loads(sidecar_path(path).read_bytes())
        self.assertEqual(data["frame_id"], 1)
        self.assertFalse(data["detections"][0]["metadata_available"])
        self.assertNotIn("confidence", data["detections"][0])

    def test_worker_debug_write_failure_does_not_fail_inference(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing" / "frame.jpg"
            replies = self.run_worker(DiagnosticConfig(True), path, [])
        self.assertEqual([r["person_count"] for r in replies], [1, 1])


if __name__ == "__main__":
    unittest.main()
