"""Actual worker entry points, with every model boundary mocked."""

import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from contextlib import redirect_stderr

import bbox_visual_metadata as metadata
import hybrid_v7_stream_worker as hybrid
import test_hybrid_person_confidence_trial as hybrid_tests
import test_bbox_detail_overlay as fixtures


class TestVisualWorkerExports(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = fixtures.synthetic(self.temp.name)
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("HTTM_PERSON_TRIAL_MIN_CONF", None)
        self.enterContext(redirect_stderr(io.StringIO()))

    def run_hybrid(self, enabled):
        worker = hybrid_tests.make_worker()
        worker.visualization_enabled = enabled
        with (patch.object(hybrid, "prepare_objects_v7", return_value=self.f.objects) as prepare,
              patch.object(hybrid, "infer_v3", return_value={"detections": self.f.detections}) as infer,
              patch.object(hybrid, "add_sleeping_highres_proposals_v7", return_value=([], {})) as highres,
              patch.object(hybrid, "spatial_v7", return_value={"people": self.f.raw_people}) as spatial):
            reply = worker.process(self.f.path, 7)
        for boundary in (prepare, infer, highres, spatial):
            boundary.assert_called_once()
        self.assertEqual(infer.call_args.kwargs["conf"], .05)
        self.assertEqual(infer.call_args.kwargs["iou"], .50)
        return reply

    def test_hybrid_overlay_on_preserves_protocol_decisions_and_all_evidence(self):
        original = copy.deepcopy((self.f.objects, self.f.detections, self.f.raw_people))
        baseline = self.run_hybrid(False)
        self.assertFalse(metadata.sidecar_path(self.f.path, "hybrid").exists())
        actual = self.run_hybrid(True)
        self.assertEqual(actual, baseline)
        self.assertEqual(set(actual), {"frame_id", "image", "mode", "person_count", "people"})
        packet = json.loads(metadata.sidecar_path(self.f.path, "hybrid").read_bytes())
        self.assertEqual(packet["phones"][0]["confidence"], .27)
        self.assertEqual(packet["candidates"][1]["confidence"], .81)
        self.assertEqual(original, (self.f.objects, self.f.detections, self.f.raw_people))

    def test_hybrid_off_does_not_build_or_write_metadata(self):
        with patch.object(hybrid, "write_source", side_effect=AssertionError("OFF")) as write:
            self.run_hybrid(False)
        write.assert_not_called()

    def test_hybrid_visual_write_failure_keeps_protocol(self):
        baseline = self.run_hybrid(False)
        with patch.object(Path, "write_bytes", side_effect=OSError("synthetic IO failure")):
            actual = self.run_hybrid(True)
        self.assertEqual(actual, baseline)

    def face_module(self):
        # Avoid importing InsightFace in this environment. Only its boundary
        # functions are injected; the actual worker main/loop runs unchanged.
        api = SimpleNamespace(load_database=Mock(return_value={}), load_face_model=Mock(), recognize=Mock())
        spec = importlib.util.spec_from_file_location("isolated_visual_face_worker", Path(__file__).with_name("face_id_stream_worker.py"))
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {"recognize_camera_tracking": api}):
            spec.loader.exec_module(module)
        return module

    def run_face(self, enabled):
        module = self.face_module()
        if enabled:
            os.environ["HTTM_PERSON_SNAPSHOT_OVERLAY"] = "1"
        else:
            os.environ.pop("HTTM_PERSON_SNAPSHOT_OVERLAY", None)
        face = SimpleNamespace(bbox=[30, 20, 70, 60], embedding="not exported")
        model = Mock(); model.get.return_value = [face]
        module.load_face_model.return_value = model
        module.recognize.return_value = ("SV001", .72)
        output = io.StringIO()
        request = json.dumps({"frame_id": 7, "image": str(self.f.path)}) + "\n"
        with patch.object(sys, "stdin", io.StringIO(request)), patch.object(sys, "stdout", output):
            module.main()
        model.get.assert_called_once()
        module.recognize.assert_called_once_with("not exported", {}, .45)
        return json.loads(output.getvalue())

    def test_face_on_preserves_reply_recognition_threshold_and_model_call_count(self):
        baseline = self.run_face(False)
        self.assertFalse(metadata.sidecar_path(self.f.path, "face").exists())
        actual = self.run_face(True)
        self.assertEqual(actual, baseline)
        packet = json.loads(metadata.sidecar_path(self.f.path, "face").read_bytes())
        self.assertEqual(packet["faces"][0]["score"], .72)
        self.assertNotIn("embedding", repr(packet))
        self.assertEqual(set(actual), {"frame_id", "image", "face_count", "faces"})

    def test_face_write_failure_keeps_recognition_output(self):
        baseline = self.run_face(False)
        with patch.object(Path, "write_bytes", side_effect=OSError("synthetic IO failure")):
            actual = self.run_face(True)
        self.assertEqual(actual, baseline)

    def test_environment_requires_exact_opt_in(self):
        for value in (None, "0", "true", "1"):
            if value is None:
                os.environ.pop("HTTM_PERSON_SNAPSHOT_OVERLAY", None)
            else:
                os.environ["HTTM_PERSON_SNAPSHOT_OVERLAY"] = value
            self.assertEqual(metadata.enabled(), value == "1")


if __name__ == "__main__":
    unittest.main()
