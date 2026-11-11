"""Checkpoint selection only; model, inference and camera boundaries are mocked."""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

AI = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI))
from src.behavior_checkpoint import CHECKPOINT_ENV, behavior_checkpoint_path


class TestBehaviorCheckpointSelection(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(CHECKPOINT_ENV, None)

    def test_default_is_existing_v1_4(self):
        self.assertEqual(behavior_checkpoint_path(), AI / "models/best_behavior_v1_4.pt")

    def test_explicit_v1_4_and_v2_rollback(self):
        for name in ("best_behavior_v1_4.pt", "best_behavior_v2.pt"):
            with self.subTest(name=name):
                os.environ[CHECKPOINT_ENV] = name
                self.assertEqual(behavior_checkpoint_path(), AI / "models" / name)

    def test_unapproved_or_empty_selection_fails(self):
        for name in ("", "v1.4", "missing.pt", "../best_behavior_v1_4.pt"):
            with self.subTest(name=name):
                os.environ[CHECKPOINT_ENV] = name
                with self.assertRaisesRegex(ValueError, CHECKPOINT_ENV):
                    behavior_checkpoint_path()

    def test_missing_checkpoint_fails_before_ultralytics_can_download(self):
        with patch.object(Path, "is_file", return_value=False):
            with self.assertRaisesRegex(FileNotFoundError, "Behavior checkpoint not found"):
                behavior_checkpoint_path()


class TestBehaviorCheckpointWorker(unittest.TestCase):
    def setUp(self):
        import hybrid_v7_stream_worker
        self.module = hybrid_v7_stream_worker
        self.enterContext(patch.dict(os.environ))
        for name in (CHECKPOINT_ENV, "HTTM_PERSON_TRIAL_MIN_CONF", "HTTM_PERSON_DEBUG"):
            os.environ.pop(name, None)

    def test_live_worker_loads_selected_checkpoint_and_preserves_settings(self):
        for name in (None, "best_behavior_v2.pt"):
            with self.subTest(name=name):
                if name is None:
                    os.environ.pop(CHECKPOINT_ENV, None)
                else:
                    os.environ[CHECKPOINT_ENV] = name
                with (patch.object(self.module, "YOLO", return_value=Mock(names={0: "sleeping", 1: "using_phone"})) as load,
                      patch.object(self.module, "UsingPhonePrototype") as objects,
                      patch.object(self.module.FaceFeatureExtractor, "availability", return_value=(False, "mock")),
                      redirect_stderr(io.StringIO()) as log):
                    worker = self.module.HybridV7Stream()
                    selected = name or "best_behavior_v1_4.pt"
                    load.assert_called_once_with(str(AI / "models" / selected))
                    self.assertIn(selected, log.getvalue())
                    self.assertEqual(worker.config["inference_conf"], .05)
                    self.assertEqual(worker.config["candidate_model_confidence_min"], .25)
                    self.assertTrue(worker.config["person_roi_enabled"])
                    self.assertEqual(objects.call_args.kwargs["conf"], .12)
                    worker.close()

    def test_invalid_selection_does_not_load_any_model(self):
        os.environ[CHECKPOINT_ENV] = "unapproved.pt"
        with patch.object(self.module, "YOLO") as load, patch.object(self.module, "UsingPhonePrototype") as objects:
            with self.assertRaisesRegex(ValueError, CHECKPOINT_ENV):
                self.module.HybridV7Stream()
        load.assert_not_called()
        objects.assert_not_called()


if __name__ == "__main__":
    unittest.main()
