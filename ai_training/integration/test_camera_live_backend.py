"""Camera Live backend selection and cleanup; no real camera, GUI or AI worker."""

import io
import os
import unittest
from contextlib import redirect_stdout
from unittest.mock import Mock, call, patch

import numpy as np

import camera_coordinator as coordinator
import camera_coordinator_live as live


class TestCameraLiveBackend(unittest.TestCase):
    camera_index = 0

    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop("HTTM_FACE_TRACKER_TRIAL_MAX_AGE", None)
        # Camera Live imports the configured value here. Patch its lookup site
        # so backend behavior is checked independently of physical device order.
        self.enterContext(patch.object(live, "CAMERA_INDEX", self.camera_index))
        self.enterContext(patch.object(live.Path, "is_file", return_value=True))
        self.camera = self.enterContext(patch.object(live.cv2, "VideoCapture"))
        self.thread = self.enterContext(patch.object(live.threading, "Thread"))
        self.thread.return_value.is_alive.return_value = False
        self.window = self.enterContext(patch.object(live.cv2, "namedWindow"))
        for operation in ("resizeWindow", "imshow", "destroyAllWindows", "putText"):
            self.enterContext(patch.object(live.cv2, operation))
        self.enterContext(patch.object(live.cv2, "waitKey", return_value=ord("q")))
        self.enterContext(patch.object(live, "draw_ai_panel", side_effect=lambda frame, *a, **k: frame))
        self.output = self.enterContext(redirect_stdout(io.StringIO()))
        self.timeline = Mock()
        self.timeline.attach_mock(self.camera, "open")

    def capture(self, name, opened):
        cap = Mock()
        cap.isOpened.return_value = opened
        cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
        self.timeline.attach_mock(cap, name)
        return cap

    def test_dshow_success_does_not_attempt_msmf_and_releases_on_quit(self):
        dshow = self.capture("dshow", True)
        self.camera.return_value = dshow

        live.main()

        self.camera.assert_called_once_with(self.camera_index, live.cv2.CAP_DSHOW)
        dshow.read.assert_called_once_with()
        dshow.release.assert_called_once_with()
        self.thread.return_value.start.assert_called_once_with()
        self.assertIn(f"[CAMERA] Opened camera index {self.camera_index} with backend DSHOW", self.output.getvalue())
        self.assertNotIn("MSMF", self.output.getvalue())

    def test_dshow_failure_releases_before_msmf_at_same_index_and_releases_on_quit(self):
        dshow = self.capture("dshow", False)
        msmf = self.capture("msmf", True)
        self.camera.side_effect = [dshow, msmf]

        live.main()

        self.assertEqual(self.timeline.mock_calls[:5], [
            call.open(self.camera_index, live.cv2.CAP_DSHOW), call.dshow.isOpened(), call.dshow.release(),
            call.open(self.camera_index, live.cv2.CAP_MSMF), call.msmf.isOpened(),
        ])
        self.assertEqual(self.camera.call_count, 2)
        dshow.release.assert_called_once_with()
        dshow.read.assert_not_called()
        msmf.read.assert_called_once_with()
        msmf.release.assert_called_once_with()
        self.assertIn(f"[CAMERA] Opened camera index {self.camera_index} with backend MSMF", self.output.getvalue())
        self.assertNotIn("with backend DSHOW", self.output.getvalue())

    def test_both_backends_fail_releases_both_without_starting_gui_or_worker(self):
        dshow = self.capture("dshow", False)
        msmf = self.capture("msmf", False)
        self.camera.side_effect = [dshow, msmf]

        with self.assertRaisesRegex(RuntimeError,
                                   f"Cannot open camera index {self.camera_index}: DSHOW and MSMF both failed"):
            live.main()

        self.assertEqual(self.timeline.mock_calls, [
            call.open(self.camera_index, live.cv2.CAP_DSHOW), call.dshow.isOpened(), call.dshow.release(),
            call.open(self.camera_index, live.cv2.CAP_MSMF), call.msmf.isOpened(), call.msmf.release(),
        ])
        dshow.read.assert_not_called()
        msmf.read.assert_not_called()
        self.thread.assert_not_called()
        self.window.assert_not_called()
        self.assertNotIn("[CAMERA] Opened", self.output.getvalue())


class TestCameraLiveBackendIndexTwo(TestCameraLiveBackend):
    """The same backend/resource contract must also hold at mock index 2."""

    camera_index = 2


class TestCameraIndexContract(unittest.TestCase):
    def test_camera_live_uses_coordinator_configuration(self):
        # No particular index identifies iVCam on every machine.
        self.assertEqual(live.CAMERA_INDEX, coordinator.CAMERA_INDEX)


if __name__ == "__main__":
    unittest.main()
