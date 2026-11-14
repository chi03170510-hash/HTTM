"""Tests for AI-Smart-Classroom-Monitoring-System core integration into HTTM Camera Live."""

import os
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

INTEGRATION = Path(__file__).resolve().parent
AI_TRAINING = INTEGRATION.parent
HTTM_ROOT = AI_TRAINING.parent

import sys

if str(INTEGRATION) not in sys.path:
    sys.path.insert(0, str(INTEGRATION))
if str(AI_TRAINING) not in sys.path:
    sys.path.insert(0, str(AI_TRAINING))

import camera_coordinator_live as live
import camera_live_preflight as preflight
from camera_live_runtime import LiveWorker
from live_bbox_overlay import HEADER_HEIGHT, draw_live_bboxes
from src.behavior_checkpoint import CHECKPOINT_ENV, DEFAULT_CHECKPOINT, behavior_checkpoint_path

try:
    import ultralytics  # noqa: F401
    HAS_ULTRALYTICS = True
except ImportError:
    HAS_ULTRALYTICS = False


class TestBehaviorCheckpoint(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(CHECKPOINT_ENV, None)

    def test_default_checkpoint_is_behavior_yolo(self):
        self.assertEqual(DEFAULT_CHECKPOINT, "behavior_yolo.pt")
        path = behavior_checkpoint_path()
        self.assertEqual(path.name, "behavior_yolo.pt")
        self.assertTrue(path.is_file())

    def test_invalid_checkpoint_raises(self):
        os.environ[CHECKPOINT_ENV] = "invalid_model.pt"
        with self.assertRaisesRegex(ValueError, CHECKPOINT_ENV):
            behavior_checkpoint_path()


@unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics is installed in .venv-hybrid")
class TestSmartClassroomCore(unittest.TestCase):
    def test_backpack_is_filtered_without_removing_other_objects(self):
        from src.smart_classroom_core import ObjectDetector

        detector = ObjectDetector.__new__(ObjectDetector)
        detector.conf_thresh = 0.35
        detector.cue_conf_thresh = 0.15
        detector.imgsz = 640
        detector.device = "cpu"
        boxes = []
        for class_id in (0, 24, 67):  # COCO person, backpack, cell phone
            box = Mock()
            box.cls = np.array([class_id])
            box.conf = np.array([0.90])
            box.xyxy = np.array([[20, 30, 120, 200]])
            boxes.append(box)
        detector.model = Mock()
        detector.model.predict.return_value = [Mock(
            boxes=boxes, names={0: "person", 24: "backpack", 67: "cell phone"},
        )]

        result = detector.detect(np.zeros((480, 640, 3), dtype=np.uint8))
        self.assertEqual([item["class_name"] for item in result["detections"]],
                         ["Person", "Cell Phone"])
        self.assertEqual(result["occupancy"], 1)
        self.assertEqual(len(result["raw_cues"]["phones"]), 1)

    def test_pipeline_loads_and_runs_with_3_classes(self):
        from src.smart_classroom_core import BEHAVIOR_CLASSES, UnifiedPipeline

        expected_classes = {1: "Sleeping", 2: "Using Phone", 6: "Eating", 7: "Turning / Talking"}
        self.assertEqual(BEHAVIOR_CLASSES, expected_classes)
        self.assertEqual(live.BEHAVIOR_NAMES, ("Using Phone", "Eating", "Sleeping", "Turning / Talking"))

        pipeline = UnifiedPipeline()
        self.assertEqual(pipeline.behavior_detector.behavior_classes, expected_classes)
        self.assertEqual(len(BEHAVIOR_CLASSES), 4)

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = pipeline.process_frame(frame, classroom_id="H305", annotate=True)
        self.assertEqual(result["classroom_id"], "H305")
        self.assertIn("objects", result)
        self.assertIn("behaviors", result)
        self.assertIn("occupancy", result)
        self.assertEqual(result["annotated_frame"].shape, frame.shape)


@unittest.skipUnless(HAS_ULTRALYTICS, "ultralytics is installed in .venv-hybrid")
class TestSleepingFusion(unittest.TestCase):
    def setUp(self):
        from src.smart_classroom_core import BehaviorDetector

        # Exercise fusion without loading weights or accessing a camera.
        self.detector = BehaviorDetector.__new__(BehaviorDetector)
        self.detector.conf_thresh = 0.15
        self.detector._extract_person_face_features = Mock(return_value=None)
        self.detector._extract_person_pose_turn = Mock(return_value=None)
        self.frame = np.zeros((480, 640, 3), dtype=np.uint8)
        self.persons = [{"bbox": [20, 40, 160, 300]}, {"bbox": [300, 40, 440, 300]}]

    def sleeping(self, bbox, confidence=0.76):
        return {"class_id": 1, "class_name": "Sleeping", "confidence": confidence, "bbox": bbox}

    def fuse(self, primary=(), aux=(), persons=None):
        return self.detector._fuse_behavior_cues(
            self.frame, list(primary), list(aux),
            {"persons": self.persons if persons is None else persons},
        )

    def test_one_prediction_does_not_spread_to_hidden_faces(self):
        for auxiliary in (False, True):
            with self.subTest(auxiliary=auxiliary):
                detections = [self.sleeping([25, 45, 155, 295])]
                result = self.fuse(aux=detections) if auxiliary else self.fuse(primary=detections)
                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["bbox"], self.persons[0]["bbox"])
                self.assertEqual(result[0]["confidence"], 0.76)

    def test_low_confidence_is_not_boosted(self):
        detection = self.sleeping(self.persons[0]["bbox"], 0.36)
        self.assertEqual(self.fuse(primary=[detection], aux=[detection]), [])

    def test_unrelated_box_is_not_assigned(self):
        self.assertEqual(self.fuse(primary=[self.sleeping([480, 40, 620, 300])]), [])

    def test_ambiguous_overlap_is_rejected(self):
        persons = [{"bbox": [20, 40, 160, 300]}, {"bbox": [30, 40, 170, 300]}]
        self.assertEqual(self.fuse(primary=[self.sleeping([25, 40, 165, 300])], persons=persons), [])

    def test_two_separate_sleepers_keep_original_scores(self):
        result = self.fuse(primary=[self.sleeping(p["bbox"], 0.61 + i * 0.1)
                                    for i, p in enumerate(self.persons)])
        self.assertEqual(len(result), 2)
        self.assertEqual(sorted(r["confidence"] for r in result), [0.61, 0.71])

    def test_head_tilt_and_hidden_wide_torso_do_not_generate_sleeping(self):
        persons = [{"bbox": [0, 160, 640, 480]}]
        self.assertEqual(self.fuse(persons=persons), [])
        self.detector._extract_person_face_features.return_value = {
            "head_pose": {"valid": True, "pitch_deg": 60, "roll_deg": 0},
        }
        self.assertEqual(self.fuse(persons=persons), [])

    def test_upright_head_vetoes_sleeping(self):
        self.detector._extract_person_face_features.return_value = {
            "head_pose": {"valid": True, "pitch_deg": 0, "roll_deg": 0},
        }
        self.assertEqual(self.fuse(primary=[self.sleeping(self.persons[0]["bbox"])]), [])

    def test_turning_still_works_with_sleeping_fusion(self):
        self.detector._extract_person_face_features.return_value = {
            "head_pose": {"valid": True, "pitch_deg": 0, "roll_deg": 0, "yaw_deg": 35},
        }
        result = self.fuse(persons=[self.persons[0]])
        self.assertEqual([item["class_name"] for item in result], ["Turning / Talking"])
        self.detector._extract_person_pose_turn.assert_not_called()

    def test_no_person_means_no_sleeping_attribution(self):
        detection = self.sleeping([20, 40, 160, 300])
        self.assertEqual(self.fuse(primary=[detection], aux=[detection], persons=[]), [])


class TestLiveBBoxOverlay(unittest.TestCase):
    def _sample_result(self, *, captured_at=10.0):
        behaviors = [
            {"class_id": 2, "class_name": "Using Phone", "confidence": 0.88, "bbox": [30, 40, 110, 180]},
            {"class_id": 6, "class_name": "Eating", "confidence": 0.82, "bbox": [120, 30, 240, 210]},
        ]
        return {
            "frame_id": 1,
            "captured_at": captured_at,
            "completed_at": captured_at + 0.1,
            "occupancy": 2,
            "person_count": 2,
            "objects": [
                {"class_id": 0, "class_name": "Person", "confidence": 0.95, "bbox": [20, 30, 120, 200]},
                {"class_id": 67, "class_name": "Cell Phone", "confidence": 0.72, "bbox": [60, 90, 90, 130]},
            ],
            "behaviors": behaviors,
            "inference_runtime": {
                "backend": "cpu",
                "models": [{"device": "cpu", "precision": "float32"}] * 2,
                "device_name": "CPU",
            },
            "source_size": (320, 240),
            "inference_time_ms": 85.0,
            "elapsed": 0.1,
        }

    def test_draws_objects_and_behaviors_without_mutating_frame(self):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        result = self._sample_result()
        with (
            patch.object(cv2, "rectangle", wraps=cv2.rectangle) as rectangle,
            patch.object(cv2, "putText", wraps=cv2.putText) as put_text,
        ):
            output = draw_live_bboxes(frame, result, now=10.2, view_width=live.VIEW_WIDTH)

        self.assertEqual(output.shape, (960 + HEADER_HEIGHT, live.VIEW_WIDTH, 3))
        np.testing.assert_array_equal(frame, np.full_like(frame, 80))

        rect_colors = [call.args[3] for call in rectangle.call_args_list]
        self.assertIn((0, 255, 0), rect_colors)     # Person
        self.assertIn((255, 165, 0), rect_colors)   # Cell Phone
        self.assertIn((255, 0, 255), rect_colors)   # Using Phone / Eating
        self.assertNotIn((0, 0, 255), rect_colors)

        labels = [call.args[1] for call in put_text.call_args_list]
        self.assertTrue(any("Occupancy: 2" in lbl for lbl in labels))
        self.assertFalse(any(lbl.startswith("Person ") for lbl in labels))
        self.assertFalse(any("FIGHTING" in lbl for lbl in labels))
        self.assertTrue(any("Using Phone, Eating, Sleeping" in lbl for lbl in labels))
        self.assertTrue(any("YOLO (yolov8n + behavior_yolo): CPU" in lbl for lbl in labels))

    def test_stale_result_hides_boxes(self):
        frame = np.full((240, 320, 3), 80, dtype=np.uint8)
        result = self._sample_result(captured_at=10.0)
        with patch.object(cv2, "rectangle") as rectangle:
            output = draw_live_bboxes(frame, result, now=16.5, view_width=320)
        rectangle.assert_not_called()
        np.testing.assert_array_equal(output[HEADER_HEIGHT:], frame)


class TestCameraCoordinatorLive(unittest.TestCase):
    def test_ai_loop_processes_frames_and_shuts_down_cleanly(self):
        self.assertEqual(live.BEHAVIOR_NAMES, ("Using Phone", "Eating", "Sleeping", "Turning / Talking"))
        fake_worker = Mock()
        fake_worker.process_frame.side_effect = [
            {
                "frame_id": 1,
                "mode": "AI_SMART_UNIFIED",
                "inference_runtime": {"backend": "cpu", "models": [], "device_name": "CPU"},
                "occupancy": 1,
                "person_count": 1,
                "objects": [{"class_id": 0, "class_name": "Person", "confidence": 0.9, "bbox": [10, 10, 80, 100]}],
                "behaviors": [{"class_id": 6, "class_name": "Eating", "confidence": 0.85, "bbox": [12, 12, 78, 98]}],
                "object_inference_time_ms": 20.0,
                "behavior_inference_time_ms": 25.0,
                "inference_time_ms": 45.0,
            },
            {
                "frame_id": 2,
                "mode": "AI_SMART_UNIFIED",
                "inference_runtime": {"backend": "cpu", "models": [], "device_name": "CPU"},
                "occupancy": 2,
                "person_count": 2,
                "objects": [],
                "behaviors": [{"class_id": 1, "class_name": "Sleeping", "confidence": 0.77, "bbox": [20, 20, 90, 110]}],
                "object_inference_time_ms": 18.0,
                "behavior_inference_time_ms": 22.0,
                "inference_time_ms": 40.0,
            },
        ]

        frame_queue = queue.Queue(maxsize=2)
        pixels = np.full((120, 160, 3), 70, dtype=np.uint8)
        frame_queue.put((10.0, pixels))
        frame_queue.put((11.0, pixels))
        result_queue = queue.Queue()
        stop_event = threading.Event()

        with patch.object(live, "Worker", return_value=fake_worker):
            live.ai_loop(frame_queue, result_queue, stop_event, max_observations=2)

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get_nowait())

        self.assertTrue(stop_event.is_set())
        self.assertEqual(messages[-1], {"finished": True})
        fake_worker.close.assert_called_once()

        results = [m for m in messages if "frame_id" in m]
        self.assertEqual(len(results), 2)
        self.assertNotIn("active_fighting", results[0])
        self.assertEqual(results[0]["behaviors"][0]["class_name"], "Eating")
        self.assertEqual(results[0]["occupancy"], 1)
        self.assertEqual(results[1]["behaviors"][0]["class_name"], "Sleeping")
        self.assertEqual(results[1]["occupancy"], 2)

    def test_draw_ai_panel_renders_70_30_split_without_mutating_frame(self):
        frame = np.full((240, 320, 3), 60, dtype=np.uint8)
        result = {
            "frame_id": 3,
            "captured_at": 10.0,
            "completed_at": 10.1,
            "occupancy": 2,
            "person_count": 2,
            "objects": [{"class_id": 0, "class_name": "Person", "confidence": 0.92, "bbox": [10, 20, 100, 180]}],
            "behaviors": [{"class_id": 2, "class_name": "Using Phone", "confidence": 0.81, "bbox": [15, 25, 95, 175]}],
            "inference_time_ms": 50.0,
            "elapsed": 0.1,
        }
        panel_img = live.draw_ai_panel(frame, result, page=0, now=10.2)
        self.assertEqual(panel_img.shape[1], live.VIEW_WIDTH)
        self.assertGreaterEqual(panel_img.shape[0], 540)
        np.testing.assert_array_equal(frame, np.full_like(frame, 60))

    def test_main_preserves_camera_dshow_msmf_protocol(self):
        fake_cap = Mock()
        fake_cap.isOpened.side_effect = [False, True]
        fake_cap.read.return_value = (True, np.zeros((240, 320, 3), dtype=np.uint8))

        class FakeThread:
            def __init__(self, target, args, daemon):
                self.results = args[1]

            def start(self):
                self.results.put({"finished": True})

            def is_alive(self):
                return False

        with (
            patch.object(live.Path, "is_file", return_value=True),
            patch.object(live, "observation_limit_from_environment", return_value=None),
            patch.object(live.cv2, "VideoCapture", return_value=fake_cap) as vc,
            patch.object(live.threading, "Thread", FakeThread),
            patch.object(live.cv2, "namedWindow"),
            patch.object(live.cv2, "resizeWindow"),
            patch.object(live.cv2, "imshow"),
            patch.object(live.cv2, "waitKey", return_value=ord("q")),
            patch.object(live.cv2, "destroyAllWindows"),
        ):
            live.main()

        self.assertEqual(vc.call_count, 2)
        self.assertEqual(vc.call_args_list[0].args, (live.CAMERA_INDEX, cv2.CAP_DSHOW))
        self.assertEqual(vc.call_args_list[1].args, (live.CAMERA_INDEX, cv2.CAP_MSMF))
        fake_cap.release.assert_called()


class TestSubprocessWorkerIPC(unittest.TestCase):
    def test_live_worker_runs_smart_classroom_stream_worker(self):
        hybrid_python = HTTM_ROOT / ".venv-hybrid/Scripts/python.exe"
        venv_python = HTTM_ROOT / ".venv/Scripts/python.exe"
        worker_python = (
            hybrid_python if hybrid_python.is_file()
            else venv_python if venv_python.is_file()
            else Path(sys.executable)
        )
        worker_script = INTEGRATION / "smart_classroom_stream_worker.py"
        if not worker_python.is_file():
            self.skipTest("worker python not present")

        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / "synthetic.jpg"
            cv2.imwrite(str(img_path), np.zeros((240, 320, 3), dtype=np.uint8))
            worker = LiveWorker(worker_python, worker_script)
            try:
                res = worker.process_frame(img_path, 1)
            finally:
                worker.close()

        self.assertEqual(res["frame_id"], 1)
        self.assertEqual(res["mode"], "AI_SMART_UNIFIED")
        self.assertIsInstance(res["occupancy"], int)
        self.assertIsInstance(res["objects"], list)
        self.assertIsInstance(res["behaviors"], list)


class TestPreflight(unittest.TestCase):
    def test_preflight_passes_or_warns_without_fail(self):
        report = preflight.run_preflight(HTTM_ROOT)
        self.assertIn(report["status"], ("PASS", "WARN"))
        self.assertEqual(report["counts"].get("FAIL", 0), 0)


if __name__ == "__main__":
    unittest.main()
