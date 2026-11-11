"""Opt-in boundaries, truthful provenance, bounded storage and live integration."""

import copy
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import camera_coordinator_live as live
from person_tracker import PersonTracker
from person_tracker_diagnostics import (
    DiagnosticConfig, PersonTrackerDiagnostics, detection_provenance, sidecar_path, write_detection_sidecar,
)
from test_ch20_duration_pipeline import PipelineHarness, observation
from test_person_tracker_diagnostic_replay import diagnostic_result


class TestPersonTrackerDiagnostics(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tracker = PersonTracker(iou_threshold=.3, max_age=5., keep_unmatched=False)
        self.people = [{"person_ref": "person_0", "bbox_xyxy": [20, 40, 180, 420],
                        "behaviors": {"sleeping": {"model_confidence": .99}}}]
        self.result = diagnostic_result(1, 98., 100.1)

    def record(self, config, *, result=None, raw=None):
        logger = PersonTrackerDiagnostics(config)
        before = logger.before_update(self.tracker, 100.)
        self.tracker.update(self.people, 100.)
        if raw is not None:
            write_detection_sidecar(config, self.root / "frame.jpg", 1, self.result["snapshot"].image.shape,
                                    self.people, raw)
        return logger, logger.record(before, self.tracker, self.people, result or self.result, self.root / "frame.jpg")

    def test_defaults_disabled_and_snapshot_flag_alone_cannot_enable_debug(self):
        with patch.dict(os.environ, {"HTTM_PERSON_DEBUG_SNAPSHOTS": "1"}, clear=True):
            config = DiagnosticConfig.from_environment()
        self.assertFalse(config.enabled)
        self.assertFalse(config.snapshots)
        with (patch.object(Path, "mkdir", side_effect=AssertionError("disabled must not write")),
              patch.object(cv2, "imencode", side_effect=AssertionError("disabled must not encode"))):
            logger, result = self.record(config)
        self.assertIsNone(result)
        self.assertIsNone(logger.session_dir)

    def test_environment_limits_are_bounded_and_explicit(self):
        with patch.dict(os.environ, {"HTTM_PERSON_DEBUG": "1", "HTTM_PERSON_DEBUG_MAX_FRAMES": "9999",
                                    "HTTM_PERSON_DEBUG_MAX_BYTES": "999999999", "HTTM_PERSON_DEBUG_SNAPSHOTS": "1"}, clear=True):
            config = DiagnosticConfig.from_environment(self.root)
        self.assertTrue(config.enabled and config.snapshots)
        self.assertEqual((config.max_frames, config.max_bytes), (100, 100 * 1024 * 1024))
        with patch.dict(os.environ, {"HTTM_PERSON_DEBUG": "1", "HTTM_PERSON_DEBUG_MAX_BYTES": "invalid"}, clear=True):
            self.assertEqual(DiagnosticConfig.from_environment().max_bytes, 20 * 1024 * 1024)

    def test_missing_person_confidence_is_not_replaced_by_behavior_confidence(self):
        config = DiagnosticConfig(True, output_dir=self.root / "debug")
        logger, record = self.record(config)
        self.assertEqual(record["detection_metadata_status"], "not_provided_by_worker")
        self.assertFalse(record["detections"][0]["metadata_available"])
        self.assertNotIn("confidence", record["detections"][0])
        self.assertNotIn("sources", record["detections"][0])
        self.assertEqual([p.name for p in logger.session_dir.iterdir()], ["frames.jsonl"])

    def test_actual_source_fields_are_projected_without_personal_extras(self):
        raw = [{"detection_id": "person_0", "confidence": .61, "class_name": "person", "class_id": 0,
                "sources": ["coco_person_640", "coco_person_fullframe_highres"],
                "person_quality": {"quality": "usable", "area_frame_ratio": .1},
                "face_embedding": [123], "student_name": "FORBIDDEN"}]
        original = copy.deepcopy(raw)
        config = DiagnosticConfig(True, output_dir=self.root / "debug")
        _, record = self.record(config, raw=raw)
        item = record["detections"][0]
        self.assertEqual((item["confidence"], item["sources"]), (.61, raw[0]["sources"]))
        self.assertEqual(record["detection_metadata_status"], "validated_sidecar")
        self.assertEqual(raw, original)
        self.assertNotIn("FORBIDDEN", json.dumps(record))
        self.assertNotIn("face_embedding", json.dumps(record))

    def test_ambiguous_source_reference_does_not_fabricate_metadata(self):
        raw = [{"detection_id": "person_0", "confidence": .8}] * 2
        row = detection_provenance(self.people, raw)[0]
        self.assertFalse(row["metadata_available"])
        self.assertNotIn("confidence", row)

    def test_stale_wrong_frame_or_dimensions_sidecar_is_not_attached(self):
        for field, value in (("frame_id", 99), ("width", 999), ("image", "wrong.jpg")):
            with self.subTest(field=field):
                self.tracker = PersonTracker(max_age=5., keep_unmatched=False)
                config = DiagnosticConfig(True, output_dir=self.root / field)
                write_detection_sidecar(config, self.root / "frame.jpg", 1, self.result["snapshot"].image.shape,
                                        self.people, [{"detection_id": "person_0", "confidence": .6}])
                path = sidecar_path(self.root / "frame.jpg")
                data = json.loads(path.read_bytes())
                data[field] = value
                path.write_text(json.dumps(data))
                _, record = self.record(config)
                self.assertEqual(record["detection_metadata_status"], "sidecar_metadata_mismatch")
                self.assertNotIn("confidence", record["detections"][0])

    def test_opt_in_snapshot_uses_owned_inference_pixels_after_jpeg_path_overwrite(self):
        pixels = np.random.default_rng(21).integers(0, 256, (90, 120, 3), dtype=np.uint8)
        path = self.root / "frame.jpg"
        cv2.imwrite(str(path), pixels)
        inference_pixels = cv2.imread(str(path))
        result = diagnostic_result(1, 98., 100.1, pixels=inference_pixels)
        cv2.imwrite(str(path), np.zeros_like(pixels))
        config = DiagnosticConfig(True, True, self.root / "debug")
        with patch.object(cv2, "imread", side_effect=AssertionError("must not reread JPEG")):
            logger, record = self.record(config, result=result)
        saved = cv2.imread(str(logger.session_dir / record["snapshot"]["file"]))
        np.testing.assert_array_equal(saved, inference_pixels)
        self.assertEqual((record["snapshot"]["frame_id"], record["snapshot"]["captured_at"]), (1, 98.))

    def test_snapshot_metadata_mismatch_stops_debug_not_tracker(self):
        config = DiagnosticConfig(True, True, self.root / "debug")
        changed = dict(self.result, frame_id=2)
        logger, record = self.record(config, result=changed)
        self.assertIsNone(record)
        self.assertFalse(logger.active)
        self.assertEqual(set(self.tracker.tracks), {1})
        self.assertFalse((self.root / "debug").exists())

    def test_frame_and_byte_caps_stop_persistent_logging(self):
        logger, record = self.record(DiagnosticConfig(True, output_dir=self.root / "frames", max_frames=1))
        self.assertIsNotNone(record)
        self.assertFalse(logger.active)
        self.assertIsNone(logger.before_update(self.tracker, 101.))
        self.assertEqual(logger.frames_written, 1)
        self.assertEqual(logger.bytes_written, sum(p.stat().st_size for p in logger.session_dir.iterdir()))
        self.tracker = PersonTracker(max_age=5., keep_unmatched=False)
        logger, record = self.record(DiagnosticConfig(True, output_dir=self.root / "bytes", max_bytes=100))
        self.assertIsNone(record)
        self.assertFalse(logger.active)
        self.assertFalse((self.root / "bytes").exists())

    def test_image_is_skipped_when_only_metadata_fits_byte_budget(self):
        pixels = np.random.default_rng(3).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        config = DiagnosticConfig(True, True, self.root / "debug", max_bytes=10000)
        logger, record = self.record(config, result=diagnostic_result(1, 98., 100.1, pixels=pixels))
        self.assertEqual(record["snapshot"], {"saved": False, "reason": "byte_limit"})
        self.assertLessEqual(sum(p.stat().st_size for p in logger.session_dir.iterdir()), 10000)

    def test_output_failure_does_not_undo_real_tracker_update(self):
        blocked = self.root / "blocked"
        blocked.write_text("existing user file")
        logger, record = self.record(DiagnosticConfig(True, output_dir=blocked))
        self.assertIsNone(record)
        self.assertFalse(logger.active)
        self.assertEqual(set(self.tracker.tracks), {1})
        self.assertEqual(blocked.read_text(), "existing user file")

    def test_duplicate_detection_selection_is_not_guessed_and_greedy_loss_is_distinguished(self):
        config = DiagnosticConfig(True, output_dir=self.root / "debug")
        logger, _ = self.record(config)
        duplicate = self.people + copy.deepcopy(self.people)
        before = logger.before_update(self.tracker, 101.)
        self.tracker.update(duplicate, 101.)
        record = logger.record(before, self.tracker, duplicate, diagnostic_result(2, 99., 101.1), self.root / "frame.jpg")
        self.assertEqual([p["selected"] for p in record["candidate_ious"]], [None, None])
        before = logger.before_update(self.tracker, 102.)
        self.tracker.update(self.people, 102.)
        record = logger.record(before, self.tracker, self.people, diagnostic_result(3, 100., 102.1), self.root / "frame.jpg")
        removed = [d for d in record["decisions"] if d["action"] == "unmatched_removed"]
        self.assertEqual([d["reason"] for d in removed], ["not_selected_by_greedy_matching"])

    def test_temporary_provenance_size_limit(self):
        config = DiagnosticConfig(True, max_bytes=10)
        self.assertFalse(write_detection_sidecar(config, self.root / "frame.jpg", 1, (480, 640, 3), self.people, []))
        self.assertFalse(sidecar_path(self.root / "frame.jpg").exists())

    def test_unchanged_live_pipeline_schema_and_results_with_debug_on_or_off(self):
        observations = [observation(i, 198.5 + (i - 1) * 1.5) for i in range(1, 4)]
        with patch.object(live.DiagnosticConfig, "from_environment", return_value=DiagnosticConfig()):
            ordinary = PipelineHarness().run(observations)
        config = DiagnosticConfig(True, output_dir=self.root / "debug")
        with patch.object(live.DiagnosticConfig, "from_environment", return_value=config):
            debug = PipelineHarness().run(observations)
        for normal, extra in zip(ordinary.results, debug.results):
            self.assertEqual(normal.keys(), extra.keys())
            for key in ("tracked_people", "tracked_faces", "matches", "behavior_frame", "warning_frame"):
                self.assertEqual(normal[key], extra[key])
            self.assertEqual(extra["warning_frame"].enabled_behaviors, ())
        session = next((self.root / "debug").iterdir())
        records = [json.loads(line) for line in (session / "frames.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 3)
        self.assertEqual(records[0]["associations"][0]["student_id"], "SV001")
        self.assertEqual(records[0]["associations"][0]["face_track_id"], 1)
        self.assertEqual([r["tracker_updated_at"] for r in records], [200., 201.5, 203.])
        self.assertEqual(records[1]["tracks_before"][0]["last_seen"], 200.)
        self.assertEqual(records[1]["tracks_before"][0]["age_seconds"], 1.5)
        self.assertEqual(len(list(session.iterdir())), 1)

    def test_live_pipeline_continues_when_diagnostic_disk_write_fails(self):
        config = DiagnosticConfig(True, output_dir=self.root / "debug")
        with (patch.object(live.DiagnosticConfig, "from_environment", return_value=config),
              patch.object(PersonTrackerDiagnostics, "_record", side_effect=OSError("disk unavailable"))):
            trace = PipelineHarness().run([observation(1, 198.5), observation(2, 200.)])
        self.assertEqual(len(trace.results), 2)
        self.assertFalse(trace.errors)
        self.assertFalse((self.root / "debug").exists())


if __name__ == "__main__":
    unittest.main()
