"""Real worker output gate and CH20 path; model/camera boundaries are mocked.

Run with .venv-hybrid. Real checkpoints and camera devices are never opened.
"""

import copy
import io
import json
import os
import unittest
from contextlib import contextmanager, redirect_stderr
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import camera_coordinator_live as live
import hybrid_v7_stream_worker as module
from behavior_warning import DEFAULT_WARNING_CONFIG
from person_confidence_trial import TRIAL_ENV, person_trial_min_conf_from_environment
from person_tracker_diagnostics import DiagnosticConfig
from test_ch20_duration_pipeline import PipelineHarness, _ReplayWorker, observation, phone_value
from test_person_confidence_trial import SESSIONS, fixture, session_inputs


def make_worker():
    worker = module.HybridV7Stream.__new__(module.HybridV7Stream)
    worker.trial_min_conf = person_trial_min_conf_from_environment()
    worker.model, worker.objects, worker.face = Mock(), Mock(), None
    worker.spatial_config, worker.device = object(), "cpu"
    worker.names = {0: "sleeping", 1: "using_phone"}
    worker.config = {"inference_conf": .05, "person_roi_enabled": True, "person_roi_padding": .12, "dedupe_iou": .65}
    worker.diagnostic_config, worker._diagnostic_frames = DiagnosticConfig(), 0
    return worker


@contextmanager
def inference_replies(people, raw, *, read_image=True):
    obj = {"person_detections": raw,
           "phone_detections": [{"detection_id": "phone_0", "confidence": .18, "bbox_xyxy": [50, 100, 70, 140]}],
           "food_detections": [{"detection_id": "food_0", "class_name": "apple", "confidence": .12}],
           "associations": [{"phone_detection_id": "phone_0", "person_detection_id": "person_0"}]}
    spatial = {"people": people, "phone_detections": obj["phone_detections"], "food_detections": obj["food_detections"]}
    with (patch.object(module, "prepare_objects_v7", return_value=obj),
          patch.object(module, "infer_v3", return_value={"detections": []}) as infer,
          patch.object(module, "add_sleeping_highres_proposals_v7", return_value=([], {})),
          patch.object(module, "spatial_v7", return_value=spatial)):
        if read_image:
            with patch.object(module.cv2, "imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
                yield obj, spatial, infer
        else:
            yield obj, spatial, infer


class TestHybridPersonConfidenceTrial(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for name in tuple(os.environ):
            if name == TRIAL_ENV or name.startswith(("HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG")):
                os.environ.pop(name)

    def test_constructor_reads_environment_once_and_keeps_shared_detector_confidence(self):
        for value in (None, "0.35"):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(TRIAL_ENV, None)
                else:
                    os.environ[TRIAL_ENV] = value
                model = Mock(names={0: "sleeping", 1: "using_phone"})
                with (patch.object(module, "YOLO", return_value=model),
                      patch.object(module, "UsingPhonePrototype") as objects,
                      patch.object(module.FaceFeatureExtractor, "availability", return_value=(False, "mock")),
                      redirect_stderr(io.StringIO()) as log):
                    worker = module.HybridV7Stream()
                    self.assertEqual(worker.trial_min_conf, None if value is None else .35)
                    objects.assert_called_once()
                    self.assertEqual(objects.call_args.kwargs["conf"], .12)
                    self.assertEqual(worker.config["inference_conf"], .05)
                    os.environ.pop(TRIAL_ENV, None)
                    self.assertEqual(worker.trial_min_conf, None if value is None else .35)
                    worker.close()
                self.assertEqual("[PERSON CONFIDENCE TRIAL]" in log.getvalue(), value is not None)

    def test_invalid_environment_fails_before_loading_models(self):
        os.environ[TRIAL_ENV] = "0.350"
        with patch.object(module, "YOLO") as model, patch.object(module, "UsingPhonePrototype") as objects:
            with self.assertRaisesRegex(ValueError, TRIAL_ENV):
                module.HybridV7Stream()
        model.assert_not_called()
        objects.assert_not_called()

    def test_baseline_schema_and_low_confidence_output_remain_unchanged(self):
        people, raw = map(list, zip(fixture("person_0", .1), fixture("person_1", .9)))
        # Baseline deliberately tolerates missing provenance as it did before.
        with inference_replies(people, []), redirect_stderr(io.StringIO()) as log:
            result = make_worker().process(Path("frame.jpg"), 1)
        self.assertEqual(set(result), {"frame_id", "image", "mode", "person_count", "people"})
        self.assertEqual(result["person_count"], 2)
        self.assertEqual(result["people"], people)
        self.assertNotIn("PERSON CONFIDENCE TRIAL", log.getvalue())

    def test_worker_boundary_scores_and_both_detection_sources(self):
        os.environ[TRIAL_ENV] = "0.35"
        entries = [fixture(f"person_{i}", score, highres=i >= 3)
                   for i, score in enumerate((.3499, .3500, .3501, .3499, .3500, .3501))]
        people, raw = map(list, zip(*entries))
        with inference_replies(people, raw), redirect_stderr(io.StringIO()) as log:
            result = make_worker().process(Path("frame.jpg"), 12)
        self.assertEqual([p["person_ref"] for p in result["people"]], ["person_1", "person_2", "person_4", "person_5"])
        self.assertEqual(result["person_count"], 4)
        payload = json.loads(log.getvalue().split("[PERSON CONFIDENCE TRIAL] ", 1)[1])
        self.assertEqual((payload["frame_id"], payload["min_conf"], payload["before"], payload["after"]), (12, .35, 6, 4))
        self.assertEqual([r["person_ref"] for r in payload["rejected"]], ["person_0", "person_3"])

    def test_phone_food_and_retained_behavior_are_unchanged_and_not_reinferred(self):
        people, raw = map(list, zip(fixture("person_0", .9), fixture("person_1", .2)))
        before = copy.deepcopy((people, raw))
        with inference_replies(people, raw) as (obj, spatial, infer), redirect_stderr(io.StringIO()):
            original = copy.deepcopy((obj, spatial))
            baseline = make_worker().process(Path("frame.jpg"), 1)
            os.environ[TRIAL_ENV] = "0.35"
            candidate = make_worker().process(Path("frame.jpg"), 1)
            self.assertEqual((obj, spatial), original)
            self.assertEqual(infer.call_count, 2)
            self.assertTrue(all(call.kwargs["conf"] == .05 for call in infer.call_args_list))
        self.assertEqual((people, raw), before)
        self.assertEqual(set(candidate), set(baseline))
        self.assertEqual(candidate["people"][0], baseline["people"][0])
        self.assertIs(candidate["people"][0]["behaviors"], people[0]["behaviors"])
        self.assertEqual(set(candidate["people"][0]), {"person_ref", "bbox_xyxy", "person_bbox_xyxy_clipped", "behaviors"})

    def test_missing_confidence_or_ambiguous_mapping_fails_in_worker(self):
        os.environ[TRIAL_ENV] = "0.35"
        person, raw = fixture()
        for records in ([], [raw, copy.deepcopy(raw)], [{k: v for k, v in raw.items() if k != "confidence"}]):
            with self.subTest(records=records), inference_replies([person], records), self.assertRaises(ValueError):
                make_worker().process(Path("frame.jpg"), 1)

    def test_dangling_evidence_aborts_frame_before_diagnostics_without_mutation(self):
        os.environ[TRIAL_ENV] = "0.35"
        people, raw = map(list, zip(fixture("person_0", .9), fixture("person_1", .2)))
        people[0]["behaviors"]["using_phone"]["evidence"] = {"candidate_person_ids": ["person_0", "person_1"]}
        before = copy.deepcopy(people)
        with inference_replies(people, raw), patch.object(module, "write_detection_sidecar") as write:
            with self.assertRaisesRegex(ValueError, "dangling references"):
                make_worker().process(Path("frame.jpg"), 1)
        write.assert_not_called()
        self.assertEqual(people, before)

    def test_protocol_stdout_stays_json_and_trial_logs_go_to_stderr(self):
        os.environ[TRIAL_ENV] = "0.35"
        people, raw = map(list, zip(fixture("person_0", .9), fixture("person_1", .2)))
        worker, stdout, stderr = make_worker(), io.StringIO(), io.StringIO()
        requests = io.StringIO(json.dumps({"frame_id": 1, "image": "frame.jpg"}) + "\n")
        with (inference_replies(people, raw), patch.object(module, "HybridV7Stream", return_value=worker),
              patch.object(module.sys, "stdin", requests), patch.object(module.sys, "stdout", stdout),
              patch.object(module.Path, "is_file", return_value=True), redirect_stderr(stderr)):
            module.main()
        reply = json.loads(stdout.getvalue())
        self.assertEqual((reply["frame_id"], reply["person_count"]), (1, 1))
        self.assertEqual(len(stdout.getvalue().splitlines()), 1)
        self.assertIn("[PERSON CONFIDENCE TRIAL]", stderr.getvalue())

    def test_protocol_reports_trial_metadata_error_instead_of_success(self):
        os.environ[TRIAL_ENV] = "0.35"
        person, _ = fixture()
        worker, stdout = make_worker(), io.StringIO()
        requests = io.StringIO(json.dumps({"frame_id": 7, "image": "frame.jpg"}) + "\n")
        with (inference_replies([person], []), patch.object(module, "HybridV7Stream", return_value=worker),
              patch.object(module.sys, "stdin", requests), patch.object(module.sys, "stdout", stdout),
              patch.object(module.Path, "is_file", return_value=True), redirect_stderr(io.StringIO())):
            module.main()
        reply = json.loads(stdout.getvalue())
        self.assertEqual(set(reply), {"frame_id", "error"})
        self.assertEqual(reply["frame_id"], 7)
        self.assertIn("exactly one", reply["error"])

    def test_both_recorded_sessions_through_real_worker_output_gate(self):
        for name, expected in zip(SESSIONS, ((19, 10), (24, 21))):
            for value, count in zip((None, "0.35"), expected):
                if value is None:
                    os.environ.pop(TRIAL_ENV, None)
                else:
                    os.environ[TRIAL_ENV] = value
                worker, total = make_worker(), 0
                _, _, inputs = session_inputs(name)
                for row, people, raw in inputs:
                    with inference_replies(people, raw), redirect_stderr(io.StringIO()):
                        result = worker.process(Path("recorded_frame.jpg"), row["frame_id"])
                    total += result["person_count"]
                    self.assertEqual(result["frame_id"], row["frame_id"])
                    self.assertEqual(result["person_count"], len(result["people"]))
                self.assertEqual(total, count)

    def run_live_pipeline(self, items, confidences):
        original = _ReplayWorker.process_frame
        hybrid = make_worker()

        def process(worker, path, frame_id):
            reply = original(worker, path, frame_id)
            if worker.kind != "hybrid":
                return reply
            index = len(worker.requests) - 1
            people = reply["people"]
            raw = [{"detection_id": p["person_ref"], "class_name": "person", "confidence": confidences[index][i]}
                   for i, p in enumerate(people)]
            with inference_replies(people, raw, read_image=False):
                return hybrid.process(path, frame_id)

        with (patch.object(_ReplayWorker, "process_frame", process),
              patch.object(live, "WARNING_CONFIG", DEFAULT_WARNING_CONFIG),
              patch("http.client.HTTPConnection.request", side_effect=AssertionError("API forbidden"))):
            return PipelineHarness().run(items)

    def test_real_worker_to_association_track_duration_snapshot_and_panel(self):
        items = [observation(i, 198.5 + 1.5 * (i - 1), second_student="SV002") for i in range(1, 5)]
        for value, expected in ((None, 2), ("0.35", 1)):
            with self.subTest(value=value):
                if value is None:
                    os.environ.pop(TRIAL_ENV, None)
                else:
                    os.environ[TRIAL_ENV] = value
                trace = self.run_live_pipeline(items, [[.9, .2]] * 4)
                self.assertEqual([r["person_count"] for r in trace.results], [expected] * 4)
                self.assertEqual([len(r["matches"]) for r in trace.results], [expected] * 4)
                self.assertEqual([phone_value(r) for r in trace.results], [0., 1.5, 3., 4.5])
                self.assertTrue(all(m["person_ref"] in {p["person_ref"] for p in r["tracked_people"]}
                                    for r in trace.results for m in r["matches"]))
                self.assertTrue(all(not r["warning_frame"].events and r["warning_frame"].enabled_behaviors == () for r in trace.results))
                self.assertEqual(len(trace.samples), 8)
                self.assertFalse(trace.errors)
                if value is not None:
                    self.assertTrue(all(m["student_id"] == "SV001" for r in trace.results for m in r["matches"]))
                    self.assertTrue(all("Person Track: P1" in s.labels for s in trace.samples))

    def test_filtered_person_breaks_duration_and_return_starts_new_track(self):
        os.environ[TRIAL_ENV] = "0.35"
        items = [observation(i, 198.5 + 1.5 * (i - 1)) for i in range(1, 5)]
        trace = self.run_live_pipeline(items, [[.9], [.9], [.2], [.9]])
        self.assertEqual([r["person_count"] for r in trace.results], [1, 1, 0, 1])
        self.assertEqual(trace.results[2]["behavior_frame"].people, ())
        self.assertEqual([phone_value(trace.results[i]) for i in (0, 1, 3)], [0., 1.5, 0.])
        self.assertEqual(trace.results[3]["matches"][0]["person_track_id"], "P2")
        self.assertTrue(all(not r["warning_frame"].events for r in trace.results))


if __name__ == "__main__":
    unittest.main()
