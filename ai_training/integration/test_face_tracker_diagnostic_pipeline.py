"""Real ai_loop/trackers/V4/CH20/CH21/panel; synthetic worker replies only."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import camera_coordinator_live as live
from behavior_warning import DEFAULT_WARNING_CONFIG
from face_tracker_diagnostics import FaceTrackerDiagnostics
from person_tracker_diagnostics import DiagnosticConfig
from replay_face_tracker_diagnostics import replay_records as replay_faces
from replay_person_tracker_diagnostics import replay_records as replay_people
from test_ch21_warning_pipeline import events, run_pipeline, source
from test_ch20_duration_pipeline import phone_value


def observations(count=5):
    # Each source is fresh at evaluation; actual mocked completion cadence is 3s.
    return [source(i, 198.5 + 3. * (i - 1)) for i in range(1, count + 1)]


def collect(directory, items, *, face_enabled=True, max_frames=10, max_bytes=20971520):
    config = DiagnosticConfig(True, output_dir=Path(directory), face_enabled=face_enabled,
                              max_frames=max_frames, max_bytes=max_bytes)
    with patch.object(live.DiagnosticConfig, "from_environment", return_value=config):
        trace = run_pipeline(items)
    paths = list(Path(directory).glob("session_*/frames.jsonl"))
    records = [json.loads(line) for line in paths[0].read_text().splitlines()] if paths else []
    return trace, records, paths


class TestFaceTrackerDiagnosticPipeline(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temp)

    def test_case_g_face_ids_change_but_person_duration_and_warning_episode_continue(self):
        trace, records, _ = collect(self.root, observations())
        self.assertEqual([r["matches"][0]["track_id"] for r in trace.results], [1, 2, 3, 4, 5])
        self.assertEqual([r["matches"][0]["person_track_id"] for r in trace.results], ["P1"] * 5)
        self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., 9., 12.])
        self.assertEqual([(e.frame_id, e.qualified_seconds) for e in events(trace)], [(3, 6.)])
        episode_ids = [next(v.episode_id for v in r["warning_frame"].views if v.behavior == "using_phone") for r in trace.results]
        self.assertEqual(len(set(episode_ids)), 1)
        self.assertEqual(len(trace.samples), 10)
        self.assertTrue(all(r["consistent"] for r in replay_faces(records)))
        self.assertTrue(all(r["consistent"] for r in replay_people(records)))
        self.assertEqual([r["face_tracker"]["face_update_gap"] for r in records], [None, 3., 3., 3., 3.])
        self.assertEqual([r["face_tracker"]["tracks_after"][0]["stable_label"] for r in records], ["Unknown"] * 5)
        self.assertIs(live.WARNING_CONFIG, DEFAULT_WARNING_CONFIG)

    def test_case_h_unknown_duplicate_or_lost_association_breaks_episodes(self):
        for mode in ("unknown", "duplicate", "association_lost", "missing_face"):
            with self.subTest(mode=mode):
                items = observations(7)
                kwargs = {"student": "Unknown"} if mode == "unknown" else (
                    {"second_student": "SV001"} if mode == "duplicate" else {"associated": False})
                items[3] = source(4, 207.5, **kwargs)
                if mode == "missing_face":
                    items[3] = replace(source(4, 207.5), faces=())
                trace, records, _ = collect(self.root / mode, items)
                self.assertEqual([phone_value(r) for r in trace.results], [0., 3., 6., None, 0., 3., 6.])
                self.assertEqual([e.frame_id for e in events(trace)], [3, 7])
                self.assertNotEqual(events(trace)[0].episode_id, events(trace)[1].episode_id)
                self.assertTrue(all(v.state == "CLEAR" for v in trace.results[3]["warning_frame"].views))
                self.assertTrue(all(r["consistent"] for r in replay_faces(records)))
                if mode == "duplicate":
                    self.assertEqual(len(records[3]["face_tracker"]["mapping"]), 2)

    def test_all_faces_including_unassociated_are_saved_in_same_session(self):
        item = source(1, 198.5, second_student="SV002")
        extra = {"face_bbox": [530, 50, 575, 100], "student_id": "Unknown", "score": .32,
                 "embedding": [123], "student_name": "PRIVATE"}
        item = replace(item, faces=item.faces + (extra,))
        trace, records, paths = collect(self.root, [item])
        block = records[0]["face_tracker"]
        self.assertEqual(trace.results[0]["face_count"], 3)
        self.assertEqual(block["face_detection_count"], 3)
        self.assertEqual(len(block["mapping"]), 3)
        self.assertEqual(block["mapping"][2]["associations"], [])
        self.assertEqual(block["detections"][2]["recognition_score"], .32)
        self.assertEqual(len(list(paths[0].parent.iterdir())), 1)
        self.assertNotIn("embedding", paths[0].read_text())
        self.assertNotIn("PRIVATE", paths[0].read_text())
        self.assertEqual(block["face_tracker_updated_at"], 200.)
        self.assertEqual(block["person_tracker_updated_at"], records[0]["tracker_updated_at"])
        self.assertEqual(block["processing"]["face"], trace.results[0]["face_time"])
        self.assertEqual(block["frame_id"], records[0]["frame_id"])
        self.assertEqual(block["captured_at"], records[0]["captured_at"])

    def test_face_debug_on_off_keeps_production_schema_values_and_timers_identical(self):
        items = observations(3)
        with (patch.object(FaceTrackerDiagnostics, "before_update", side_effect=AssertionError("face debug off")),
              patch.object(FaceTrackerDiagnostics, "build", side_effect=AssertionError("face debug off"))):
            ordinary, old_records, _ = collect(self.root / "off", items, face_enabled=False)
        extra, records, _ = collect(self.root / "on", items)
        for normal, debug in zip(ordinary.results, extra.results):
            self.assertEqual(normal.keys(), debug.keys())
            for key in ("tracked_faces", "tracked_people", "matches", "behavior_frame", "warning_frame",
                        "face_time", "hybrid_time", "elapsed", "captured_at", "completed_at"):
                self.assertEqual(normal[key], debug[key])
        self.assertTrue(all("face_tracker" not in r for r in old_records))
        for old, new in zip(old_records, records):
            without_face = {k: v for k, v in new.items() if k != "face_tracker"}
            self.assertEqual(old, without_face)

    def test_shared_frame_and_byte_quota_include_face_block_without_new_images(self):
        trace, records, paths = collect(self.root / "frames", observations(), max_frames=2)
        self.assertEqual(len(trace.results), 5)
        self.assertEqual(len(records), 2)
        self.assertTrue(all("face_tracker" in r for r in records))
        self.assertEqual([p.name for p in paths[0].parent.iterdir()], ["frames.jsonl"])
        trace, records, paths = collect(self.root / "bytes", observations(), max_bytes=6000)
        self.assertEqual(len(trace.results), 5)
        self.assertLess(len(records), 5)
        self.assertLessEqual(sum(p.stat().st_size for p in paths[0].parent.iterdir()) if paths else 0, 6000)

    def test_face_debug_build_failure_does_not_stop_ai_or_change_warning_defaults(self):
        with patch.object(FaceTrackerDiagnostics, "build", side_effect=ValueError("debug unavailable")):
            trace, records, _ = collect(self.root, observations(3))
        self.assertFalse(records)
        self.assertEqual(len(trace.results), 3)
        self.assertFalse(trace.errors)
        self.assertEqual([e.frame_id for e in events(trace)], [3])
        self.assertEqual(DEFAULT_WARNING_CONFIG.enabled_behaviors, ())


if __name__ == "__main__":
    unittest.main()
