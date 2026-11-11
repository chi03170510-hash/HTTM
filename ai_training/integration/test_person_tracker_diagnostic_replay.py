"""Exact camera bboxes; independent synthetic capture/update/processing clocks."""

import copy
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path

import numpy as np

from behavior_duration import BehaviorFrame
from behavior_overlay import AISnapshot
from person_tracker import PersonTracker
from person_tracker_diagnostics import DiagnosticConfig, PersonTrackerDiagnostics
from replay_person_tracker_diagnostics import replay_records


BOX4_MAIN = [91.14, 66.23, 431.39, 412.74]
BOX4_SIDE = [0.46, 237.22, 187.14, 406.70]
BOX5_MAIN = [56.25, 63.61, 638.46, 414.51]
BOX6_MAIN = [88.32, 62.43, 431.99, 413.66]
BOX6_SIDE = [0.22, 237.52, 184.16, 409.25]


def diagnostic_result(frame_id, captured_at, completed_at, *, elapsed=2., pixels=None, matches=()):
    pixels = np.full((480, 640, 3), 80, dtype=np.uint8) if pixels is None else pixels
    return {"frame_id": frame_id, "captured_at": captured_at, "completed_at": completed_at,
            "elapsed": elapsed, "face_time": .3, "hybrid_time": elapsed - .3, "matches": list(matches),
            "snapshot": AISnapshot(frame_id, captured_at, BehaviorFrame(frame_id, captured_at, ()), pixels)}


def exact_replay(directory, *, gap=4., captures=(98., 100.1, 104.1), processing=(1.9, 3.8, 2.8)):
    tracker = PersonTracker(iou_threshold=.3, max_age=5., keep_unmatched=False)
    logger = PersonTrackerDiagnostics(DiagnosticConfig(True, output_dir=Path(directory)))
    times = (100., 100. + gap, 103. + gap)
    boxes = ((BOX4_MAIN, BOX4_SIDE), (BOX5_MAIN,), (BOX6_MAIN, BOX6_SIDE))
    records, outputs = [], []
    for index, (frame_id, timestamp, frame_boxes) in enumerate(zip((4, 5, 6), times, boxes)):
        people = [{"person_ref": f"frame_{frame_id}_person_{i}", "bbox_xyxy": list(box)}
                  for i, box in enumerate(frame_boxes)]
        before = logger.before_update(tracker, timestamp)
        output = tracker.update(people, timestamp)
        result = diagnostic_result(frame_id, captures[index], timestamp + .1, elapsed=processing[index],
                                   matches=[{"person_ref": people[0]["person_ref"], "track_id": 1,
                                             "person_track_id": next(p["track_id"] for p in output if p["person_ref"] == people[0]["person_ref"]),
                                             "association_status": "matched", "student_id": "SV001"}])
        records.append(logger.record(before, tracker, people, result, Path(directory) / "frame.jpg"))
        outputs.append(output)
    return records, outputs, logger


class TestPersonTrackerDiagnosticReplay(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))

    def test_case_a_within_five_preserves_p1_on_exact_wide_box_then_shrinks(self):
        with tempfile.TemporaryDirectory() as directory:
            records, outputs, _ = exact_replay(directory)
        self.assertEqual([[p["track_id"] for p in output] for output in outputs],
                         [["P1", "P2"], ["P1"], ["P1", "P3"]])
        self.assertEqual(outputs[1][0]["bbox"], tuple(BOX5_MAIN))
        self.assertAlmostEqual(records[1]["candidate_ious"][0]["iou"], .577099738, places=8)
        self.assertAlmostEqual(records[2]["candidate_ious"][0]["iou"], .58768886, places=8)
        self.assertTrue(all(r["consistent"] for r in replay_records(records)))

    def test_case_b_above_five_expires_before_matching_and_allocates_p3(self):
        with tempfile.TemporaryDirectory() as directory:
            records, outputs, _ = exact_replay(directory, gap=7.2, captures=(98., 100.1, 107.3), processing=(1.9, 7.05, 2.8))
        self.assertEqual([[p["track_id"] for p in output] for output in outputs],
                         [["P1", "P2"], ["P3"], ["P3", "P4"]])
        frame5 = records[1]
        self.assertAlmostEqual(frame5["tracker_update_gap"], 7.2)
        self.assertAlmostEqual(frame5["captured_at_gap"], 2.1)
        self.assertEqual(frame5["processing"]["elapsed"], 7.05)
        self.assertNotEqual(frame5["tracker_update_gap"], frame5["processing"]["elapsed"])
        self.assertEqual([d["track_id"] for d in frame5["decisions"] if d["action"] == "expired"], ["P1", "P2"])
        main = frame5["candidate_ious"][0]
        self.assertTrue(main["threshold_passed"])
        self.assertFalse(main["considered_by_tracker"])
        self.assertTrue(all(r["consistent"] for r in replay_records(records)))

    def test_exact_five_and_just_above_have_different_expiry(self):
        for gap, expected in ((5., "P1"), (5.001, "P3")):
            with self.subTest(gap=gap), tempfile.TemporaryDirectory() as directory:
                records, outputs, _ = exact_replay(directory, gap=gap)
                self.assertEqual(outputs[1][0]["track_id"], expected)
                self.assertTrue(all(r["consistent"] for r in replay_records(records)))

    def test_capture_gap_and_processing_do_not_drive_tracker_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            records, _, _ = exact_replay(directory)
        changed = copy.deepcopy(records)
        changed[1]["captured_at_gap"] = 50.
        changed[1]["processing"]["elapsed"] = 7.05
        self.assertEqual([r["tracked_people"] for r in replay_records(records)],
                         [r["tracked_people"] for r in replay_records(changed)])

    def test_side_track_is_unmatched_not_expired_when_gap_is_short(self):
        with tempfile.TemporaryDirectory() as directory:
            records, _, _ = exact_replay(directory)
        decision = next(d for d in records[1]["decisions"] if d["track_id"] == "P2")
        self.assertEqual((decision["action"], decision["reason"]), ("unmatched_removed", "no_eligible_iou"))
        self.assertAlmostEqual(records[1]["candidate_ious"][1]["iou"], .103779879, places=8)

    def test_offline_replay_reads_persisted_jsonl_and_detects_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, logger = exact_replay(directory)
            records = [json.loads(line) for line in (logger.session_dir / "frames.jsonl").read_text().splitlines()]
        self.assertTrue(all(r["consistent"] for r in replay_records(records)))
        records[1]["tracks_after"][0]["track_id"] = "P99"
        self.assertFalse(replay_records(records)[1]["consistent"])

    def test_incomplete_truncated_and_empty_replays_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            records, _, _ = exact_replay(directory)
        truncated = copy.deepcopy(records)
        truncated[0]["details_truncated"] = True
        for invalid in ([], records[1:], truncated):
            with self.subTest(invalid=bool(invalid)), self.assertRaises(ValueError):
                replay_records(invalid)


if __name__ == "__main__":
    unittest.main()
