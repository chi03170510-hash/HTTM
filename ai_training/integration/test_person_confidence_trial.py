"""Pure trial tests and read-only diagnostic replay; no models or camera."""

import copy
import hashlib
import json
import os
import unittest
from collections import Counter
from pathlib import Path
from unittest.mock import patch

from person_confidence_trial import TRIAL_ENV, filter_person_outputs, person_trial_min_conf_from_environment
from person_tracker import PersonTracker
from person_face_association_v4 import associate_v4
from replay_face_tracker_diagnostics import replay_records as replay_faces
from replay_person_tracker_diagnostics import replay_records as replay_people
from face_tracker import FaceTracker


DEBUG_ROOT = Path(__file__).resolve().parents[1] / "runs/person_tracker_debug"
SESSIONS = ("session_5be088f4b254495c86ecd50a4f899ee3", "session_e62b02db457743419e0352de635ee80a")


def fixture(ref="person_0", confidence=.8, *, highres=False):
    person = {"person_ref": ref, "person_bbox_xyxy_clipped": [20., 40., 180., 420.],
              "bbox_xyxy": [20, 40, 180, 420],
              "behaviors": {"using_phone": {"state": "detected", "model_confidence": .82,
                                            "evidence": {"owner": ref, "phone_id": "phone_0"}}}}
    raw = {"detection_id": ref, "class_name": "person", "confidence": confidence,
           "bbox_xyxy": [20, 40, 180, 420], "highres_only": highres,
           "sources": ["coco_person_fullframe_highres" if highres else "coco_person_640"]}
    return person, raw


def session_inputs(name):
    path = DEBUG_ROOT / name / "frames.jsonl"
    data = path.read_bytes()
    rows = tuple(json.loads(line) for line in data.decode("utf-8-sig").splitlines() if line.strip())
    if not all(r["consistent"] for r in replay_people(rows)) or not all(r["consistent"] for r in replay_faces(rows)):
        raise AssertionError("source tracker replay mismatch")
    inputs = []
    for row in rows:
        if row["details_truncated"] or row["detection_metadata_status"] != "validated_sidecar":
            raise AssertionError("complete original person metadata required")
        people, raw = [], []
        for d in row["detections"]:
            # Original behavior payload is NOT saved in JSONL. Unknown/empty,
            # never fabricated detected evidence, is used for geometry replay.
            people.append({"person_ref": d["person_ref"], "bbox_xyxy": list(d["bbox"]),
                           "person_bbox_xyxy_clipped": list(d["bbox"]), "behaviors": {}})
            raw.append({"detection_id": d["person_ref"], "bbox_xyxy": list(d["bbox"]),
                        **{k: v for k, v in d.items() if k not in ("person_ref", "bbox", "detection_index")}})
        inputs.append((row, people, raw))
    return path, hashlib.sha256(data).hexdigest(), inputs


def replay_trial_session(name, min_conf):
    path, digest, inputs = session_inputs(name)
    person_tracker = PersonTracker(**inputs[0][0]["parameters"])
    face_tracker = FaceTracker(**inputs[0][0]["face_tracker"]["parameters"])
    trace, before_total, after_total, association_total = [], 0, 0, 0
    for row, people, raw in inputs:
        original = copy.deepcopy((row, people, raw))
        kept, rejected = filter_person_outputs(people, raw, min_conf)
        tracked = person_tracker.update(kept, row["tracker_updated_at"])
        faces = row["face_tracker"]["detections"]
        output = face_tracker.update([(tuple(f["face_bbox"]), f["current_student_id"], f["recognition_score"]) for f in faces],
                                     row["face_tracker"]["face_tracker_updated_at"])
        by_box = {tuple(f["face_bbox"]): f for f in faces}
        if len(by_box) != len(faces):
            raise AssertionError("this replay requires unique recorded face boxes")
        current_faces = [(tid, box, by_box[tuple(box)]["current_student_id"], score) for tid, box, _, score in output]
        matches = associate_v4(current_faces, kept)
        mapping = {p["person_ref"]: p["track_id"] for p in tracked}
        for match in matches:
            match["person_track_id"] = mapping[match["person_ref"]]
        assert all(m["person_ref"] in mapping for m in matches)
        if min_conf is None:
            expected = sorted((t["track_id"], t["person_ref"], tuple(t["bbox"])) for t in row["tracks_after"]
                              if t["last_seen"] == row["tracker_updated_at"])
            assert sorted((t["track_id"], t["person_ref"], tuple(t["bbox"])) for t in tracked) == expected
        assert (row, people, raw) == original
        before_total += len(people)
        after_total += len(kept)
        association_total += len(matches)
        trace.append({"frame_id": row["frame_id"], "refs": [p["person_ref"] for p in kept],
                      "tracked_people": tracked, "matches": matches, "rejected": rejected})
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
    return {"before": before_total, "after": after_total, "associations": association_total,
            "frames": len(trace), "trace": trace, "source_sha256": digest}


class TestPersonConfidenceTrial(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(TRIAL_ENV, None)

    def test_baseline_environment_is_unset_and_missing_metadata_is_not_required(self):
        self.assertIsNone(person_trial_min_conf_from_environment())
        people = [fixture(confidence=.01)[0]]
        kept, rejected = filter_person_outputs(people, [])
        self.assertIs(kept, people)
        self.assertEqual(rejected, [])

    def test_only_exact_approved_environment_value_is_accepted(self):
        os.environ[TRIAL_ENV] = "0.35"
        self.assertEqual(person_trial_min_conf_from_environment(), .35)
        del os.environ[TRIAL_ENV]
        self.assertIsNone(person_trial_min_conf_from_environment())

    def test_invalid_environment_values_are_rejected(self):
        for value in ("", "0.30", "0.350", "0.4", "0.35 ", " 0.35", ".35", "nan", "inf", "-1", "true"):
            with self.subTest(value=value), patch.dict(os.environ, {TRIAL_ENV: value}):
                with self.assertRaisesRegex(ValueError, TRIAL_ENV):
                    person_trial_min_conf_from_environment()

    def test_exact_confidence_boundaries_from_both_sources(self):
        for highres in (False, True):
            with self.subTest(highres=highres):
                entries = [fixture(f"person_{i}", score, highres=highres) for i, score in enumerate((.3499, .3500, .3501))]
                people, raw = map(list, zip(*entries))
                kept, rejected = filter_person_outputs(people, raw, .35)
                self.assertEqual([p["person_ref"] for p in kept], ["person_1", "person_2"])
                self.assertEqual(rejected[0]["confidence"], .3499)

    def test_output_order_identity_behaviors_and_inputs_are_unchanged(self):
        people, raw = map(list, zip(fixture("person_2", .9), fixture("person_0", .2), fixture("person_1", .4)))
        before = copy.deepcopy((people, raw))
        kept, _ = filter_person_outputs(people, raw[::-1], .35)
        self.assertEqual([p["person_ref"] for p in kept], ["person_2", "person_1"])
        self.assertIs(kept[0], people[0])
        self.assertIs(kept[1]["behaviors"], people[2]["behaviors"])
        self.assertEqual((people, raw), before)

    def test_confidence_is_not_borrowed_from_behavior_quality_or_highres_metadata(self):
        person, raw = fixture(confidence=.2)
        raw.update(person_quality={"confidence_object_detector": .99}, highres_match_iou=.99)
        kept, _ = filter_person_outputs([person], [raw], .35)
        self.assertEqual(kept, [])
        del raw["confidence"]
        with self.assertRaisesRegex(ValueError, "confidence"):
            filter_person_outputs([person], [raw], .35)

    def test_missing_nonfinite_out_of_range_boolean_and_text_confidence_fail(self):
        for value in (None, float("nan"), float("inf"), -.01, 1.01, True, ".9"):
            with self.subTest(value=value):
                person, raw = fixture(confidence=value)
                with self.assertRaisesRegex(ValueError, "confidence"):
                    filter_person_outputs([person], [raw], .35)

    def test_missing_or_duplicate_raw_mapping_fails(self):
        person, raw = fixture()
        for records in ([], [raw, copy.deepcopy(raw)]):
            with self.subTest(records=records), self.assertRaisesRegex(ValueError, "exactly one"):
                filter_person_outputs([person], records, .35)

    def test_duplicate_or_invalid_output_reference_fails(self):
        person, raw = fixture()
        for people in ([person, copy.deepcopy(person)], [{**person, "person_ref": None}], [{**person, "person_ref": ""}]):
            with self.subTest(people=people), self.assertRaisesRegex(ValueError, "person_ref"):
                filter_person_outputs(people, [raw], .35)

    def test_non_person_metadata_is_not_accepted_as_person_confidence(self):
        person, raw = fixture()
        raw["class_name"] = "cell phone"
        with self.assertRaisesRegex(ValueError, "not a person"):
            filter_person_outputs([person], [raw], .35)

    def test_empty_and_all_rejected_outputs_have_no_dangling_references(self):
        self.assertEqual(filter_person_outputs([], [], .35), ([], []))
        person, raw = fixture(confidence=.1)
        kept, rejected = filter_person_outputs([person], [raw], .35)
        self.assertEqual(kept, [])
        self.assertEqual(len(rejected), 1)

    def test_nested_references_to_removed_people_fail_instead_of_mutating_behavior(self):
        first, first_raw = fixture("person_0", .9)
        second, second_raw = fixture("person_1", .1)
        for reference in ({"person_detection_id": "person_1"}, {"candidate_person_ids": ["person_0", "person_1"]},
                          {"roi_provenance": {"person_ref": "person_1"}}, {"owners": {"person_1": {}}}):
            first["behaviors"]["using_phone"]["evidence"] = reference
            before = copy.deepcopy(first)
            with self.subTest(reference=reference), self.assertRaisesRegex(ValueError, "dangling references"):
                filter_person_outputs([first, second], [first_raw, second_raw], .35)
            self.assertEqual(first, before)

    def test_links_between_kept_people_and_null_owners_are_preserved(self):
        first, raw_first = fixture("person_0", .9)
        second, raw_second = fixture("person_1", .8)
        first["behaviors"]["using_phone"]["evidence"] = {"candidate_person_ids": ["person_1"], "owner": None}
        kept, _ = filter_person_outputs([first, second], [raw_first, raw_second], .35)
        self.assertIs(kept[0]["behaviors"], first["behaviors"])

    def test_only_approved_numeric_threshold_is_supported(self):
        for value in (.3, .4, True, float("nan")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                filter_person_outputs([], [], value)

    def test_single_person_session_baseline_and_candidate_replay(self):
        baseline, trial = (replay_trial_session(SESSIONS[0], value) for value in (None, .35))
        self.assertEqual((baseline["before"], baseline["after"], baseline["associations"]), (19, 19, 10))
        self.assertEqual((trial["before"], trial["after"], trial["associations"]), (19, 10, 10))
        self.assertTrue(all(len(r["tracked_people"]) == 1 for r in trial["trace"]))
        self.assertEqual([r["matches"] for r in trial["trace"]], [r["matches"] for r in baseline["trace"]])
        self.assertEqual(Counter(m["student_id"] for r in baseline["trace"] for m in r["matches"]),
                         Counter(m["student_id"] for r in trial["trace"] for m in r["matches"]))

    def test_d3_t1_session_keeps_targets_and_does_not_label_background_as_false_positive(self):
        baseline, trial = (replay_trial_session(SESSIONS[1], value) for value in (None, .35))
        self.assertEqual((baseline["before"], baseline["after"], baseline["associations"]), (24, 24, 17))
        self.assertEqual((trial["before"], trial["after"], trial["associations"]), (24, 21, 17))
        self.assertTrue(all(len(r["tracked_people"]) >= 2 for r in trial["trace"]))
        self.assertEqual([r["matches"] for r in trial["trace"]], [r["matches"] for r in baseline["trace"]])
        self.assertEqual([(r["frame_id"], x["confidence"]) for r in trial["trace"] for x in r["rejected"]],
                         [(7, .1805), (7, .1259), (8, .1743)])
        self.assertIn("person_3", trial["trace"][0]["refs"])  # Composite bbox remains.


if __name__ == "__main__":
    unittest.main()
