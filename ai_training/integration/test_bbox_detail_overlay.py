"""B3+B4 synthetic evidence only; no physical camera or model inference."""

import copy
from dataclasses import FrozenInstanceError, replace
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

import cv2
import numpy as np

import bbox_visual_metadata as metadata
import behavior_overlay as drawing
import person_snapshot_overlay as overlay
import snapshot_detail_overlay as details
from behavior_duration import BehaviorDuration, BehaviorFrame, PersonBehaviorView
import test_ch20_duration_pipeline as pipeline
import test_identity_continuity_shadow as shadow_tests


def synthetic(directory):
    """Independent display fixture, NOT camera data or detection ground truth."""
    image_path = Path(directory) / "frame.jpg"
    pixels = np.full((240, 320, 3), 55, dtype=np.uint8)
    cv2.rectangle(pixels, (10, 5), (145, 235), (65, 80, 95), -1)
    cv2.rectangle(pixels, (175, 5), (310, 235), (90, 80, 60), -1)
    cv2.imwrite(str(image_path), pixels)
    pixels = cv2.imread(str(image_path))
    people = (
        PersonBehaviorView("P1", "person_a", (10, 5, 145, 235), "SV001", 11, "matched", "matched",
                           (BehaviorDuration("sleeping", "detected", 3.), BehaviorDuration("using_phone", "not_detected", None))),
        PersonBehaviorView("P2", "person_b", (175, 5, 310, 235), "SV002", 22, "matched", "matched",
                           (BehaviorDuration("sleeping", "not_detected", None), BehaviorDuration("using_phone", "detected", 5.))),
    )
    frame = BehaviorFrame(7, 10., people)
    snapshot = drawing.AISnapshot(7, 10., frame, pixels)
    faces = [{"face_bbox": [30, 20, 70, 60], "student_id": "SV001", "score": .72},
             {"face_bbox": [220, 20, 260, 60], "student_id": "SV002", "score": .79}]
    tracks = [(11, faces[0]["face_bbox"], "STABLE_MUST_NOT_LEAK", .72),
              (22, faces[1]["face_bbox"], "STABLE_MUST_NOT_LEAK", .79)]
    detections = [{"class_name": "sleeping", "bbox_xyxy": [20, 40, 130, 150], "confidence": .93,
                   "source": "full_frame_highres", "person_ref": "person_a", "association_status": "associated"},
                  {"class_name": "using_phone", "bbox_xyxy": [180, 60, 300, 210], "confidence": .81,
                   "source": "person_roi", "person_ref": "person_b", "association_status": "associated_by_person_roi"}]
    objects = {"phone_detections": [{"detection_id": "phone_1", "bbox_xyxy": [230, 110, 250, 145],
                                     "confidence": .27, "source": "coco"}],
               "v6_phone_person_associations": [{"phone_detection_id": "phone_1", "person_detection_id": "person_b",
                                                  "status": "ASSOCIATED_GEOMETRICALLY"}]}
    raw_people = [{"person_ref": "person_a", "behaviors": {"sleeping": {"state": "detected", "bounding_box": [20, 40, 130, 150]}}},
                  {"person_ref": "person_b", "behaviors": {"using_phone": {"state": "detected", "bounding_box": [230, 110, 250, 145]}}}]
    return SimpleNamespace(path=image_path, snapshot=snapshot, faces=faces, tracks=tracks,
                           detections=detections, objects=objects, raw_people=raw_people)


def export(fixture):
    for role, kwargs in (("face", {"faces": fixture.faces}), ("hybrid", {"detections": fixture.detections,
                           "objects": fixture.objects, "people": fixture.raw_people})):
        assert metadata.write_source(fixture.path, fixture.snapshot.frame_id, fixture.snapshot.image.shape, role, **kwargs)
    return {"frame_id": fixture.snapshot.frame_id, "image": str(fixture.path.resolve()), "faces": copy.deepcopy(fixture.faces)}


def project(fixture):
    reply = export(fixture)
    return metadata.collect(fixture.snapshot, fixture.path, reply, fixture.tracks)


def result(snapshot):
    return {"frame_id": snapshot.frame_id, "captured_at": snapshot.captured_at, "completed_at": 11.,
            "behavior_frame": snapshot.observations, "snapshot": snapshot}


class TestVisualMetadata(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.f = synthetic(self.temp.name)
        self.enterContext(redirect_stderr(io.StringIO()))

    def edit(self, source_role, **changes):
        path = metadata.sidecar_path(self.f.path, source_role)
        packet = json.loads(path.read_bytes())
        packet.update(changes)
        path.write_text(json.dumps(packet), encoding="utf-8")

    def test_two_people_keys_scores_and_actual_boxes(self):
        original = copy.deepcopy((self.f.faces, self.f.tracks, self.f.detections, self.f.objects, self.f.raw_people))
        view = project(self.f)
        self.assertFalse(view.issues)
        self.assertEqual([b.owner_pid for b in view.boxes], ["P1", "P2", "P1", "P2", "P2"])
        self.assertEqual([b.score for b in view.boxes], [.72, .79, .93, .81, .27])
        self.assertEqual(view.boxes[-2].bbox, (180, 60, 300, 210))
        self.assertEqual(view.boxes[-1].bbox, (230, 110, 250, 145))
        self.assertTrue(view.boxes[2].selected)
        self.assertFalse(view.boxes[3].selected)
        self.assertTrue(view.boxes[4].selected)
        self.assertEqual([b.duration for b in view.boxes[2:4]], [3., 5.])
        self.assertEqual(original, (self.f.faces, self.f.tracks, self.f.detections, self.f.objects, self.f.raw_people))

    def test_reordered_sources_and_production_views_use_keys_not_position(self):
        a = project(self.f)
        self.f.faces.reverse(); self.f.tracks.reverse(); self.f.detections.reverse(); self.f.raw_people.reverse()
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations,
                                            people=tuple(reversed(self.f.snapshot.observations.people))))
        b = project(self.f)
        self.assertEqual(set(a.boxes), set(b.boxes))

    def test_single_person(self):
        self.f.faces = self.f.faces[:1]; self.f.tracks = self.f.tracks[:1]
        self.f.detections = self.f.detections[:1]; self.f.objects = {}; self.f.raw_people = self.f.raw_people[:1]
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations,
                                            people=self.f.snapshot.observations.people[:1]))
        self.assertEqual([b.owner_pid for b in project(self.f).boxes], ["P1", "P1"])

    def test_face_unknown_is_detection_without_claiming_recognition_or_sid(self):
        self.f.faces[0]["student_id"] = "Unknown"
        face = project(self.f).boxes[0]
        self.assertEqual(face.face_track_id, 11)
        self.assertEqual(face.recognition, "Unknown")
        self.assertIsNone(face.owner_pid)
        self.assertEqual(face.identity, "Unknown")

    def test_recognized_face_without_accepted_production_sid_does_not_leak_raw_sid(self):
        self.f.faces[0]["student_id"] = "RAW_SID_MUST_NOT_LEAK"
        p = replace(self.f.snapshot.observations.people[0], student_id=None, identity_status="unassociated")
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations,
                                    people=(p, self.f.snapshot.observations.people[1])))
        item = project(self.f).boxes[0]
        self.assertEqual(item.identity, "Unknown")
        self.assertNotIn("RAW_SID", repr(item))

    def test_current_face_sid_conflict_is_not_hidden_by_production_view(self):
        self.f.faces[0]["student_id"] = "DIFFERENT"
        item = project(self.f).boxes[0]
        self.assertIsNone(item.owner_pid)
        self.assertEqual(item.identity, "Unknown")

    def test_unreliable_face_track_is_omitted(self):
        self.f.tracks[0] = (11, (1, 2, 3, 4), "SV001", .72)
        face = project(self.f).boxes[0]
        self.assertIsNone(face.face_track_id)
        self.assertIsNone(face.owner_pid)

    def test_duplicate_face_geometry_does_not_join_to_face_track(self):
        self.f.faces.append(copy.deepcopy(self.f.faces[0]))
        faces = [b for b in project(self.f).boxes if b.bbox == tuple(self.f.faces[0]["face_bbox"])]
        self.assertTrue(all(f.face_track_id is None and f.owner_pid is None for f in faces))

    def test_duplicate_face_track_ids_cannot_establish_ownership(self):
        self.f.tracks[1] = (11, *self.f.tracks[1][1:])
        self.assertTrue(all(b.face_track_id is None for b in project(self.f).boxes[:2]))

    def test_duplicate_sid_is_labelled_ambiguous(self):
        people = self.f.snapshot.observations.people
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations,
                                    people=(people[0], replace(people[1], student_id="SV001"))))
        items = project(self.f).boxes
        self.assertEqual(items[2].identity, "Ambiguous")
        self.assertEqual(items[3].identity, "Ambiguous")

    def test_ambiguous_production_identity_has_no_accepted_sid(self):
        people = self.f.snapshot.observations.people
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations,
                             people=(replace(people[0], student_id=None, identity_status="ambiguous_identity"), people[1])))
        self.assertEqual(project(self.f).boxes[2].identity, "Ambiguous")

    def test_sleeping_candidate_not_promoted_when_production_state_differs(self):
        people = self.f.snapshot.observations.people
        p = replace(people[0], behaviors=(BehaviorDuration("sleeping", "suspected", None),))
        self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations, people=(p, people[1])))
        b = project(self.f).boxes[2]
        self.assertFalse(b.selected)
        self.assertEqual(b.state, "suspected")
        self.assertIsNone(b.duration)

    def test_unassigned_candidate_does_not_acquire_student(self):
        self.f.detections[0].update(association_status="unassigned")
        b = project(self.f).boxes[2]
        self.assertIsNone(b.owner_pid)
        self.assertEqual(b.identity, "Unknown")
        self.assertIsNone(b.state)
        self.assertFalse(b.selected)

    def test_ambiguous_candidate_does_not_acquire_student(self):
        self.f.detections[0].update(association_status="ambiguous", association_ambiguous=True)
        b = project(self.f).boxes[2]
        self.assertIsNone(b.owner_pid)
        self.assertEqual(b.identity, "Ambiguous")

    def test_missing_person_ref_is_not_repaired_by_geometry(self):
        self.f.detections[0].pop("person_ref")
        self.assertIsNone(project(self.f).boxes[2].owner_pid)

    def test_duplicate_ref_and_pid_cannot_link_candidates(self):
        people = self.f.snapshot.observations.people
        for p in (replace(people[1], person_ref="person_a"), replace(people[1], person_track_id="P1")):
            with self.subTest(p=p):
                self.f.snapshot = replace(self.f.snapshot, observations=replace(self.f.snapshot.observations, people=(people[0], p)))
                self.assertIsNone(project(self.f).boxes[2].owner_pid)

    def test_duplicate_sleeping_boxes_are_not_claimed_as_selected(self):
        self.f.detections.append(copy.deepcopy(self.f.detections[0]))
        self.assertFalse(any(b.selected for b in project(self.f).boxes if b.kind == "sleeping_candidate"))

    def test_phone_ambiguous_or_missing_association_cannot_link(self):
        for rows in ([], [{"phone_detection_id": "phone_1", "person_detection_id": "person_b", "status": "AMBIGUOUS"}],
                     [{"phone_detection_id": "phone_1", "person_detection_id": "missing_person", "status": "ASSOCIATED_GEOMETRICALLY"}]):
            with self.subTest(rows=rows):
                self.f.objects["v6_phone_person_associations"] = rows
                b = project(self.f).boxes[-1]
                self.assertIsNone(b.owner_pid)
                self.assertFalse(b.selected)
                self.assertEqual(b.identity, "Ambiguous" if rows and rows[0]["status"] == "AMBIGUOUS" else "Unknown")

    def test_phone_duplicate_detection_or_association_ids_cannot_link(self):
        for key in ("phone_detections", "v6_phone_person_associations"):
            with self.subTest(key=key):
                original = copy.deepcopy(self.f.objects)
                self.f.objects[key].append(copy.deepcopy(self.f.objects[key][0]))
                self.assertTrue(all(b.owner_pid is None for b in project(self.f).boxes if b.kind == "phone_object"))
                self.f.objects = original

    def test_missing_scores_stay_na_and_do_not_inherit_other_confidence(self):
        self.f.detections[1].pop("confidence"); self.f.objects["phone_detections"][0].pop("confidence")
        items = project(self.f).boxes
        self.assertIsNone(items[-1].score)
        self.assertIsNone(items[-2].score)

    def test_missing_sidecars_leave_person_snapshot_available(self):
        value = metadata.collect(self.f.snapshot, self.f.path, {}, [])
        self.assertFalse(value.boxes)
        self.assertEqual(len(value.issues), 2)

    def test_frame_size_hash_path_role_schema_mismatch_drop_only_source_layer(self):
        for changes in ({"frame_id": 6}, {"frame_id": True}, {"source_size": [640, 480]},
                        {"jpeg_sha256": "wrong"}, {"image": "different.jpg"}, {"role": "face"}, {"schema": "old"}):
            with self.subTest(changes=changes):
                reply = export(self.f); self.edit("hybrid", **changes)
                value = metadata.collect(self.f.snapshot, self.f.path, reply, self.f.tracks)
                self.assertEqual(len(value.boxes), 2)
                self.assertTrue(value.issues)

    def test_face_reply_and_sidecar_must_match(self):
        reply = export(self.f); reply["faces"][0]["score"] = .99
        value = metadata.collect(self.f.snapshot, self.f.path, reply, self.f.tracks)
        self.assertEqual(len(value.boxes), 3)
        self.assertTrue(value.issues)

    def test_replaced_jpeg_invalidates_sidecars(self):
        reply = export(self.f)
        cv2.imwrite(str(self.f.path), np.full_like(self.f.snapshot.image, 200))
        snapshot = replace(self.f.snapshot, image=cv2.imread(str(self.f.path)))
        value = metadata.collect(snapshot, self.f.path, reply, self.f.tracks)
        self.assertFalse(value.boxes)

    def test_same_size_wrong_snapshot_pixels_rejected(self):
        reply = export(self.f)
        with self.assertRaisesRegex(ValueError, "pixels"):
            metadata.collect(replace(self.f.snapshot, image=np.zeros_like(self.f.snapshot.image)), self.f.path, reply, self.f.tracks)

    def test_malformed_json_and_row_limit_are_safe(self):
        for bad in ("{", json.dumps({"schema": metadata.SCHEMA, "role": "hybrid"})):
            reply = export(self.f)
            metadata.sidecar_path(self.f.path, "hybrid").write_text(bad)
            self.assertEqual(len(metadata.collect(self.f.snapshot, self.f.path, reply, self.f.tracks).boxes), 2)
        with patch.object(metadata, "MAX_ROWS", 1):
            self.assertFalse(metadata.write_source(self.f.path, 7, (240, 320), "face", faces=self.f.faces))

    def test_invalid_bbox_is_omitted_and_reported(self):
        self.f.detections[0]["bbox_xyxy"] = [0, 0, 0, 10]
        value = project(self.f)
        self.assertEqual(len(value.boxes), 4)
        self.assertTrue(value.issues)

    def test_metadata_is_frozen_and_independent_of_mutable_worker_results(self):
        value = project(self.f)
        self.f.detections[0]["bbox_xyxy"][0] = 999
        self.assertEqual(value.boxes[2].bbox, (20, 40, 130, 150))
        with self.assertRaises(FrozenInstanceError):
            value.frame_id = 3

    def test_sidecar_excludes_biometrics_and_unrelated_phone_food_data(self):
        self.f.faces[0]["embedding"] = "NEVER_EXPORT"
        self.f.objects["food_detections"] = [{"secret": "NEVER_EXPORT"}]
        export(self.f)
        for role in ("face", "hybrid"):
            self.assertNotIn("NEVER_EXPORT", metadata.sidecar_path(self.f.path, role).read_text())


class TestDetailRendering(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.f = synthetic(self.temp.name)
        self.snapshot = replace(self.f.snapshot, render_metadata=project(self.f))
        self.enterContext(redirect_stderr(io.StringIO()))

    def paint(self, snapshot=None, *, max_size=(640, 480), now=12.):
        snapshot = self.snapshot if snapshot is None else snapshot
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            image = overlay.PersonSnapshotOverlay(stale_after=5., max_size=max_size).render(result(snapshot), now=now)
        return image, [c.args[1] for c in text.call_args_list]

    def test_legend_distinguishes_sources_and_similarity_from_probability(self):
        image, texts = self.paint()
        combined = "\n".join(texts)
        self.assertIn("Behavior YOLO 0.810", combined)
        self.assertIn("COCO score 0.270", combined)
        self.assertIn("similarity 0.720", combined)
        self.assertIn("not identity probability", combined)
        self.assertIn("Sleeping production detected box", combined)
        self.assertIn("Using Phone YOLO candidate", combined)
        self.assertIn("COCO phone OBJECT", combined)
        self.assertEqual(image.shape, (560, 1180, 3))
        self.assertNotIn("STABLE_MUST_NOT_LEAK", combined)

    def test_no_metadata_keeps_legacy_person_only_size(self):
        self.assertEqual(self.paint(self.f.snapshot)[0].shape, (560, 640, 3))

    def test_metadata_mismatch_preserves_person_layer_and_omits_details(self):
        for value in (replace(self.snapshot.render_metadata, frame_id=99),
                      replace(self.snapshot.render_metadata, captured_at=11.),
                      replace(self.snapshot.render_metadata, source_size=(10, 10)),
                      replace(self.snapshot.render_metadata, pixels_sha256="wrong"), object()):
            with self.subTest(value=value):
                image, texts = self.paint(replace(self.snapshot, render_metadata=value))
                self.assertEqual(image.shape, (560, 640, 3))
                self.assertIn("P1 | SV001", texts)
                self.assertFalse(any("COCO score" in t for t in texts))

    def test_stale_observation_age_labels_stay_visible(self):
        _, texts = self.paint(now=30.)
        self.assertIn("STALE: capture + completion", texts)

    def test_detail_geometry_resize_letterbox_clipping_and_invalid(self):
        for bbox in ((-20, -20, 319, 239), (319, 0, 400, 30), (0, 0, float("nan"), 40),
                     (0, 0, float("inf"), 40), (0, 0, 0, 10), (320, 0, 330, 20)):
            with self.subTest(bbox=bbox):
                item = replace(self.snapshot.render_metadata.boxes[0], bbox=bbox)
                snap = replace(self.snapshot, render_metadata=replace(self.snapshot.render_metadata, boxes=(item,)))
                with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as boxes:
                    self.paint(snap, max_size=(200, 200))
                outlines = [c for c in boxes.call_args_list if c.args[4] == 2 and c.args[3] == details.COLORS["face"]]
                if bbox[0] < 320 and all(np.isfinite(bbox)) and bbox[2] > bbox[0]:
                    self.assertEqual(len(outlines), 1)
                    x1, y1 = outlines[0].args[1]; x2, y2 = outlines[0].args[2]
                    roi = outlines[0].args[0]
                    ox, oy = (200 - roi.shape[1]) // 2, 80 + (200 - roi.shape[0]) // 2
                    x1, x2, y1, y2 = x1 + ox, x2 + ox, y1 + oy, y2 + oy
                    self.assertTrue(0 <= x1 <= x2 < 200 and 105 <= y1 <= y2 < 255)
                else:
                    self.assertFalse(outlines)

    def test_cached_projection_drawn_once_ages_change_only_header(self):
        display = overlay.PersonSnapshotOverlay(stale_after=5.)
        value = result(self.snapshot)
        with patch.object(overlay, "render_details", wraps=details.render_details) as render:
            a = display.render(value, now=12.)
            b = display.render(value, now=20.)
        render.assert_called_once()
        np.testing.assert_array_equal(a[80:], b[80:])

    def test_renderer_failure_drops_only_detail_layers(self):
        with patch.object(details, "label_lines", side_effect=RuntimeError("synthetic detail failure")):
            image, texts = self.paint()
        self.assertEqual(image.shape, (560, 640, 3))
        self.assertIn("P1 | SV001", texts)

    def test_missing_confidence_is_na(self):
        b = replace(self.snapshot.render_metadata.boxes[-1], score=None)
        self.assertIn("COCO score N/A", details.label_lines(b, "O1")[1])


class TestDetailPipelineIsolation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.enterContext(patch.dict(os.environ))
        for name in (overlay.OVERLAY_ENV, "HTTM_IDENTITY_SHADOW", "HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG"):
            os.environ.pop(name, None)
        os.environ["HTTM_IDENTITY_SHADOW_DIR"] = self.temp.name
        self.checker = shadow_tests.TestIdentityContinuityShadow()

    @staticmethod
    def with_source(original):
        def process(worker, path, fid):
            reply = original(worker, path, fid)
            if metadata.enabled():
                shape = cv2.imread(str(path)).shape
                if worker.kind == "face":
                    metadata.write_source(path, fid, shape, "face", faces=reply["faces"])
                else:
                    # Synthetic worker evidence. No inference or production bbox is fabricated.
                    ds = [{"class_name": "using_phone", "bbox_xyxy": p["person_bbox_xyxy_clipped"],
                           "confidence": .81, "person_ref": p["person_ref"], "association_status": "associated"}
                          for p in reply["people"]]
                    metadata.write_source(path, fid, shape, "hybrid", detections=ds, people=reply["people"])
            return reply
        return process

    def run_mode(self, observations, enabled, shadow):
        if enabled:
            os.environ[overlay.OVERLAY_ENV] = "1"
        else:
            os.environ.pop(overlay.OVERLAY_ENV, None)
        return self.checker.run_mode(observations, shadow)

    def test_real_ai_loop_overlay_shadow_matrix_preserves_all_production_arguments_and_panel(self):
        observations = [pipeline.observation(i, 100. + (i - 1) * 1.5, second_student="SV002", second_state="not_detected")
                        for i in range(1, 4)]
        for o in observations:
            o.people[1]["behaviors"]["sleeping"]["state"] = "detected"
        with patch.object(pipeline._ReplayWorker, "process_frame", self.with_source(pipeline._ReplayWorker.process_frame)):
            baseline = self.run_mode(observations, False, False)
            for enabled, shadow in ((False, True), (True, False), (True, True)):
                with self.subTest(overlay=enabled, shadow=shadow):
                    actual = self.run_mode(observations, enabled, shadow)
                    self.checker.assert_unchanged(baseline, actual)
                    if enabled:
                        self.assertTrue(actual[0].results[-1]["snapshot"].render_metadata.boxes)
        self.assertEqual(baseline[0].results[-1]["behavior_frame"].people[0].behaviors[1].estimated_seconds, 3.)

    def test_overlay_off_never_collects_visual_metadata(self):
        with patch.object(pipeline.live, "attach_to_result", side_effect=AssertionError("OFF")) as attach:
            trace = self.run_mode([pipeline.observation(1, 100.)], False, False)
        attach.assert_not_called()
        self.assertIsNone(trace[0].results[0]["snapshot"].render_metadata)

    def test_missing_metadata_or_export_failure_does_not_change_duration_and_warning(self):
        observations = [pipeline.observation(i, 100. + i, student="Unknown" if i == 2 else "SV001") for i in range(1, 4)]
        baseline = self.run_mode(observations, False, False)
        with patch.object(metadata, "collect", side_effect=OSError("synthetic metadata IO failure")):
            actual = self.run_mode(observations, True, True)
        self.checker.assert_unchanged(baseline, actual)
        self.assertIn("metadata omitted", actual[0].output)


if __name__ == "__main__":
    unittest.main()
