"""B1+B2: source-coordinate rendering and real Camera Live with mocked devices."""

import ast
import copy
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
from unittest.mock import Mock, patch

import cv2
import numpy as np

import behavior_overlay as drawing
import camera_coordinator_live as live
import person_snapshot_overlay as overlay
import test_identity_continuity_shadow as shadow_tests
from behavior_duration import BehaviorDuration, BehaviorFrame, PersonBehaviorView
from behavior_warning import DEFAULT_WARNING_CONFIG
from identity_continuity import IdentityContinuityResolver
from test_ch20_duration_pipeline import observation


def person(pid="P1", sid="SV001", bbox=(10., 10., 80., 40.), identity="matched", association="matched"):
    return PersonBehaviorView(pid, "synthetic_" + pid, bbox, sid, 99, association, identity,
                              (BehaviorDuration("using_phone", "detected", 3.),))


def result(*, fid=1, captured=10., completed=11., people=None, size=(100, 50), color=70):
    people = (person(),) if people is None else tuple(people)
    frame = BehaviorFrame(fid, captured, people)
    image = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    return {"frame_id": fid, "captured_at": captured, "completed_at": completed, "behavior_frame": frame,
            "snapshot": drawing.AISnapshot(fid, captured, frame, image)}


class TestPersonSnapshotProjection(unittest.TestCase):
    def draw(self, value, max_size=(100, 50)):
        with (patch.object(drawing.cv2, "rectangle", wraps=cv2.rectangle) as boxes,
              patch.object(drawing.cv2, "putText", wraps=cv2.putText) as text):
            rendered = drawing.render_snapshot(value["snapshot"], expected_frame_id=value["frame_id"],
                expected_captured_at=value["captured_at"], max_size=max_size, person_only=True)
        # B5 draws in an image ROI to clip stroke thickness too. Verify final
        # display coordinates independently of the drawing target's origin.
        outlines = []
        for c in boxes.call_args_list:
            if c.args[4] != 2:
                continue
            roi = c.args[0]
            ox = (max_size[0] - roi.shape[1]) // 2
            oy = drawing.SNAPSHOT_HEADER + (max_size[1] - roi.shape[0]) // 2
            outlines.append(tuple((x + ox, y + oy) for x, y in (c.args[1], c.args[2])))
        return rendered, outlines, [c.args[1] for c in text.call_args_list]

    def test_actual_hybrid_clip_endpoint_reaches_last_display_pixel(self):
        path = Path(live.ROOT) / "ai_training/src/hybrid_v3.py"
        node = next(n for n in ast.parse(path.read_text(encoding="utf-8-sig")).body
                    if isinstance(n, ast.FunctionDef) and n.name == "clip_box")
        scope = {"math": math}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        bbox = scope["clip_box"]([0, 0, 100, 50], 100, 50)
        original = copy.deepcopy(bbox)
        _, boxes, _ = self.draw(result(people=[person(bbox=bbox)]))
        self.assertEqual(boxes, [((0, 80), (99, 129))])
        self.assertEqual(bbox, original, "Rendering cannot alter detector geometry")

    def test_negative_partial_and_fractional_boxes_clip_to_enclosing_pixels(self):
        for bbox in ((-10., -20., 99., 49.), (.2, .1, 99.8, 49.9), (-100., -100., 200., 200.)):
            with self.subTest(bbox=bbox):
                _, boxes, _ = self.draw(result(people=[person(bbox=bbox)]))
                self.assertEqual(boxes, [((0, 80), (99, 129))])

    def test_nan_inf_zero_area_reversed_outside_and_malformed_boxes_are_omitted(self):
        invalid = ((0, 0, float("nan"), 10), (0, 0, float("inf"), 10),
                   (0, -float("inf"), 20, 10), (10, 0, 10, 20), (20, 0, 10, 20),
                   (100, 0, 120, 20), (-20, 0, -1, 20), (0, 50, 10, 60), (0, 0, 1), None)
        for bbox in invalid:
            with self.subTest(bbox=bbox):
                _, boxes, _ = self.draw(result(people=[person(bbox=bbox)]))
                self.assertEqual(boxes, [])

    def test_exact_last_source_column_can_render_a_single_visible_column(self):
        _, boxes, _ = self.draw(result(people=[person(bbox=(99., 0., 110., 20.))]))
        self.assertEqual(boxes, [((99, 80), (99, 100))])

    def test_upscale_downscale_and_letterbox_use_same_image_transform(self):
        for source, viewport, bbox, expected in (
            ((100, 50), (200, 100), (10, 10, 90, 40), ((20, 100), (181, 161))),
            ((200, 100), (100, 50), (20, 20, 180, 80), ((10, 90), (90, 120))),
            ((100, 100), (200, 100), (0, 0, 99, 99), ((50, 80), (149, 179))),
            ((50, 100), (200, 100), (0, 0, 49, 99), ((75, 80), (124, 179))),
        ):
            with self.subTest(source=source, viewport=viewport):
                _, boxes, _ = self.draw(result(people=[person(bbox=bbox)], size=source), viewport)
                self.assertEqual(boxes, [expected])

    def test_source_resolution_changes_have_independent_transform(self):
        a = result(people=[person(bbox=(0, 0, 99, 49))])
        b = result(fid=2, captured=12., completed=13., people=[person(bbox=(0, 0, 49, 99))], size=(50, 100))
        self.assertEqual(self.draw(a, (200, 100))[1], [((0, 80), (199, 179))])
        self.assertEqual(self.draw(b, (200, 100))[1], [((75, 80), (124, 179))])

    def test_legacy_edge_renderer_keeps_original_coordinate_contract(self):
        value = result(people=[person(bbox=(10, 10, 90, 40))])
        with patch.object(drawing.cv2, "rectangle", wraps=cv2.rectangle) as boxes:
            drawing.render_snapshot(value["snapshot"], expected_frame_id=1, expected_captured_at=10., max_size=(200, 100))
        self.assertEqual([(c.args[1], c.args[2]) for c in boxes.call_args_list if c.args[4] == 2],
                         [((20, 100), (179, 159))])

    def test_accepted_production_sid_and_pid_are_the_only_person_label(self):
        _, boxes, labels = self.draw(result(), (640, 480))
        self.assertIn("P1 | SV001", labels)
        self.assertEqual(len(boxes), 1)
        self.assertFalse(any("Using Phone" in s or "Sleeping" in s or "0.8" in s or "T99" in s for s in labels))

    def test_unknown_or_unaccepted_sid_is_not_promoted(self):
        for sid, identity, association in ((None, "unknown", "matched"), ("Unknown", "matched", "matched"),
                                           ("RAW_LEAK", "unassociated", "unmatched")):
            with self.subTest(identity=identity):
                _, _, labels = self.draw(result(people=[person(sid=sid, identity=identity, association=association)]), (640, 480))
                self.assertIn("P1 | Unknown", labels)
                self.assertFalse(any("RAW_LEAK" in s for s in labels))

    def test_ambiguous_association_or_duplicate_student_is_explicitly_ambiguous(self):
        for people in ([person(sid="RAW_LEAK", identity="ambiguous_association")],
                       [person(), person("P2", "SV001", bbox=(20, 20, 70, 45))]):
            with self.subTest(count=len(people)):
                _, boxes, labels = self.draw(result(people=people), (640, 480))
                self.assertEqual(len(boxes), len(people))
                self.assertIn("P1 | Ambiguous", labels)
                self.assertFalse(any("RAW_LEAK" in s or "SV001" in s for s in labels))

    def test_duplicate_pid_cannot_claim_an_accepted_sid(self):
        _, _, labels = self.draw(result(people=[person(), person(sid="SV002")]), (640, 480))
        self.assertEqual(labels.count("P1 | Ambiguous"), 2)

    def test_two_people_reorder_preserves_each_bbox_and_identity(self):
        people = [person(), person("P2", "SV002", (30, 20, 90, 45))]
        _, a, labels_a = self.draw(result(people=people), (640, 480))
        _, b, labels_b = self.draw(result(people=reversed(people)), (640, 480))
        self.assertEqual(a, list(reversed(b)))
        self.assertTrue({"P1 | SV001", "P2 | SV002"} <= set(labels_a) & set(labels_b))

    def test_pixels_and_production_views_are_not_modified(self):
        value = result()
        image, observation = value["snapshot"].image.copy(), value["behavior_frame"]
        self.draw(value)
        self.assertIs(value["behavior_frame"], observation)
        np.testing.assert_array_equal(value["snapshot"].image, image)
        self.assertFalse(value["snapshot"].image.flags.writeable)


class TestPersonSnapshotController(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(overlay.OVERLAY_ENV, None)
        self.display = overlay.PersonSnapshotOverlay(stale_after=live.PERSON_TRACK_MAX_AGE)

    def texts(self, value, now):
        with patch.object(overlay.cv2, "putText", wraps=cv2.putText) as text:
            image = self.display.render(value, now=now)
        return image, [c.args[1] for c in text.call_args_list]

    def test_off_never_constructs_renderer_for_unset_or_nonoptin_values(self):
        with patch.object(overlay, "PersonSnapshotOverlay", side_effect=AssertionError("OFF")):
            for value in (None, "0", "true", "", "yes"):
                if value is None:
                    os.environ.pop(overlay.OVERLAY_ENV, None)
                else:
                    os.environ[overlay.OVERLAY_ENV] = value
                self.assertIsNone(overlay.overlay_from_environment(stale_after=5.))

    def test_exact_one_enables_only_display_object_and_logs_it(self):
        os.environ[overlay.OVERLAY_ENV] = "1"
        with (patch.object(cv2, "namedWindow", side_effect=AssertionError("not yet")), redirect_stdout(io.StringIO()) as log):
            display = overlay.overlay_from_environment(stale_after=5.)
        self.assertIsInstance(display, overlay.PersonSnapshotOverlay)
        self.assertIn("separate historical window", log.getvalue())

    def test_no_ai_data_shows_waiting_without_any_boxes(self):
        with patch.object(cv2, "rectangle", wraps=cv2.rectangle) as boxes:
            _, labels = self.texts(None, 12.)
        boxes.assert_not_called()
        self.assertIn("Waiting for first completed AI observation", labels)

    def test_snapshot_pixels_are_used_instead_of_live_or_temp_image_path(self):
        value = result(size=(640, 480), color=37)
        value["image"] = "a_later_unrelated_frame.jpg"
        with patch.object(cv2, "imread", side_effect=AssertionError("no path reload")):
            image = self.display.render(value, now=12.)
        np.testing.assert_array_equal(image[80 + 450, 600], value["snapshot"].image[450, 600])
        self.assertEqual(image[80 + 450, 600, 0], 37)

    def test_frame_capture_and_behavior_mismatch_fail_closed(self):
        for key, bad in (("frame_id", 2), ("captured_at", 9.), ("behavior_frame", BehaviorFrame(1, 10., ()))):
            with self.subTest(key=key), redirect_stderr(io.StringIO()):
                self.display = overlay.PersonSnapshotOverlay(stale_after=5.)
                value = result(); value[key] = bad
                with patch.object(overlay, "render_snapshot", side_effect=AssertionError("must not render")):
                    _, labels = self.texts(value, 12.)
                self.assertIn("Render unavailable; live preview continues", labels)

    def test_capture_and_completion_ages_and_staleness_are_distinct(self):
        value = result(captured=10., completed=20.)
        _, labels = self.texts(value, 21.)
        self.assertIn("Capture age: 11.0s | Completion age: 1.0s", labels)
        self.assertIn("STALE: capture", labels)
        _, labels = self.texts(value, 26.)
        self.assertIn("STALE: capture + completion", labels)

    def test_completion_age_is_unavailable_if_not_provided(self):
        value = result(); value.pop("completed_at")
        self.assertIn("Capture age: 2.0s | Completion age: --", self.texts(value, 12.)[1])

    def test_invalid_now_or_completion_never_draws_a_cached_person(self):
        for now, completed in ((9., 11.), (12., 13.), (12., float("inf")), (float("nan"), 11.)):
            with self.subTest(now=now, completed=completed), redirect_stderr(io.StringIO()):
                self.display = overlay.PersonSnapshotOverlay(stale_after=5.)
                _, labels = self.texts(result(completed=completed), now)
                self.assertIn("Render unavailable; live preview continues", labels)

    def test_geometry_renders_once_while_ages_refresh_without_duration_updates(self):
        value = result(); original = value["behavior_frame"]
        with patch.object(overlay, "render_snapshot", wraps=overlay.render_snapshot) as render:
            first = self.display.render(value, now=12.)
            second = self.display.render(value, now=13.)
        render.assert_called_once()
        self.assertIs(value["behavior_frame"], original)
        np.testing.assert_array_equal(first[80:], second[80:])
        self.assertFalse(np.array_equal(first[:80], second[:80]))

    def test_new_jpeg_or_resolution_with_same_ids_invalidates_cache(self):
        with patch.object(overlay, "render_snapshot", wraps=overlay.render_snapshot) as render:
            a = result(color=10)
            b = result(color=90, size=(200, 100))
            self.display.render(a, now=12.)
            self.display.render(b, now=12.)
        self.assertEqual(render.call_count, 2)

    def test_renderer_failure_is_bounded_and_recovers_on_new_result(self):
        value = result()
        with (patch.object(overlay, "render_snapshot", side_effect=RuntimeError("synthetic renderer failure")) as render,
              redirect_stderr(io.StringIO()) as errors):
            self.display.render(value, now=12.)
            self.display.render(value, now=13.)
        render.assert_called_once()
        self.assertEqual(errors.getvalue().count("synthetic renderer failure"), 1)
        self.assertIn("P1 | SV001", self.texts(result(fid=2, captured=12., completed=13.), 14.)[1])

    def test_renderer_metadata_mismatch_never_leaks_old_cached_boxes(self):
        self.display.render(result(), now=12.)
        wrong = drawing.RenderedSnapshot(999, 12., np.zeros((560, 640, 3), dtype=np.uint8))
        with patch.object(overlay, "render_snapshot", return_value=wrong), redirect_stderr(io.StringIO()):
            _, labels = self.texts(result(fid=2, captured=12., completed=13.), 14.)
        self.assertIn("Render unavailable; live preview continues", labels)
        self.assertFalse(any("SV001" in label for label in labels))

    def test_gui_failure_is_caught_and_not_retried_on_every_live_frame(self):
        with patch.object(cv2, "namedWindow", side_effect=RuntimeError("synthetic GUI failure")) as window, redirect_stderr(io.StringIO()):
            self.assertFalse(self.display.show(None, now=12.))
            self.assertFalse(self.display.show(None, now=13.))
        window.assert_called_once()

    def test_raw_association_and_shadow_proposal_cannot_supply_display_identity(self):
        value = result(people=[person(sid=None, identity="unassociated", association="unmatched")])
        value.update(matches=[{"student_id": "RAW_LEAK"}], shadow_proposed_student_id="SHADOW_LEAK",
                     tracked_faces=[(99, (0, 0, 10, 10), "FACE_LEAK", .99)])
        _, labels = self.texts(value, 12.)
        self.assertIn("P1 | Unknown", labels)
        self.assertFalse(any("LEAK" in label for label in labels))


class TestPersonOverlayCameraIntegration(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        for name in (overlay.OVERLAY_ENV, "HTTM_IDENTITY_SHADOW", "HTTM_PERSON_DEBUG", "HTTM_FACE_DEBUG"):
            os.environ.pop(name, None)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        os.environ["HTTM_IDENTITY_SHADOW_DIR"] = self.temp.name
        self.checker = shadow_tests.TestIdentityContinuityShadow()

    def run_mode(self, observations, *, enabled, shadow):
        if enabled:
            os.environ[overlay.OVERLAY_ENV] = "1"
        else:
            os.environ.pop(overlay.OVERLAY_ENV, None)
        return self.checker.run_mode(observations, shadow)

    def test_overlay_off_retains_old_panel_geometry_and_does_not_construct_controller(self):
        with patch.object(overlay, "PersonSnapshotOverlay", side_effect=AssertionError("OFF")):
            trace = self.run_mode([observation(1, 100.)], enabled=False, shadow=False)
        self.assertTrue(all(sample.image_shape == (720, 1280, 3) for sample in trace[0].samples))

    def test_overlay_and_shadow_matrix_preserves_production_inputs_results_and_panel(self):
        observations = [observation(i, 100. + 1.5 * (i - 1), second_student="SV002", second_state="not_detected")
                        for i in range(1, 4)]
        for item in observations:
            item.people[1]["behaviors"]["sleeping"]["state"] = "detected"
        original = copy.deepcopy(observations)
        baseline = self.run_mode(observations, enabled=False, shadow=False)
        for enabled, shadow in ((True, False), (False, True), (True, True)):
            with self.subTest(overlay=enabled, shadow=shadow):
                actual = self.run_mode(observations, enabled=enabled, shadow=shadow)
                self.checker.assert_unchanged(baseline, actual)
        self.assertEqual(observations, original)
        self.assertFalse(live.WARNING_CONFIG.enabled_behaviors)

    def test_duplicate_unknown_and_lost_association_keep_duration_and_identity_rules(self):
        for kwargs in ({"second_student": "SV001"}, {"student": "Unknown"}, {"associated": False}):
            with self.subTest(kwargs=kwargs):
                observations = [observation(1, 100.), observation(2, 101.5, **kwargs), observation(3, 103.)]
                baseline = self.run_mode(observations, enabled=False, shadow=False)
                actual = self.run_mode(observations, enabled=True, shadow=True)
                self.checker.assert_unchanged(baseline, actual)
                self.assertIsNone(actual[0].results[1]["behavior_frame"].people[0].student_id)

    def test_correct_worker_jpeg_is_the_only_snapshot_render_input(self):
        seen = []
        original = overlay.render_snapshot
        def render(snapshot, **kwargs):
            seen.append(snapshot)
            self.assertTrue(kwargs["person_only"])
            return original(snapshot, **kwargs)
        with patch.object(overlay, "render_snapshot", side_effect=render):
            trace = self.run_mode([observation(i, 100. + i) for i in range(1, 4)], enabled=True, shadow=False)
        self.assertEqual(len(seen), 3, "Repeated UI paints must reuse cached geometry")
        for snapshot, completed in zip(seen, trace[0].results):
            self.assertIs(snapshot, completed["snapshot"])
            self.assertEqual(snapshot.frame_id, completed["frame_id"])
            self.assertEqual(snapshot.captured_at, completed["captured_at"])
            self.assertFalse(snapshot.image.flags.writeable)

    def test_renderer_exception_never_stops_camera_pipeline(self):
        observations = [observation(1, 100.), observation(2, 101.5)]
        baseline = self.run_mode(observations, enabled=False, shadow=False)
        with patch.object(overlay, "render_snapshot", side_effect=RuntimeError("synthetic render error")):
            actual = self.run_mode(observations, enabled=True, shadow=False)
        self.checker.assert_unchanged(baseline, actual)
        self.assertFalse(actual[0].errors)
        self.assertIn("synthetic render error", actual[0].output)

    def test_unexpected_optional_show_exception_is_contained_in_main(self):
        baseline = self.run_mode([observation(1, 100.)], enabled=False, shadow=False)
        with patch.object(overlay.PersonSnapshotOverlay, "show", side_effect=RuntimeError("optional display boundary")):
            actual = self.run_mode([observation(1, 100.)], enabled=True, shadow=False)
        self.checker.assert_unchanged(baseline, actual)
        self.assertIn("disabled after display error", actual[0].output)

    def test_forced_wrong_shadow_sid_cannot_be_rendered_as_production(self):
        observations = [observation(1, 100.)]
        baseline = self.run_mode(observations, enabled=False, shadow=False)
        update = IdentityContinuityResolver.update
        def forced(resolver, *args, **kwargs):
            frame = update(resolver, *args, **kwargs)
            return replace(frame, observations=tuple(replace(row, proposed_student_id="SHADOW_LEAK", status="track_retained")
                                                     for row in frame.observations))
        with (patch.object(IdentityContinuityResolver, "update", forced),
              patch.object(cv2, "putText", wraps=cv2.putText) as text):
            actual = self.run_mode(observations, enabled=True, shadow=True)
        self.checker.assert_unchanged(baseline, actual)
        self.assertFalse(any("SHADOW_LEAK" in call.args[1] for call in text.call_args_list))

    def test_enabled_waiting_window_is_separate_from_live_preview(self):
        os.environ[overlay.OVERLAY_ENV] = "1"
        cap = Mock(); cap.isOpened.return_value = True
        cap.read.return_value = (True, np.full((480, 640, 3), 90, dtype=np.uint8))
        with (patch.object(cv2, "VideoCapture", return_value=cap), patch.object(live.Path, "is_file", return_value=True),
              patch.object(live.threading, "Thread") as thread, patch.object(cv2, "namedWindow") as window,
              patch.object(cv2, "resizeWindow"), patch.object(cv2, "imshow") as show,
              patch.object(cv2, "waitKey", return_value=ord("q")), patch.object(cv2, "destroyAllWindows"),
              redirect_stdout(io.StringIO())):
            thread.return_value.is_alive.return_value = False
            live.main()
        names = [call.args[0] for call in window.call_args_list]
        self.assertEqual(names, [live.WINDOW_NAME, overlay.WINDOW_NAME])
        self.assertEqual([call.args[0] for call in show.call_args_list], names)
        self.assertEqual(show.call_args_list[0].args[1].shape, (720, 1280, 3))
        self.assertEqual(show.call_args_list[1].args[1].shape, (560, 640, 3))
        cap.release.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
