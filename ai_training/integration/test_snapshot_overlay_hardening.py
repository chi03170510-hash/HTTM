"""B5 synthetic frames only. Camera/AI production matrix is in B1/B4 tests."""

from contextlib import redirect_stderr
from dataclasses import replace
import io
import unittest
from unittest.mock import patch

import cv2
import numpy as np

import behavior_overlay as drawing
import person_snapshot_overlay as overlay
import snapshot_detail_overlay as detail
from bbox_visual_metadata import VisualBox
from benchmark_snapshot_overlay import synthetic_scene
from snapshot_label_layout import fit_marked, identity_text, overlaps, place_label, text_width


class TestLabelHardening(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))

    def render(self, value, viewport=(640, 480), now=12.):
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            image = overlay.PersonSnapshotOverlay(stale_after=5., max_size=viewport).render(value, now=now)
        return image, [c.args[1] for c in text.call_args_list]

    def test_one_to_twenty_person_plates_do_not_collide_even_with_overlapping_boxes(self):
        for count in (1, 2, 5, 10, 20):
            for overlap in (False, True):
                with self.subTest(count=count, overlap=overlap):
                    snap = synthetic_scene(count, overlap=overlap)["snapshot"]
                    base = drawing.render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., person_only=True)
                    self.assertEqual(len(base.label_regions), count)
                    for i, a in enumerate(base.label_regions):
                        self.assertTrue(0 <= a[0] < a[2] <= 640 and 80 <= a[1] < a[3] <= 560)
                        self.assertFalse(any(overlaps(a, b) for b in base.label_regions[i + 1:]))

    def test_small_crowded_view_reports_hidden_identity_and_never_overlaps(self):
        value = synthetic_scene(20, overlap=True)
        base = drawing.render_snapshot(value["snapshot"], expected_frame_id=7, expected_captured_at=10.,
                                       person_only=True, max_size=(80, 30))
        self.assertTrue(base.layout_notes)
        self.assertLess(len(base.label_regions), 20)
        for i, a in enumerate(base.label_regions):
            self.assertFalse(any(overlaps(a, b) for b in base.label_regions[i + 1:]))
        image, texts = self.render(value, (80, 30))
        self.assertEqual(image.shape, (110, 620, 3))
        self.assertTrue(any("IDs hidden: 20" in t for t in texts))

    def test_long_sid_uses_explicit_truncation_marker(self):
        _, texts = self.render(synthetic_scene(1, long_sid=True), (100, 100))
        self.assertTrue(any("[SID clipped]" in t for t in texts))
        self.assertTrue(all("clipped" in t or "..." in t for t in texts if "LONG_IDENTITY" in t))
        for width in (4, 10, 40, 100, 520):
            text = identity_text("SV001" + "X" * 500, "P1", width)
            self.assertLessEqual(text_width(text, .38), width)
            self.assertTrue(not text or text == "!" or "[SID clipped]" in text)

    def test_all_twenty_identities_precede_secondary_details(self):
        _, texts = self.render(synthetic_scene(20))
        primary = [t for t in texts if t.startswith(("SV", "Unknown |", "Ambiguous |"))]
        self.assertEqual(len(primary), 20)
        self.assertTrue(any(t.startswith("SV001 | P1") for t in primary))
        self.assertTrue(any("IDs hidden: 0" in t and "details hidden:" in t for t in texts))

    def test_behavior_states_precede_duration_and_confidence(self):
        _, texts = self.render(synthetic_scene(1))
        primary = next(t for t in texts if t.startswith("SV001 | P1"))
        self.assertLess(primary.index("Sleeping:detected"), primary.index("~3.0s"))
        self.assertLess(primary.index("Phone:suspected"), primary.index("~3.0s"))
        self.assertNotIn("0.930", primary)

    def test_long_text_fitting_has_marker_and_bounded_pixel_width(self):
        for width in (1, 8, 20, 100, 500):
            value = fit_marked("A" * 1000, width, .38)
            self.assertLessEqual(text_width(value, .38), width)
            self.assertTrue(not value or value == "!" or value.endswith("..."))

    def test_outside_plates_do_not_change_local_collision_placement(self):
        kwargs = ((10, 10), (20, 18), (0, 0, 100, 80))
        local = [(5, 5, 40, 30)]
        self.assertEqual(place_label(*kwargs, local),
                         place_label(*kwargs, local + [(200, 200, 300, 300)] * 100))

    def test_detail_tags_cannot_cover_identity_plates_or_each_other(self):
        snap = synthetic_scene(20, overlap=True)["snapshot"]
        base = drawing.render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., person_only=True)
        rendered = detail.render_details(snap, base, max_size=(640, 480))
        for i, a in enumerate(rendered.label_regions):
            self.assertFalse(any(overlaps(a, b) for b in rendered.label_regions[i + 1:]))
        for x1, y1, x2, y2 in base.label_regions:
            np.testing.assert_array_equal(rendered.image[y1:y2, x1:x2], base.image[y1:y2, x1:x2])

    def test_duplicate_production_sid_is_ambiguous_without_old_metadata_leak(self):
        value = synthetic_scene(2)
        snap = value["snapshot"]
        frame = replace(snap.observations, people=tuple(replace(p, student_id="DUPLICATE_SID") for p in snap.observations.people))
        value.update(behavior_frame=frame, snapshot=replace(snap, observations=frame))
        _, texts = self.render(value)
        combined = "\n".join(texts)
        self.assertIn("Ambiguous", combined)
        self.assertFalse(any(s in combined for s in ("DUPLICATE_SID", "SV001", "SV002")))

    def test_missing_metadata_long_identity_still_has_marked_legend(self):
        value = synthetic_scene(5, long_sid=True)
        value["snapshot"] = replace(value["snapshot"], render_metadata=None)
        _, texts = self.render(value, (320, 240))
        self.assertTrue(any("[SID clipped]" in t for t in texts))
        self.assertFalse(any("Behavior YOLO" in t for t in texts))

    def test_text_stays_inside_canvas_at_small_and_normal_resolutions(self):
        for viewport in ((100, 100), (320, 240), (640, 480)):
            with patch.object(cv2, "putText", wraps=cv2.putText) as calls:
                overlay.PersonSnapshotOverlay(stale_after=5., max_size=viewport).render(synthetic_scene(20, long_sid=True), now=30.)
            for call in calls.call_args_list:
                image, text, (x, y), font, scale = call.args[:5]
                (w, h), baseline = cv2.getTextSize(text, font, scale, 1)
                if text:
                    self.assertTrue(0 <= x and x + w <= image.shape[1], (text, viewport))
                    self.assertTrue(h <= y and y + baseline < image.shape[0], (text, viewport))


class TestCoordinateHardening(unittest.TestCase):
    def setUp(self):
        self.enterContext(redirect_stderr(io.StringIO()))

    def test_all_detail_strokes_and_tags_stay_inside_image_roi(self):
        for viewport in ((200, 200), (400, 100), (100, 400)):
            snap = synthetic_scene(1, source_size=(100, 50))["snapshot"]
            boxes = tuple(VisualBox(kind, (-100., -100., 200., 100.), .5) for kind in detail.COLORS)
            snap = replace(snap, observations=replace(snap.observations, people=()),
                           render_metadata=replace(snap.render_metadata, boxes=boxes))
            base = drawing.render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., person_only=True, max_size=viewport)
            result = detail.render_details(snap, base, max_size=viewport)
            scale = min(viewport[0] / 100, viewport[1] / 50)
            w, h = round(100 * scale), round(50 * scale)
            x, y = (viewport[0] - w) // 2, 80 + (viewport[1] - h) // 2
            difference = np.any(result.image[:, :viewport[0]] != base.image, axis=2)
            difference[y:y + h, x:x + w] = False
            self.assertFalse(difference.any())

    def test_invalid_bbox_for_every_detail_kind_is_omitted(self):
        for kind in detail.COLORS:
            for bbox in ((0, 0, float("nan"), 4), (0, 0, float("inf"), 4), (0, 0, 0, 5),
                         (-100, -100, -1, -1), (1280, 0, 1400, 30), (10, 0, 5, 30), None):
                with self.subTest(kind=kind, bbox=bbox):
                    snap = synthetic_scene(1)["snapshot"]
                    snap = replace(snap, render_metadata=replace(snap.render_metadata, boxes=(VisualBox(kind, bbox, .5),)))
                    base = drawing.render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., person_only=True)
                    self.assertIs(detail.render_details(snap, base, max_size=(640, 480)), base)

    def test_person_thick_stroke_does_not_bleed_into_letterbox(self):
        snap = synthetic_scene(1, source_size=(100, 50))["snapshot"]
        p = replace(snap.observations.people[0], bbox=(0, 0, 99, 49))
        snap = replace(snap, observations=replace(snap.observations, people=(p,)))
        with patch.object(drawing, "place_label", return_value=None):
            result = drawing.render_snapshot(snap, expected_frame_id=7, expected_captured_at=10., person_only=True, max_size=(200, 200))
        for y in (129, 230):
            np.testing.assert_array_equal(result.image[y], np.full_like(result.image[y], 25))


class TestCacheHardening(unittest.TestCase):
    def setUp(self):
        self.display = overlay.PersonSnapshotOverlay(stale_after=5.)
        self.enterContext(redirect_stderr(io.StringIO()))

    def test_clock_and_completion_input_errors_recover_on_same_observation(self):
        for change in ({"now": 9.}, {"completed_at": 99.}):
            value = synthetic_scene(1)
            display = overlay.PersonSnapshotOverlay(stale_after=5.)
            value["completed_at"] = change.get("completed_at", 11.)
            display.render(value, now=change.get("now", 12.))
            value["completed_at"] = 11.
            with patch.object(cv2, "putText", wraps=cv2.putText) as text:
                display.render(value, now=12.)
            self.assertTrue(any("SV001" in c.args[1] for c in text.call_args_list))

    def test_new_metadata_invalidates_cached_geometry(self):
        value = synthetic_scene(1)
        with patch.object(overlay, "render_snapshot", wraps=drawing.render_snapshot) as render:
            self.display.render(value, now=12.)
            old = value["snapshot"]
            value["snapshot"] = replace(old, render_metadata=replace(old.render_metadata, boxes=()))
            self.display.render(value, now=13.)
        self.assertEqual(render.call_count, 2)

    def test_new_frame_never_reuses_historical_pixels(self):
        value = synthetic_scene(1)
        self.display.render(value, now=12.)
        old = value["snapshot"]; frame = replace(old.observations, frame_id=8)
        value.update(frame_id=8, behavior_frame=frame, snapshot=replace(old, frame_id=8, observations=frame,
                     render_metadata=None, image=np.full_like(old.image, 99)))
        with patch.object(overlay, "render_snapshot", wraps=drawing.render_snapshot) as render:
            image = self.display.render(value, now=13.)
        render.assert_called_once()
        self.assertEqual(image[400, 500, 0], 99)

    def test_frame_mismatch_omits_details_instead_of_using_cache(self):
        value = synthetic_scene(1)
        self.display.render(value, now=12.)
        snap = value["snapshot"]
        value["snapshot"] = replace(snap, render_metadata=replace(snap.render_metadata, frame_id=99))
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            self.display.render(value, now=13.)
        self.assertFalse(any("COCO score" in c.args[1] for c in text.call_args_list))

    def test_stale_ages_and_future_completion_do_not_infer_clock_offsets(self):
        value = synthetic_scene(1)
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            self.display.render(value, now=30.)
        labels = [c.args[1] for c in text.call_args_list]
        self.assertIn("Capture age: 20.0s | Completion age: 19.0s", labels)
        self.assertIn("STALE: capture + completion", labels)
        value["completed_at"] = 31.
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            self.display.render(value, now=30.)
        self.assertTrue(any("Render unavailable" in c.args[1] for c in text.call_args_list))

    def test_stale_visible_at_small_viewport(self):
        with patch.object(cv2, "putText", wraps=cv2.putText) as text:
            overlay.PersonSnapshotOverlay(stale_after=5., max_size=(100, 100)).render(synthetic_scene(1), now=30.)
        self.assertIn("STALE", [c.args[1] for c in text.call_args_list])

    def test_renderer_exception_is_bounded_and_new_observation_recovers(self):
        value = synthetic_scene(1)
        with patch.object(overlay, "render_snapshot", side_effect=RuntimeError("synthetic failure")) as render:
            self.display.render(value, now=12.); self.display.render(value, now=13.)
        render.assert_called_once()
        self.assertIsNotNone(self.display._source_snapshot)
        with patch.object(overlay, "render_snapshot", wraps=drawing.render_snapshot) as render:
            self.display.render(synthetic_scene(2), now=14.)
        render.assert_called_once()

    def test_invalid_detail_renderer_size_cannot_poison_cache(self):
        wrong = drawing.RenderedSnapshot(7, 10., np.zeros((2, 2, 3), dtype=np.uint8))
        with patch.object(overlay, "render_details", return_value=wrong):
            image = self.display.render(synthetic_scene(1), now=12.)
        self.assertEqual(image.shape, (560, 640, 3)); self.assertIsNone(self.display._cached)

    def test_malformed_result_does_not_escape_renderer(self):
        self.assertEqual(self.display.render([], now=12.).shape, (560, 640, 3))


if __name__ == "__main__":
    unittest.main()
