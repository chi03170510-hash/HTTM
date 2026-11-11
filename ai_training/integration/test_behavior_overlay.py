"""CH-20 snapshot geometry/metadata tests using synthetic BGR images only."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import cv2
import numpy as np

from behavior_duration import BehaviorDurationTracker
from behavior_overlay import AISnapshot, SNAPSHOT_HEADER, person_label_lines, render_snapshot


def snapshot_fixture(bbox=(0, 0, 100, 50), image=None, frame_id=1, captured_at=10.0):
    if image is None:
        image = np.full((50, 100, 3), 70, dtype=np.uint8)
    observations = BehaviorDurationTracker().update(
        frame_id, captured_at,
        [{"person_ref": "person_0", "behaviors": {"using_phone": {"state": "detected"}}}],
        [{"person_ref": "person_0", "track_id": "P1", "bbox": bbox}],
        [{"person_ref": "person_0", "person_track_id": "P1", "student_id": "SYNTHETIC_001",
          "track_id": 1, "association_status": "matched"}],
    )
    return AISnapshot(frame_id, captured_at, observations, image)


class TestBehaviorOverlay(unittest.TestCase):
    def boxes(self, snapshot, max_size):
        with patch("behavior_overlay.cv2.rectangle", wraps=cv2.rectangle) as rectangle:
            rendered = render_snapshot(snapshot, expected_frame_id=snapshot.frame_id,
                                       expected_captured_at=snapshot.captured_at, max_size=max_size)
        outlines = [(call.args[1], call.args[2]) for call in rectangle.call_args_list if call.args[4] == 2]
        return rendered, outlines

    def test_edge_and_outside_boxes_are_clipped_to_image_pixels(self):
        for box in ((0, 0, 100, 50), (-20, -30, 120, 80), (0.2, .1, 100, 50)):
            with self.subTest(box=box):
                _, outlines = self.boxes(snapshot_fixture(box), (100, 50))
                self.assertEqual(outlines, [((0, SNAPSHOT_HEADER), (99, SNAPSHOT_HEADER + 49))])

    def test_downscale_changes_pixels_and_person_box_together(self):
        image = np.full((100, 200, 3), (40, 80, 120), dtype=np.uint8)
        rendered, outlines = self.boxes(snapshot_fixture((20, 20, 180, 80), image), (100, 50))
        self.assertEqual(rendered.image.shape, (SNAPSHOT_HEADER + 50, 100, 3))
        self.assertEqual(outlines, [((10, SNAPSHOT_HEADER + 10), (89, SNAPSHOT_HEADER + 39))])
        snapshot = snapshot_fixture((20, 20, 180, 80), image)
        empty = replace(snapshot, observations=replace(snapshot.observations, people=()))
        background, _ = self.boxes(empty, (100, 50))
        np.testing.assert_array_equal(background.image[SNAPSHOT_HEADER + 1, 1], image[2, 2])

    def test_letterbox_offset_is_applied_to_boxes(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        _, outlines = self.boxes(snapshot_fixture((0, 0, 100, 100), image), (200, 100))
        self.assertEqual(outlines, [((50, SNAPSHOT_HEADER), (149, SNAPSHOT_HEADER + 99))])

    def test_upscale_maps_box_to_resized_snapshot(self):
        _, outlines = self.boxes(snapshot_fixture((10, 10, 90, 40)), (200, 100))
        self.assertEqual(outlines, [((20, SNAPSHOT_HEADER + 20), (179, SNAPSHOT_HEADER + 79))])

    def test_invalid_or_fully_outside_boxes_are_not_drawn(self):
        for box in ((120, 0, 150, 20), (10, 10, 10, 20), (0, 0, float("nan"), 20)):
            with self.subTest(box=box):
                _, outlines = self.boxes(snapshot_fixture(box), (100, 50))
                self.assertEqual(outlines, [])

    def test_snapshot_owns_read_only_pixels_and_survives_source_reuse(self):
        image = np.full((50, 100, 3), 70, dtype=np.uint8)
        snapshot = snapshot_fixture(image=image)
        image[:] = 200
        np.testing.assert_array_equal(snapshot.image, np.full_like(image, 70))
        self.assertFalse(snapshot.image.flags.writeable)
        with self.assertRaises(ValueError):
            snapshot.image[0, 0] = 0
        original = snapshot.image.copy()
        self.boxes(snapshot, (100, 50))
        np.testing.assert_array_equal(snapshot.image, original)

    def test_frame_id_or_capture_time_mismatch_is_rejected(self):
        snapshot = snapshot_fixture()
        for frame_id, capture in ((2, 10.0), (1, 11.0)):
            with self.subTest(frame_id=frame_id, captured_at=capture):
                with self.assertRaises(ValueError):
                    AISnapshot(frame_id, capture, snapshot.observations, snapshot.image)
                with self.assertRaises(ValueError):
                    render_snapshot(snapshot, expected_frame_id=frame_id, expected_captured_at=capture)

    def test_rendered_snapshot_metadata_is_preserved_and_cache_is_checked(self):
        snapshot = snapshot_fixture(frame_id=8, captured_at=245676.420614)
        rendered, _ = self.boxes(snapshot, (100, 50))
        self.assertEqual(rendered.frame_id, 8)
        self.assertEqual(rendered.captured_at, 245676.420614)
        rendered.validate(8, 245676.420614)
        with self.assertRaises(ValueError):
            rendered.validate(9, 245676.420614)

    def test_labels_use_person_box_and_estimates_without_warning(self):
        snapshot = snapshot_fixture()
        lines = person_label_lines(snapshot.observations.people[0])
        self.assertIn("SYNTHETIC_001 | P1", lines)
        self.assertIn("Using Phone: detected ~0.0s", lines)
        self.assertIn("Sleeping: unknown --", lines)
        self.assertFalse(any("WARNING" in text for text in lines))
        person = replace(snapshot.observations.people[0], student_id=None)
        self.assertEqual(person_label_lines(person)[0], "Unknown | P1")

    def test_empty_snapshot_and_invalid_dimensions(self):
        snapshot = snapshot_fixture()
        empty = replace(snapshot, observations=replace(snapshot.observations, people=()))
        _, outlines = self.boxes(empty, (100, 50))
        self.assertEqual(outlines, [])
        with self.assertRaises(ValueError):
            render_snapshot(snapshot, expected_frame_id=1, expected_captured_at=10.0, max_size=(0, 50))
        with self.assertRaises(ValueError):
            AISnapshot(1, 10.0, snapshot.observations, np.zeros((0, 0, 3), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main()
