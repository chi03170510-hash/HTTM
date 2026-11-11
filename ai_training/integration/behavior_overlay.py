"""Historical AI snapshots: source pixels, metadata and person boxes stay together.

This module never reads an image path, runs inference, or draws on live frames.
"""

import math
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from behavior_duration import BehaviorFrame, behavior_label, duration_text
from snapshot_label_layout import fit_marked, identity_text, place_label


SNAPSHOT_HEADER = 80


@dataclass(frozen=True)
class AISnapshot:
    frame_id: int
    captured_at: float
    observations: BehaviorFrame
    image: np.ndarray = field(repr=False, compare=False)
    render_metadata: object | None = field(default=None, repr=False, compare=False)

    def __post_init__(self):
        if (self.frame_id != self.observations.frame_id
                or self.captured_at != self.observations.captured_at
                or not math.isfinite(self.captured_at)):
            raise ValueError("snapshot and behavior metadata must identify the same frame")
        if (not isinstance(self.image, np.ndarray) or self.image.dtype != np.uint8
                or self.image.ndim != 3 or self.image.shape[2] != 3
                or min(self.image.shape[:2]) < 1):
            raise ValueError("snapshot requires a nonempty uint8 BGR image")
        owned = self.image.copy()
        owned.setflags(write=False)
        object.__setattr__(self, "image", owned)


@dataclass(frozen=True)
class RenderedSnapshot:
    frame_id: int
    captured_at: float
    image: np.ndarray = field(repr=False, compare=False)
    label_regions: tuple = ()
    layout_notes: tuple[str, ...] = ()

    def validate(self, frame_id, captured_at):
        if self.frame_id != frame_id or self.captured_at != captured_at:
            raise ValueError("cached snapshot does not belong to the displayed AI result")


def person_label_lines(person):
    return (f"{person.student_id or 'Unknown'} | {person.person_track_id}",) + tuple(
        f"{behavior_label(b.name)}: {b.state} {duration_text(b)}" for b in person.behaviors
    )


def _fit_text(text, width, scale=.43):
    while text and cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0] > width:
        text = text[:-1]
    return text


def _scaled_box(bbox, source_size, display_size, offset):
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    width, height = source_size
    out_width, out_height = display_size
    x1, x2 = max(0., x1), min(float(width), x2)
    y1, y2 = max(0., y1), min(float(height), y2)
    if x2 <= x1 or y2 <= y1:
        return None
    # xyxy uses an exclusive far edge; OpenCV rectangle uses inclusive pixels.
    left = min(out_width - 1, math.floor(x1 * out_width / width))
    top = min(out_height - 1, math.floor(y1 * out_height / height))
    right = max(left, min(out_width - 1, math.ceil(x2 * out_width / width) - 1))
    bottom = max(top, min(out_height - 1, math.ceil(y2 * out_height / height) - 1))
    return left + offset[0], top + offset[1], right + offset[0], bottom + offset[1]


def person_box_for_display(bbox, source_size):
    """Render-only rasterization of source pixel endpoints to half-open bounds.

    Source tracking uses continuous xyxy and Hybrid clips endpoints to W-1/H-1,
    without an inclusive/exclusive flag. The opt-in Person view encloses both
    endpoint pixels: floor near/far, then add one to the far pixel ONLY here.
    Existing detection/tracking coordinates and legacy edge rendering stay intact.
    """
    if not isinstance(bbox, (tuple, list)) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = map(float, bbox)
    except (TypeError, ValueError, OverflowError):
        return None
    width, height = source_size
    if (not all(math.isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1
            or x2 < 0 or y2 < 0 or x1 > width - 1 or y1 > height - 1):
        return None
    return (max(0, math.floor(x1)), max(0, math.floor(y1)),
            min(width, math.floor(x2) + 1), min(height, math.floor(y2) + 1))


def _production_person_labels(people):
    """Read accepted production views only; never inspect matches or shadow SID."""
    tracks = Counter(p.person_track_id for p in people)
    students = Counter(p.student_id for p in people if isinstance(p.student_id, str) and p.student_id.strip())
    labels = {}
    for p in people:
        student = p.student_id.strip() if isinstance(p.student_id, str) else None
        ambiguous = (p.identity_status.startswith("ambiguous") or tracks[p.person_track_id] != 1
                     or (student is not None and students[p.student_id] > 1))
        accepted = (p.identity_status == "matched" and p.association_status in {"matched", "matched_v4_heuristic"}
                    and student and student.casefold() not in {"unknown", "none", "null"})
        identity = "Ambiguous" if ambiguous else student if accepted else "Unknown"
        color = (0, 180, 255) if ambiguous else (255, 200, 0) if accepted else (180, 180, 180)
        labels[p.person_track_id] = ((f"{p.person_track_id} | {identity}",), color)
    return labels


def render_snapshot(snapshot, *, expected_frame_id, expected_captured_at, max_size=(640, 480), person_only=False):
    """Render once per AI result; resize pixels and boxes with the same transform."""
    if (snapshot.frame_id != expected_frame_id or snapshot.captured_at != expected_captured_at
            or snapshot.observations.frame_id != expected_frame_id
            or snapshot.observations.captured_at != expected_captured_at):
        raise ValueError("snapshot, observations and AI result metadata disagree")
    max_width, max_height = max_size
    if (not isinstance(max_width, int) or not isinstance(max_height, int)
            or max_width < 1 or max_height < 1):
        raise ValueError("max_size must contain positive integer dimensions")
    height, width = snapshot.image.shape[:2]
    scale = min(max_width / width, max_height / height)
    out_width = max(1, min(max_width, round(width * scale)))
    out_height = max(1, min(max_height, round(height * scale)))
    offset = ((max_width - out_width) // 2, SNAPSHOT_HEADER + (max_height - out_height) // 2)
    canvas = np.full((SNAPSHOT_HEADER + max_height, max_width, 3), 25, dtype=np.uint8)
    canvas[offset[1]:offset[1] + out_height, offset[0]:offset[0] + out_width] = cv2.resize(
        snapshot.image, (out_width, out_height),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )
    for text, y in [("AI SNAPSHOT - HISTORICAL", 22),
                    (f"Frame {snapshot.frame_id} | " + ("Person PID / production SID" if person_only else "~ estimated duration"), 44)]:
        cv2.putText(canvas, _fit_text(text, max_width - 16, .48), (8, y),
                    cv2.FONT_HERSHEY_SIMPLEX, .48, (0, 220, 255), 1, cv2.LINE_AA)

    occupied, pending_labels, layout_notes = [], [], []
    identities = _production_person_labels(snapshot.observations.people) if person_only else None
    for person in snapshot.observations.people:
        source_box = person_box_for_display(person.bbox, (width, height)) if person_only else person.bbox
        box = _scaled_box(source_box, (width, height), (out_width, out_height), offset)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        if identities is None:
            lines = person_label_lines(person)
            color = ((0, 220, 0) if any(b.estimated_seconds is not None for b in person.behaviors)
                     else (180, 180, 180))
        else:
            lines, color = identities[person.person_track_id]
        if person_only:
            # Clip the stroke thickness, including letterbox/far-edge cases.
            roi = canvas[offset[1]:offset[1] + out_height, offset[0]:offset[0] + out_width]
            cv2.rectangle(roi, (x1 - offset[0], y1 - offset[1]),
                          (x2 - offset[0], y2 - offset[1]), color, 2)
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)
        label_width = min(max_width, max(
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0] for text in lines
        ) + 10)
        label_height = min(max_height, len(lines) * 18 + 6)
        px = max(0, min(x1, max_width - label_width))
        py = y1 - label_height - 3 if y1 - label_height >= SNAPSHOT_HEADER else y2 + 3
        py = max(SNAPSHOT_HEADER, min(py, canvas.shape[0] - label_height))
        if person_only:
            original = lines[0]
            if cv2.getTextSize(original, cv2.FONT_HERSHEY_SIMPLEX, .43, 1)[0][0] > label_width - 8:
                pid, identity = original.split(" | ", 1)
                lines = (identity_text(identity, pid, label_width - 8, .43),)
                layout_notes.append("Person text clipped; see identity legend (markers ... / !)")
            rect = place_label((px, py), (label_width, 24),
                               (0, SNAPSHOT_HEADER, max_width, canvas.shape[0]), occupied)
            if rect is None:
                layout_notes.append("Person label omitted; see identity legend")
                continue
            occupied.append(rect)
            px, py = rect[:2]
            cv2.line(canvas, (x1, y1), (px, py), color, 1)
            pending_labels.append((rect, lines, color))
            continue
        # Prefer a free vertical slot; the connector and P ID keep ownership
        # explicit when a crowded snapshot leaves no completely free slot.
        candidates = [py] + list(range(SNAPSHOT_HEADER, canvas.shape[0] - label_height + 1, 18))
        for candidate_y in sorted(candidates, key=lambda value: abs(value - py)):
            if all(not (px < ox2 and px + label_width > ox1
                        and candidate_y < oy2 and candidate_y + label_height > oy1)
                   for ox1, oy1, ox2, oy2 in occupied):
                py = candidate_y
                break
        occupied.append((px, py, px + label_width, py + label_height))
        cv2.line(canvas, (x1, y1), (px, py), color, 1)
        cv2.rectangle(canvas, (px, py),
                      (min(max_width - 1, px + label_width - 1), py + label_height - 1), (25, 25, 25), -1)
        for index, text in enumerate(lines):
            if py + 17 + index * 18 >= canvas.shape[0]:
                break
            cv2.putText(canvas, _fit_text(text, label_width - 8), (px + 4, py + 17 + index * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    # Later rectangles/connectors cannot obscure an earlier SID plate.
    for (px, py, right, bottom), lines, color in pending_labels:
        cv2.rectangle(canvas, (px, py), (right - 1, bottom - 1), (25, 25, 25), -1)
        for index, text in enumerate(lines):
            cv2.putText(canvas, fit_marked(text, right - px - 8), (px + 4, py + 17 + index * 18),
                        cv2.FONT_HERSHEY_SIMPLEX, .43, color, 1, cv2.LINE_AA)
    canvas.setflags(write=False)
    return RenderedSnapshot(snapshot.frame_id, snapshot.captured_at, canvas,
                            tuple(occupied) if person_only else (), tuple(layout_notes))
