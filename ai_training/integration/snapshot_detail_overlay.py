"""Face, behavior proposal and COCO object layers on one historical snapshot."""

import cv2
import numpy as np

from behavior_overlay import (RenderedSnapshot, SNAPSHOT_HEADER,
                              _scaled_box, person_box_for_display, _production_person_labels)
from bbox_visual_metadata import VisualMetadata, diagnostic
from snapshot_label_layout import fit_marked, identity_text, place_label, text_width


COLORS = {"face": (240, 180, 80), "sleeping_candidate": (180, 120, 255),
          "using_phone_candidate": (80, 240, 200), "phone_object": (60, 160, 255)}


def label_lines(item, tag):
    owner = f"{item.owner_pid} | {item.identity}" if item.owner_pid else f"Unassigned | {item.identity}"
    score = "N/A" if item.score is None else f"{item.score:.3f}"
    if item.kind == "face":
        track = "N/A" if item.face_track_id is None else str(item.face_track_id)
        return (f"{tag} Face detection | FT {track} | {owner}",
                f"Recognition: {item.recognition}; similarity {score}")
    duration = "N/A" if item.duration is None else f"~{item.duration:.1f}s"
    state = item.state or "N/A"
    if item.kind == "phone_object":
        return (f"{tag} COCO phone OBJECT | {owner}",
                f"COCO score {score}; phone support {'yes' if item.selected else 'unconfirmed'}")
    name = "Sleeping" if item.kind == "sleeping_candidate" else "Using Phone"
    role = "production detected box" if item.selected else "YOLO candidate"
    return (f"{tag} {name} {role} | {owner}",
            f"Behavior YOLO {score}; prod {state}; duration {duration}")


def render_details(snapshot, base, *, max_size):
    """Metadata failure drops only these layers; Person display remains usable."""
    metadata = snapshot.render_metadata
    if metadata is None and not base.layout_notes:
        return base
    try:
        if metadata is not None:
            try:
                if not isinstance(metadata, VisualMetadata):
                    raise ValueError("unexpected visual metadata type")
                metadata.validate(snapshot)
            except Exception as exc:
                diagnostic(f"frame={snapshot.frame_id} detail metadata skipped: {type(exc).__name__}: {exc}")
                metadata = None
        if (metadata is None or not metadata.boxes) and not base.layout_notes:
            return base
        width, height = max_size
        sh, sw = snapshot.image.shape[:2]
        scale = min(width / sw, height / sh)
        output_size = (max(1, min(width, round(sw * scale))), max(1, min(height, round(sh * scale))))
        offset = ((width - output_size[0]) // 2, SNAPSHOT_HEADER + (height - output_size[1]) // 2)
        people = snapshot.observations.people
        identities = _production_person_labels(people)
        visible = []
        for item in metadata.boxes if metadata is not None else ():
            if item.kind not in COLORS:
                diagnostic(f"frame={snapshot.frame_id} unknown visual kind omitted")
                continue
            if item.owner_pid is not None:
                owners = [p for p in people if p.person_track_id == item.owner_pid]
                p = owners[0] if len(owners) == 1 else None
                identity = identities[p.person_track_id][0][0].split(" | ", 1)[1] if p else None
                name = "sleeping" if item.kind == "sleeping_candidate" else "using_phone"
                states = [b for b in p.behaviors if b.name == name] if p else []
                conflict = p is None or item.identity != identity
                if item.kind == "face":
                    conflict = conflict or (p is not None and item.face_track_id != p.face_track_id)
                elif item.kind.endswith("_candidate"):
                    conflict = conflict or len(states) != 1 or (states[0].state, states[0].estimated_seconds) != (item.state, item.duration)
                elif item.kind == "phone_object":
                    conflict = conflict or len(states) != 1 or states[0].state != item.state
                if conflict:
                    diagnostic(f"frame={snapshot.frame_id} production view/visual owner-state mismatch; box omitted")
                    continue
            projected = person_box_for_display(item.bbox, (sw, sh))
            box = _scaled_box(projected, (sw, sh), output_size, offset) if projected else None
            if box is None:
                diagnostic(f"frame={snapshot.frame_id} invalid/outside {item.kind} box omitted")
                continue
            visible.append((item, box))
        if not visible and not base.layout_notes:
            return base
        legend_width = 540
        canvas = np.full((base.image.shape[0], width + legend_width, 3), 25, dtype=np.uint8)
        canvas[:, :width] = base.image
        rows, tags = [], []
        ox, oy = offset
        roi = canvas[oy:oy + output_size[1], ox:ox + output_size[0]]
        for number, (item, (x1, y1, x2, y2)) in enumerate(visible, 1):
            tag = ("F" if item.kind == "face" else "O" if item.kind == "phone_object" else "B") + str(number)
            color = COLORS[item.kind]
            cv2.rectangle(roi, (x1 - ox, y1 - oy), (x2 - ox, y2 - oy), color, 2)
            tags.append((tag, (x1, y1, x2, y2), color))
            priority = 0 if item.kind in {"sleeping_candidate", "using_phone_candidate"} else 1 if item.kind == "phone_object" else 2
            rows.append((priority, label_lines(item, tag), color))
        for left, top, right, bottom in base.label_regions:
            canvas[top:bottom, left:right] = base.image[top:bottom, left:right]
        occupied = list(base.label_regions)
        tags_shown = 0
        for tag, (x1, y1, x2, y2), color in tags:
            tw = text_width(tag, .4) + 6
            # A tag must remain in its own box; never move it over another
            # actor's box simply to maximize the number of visible tags.
            rect = place_label((x1, y1), (tw, 18), (x1, y1, x2 + 1, y2 + 1), occupied)
            if rect is None:
                continue
            occupied.append(rect)
            left, top, right, bottom = rect
            cv2.rectangle(canvas, (left, top), (right - 1, bottom - 1), (25, 25, 25), -1)
            cv2.putText(canvas, tag, (left + 3, top + 12), cv2.FONT_HERSHEY_SIMPLEX, .4, color, 1, cv2.LINE_AA)
            tags_shown += 1
        people_capacity = max(0, (canvas.shape[0] - 110) // 18)
        shown_people = people[:people_capacity]
        y = 84
        for p in shown_people:
            original, color = identities[p.person_track_id]
            pid, sid = original[0].split(" | ", 1)
            primary = identity_text(sid, pid, legend_width - 16)
            states, durations = [], []
            for b in p.behaviors:
                if b.name in {"sleeping", "using_phone"}:
                    name = "Sleeping" if b.name == "sleeping" else "Phone"
                    states.append(f"{name}:{b.state}")
                    durations.append(f"{name}:" + ("N/A" if b.estimated_seconds is None else f"~{b.estimated_seconds:.1f}s"))
            line = primary
            if primary == f"{sid} | {pid}":
                extra = " | ".join(states + durations)
                if extra:
                    line = fit_marked(primary + " | " + extra, legend_width - 16, .38)
            cv2.putText(canvas, line, (width + 8, y), cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
            y += 18
        detail_start = y + 26
        capacity = max(0, (canvas.shape[0] - 28 - detail_start) // 44)
        header = ("SOURCE LABELS | historical evidence", "Face similarity is not identity probability",
                  f"People {len(shown_people)}/{len(people)} | Tags {tags_shown}/{len(tags)} | SID first; Phone=Using Phone")
        for text, y in zip(header, (20, 40, 60)):
            cv2.putText(canvas, fit_marked(text, legend_width - 16, .38), (width + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1, cv2.LINE_AA)
        if detail_start - 20 < canvas.shape[0] - 18:
            cv2.putText(canvas, fit_marked(f"Showing {min(capacity, len(rows))}/{len(rows)} details; candidate != accepted behavior",
                                         legend_width - 16, .38),
                        (width + 8, detail_start - 20), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1, cv2.LINE_AA)
        for i, (_, lines, color) in enumerate(sorted(rows, key=lambda row: row[0])[:capacity]):
            for j, text in enumerate(lines):
                cv2.putText(canvas, fit_marked(text, legend_width - 16, .38),
                            (width + 8, detail_start + i * 44 + j * 18), cv2.FONT_HERSHEY_SIMPLEX,
                            .38, color, 1, cv2.LINE_AA)
        footer = (f"... / ! = clipped | IDs hidden: {len(people) - len(shown_people)} | "
                  f"details hidden: {max(0, len(rows) - capacity)}")
        cv2.putText(canvas, fit_marked(footer, legend_width - 16, .38),
                    (width + 8, canvas.shape[0] - 9), cv2.FONT_HERSHEY_SIMPLEX, .38, (220, 220, 220), 1, cv2.LINE_AA)
        canvas.setflags(write=False)
        return RenderedSnapshot(base.frame_id, base.captured_at, canvas, tuple(occupied), base.layout_notes)
    except Exception as exc:
        diagnostic(f"frame={snapshot.frame_id} detail renderer skipped: {type(exc).__name__}: {exc}")
        return base
