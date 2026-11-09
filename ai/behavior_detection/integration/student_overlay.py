"""Draw CH-20 student labels using only controller-provided timer values."""

import cv2


BEHAVIOR_LABELS = {"using_phone": "Using Phone", "sleeping": "Sleeping"}


def student_label_lines(view):
    behavior = BEHAVIOR_LABELS.get(view.behavior, "None")
    if view.behavior is not None and view.status != "CONFIRMED":
        behavior += " (suspected)"
    elif view.behavior is not None and view.duration is None:
        behavior += " (paused)"
    return (
        f"Student ID: {view.student_id if view.student_id is not None else 'Unknown'}",
        f"Track ID: {view.track_id}",
        f"Behavior: {behavior}",
        f"Duration: {view.duration:.1f}s" if view.duration is not None else "Duration: --",
    )


def draw_student_overlays(frame, views, cv2_module=cv2, hud_height=180):
    """Keep face coordinates unchanged and place a panel beside each face."""
    height, width = frame.shape[:2]
    occupied = []
    for view in views:
        x1, y1, x2, y2 = view.bbox
        color = ((0, 0, 255) if view.warned else (0, 220, 0) if view.duration is not None
                 else (0, 165, 255) if view.behavior else (0, 220, 255))
        cv2_module.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        lines = list(student_label_lines(view))
        if view.warned:
            lines.append("Possible warning: 5s timer")
        panel_w = min(width, max(cv2_module.getTextSize(line, cv2_module.FONT_HERSHEY_SIMPLEX, .46, 1)[0][0]
                                 for line in lines) + 12)
        panel_h = min(height, len(lines) * 19 + 10)
        px = max(0, min(x1, width - panel_w))
        py = y1 - panel_h - 4 if y1 >= panel_h + 4 else y2 + 4
        if px < 380 and py < hud_height:
            py = max(py, hud_height + 4)
        py = max(0, min(py, height - panel_h))
        # Try to keep multiple panels readable; track IDs and connector lines
        # still identify the owner when a crowded image leaves no free space.
        for other in occupied:
            if px < other[2] and px + panel_w > other[0] and py < other[3] and py + panel_h > other[1]:
                py = min(height - panel_h, other[3] + 3)
        occupied.append((px, py, px + panel_w, py + panel_h))
        cv2_module.line(frame, (x1, y2), (px, py), color, 1)
        cv2_module.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        for index, line in enumerate(lines):
            cv2_module.putText(frame, line, (px + 5, py + 18 + index * 19),
                               cv2_module.FONT_HERSHEY_SIMPLEX, .46, color, 1, cv2_module.LINE_AA)
    return frame
