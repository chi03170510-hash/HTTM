"""OpenCV overlay for the unified single-frame behavior evidence API."""
from __future__ import annotations

from typing import Any


RAISE_COLORS = {
    "RAISE_HAND_EVIDENCE": (40, 210, 40),
    "NOT_RAISING_HAND": (190, 190, 190),
    "UNCERTAIN": (0, 190, 255),
    "NOT_ASSESSABLE": (150, 150, 150),
}


def _box(value: Any) -> tuple[int, int, int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return tuple(int(round(float(x))) for x in value)
    except (TypeError, ValueError):
        return None


def _draw_box(frame: Any, cv2: Any, box: Any, label: str, color: tuple[int, int, int]) -> None:
    coords = _box(box)
    if coords is None:
        return
    x1, y1, x2, y2 = coords
    h, w = frame.shape[:2]
    x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
    y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if label:
        text_y = max(16, y1 - 5)
        cv2.putText(frame, label[:100], (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, color, 1, cv2.LINE_AA)


def draw_unified_result(frame: Any, result: dict[str, Any], cv2_module: Any = None) -> Any:
    """Draw branch boxes and a compact status panel; abstentions stay explicit."""
    if cv2_module is None:
        import cv2 as cv2_module
    cv2 = cv2_module

    # Raising-hand boxes carry the state color; the compact panel identifies
    # each status without placing long text over crowded detections.
    raising = result.get("raising_hand", {})
    records = raising.get("evidence", []) if isinstance(raising, dict) else []
    for record in records:
        status = str(record.get("raising_hand_status", "NOT_ASSESSABLE"))
        color = RAISE_COLORS.get(status, RAISE_COLORS["NOT_ASSESSABLE"])
        _draw_box(frame, cv2, record.get("bbox_xyxy"), "", color)

    # Sleeping API currently supplies face geometry and a frame-level status,
    # not a calibrated sleeping classifier. Label it as evidence context only.
    sleeping = result.get("sleeping", {})
    if not isinstance(sleeping, dict):
        sleeping = {}
    if isinstance(sleeping, dict):
        for face in sleeping.get("faces", []):
            _draw_box(frame, cv2, face.get("bbox_xyxy_pixel"), "", (255, 220, 40))

    # Phone branch: draw person and detected phone boxes, then surface interaction
    # evidence on the linked person only when the branch has an association.
    phone = result.get("using_phone", {})
    if not isinstance(phone, dict):
        phone = {}
    if isinstance(phone, dict):
        interactions = phone.get("interaction_evidence", [])
        by_detection: dict[str, list[str]] = {}
        for item in interactions:
            det_id = item.get("person_detection_id")
            if det_id:
                by_detection.setdefault(str(det_id), []).append(str(item.get("status", "UNCERTAIN")))
        # Draw a person box for explicit per-person interaction evidence only;
        # drawing every branch's full-body boxes duplicates the same people.
        for person in phone.get("person_detections", []):
            det_id = str(person.get("detection_id", ""))
            if by_detection.get(det_id):
                _draw_box(frame, cv2, person.get("bbox_xyxy"), "", (40, 120, 255))
        for detection in phone.get("phone_detections", []):
            _draw_box(frame, cv2, detection.get("bbox_xyxy"), "", (255, 60, 220))

    h, w = frame.shape[:2]
    from collections import Counter
    counts = Counter(str(r.get("raising_hand_status", "NOT_ASSESSABLE")) for r in records)
    lines = [
        "Raising hand (person bbox; IDs are frame-local, not tracks):",
        f"EVIDENCE {counts['RAISE_HAND_EVIDENCE']} | NOT_RAISING {counts['NOT_RAISING_HAND']} | "
        f"UNCERTAIN {counts['UNCERTAIN']} | NOT_ASSESSABLE {counts['NOT_ASSESSABLE']}",
        f"Sleeping: {sleeping.get('sleeping_evidence_status', 'NOT_ASSESSABLE')} | "
        "face/EAR geometry only; not sleep ground truth",
        f"Using phone: {phone.get('using_phone_status', 'NOT_ASSESSABLE')} | "
        f"phone boxes {len(phone.get('phone_detections', []))}; interaction evidence "
        f"{len(phone.get('interaction_evidence', []))}",
        "Box colors: raise evidence=green, not raised=gray, uncertain=amber, "
        "not assessable=slate, face=yellow, phone=magenta, phone person=blue",
    ]
    panel_h = 22 * len(lines) + 12
    panel_w = min(w - 12, 850)
    cv2.rectangle(frame, (6, 6), (6 + panel_w, 6 + panel_h), (25, 25, 25), -1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (12, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (245, 245, 245), 1, cv2.LINE_AA)
    return frame
