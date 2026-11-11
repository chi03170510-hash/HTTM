"""Associate InsightFace tracks with Hybrid V7 person detections.

This module does not load models, access cameras, or modify input records.
Ambiguous associations are deliberately left unmatched.
"""


def _valid_box(box):
    if box is None or len(box) != 4:
        return False
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError):
        return False
    import math
    return all(math.isfinite(v) for v in (x1, y1, x2, y2)) and x2 > x1 and y2 > y1


def associate_faces_to_people(face_tracks, people):
    """Return associations between face tracks and Hybrid V7 people.

    face_tracks: [(track_id, face_bbox, student_id, score), ...]
    people: Hybrid V7 spatial["people"] dictionaries.

    A match is accepted only when:
    - The face center falls within exactly one person box.
    - The face center is in the upper 60% of that person box.
    - No second face claims the same person.
    - No second person claims the same face.

    This is a conservative initial geometry-only rule, not identity proof.
    """
    candidates = []

    for track_id, face_box, student_id, score in face_tracks:
        if not _valid_box(face_box):
            continue

        fx1, fy1, fx2, fy2 = map(float, face_box)
        cx = (fx1 + fx2) / 2
        cy = (fy1 + fy2) / 2

        matches = []

        for index, person in enumerate(people):
            person_box = (
                person.get("person_bbox_xyxy_clipped")
                or person.get("bbox_xyxy")
            )

            if not _valid_box(person_box):
                continue

            px1, py1, px2, py2 = map(float, person_box)
            upper_limit = py1 + 0.60 * (py2 - py1)

            if px1 <= cx <= px2 and py1 <= cy <= upper_limit:
                matches.append(index)

        if len(matches) == 1:
            candidates.append(
                (track_id, face_box, student_id, score, matches[0])
            )

    # Reject people claimed by multiple face tracks.
    from collections import Counter
    counts = Counter(item[4] for item in candidates)
    track_counts = Counter(item[0] for item in candidates)

    results = []

    for track_id, face_box, student_id, score, person_index in candidates:
        if counts[person_index] != 1 or track_counts[track_id] != 1:
            continue

        person = people[person_index]

        results.append({
            "track_id": track_id,
            "student_id": student_id,
            "face_score": score,
            "face_bbox": list(face_box),
            "person_ref": person.get("person_ref"),
            "person_bbox": (
                person.get("person_bbox_xyxy_clipped")
                or person.get("bbox_xyxy")
            ),
            "behaviors": person.get("behaviors", {}),
            "association_status": "matched",
        })

    return results
