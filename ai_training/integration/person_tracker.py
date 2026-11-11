import math
from dataclasses import dataclass


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0.0


@dataclass
class PersonTrack:
    track_id: int
    bbox: tuple
    last_seen: float
    person_ref: str


class PersonTracker:
    """IoU tracking with an optional conservative policy for observed misses.

    max_age bounds elapsed seconds before matching. With keep_unmatched=False,
    each processed result also retires any track with no matching detection.
    """

    def __init__(self, iou_threshold=0.3, max_age=3.0, keep_unmatched=True):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.keep_unmatched = keep_unmatched
        self.tracks = {}
        self.next_id = 1

    def update(self, people, timestamp):
        # Remove expired tracks.
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen > self.max_age
        ]

        for track_id in expired:
            del self.tracks[track_id]

        detections = []

        for person in people:
            box = (
                person.get("person_bbox_xyxy_clipped")
                or person.get("bbox_xyxy")
            )

            # Stream workers supply JSON arrays; reject scalar/text/object boxes.
            if not isinstance(box, (list, tuple)) or len(box) != 4:
                continue

            try:
                bbox = tuple(float(v) for v in box)
            except (TypeError, ValueError, OverflowError):
                continue

            x1, y1, x2, y2 = bbox

            if (
                not all(math.isfinite(v) for v in bbox)
                or x2 <= x1
                or y2 <= y1
            ):
                continue

            detections.append({
                "bbox": bbox,
                "person_ref": person["person_ref"],
            })

        candidates = []

        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                iou = calculate_iou(
                    track.bbox,
                    detection["bbox"],
                )

                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, index))

        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()
        results = []

        for iou, track_id, index in candidates:
            if track_id in matched_tracks:
                continue

            if index in matched_detections:
                continue

            detection = detections[index]
            track = self.tracks[track_id]

            track.bbox = detection["bbox"]
            track.person_ref = detection["person_ref"]
            track.last_seen = timestamp

            matched_tracks.add(track_id)
            matched_detections.add(index)

            results.append({
                "track_id": f"P{track_id}",
                "person_ref": track.person_ref,
                "bbox": track.bbox,
            })

        if not self.keep_unmatched:
            # An observed miss must not leave a stale ID for a later arrival.
            for track_id in set(self.tracks) - matched_tracks:
                del self.tracks[track_id]

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue

            track_id = self.next_id
            self.next_id += 1

            self.tracks[track_id] = PersonTrack(
                track_id=track_id,
                bbox=detection["bbox"],
                last_seen=timestamp,
                person_ref=detection["person_ref"],
            )

            results.append({
                "track_id": f"P{track_id}",
                "person_ref": detection["person_ref"],
                "bbox": detection["bbox"],
            })

        return sorted(
            results,
            key=lambda item: int(item["track_id"][1:]),
        )
