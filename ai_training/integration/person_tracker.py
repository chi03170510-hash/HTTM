import math
from dataclasses import dataclass

def _minimum_assignment(costs):
    """Rectangular Hungarian assignment (rows <= columns), without AI packages."""
    if not costs:
        return []
    n, m = len(costs), len(costs[0])
    u, v, owners = [0.0] * (n + 1), [0.0] * (m + 1), [0] * (m + 1)
    for row in range(1, n + 1):
        owners[0] = row
        column = 0
        best, used, previous = [math.inf] * (m + 1), [False] * (m + 1), [0] * (m + 1)
        while True:
            used[column] = True
            current_row = owners[column]
            delta, next_column = math.inf, 0
            for candidate in range(1, m + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1][candidate - 1] - u[current_row] - v[candidate]
                if reduced < best[candidate]:
                    best[candidate], previous[candidate] = reduced, column
                if best[candidate] < delta:
                    delta, next_column = best[candidate], candidate
            for candidate in range(m + 1):
                if used[candidate]:
                    u[owners[candidate]] += delta
                    v[candidate] -= delta
                else:
                    best[candidate] -= delta
            column = next_column
            if owners[column] == 0:
                break
        while column:
            prior = previous[column]
            owners[column] = owners[prior]
            column = prior
    return [(owners[column] - 1, column - 1) for column in range(1, m + 1) if owners[column]]


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
    velocity: tuple = (0.0, 0.0)
    observations: int = 1
    motion_at: float | None = None


def _center(box):
    return ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)


def _motion_matches(tracks, detections, timestamp, iou_threshold):
    """Global one-to-one assignment with dummy columns for unmatched tracks.

    Uses geometry/motion only, never frame-local person_ref or face identity.
    Dormant tracks are candidates, but never emitted without a real detection.
    """
    track_ids = sorted(tracks)
    if not track_ids or not detections:
        return []
    costs = []
    for track_id in track_ids:
        track = tracks[track_id]
        elapsed = max(0.0, timestamp - track.motion_at) if track.motion_at is not None else 0.0
        dx, dy = (v * elapsed for v in track.velocity)
        box = track.bbox
        predicted = (box[0] + dx, box[1] + dy, box[2] + dx, box[3] + dy)
        width, height = box[2] - box[0], box[3] - box[1]
        cx, cy = _center(predicted)
        row = []
        for detection in detections:
            other = detection["bbox"]
            ow, oh = other[2] - other[0], other[3] - other[1]
            size_ratio = max(width / ow, ow / width, height / oh, oh / height)
            ox, oy = _center(other)
            distance = math.hypot((ox - cx) / width, (oy - cy) / height)
            iou = calculate_iou(predicted, other)
            # Reject teleports and major scale changes rather than borrowing IDs.
            valid = size_ratio <= 2.5 and (iou >= iou_threshold or distance <= .65)
            score = .6 * iou + .4 * math.exp(-2 * distance)
            row.append(-score if valid else 1e6)
        costs.append(row + [0.0] * len(track_ids))
    return [(0.0, track_ids[row], column) for row, column in _minimum_assignment(costs)
            if column < len(detections) and costs[row][column] < 0]


class PersonTracker:
    """Geometry-only tracking; optional motion/global assignment for live use.

    max_age bounds elapsed seconds since a real detection. keep_unmatched=True
    retains dormant tracks for short occlusions, but output contains only current
    detections. The legacy IoU policy remains available to existing callers.
    Motion cannot establish identity after complete occlusion or prove re-ID.
    """

    def __init__(self, iou_threshold=0.3, max_age=3.0, keep_unmatched=True,
                 motion_matching=False):
        self.motion_matching = motion_matching
        self._last_timestamp = None
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.keep_unmatched = keep_unmatched
        self.tracks = {}
        self.next_id = 1

    def update(self, people, timestamp):
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._last_timestamp is not None and timestamp < self._last_timestamp:
            raise ValueError("tracking timestamps must not go backwards")
        self._last_timestamp = timestamp
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

        for track_id, track in (() if self.motion_matching else self.tracks.items()):
            for index, detection in enumerate(detections):
                iou = calculate_iou(
                    track.bbox,
                    detection["bbox"],
                )

                if iou >= self.iou_threshold:
                    candidates.append((iou, track_id, index))

        if self.motion_matching:
            candidates = _motion_matches(self.tracks, detections, timestamp, self.iou_threshold)
        else:
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

            if track.motion_at is not None and timestamp > track.motion_at:
                dt = timestamp - track.motion_at
                before, after = _center(track.bbox), _center(detection["bbox"])
                measured = tuple((new - old) / dt for old, new in zip(before, after))
                weight = 1.0 if track.observations == 1 else .7
                track.velocity = tuple(weight * new + (1 - weight) * old
                                       for new, old in zip(measured, track.velocity))
            track.observations += 1
            track.motion_at = timestamp
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
                motion_at=timestamp,
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
