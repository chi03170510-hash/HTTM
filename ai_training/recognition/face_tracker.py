from collections import Counter, deque
from dataclasses import dataclass, field


def calculate_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return intersection / union


@dataclass
class FaceTrack:
    track_id: int
    bbox: tuple
    last_seen: float
    labels: deque = field(
        default_factory=lambda: deque(maxlen=5)
    )
    score: float = 0.0
    velocity: tuple = (0.0, 0.0)

    def predict_bbox(self, timestamp):
        dt = max(0.0, float(timestamp) - float(self.last_seen))
        if dt <= 0.0 or dt > 5.0:
            return self.bbox
        vx, vy = self.velocity
        dx, dy = vx * dt, vy * dt
        x1, y1, x2, y2 = self.bbox
        return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)

    def update(self, bbox, student_id, score, timestamp):
        dt = float(timestamp) - float(self.last_seen)
        if 0.01 < dt <= 5.0 and self.bbox is not None and len(self.bbox) == 4:
            old_cx = 0.5 * (self.bbox[0] + self.bbox[2])
            old_cy = 0.5 * (self.bbox[1] + self.bbox[3])
            new_cx = 0.5 * (bbox[0] + bbox[2])
            new_cy = 0.5 * (bbox[1] + bbox[3])
            vx = (new_cx - old_cx) / dt
            vy = (new_cy - old_cy) / dt
            self.velocity = (0.5 * self.velocity[0] + 0.5 * vx, 0.5 * self.velocity[1] + 0.5 * vy)
        self.bbox = bbox
        self.last_seen = timestamp
        self.labels.append(student_id)
        self.score = score

    @property
    def stable_label(self):
        if not self.labels:
            return "Unknown"

        counts = Counter(self.labels)
        label, count = counts.most_common(1)[0]

        # Require at least 3 matching observations.
        if count < 3:
            return "Unknown"

        return label


class FaceTracker:
    def __init__(self, iou_threshold=0.3, max_age=2.0, motion_matching=False):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.motion_matching = motion_matching
        self.tracks = {}
        self.next_id = 1

    def update(self, detections, timestamp):
        expired = [
            track_id
            for track_id, track in self.tracks.items()
            if timestamp - track.last_seen > self.max_age
        ]

        for track_id in expired:
            del self.tracks[track_id]

        candidates = []

        for track_id, track in self.tracks.items():
            for detection_index, detection in enumerate(detections):
                bbox, student_id, score = detection
                iou = calculate_iou(track.bbox, bbox)
                if self.motion_matching:
                    iou = max(iou, calculate_iou(track.predict_bbox(timestamp), bbox))

                if iou >= self.iou_threshold:
                    candidates.append(
                        (iou, track_id, detection_index)
                    )

        candidates.sort(reverse=True)

        matched_tracks = set()
        matched_detections = set()

        for iou, track_id, detection_index in candidates:
            if track_id in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            bbox, student_id, score = detections[detection_index]

            self.tracks[track_id].update(
                bbox, student_id, score, timestamp
            )

            matched_tracks.add(track_id)
            matched_detections.add(detection_index)

        for index, (bbox, student_id, score) in enumerate(detections):
            if index in matched_detections:
                continue

            track = FaceTrack(
                track_id=self.next_id,
                bbox=bbox,
                last_seen=timestamp,
            )

            track.update(
                bbox, student_id, score, timestamp
            )

            self.tracks[self.next_id] = track
            self.next_id += 1

        return [
            (
                track.track_id,
                track.bbox,
                track.stable_label,
                track.score,
            )
            for track in self.tracks.values()
            if track.last_seen == timestamp
        ]
