"""Synthetic actor truth ONLY. Never imported by Camera Live or the resolver.

actor_labels are authored independently of face recognition and tracking. They
are stripped before calling the real tracker/resolver. No retention policy or
production T_hold is configured by these fixtures.
"""

from dataclasses import dataclass


ACTOR_STUDENTS = {"actor_a": "SV001", "actor_b": "SV002"}
LEFT = (20., 40., 180., 420.)
RIGHT = (320., 40., 480., 420.)
OVERLAP = (100., 40., 260., 420.)


@dataclass(frozen=True)
class Detection:
    bbox: tuple = LEFT
    face_kind: str = "absent"  # absent, present, unavailable
    recognized_student_id: str | None = None
    association_status: str | None = None


@dataclass(frozen=True)
class SafetyFrame:
    frame_id: int
    captured_at: float
    updated_at: float
    detections: tuple[Detection, ...]
    actor_labels: tuple[str | None, ...]


@dataclass(frozen=True)
class SafetyFixture:
    name: str
    description: str
    frames: tuple[SafetyFrame, ...]
    synthetic: bool = True


def seen(bbox=LEFT, student="SV001", status="matched"):
    return Detection(bbox, "present", student, status)


def frame(fid, t, detections, actors, *, updated=None):
    return SafetyFrame(fid, t, 100. + t if updated is None else updated, tuple(detections), tuple(actors))


def safety_fixtures():
    a = frame(1, 0., [seen()], ["actor_a"])
    two = frame(1, 0., [seen(), seen(RIGHT, "SV002")], ["actor_a", "actor_b"])
    return (
        SafetyFixture("same_track_face_absent", "Same labelled actor; face temporarily absent, PID survives.", (
            a, frame(2, 1., [Detection()], ["actor_a"]), frame(3, 1.5, [Detection()], ["actor_a"]))),
        SafetyFixture("new_track_face_absent", "Same actor moves without overlap; new PID cannot inherit SID.", (
            a, frame(2, 1., [Detection(RIGHT)], ["actor_a"]))),
        SafetyFixture("swap_positions_unobserved", "Two actors swap between observations; geometry/PIDs stay in their old slots.", (
            two, frame(2, 1., [Detection(), Detection(RIGHT)], ["actor_b", "actor_a"]))),
        SafetyFixture("overlapping_people", "Overlapping current boxes; caches must be rejected.", (
            two, frame(2, 1., [Detection(), Detection(OVERLAP)], ["actor_a", "actor_b"]))),
        SafetyFixture("observed_exit_replacement", "An observed empty frame separates departure and replacement.", (
            a, frame(2, .5, [], []), frame(3, 1., [Detection()], ["actor_b"]))),
        SafetyFixture("fresh_face_changed_actor", "New actor at same position has a fresh correct different SID.", (
            a, frame(2, 1., [seen(student="SV002")], ["actor_b"]),
            frame(3, 1.5, [Detection()], ["actor_b"]))),
        SafetyFixture("fresh_face_misrecognition", "Actor A remains; recognition incorrectly reports SV002, then face disappears.", (
            a, frame(2, 1., [seen(student="SV002")], ["actor_a"]),
            frame(3, 1.5, [Detection()], ["actor_a"]))),
        SafetyFixture("duplicate_recognized_sid", "Two different actors are simultaneously recognized as SV001.", (
            frame(1, 0., [seen(), seen(RIGHT, "SV001")], ["actor_a", "actor_b"]),
            frame(2, 1., [Detection(), Detection(RIGHT)], ["actor_a", "actor_b"]))),
        SafetyFixture("capture_timestamp_not_increasing", "A new frame has repeated capture time, although tracker time increases.", (
            a, frame(2, 0., [Detection()], ["actor_a"], updated=101.))),
        SafetyFixture("capture_gap_exceeded", "Capture gap exceeds existing max_age, but the tracker update gap does not.", (
            a, frame(2, 5.116, [Detection()], ["actor_a"], updated=101.))),
        SafetyFixture("track_expired_recreated", "Tracker expires at update gap 5.116; small capture gap cannot rescue SID.", (
            a, frame(2, 1., [Detection()], ["actor_a"], updated=105.116))),
        SafetyFixture("same_slot_unobserved_replacement", "Actor B replaces A with identical observations except independent actor truth.", (
            a, frame(2, 1., [Detection()], ["actor_b"]), frame(3, 1.5, [Detection()], ["actor_b"]))),
        SafetyFixture("face_unknown", "A face is present but recognition fails; old identity must not hide Unknown.", (
            a, frame(2, 1., [seen(student="Unknown")], ["actor_a"]), frame(3, 1.5, [Detection()], ["actor_a"]))),
        SafetyFixture("face_data_unavailable", "Missing face result is not an observed absence.", (
            a, frame(2, 1., [Detection(face_kind="unavailable")], ["actor_a"]),
            frame(3, 1.5, [Detection()], ["actor_a"]))),
        SafetyFixture("face_without_association", "Present recognized face has no association; revoke old anchor.", (
            a, frame(2, 1., [seen(status=None)], ["actor_a"]), frame(3, 1.5, [Detection()], ["actor_a"]))),
        SafetyFixture("heuristic_cannot_seed", "A heuristic-only face match cannot authorize retention.", (
            frame(1, 0., [seen(status="matched_v4_heuristic")], ["actor_a"]),
            frame(2, 1., [Detection()], ["actor_a"]))),
        SafetyFixture("fresh_face_recovers_new_track", "Same actor on a new PID has new face recognition, not cached transfer.", (
            a, frame(2, 1., [seen(RIGHT)], ["actor_a"]), frame(3, 1.5, [Detection(RIGHT)], ["actor_a"]))),
        SafetyFixture("hold_not_refreshed_by_bbox", "Face age grows despite repeated identical boxes; test hold boundary only.", (
            a, *(frame(i + 2, t, [Detection()], ["actor_a"]) for i, t in enumerate((.5, 1., 1.5, 2., 2.001))))),
        SafetyFixture("actor_truth_unavailable", "Unknown physical actor; stable PID cannot establish identity correctness.", (
            a, frame(2, 1., [Detection()], [None]))),
        SafetyFixture("overlap_then_swap", "Observed overlap revokes anchors before actors separate into swapped slots.", (
            two, frame(2, .5, [Detection(), Detection(OVERLAP)], ["actor_a", "actor_b"]),
            frame(3, 1., [Detection(), Detection(RIGHT)], ["actor_b", "actor_a"]))),
        SafetyFixture("fresh_face_wrong_actor_new_track", "Actor B on a new PID is misrecognized as A; face_recovered is not actor truth.", (
            a, frame(2, 1., [seen(RIGHT)], ["actor_b"]), frame(3, 1.5, [Detection(RIGHT)], ["actor_b"]))),
        SafetyFixture("detection_order_reversed", "Only detection order/refs change; actor locations and identity stay fixed.", (
            two, frame(2, 1., [Detection(RIGHT), Detection()], ["actor_b", "actor_a"]))),
    )
