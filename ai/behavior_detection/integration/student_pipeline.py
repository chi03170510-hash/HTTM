"""CH-20: same-frame face/identity/behavior observations and per-track timers.

Heavy work and tracker/identity state belong to the AI worker. The main thread
alone owns BehaviorStateManager/BehaviorTimer and advances the UI clock. Sparse
model evidence is a bounded lease, not proof of continuous real-world behavior.
"""

from dataclasses import dataclass
from pathlib import Path

from ..behavior_state_manager import BehaviorStateManager
from ..behavior_timer import BehaviorTimer
from ..student_face_recognizer import DB_PATH, StudentFaceRecognizer
from .event_schema import FrameResult
from .student_tracking import (
    StableFaceTracker, StudentIdentityManager, ambiguous_student_ids,
    associate_behaviors, detect_faces, remove_duplicates, safe_student_id,
)


DEFAULT_FACE_ROOT = Path.home() / ".insightface"
WARNING_SECONDS = 5.0
MIN_CONFIRM_OBSERVATIONS = 3


@dataclass(frozen=True)
class StudentObservation:
    track_id: int
    bbox: tuple[int, int, int, int]
    student_id: str | None = None
    behavior: str | None = None
    confidence: float = 0.0
    reset_episode: bool = False


@dataclass(frozen=True)
class StudentFrameResult:
    """Internal snapshot extension; legacy detection/event JSON stays identical."""
    behavior_result: FrameResult
    students: tuple[StudentObservation, ...]
    unmatched_count: int = 0

    @property
    def metadata(self):
        return self.behavior_result.metadata

    @property
    def status(self):
        return self.behavior_result.status

    @property
    def error(self):
        return self.behavior_result.error

    @property
    def detections(self):
        return self.behavior_result.detections

    def to_dict(self):
        return self.behavior_result.to_dict()


class StudentInferenceProcessor:
    def __init__(self, adapter, face_app, recognizer):
        self.adapter, self.face_app, self.recognizer = adapter, face_app, recognizer
        self.tracker = StableFaceTracker()
        self.identities = StudentIdentityManager()
        self._face_candidates = {}

    @classmethod
    def from_local(cls, adapter, face_root=DEFAULT_FACE_ROOT, student_db=DB_PATH):
        """Use the existing buffalo_l cache; never download models implicitly."""
        pack = Path(face_root).expanduser() / "models" / "buffalo_l"
        missing = [name for name in ("det_10g.onnx", "w600k_r50.onnx") if not (pack / name).is_file()]
        if missing:
            raise FileNotFoundError(f"CH-20 needs existing InsightFace buffalo_l files in {pack}: {missing}")
        if not Path(student_db).is_file():
            raise FileNotFoundError(f"CH-20 student face database not found: {student_db}")
        from insightface.app import FaceAnalysis
        import onnxruntime as ort
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        # A direct, verified directory avoids InsightFace's package downloader.
        face_app = FaceAnalysis(name=str(pack.resolve()),
                                allowed_modules=["detection", "recognition"],
                                providers=["CPUExecutionProvider"], sess_options=options)
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
        if not {"detection", "recognition"}.issubset(face_app.models):
            raise RuntimeError("CH-20 needs both InsightFace detection and recognition models")
        return cls(adapter, face_app, StudentFaceRecognizer(student_db))

    def process(self, frame, metadata):
        behavior_result = self.adapter.process(frame, metadata)
        # Source timestamps prevent CPU inference latency from changing tracker gaps.
        now = metadata.timestamp_ms / 1000.0
        faces = detect_faces(self.face_app, frame)
        tracks, assignments = self.tracker.update(faces, now)
        self.identities.cleanup(self.tracker.records.keys())
        self._face_candidates = {tid: sid for tid, sid in self._face_candidates.items()
                                 if tid in self.tracker.records}
        identity_changes = set()
        for track_id, face in assignments.items():
            recognition = self.recognizer.recognize(face["embedding"])
            if recognition.get("status") == "MATCH" and recognition.get("student_id") is not None:
                student_id = recognition["student_id"]
                previous_id = self._face_candidates.get(track_id)
                if previous_id is not None and previous_id != student_id:
                    # A contradictory fresh identity must not inherit elapsed time
                    # or display the previous student's identity grace period.
                    self.identities.records.pop(track_id, None)
                    identity_changes.add(track_id)
                self._face_candidates[track_id] = student_id
            self.identities.update(track_id, recognition, now)
        collisions = set(ambiguous_student_ids(self.identities, tracks, now))
        behaviors = ([{"bbox": d.bbox_xyxy_px, "label": d.behavior, "confidence": d.confidence}
                      for d in behavior_result.detections] if behavior_result.status == "ok" else [])
        matches, unmatched = associate_behaviors(tracks, remove_duplicates(behaviors))
        observations = tuple(StudentObservation(
            track_id, tuple(bbox), safe_student_id(self.identities, track_id, now, collisions),
            matches[track_id]["label"] if track_id in matches else None,
            matches[track_id]["confidence"] if track_id in matches else 0.0,
            track_id in identity_changes,
        ) for track_id, bbox in sorted(tracks.items()))
        return StudentFrameResult(behavior_result, observations, len(unmatched))

    def close(self):
        self.face_app = None
        self.tracker.records.clear()
        self.identities.records.clear()
        self._face_candidates.clear()


@dataclass(frozen=True)
class StudentOverlay:
    track_id: int
    bbox: tuple[int, int, int, int]
    student_id: str | None
    behavior: str | None
    status: str
    confidence: float
    duration: float | None
    warned: bool


class StudentOverlayController:
    """Main-thread state/timers: one evidence update per distinct AI observation.

    Explicit missing behavior/face, inference error, expired evidence, or a
    known identity change resets the affected episode. Repainting a cached
    observation never increments confirmation hits. Timer elapsed advances via
    monotonic time only while a confirmed, fresh state supports that track.
    """

    def __init__(self, evidence_ttl=1.5):
        if evidence_ttl <= 0:
            raise ValueError("evidence_ttl must be greater than zero")
        self.evidence_ttl = evidence_ttl
        # Legacy default 0.8s cannot span 1 FPS. This instance uses the existing
        # video evidence TTL; legacy module defaults and CH-18 scripts stay intact.
        self.states = BehaviorStateManager(min_confirm_frames=MIN_CONFIRM_OBSERVATIONS,
                                          max_behavior_missing=evidence_ttl,
                                          max_face_missing=evidence_ttl)
        self.timer = BehaviorTimer(warning_seconds=WARNING_SECONDS)
        self._last_key = None
        self._observations = {}
        self._known_ids = {}
        self.last_warnings = ()

    def _reset_track(self, track_id):
        self.states.states.pop(track_id, None)
        self.timer.timers.pop(track_id, None)

    def _clear(self):
        self.states.states.clear()
        self.timer.timers.clear()
        self._observations.clear()
        self._known_ids.clear()

    def advance(self, snapshot, metadata, now):
        self.last_warnings = ()
        if (snapshot is None or not isinstance(snapshot.result, StudentFrameResult)
                or not snapshot.is_fresh(metadata, now, self.evidence_ttl)):
            self._clear()
            return ()
        result = snapshot.result
        key = (result.metadata.source_id, result.metadata.frame_index)
        if key != self._last_key:
            if self._last_key is not None and key[0] != self._last_key[0]:
                self._clear()
            self._last_key = key
            observations = {o.track_id: o for o in result.students}
            for track_id in set(self._observations) - set(observations):
                self._reset_track(track_id)
                self._known_ids.pop(track_id, None)
            matches = {}
            for track_id, observation in observations.items():
                previous_id = self._known_ids.get(track_id)
                if (observation.reset_episode or (previous_id is not None
                        and observation.student_id is not None and previous_id != observation.student_id)):
                    self._reset_track(track_id)
                    self._known_ids.pop(track_id, None)
                if observation.student_id is not None:
                    self._known_ids[track_id] = observation.student_id
                if (result.status == "ok" and observation.behavior in ("sleeping", "using_phone")
                        and observation.confidence >= .40):
                    previous_state = self.states.get_state(track_id)
                    if previous_state is not None and previous_state.behavior not in (None, observation.behavior):
                        # At 1 FPS the old behavior's timeout can precede three
                        # competing hits. End that episode now and confirm the
                        # new label from its own three distinct observations.
                        self._reset_track(track_id)
                    matches[track_id] = {"label": observation.behavior,
                                         "confidence": observation.confidence}
                else:
                    # An actual new negative must not reuse the previous timer.
                    self._reset_track(track_id)
            self._observations = observations
            self.states.update({tid: o.bbox for tid, o in observations.items()}, matches,
                               inference_id=result.metadata.frame_index, now=now)
        else:
            # Runs timeouts, but the same inference ID cannot count as a new hit.
            self.states.update({tid: o.bbox for tid, o in self._observations.items()}, {},
                               inference_id=result.metadata.frame_index, now=now)
        timers, warnings = self.timer.update(self.states.states, now=now)
        self.last_warnings = tuple(warnings)
        views = []
        for track_id, observation in sorted(self._observations.items()):
            state = self.states.get_state(track_id)
            timer = timers.get(track_id)
            confirmed = (state is not None and state.status == "CONFIRMED" and state.evidence_active
                         and timer is not None and timer.behavior == state.behavior and not timer.paused)
            views.append(StudentOverlay(
                track_id, observation.bbox, observation.student_id,
                state.behavior if state is not None else None,
                state.status if state is not None else "NORMAL",
                state.confidence if state is not None else 0.0,
                float(timer.elapsed) if confirmed else None,
                bool(timer.warned) if confirmed else False,
            ))
        return tuple(views)
