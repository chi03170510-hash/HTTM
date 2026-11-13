"""Controlled CH-20 integration replay; never imported by Camera Live production.

Run: python -m unittest discover -s ai_training/integration -p test_ch20_duration_pipeline.py -v

Only worker replies, virtual time/frame numbering, camera and GUI are replaced.
The real ai_loop, Association V4, both trackers, duration, result-queue consumer
in main(), draw_ai_panel and OpenCV text drawing execute unchanged.
"""

import copy
import io
import queue
import threading
import unittest
from contextlib import ExitStack, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

import camera_coordinator_live as live


@dataclass(frozen=True)
class Observation:
    frame_id: int
    captured_at: float
    people: tuple
    faces: tuple


def observation(frame_id, captured_at, state="detected", student="SV001", shift=0,
                associated=True, second_student=None, second_state="detected", present=True):
    """Synthetic model outputs, not real recognition or manually assigned P IDs."""
    people, faces = [], []
    if present:
        students = [student] + ([second_student] if second_student is not None else [])
        for index, sid in enumerate(students):
            dx = shift if index == 0 else 300
            states = {"using_phone": state if index == 0 else second_state,
                      "sleeping": "suspected", "turning_back": "unknown", "eating": "not_detected"}
            people.append({
                "person_ref": f"synthetic_person_{index}",
                "person_bbox_xyxy_clipped": [20 + dx, 40, 180 + dx, 420],
                "bbox_xyxy": [20 + dx, 40, 180 + dx, 420],
                "behaviors": {name: {"state": value,
                                     "status": "IMAGE_ONLY" if value == "detected" else value.upper(),
                                     "decision": value, "model_confidence": .8}
                              for name, value in states.items()},
            })
            face_dx = dx if associated else 460
            faces.append({"track_id": None, "face_bbox": [65 + face_dx, 65, 110 + face_dx, 115],
                          "student_id": sid, "score": .82})
    return Observation(frame_id, captured_at, tuple(people), tuple(faces))


def detected_sequence():
    return [observation(index, timestamp) for index, timestamp in
            enumerate((100.0, 101.5, 103.0, 104.5), 1)]


@dataclass(frozen=True)
class PanelSample:
    observation_index: int
    render_only: bool
    frame_id: int
    captured_at: float
    page: int
    labels: tuple
    image_shape: tuple


@dataclass(frozen=True)
class PipelineTrace:
    results: tuple
    samples: tuple[PanelSample, ...]
    errors: tuple[str, ...]
    association_calls: int
    output: str


class _Clock:
    now = 198.5

    def __call__(self):
        return self.now


class _ReplayWorker:
    def __init__(self, kind, observations, clock):
        self.kind, self.observations, self.clock = kind, observations, clock
        self.requests, self.closed = [], False

    def process_frame(self, image_path, frame_id):
        index = len(self.requests)
        source = self.observations[index]
        if source.frame_id != frame_id:
            raise AssertionError("worker request frame ID differs from the replay")
        pixels = live.cv2.imread(str(image_path))
        if pixels is None:
            raise AssertionError("worker did not receive the real temporary JPEG")
        self.requests.append((frame_id, Path(image_path), pixels))
        # Completion intervals remain 1.5s, independently of source capture gaps.
        self.clock.now = 200.0 + 1.5 * index - (.3 if self.kind == "face" else 0.)
        result = {"frame_id": frame_id, "image": str(Path(image_path).resolve())}
        if self.kind == "face":
            result.update(face_count=len(source.faces), faces=copy.deepcopy(list(source.faces)))
        else:
            result.update(mode="IMAGE_ONLY", person_count=len(source.people),
                          people=copy.deepcopy(list(source.people)))
        return result

    def close(self):
        self.closed = True


class PipelineHarness:
    def run(self, observations, *, allow_error=False):
        """Replay AI input, then deliver every real result through main's queue.

        Scheduling is deterministic and sequential; this does not measure real
        inference latency, camera performance or concurrent thread behavior.
        """
        observations = tuple(observations)
        frame_queue, result_queue = queue.Queue(), queue.Queue()
        for source in observations:
            pixels = np.full((480, 640, 3), (20 + source.frame_id * 7) % 255, dtype=np.uint8)
            frame_queue.put((source.captured_at, pixels))
        clock = _Clock()
        workers = [_ReplayWorker(kind, observations, clock) for kind in ("face", "hybrid")]
        stdout, stderr = io.StringIO(), io.StringIO()
        samples = []
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(stdout))
            stack.enter_context(redirect_stderr(stderr))
            # Guards fail rather than starting real models or writing/sending data.
            for target in ("subprocess.Popen", "sqlite3.connect", "socket.socket.connect",
                           "socket.create_connection", "urllib.request.urlopen"):
                stack.enter_context(patch(target, side_effect=AssertionError(f"forbidden side effect: {target}")))
            with (patch.object(live, "Worker", side_effect=workers),
                  patch.object(live, "MAX_AI_FRAMES", len(observations)),
                  patch.object(live, "range", return_value=iter(o.frame_id for o in observations), create=True),
                  patch.object(live.time, "perf_counter", clock),
                  patch.object(live, "associate_v4", wraps=live.associate_v4) as associate):
                # Includes real JPEG I/O, trackers, association, duration, snapshot.
                live.ai_loop(frame_queue, result_queue, threading.Event())
            messages = []
            while not result_queue.empty():
                messages.append(result_queue.get_nowait())
            if not messages or messages[-1] != {"finished": True} or not all(w.closed for w in workers):
                raise AssertionError("AI loop did not close both replay workers")
            errors = tuple(m["error"] for m in messages if "error" in m)
            results = tuple(m for m in messages if "frame_id" in m)
            if errors and not allow_error:
                raise AssertionError(errors)
            for index, result in enumerate(results):
                snapshot = result["snapshot"]
                if (snapshot.frame_id, snapshot.captured_at) != (result["frame_id"], result["captured_at"]):
                    raise AssertionError("snapshot metadata mismatch")
                for worker in workers:
                    if not np.array_equal(snapshot.image, worker.requests[index][2]):
                        raise AssertionError("snapshot pixels differ from a worker's input JPEG")
            if any(request[1].exists() for worker in workers for request in worker.requests):
                raise AssertionError("temporary JPEG was not cleaned up")
            if results:
                samples = self._run_main(results, clock)
        return PipelineTrace(results, tuple(samples), errors, associate.call_count,
                             stdout.getvalue() + stderr.getvalue())

    @staticmethod
    def _run_main(results, clock):
        # One delivery and one cached render per observation. Multiple people are
        # viewed using actual N/P key handling rather than injecting a UI page.
        events = []
        for index, result in enumerate(results):
            multiple = len(result["behavior_frame"].people) > 1
            events.extend(((index, False, ord("n") if multiple else -1),
                           (index, True, ord("p") if multiple else -1)))
        index, repeated, _ = events[-1]
        events[-1] = (index, repeated, ord("q"))
        samples, state = [], {}
        draw_panel, put_text = live.draw_ai_panel, live.cv2.putText

        class QueueDeliveryThread:
            def __init__(self, target, args, daemon):
                state["result_queue"] = args[1]

            def start(self):
                pass

            def is_alive(self):
                return False

        def read_camera():
            event_index = state.get("tick", -1) + 1
            state["tick"] = event_index
            index, repeated, _ = events[event_index]
            clock.now = results[index]["completed_at"] + (.7 if repeated else .1)
            if not repeated:
                state["result_queue"].put(results[index])
            return True, np.full((480, 640, 3), 90, dtype=np.uint8)

        def record_panel(frame, result, **kwargs):
            with patch.object(live.cv2, "putText", wraps=put_text) as text:
                rendered = draw_panel(frame, result, **kwargs)
            labels = tuple(call.args[1] for call in text.call_args_list
                           if call.args[0].shape[1] == live.VIEW_WIDTH - live.LIVE_VIEW_WIDTH)
            index, repeated, _ = events[state["tick"]]
            samples.append(PanelSample(index, repeated, result["frame_id"], result["captured_at"],
                                       kwargs["page"], labels, rendered.shape))
            if not np.any(rendered[:, live.LIVE_VIEW_WIDTH:] != 35):
                raise AssertionError("real OpenCV text drawing did not reach the panel")
            return rendered

        capture = Mock()
        capture.isOpened.return_value = True
        capture.read.side_effect = read_camera
        with (patch.object(live.cv2, "VideoCapture", return_value=capture),
              patch.object(live.threading, "Thread", QueueDeliveryThread),
              patch.object(live.Path, "is_file", return_value=True),
              patch.object(live.cv2, "namedWindow"), patch.object(live.cv2, "resizeWindow"),
              patch.object(live.cv2, "imshow"), patch.object(live.cv2, "destroyAllWindows"),
              patch.object(live.cv2, "waitKey", side_effect=lambda delay: events[state["tick"]][2]),
              patch.object(live.time, "perf_counter", clock),
              patch.object(live, "draw_ai_panel", side_effect=record_panel)):
            live.main(show_diagnostics=True)
        capture.release.assert_called_once_with()
        return samples


def phone_value(result, person_index=0):
    return next(b.estimated_seconds for b in result["behavior_frame"].people[person_index].behaviors
                if b.name == "using_phone")


class TestCH20DurationPipeline(unittest.TestCase):
    def run_scenario(self, observations, **kwargs):
        trace = PipelineHarness().run(observations, **kwargs)
        self.assertEqual(trace.association_calls, len(observations))
        self.assertNotIn("WARNING", trace.output.upper())
        for sample in trace.samples:
            self.assertEqual(sample.image_shape, (720, 1280, 3))
            self.assertFalse(any("WARNING" in label.upper() for label in sample.labels))
        return trace

    def assert_phone(self, trace, expected, states=None, persons=None, students=None):
        self.assertEqual(len(trace.results), len(expected))
        states = states or ["detected"] * len(expected)
        persons = persons or ["P1"] * len(expected)
        students = students or ["SV001"] * len(expected)
        for index, (result, seconds) in enumerate(zip(trace.results, expected)):
            with self.subTest(observation=index + 1):
                if seconds is None:
                    self.assertIsNone(phone_value(result))
                else:
                    self.assertAlmostEqual(phone_value(result), seconds)
                samples = [s for s in trace.samples if s.observation_index == index and s.page == 0]
                self.assertTrue(samples, "main() never displayed this AI observation")
                for sample in samples:
                    value = "--" if seconds is None else f"~{seconds:.1f}s"
                    self.assertIn(f"Using Phone: {states[index]} {value}", sample.labels)
                    self.assertIn(f"Person Track: {persons[index]}", sample.labels)
                    self.assertIn(f"Student: {students[index]}", sample.labels)
                    self.assertEqual(sample.frame_id, result["frame_id"])
                    self.assertEqual(sample.captured_at, result["captured_at"])

    def test_exact_detected_sequence_passes_real_pipeline_and_main_panel(self):
        trace = self.run_scenario(detected_sequence())
        self.assert_phone(trace, [0.0, 1.5, 3.0, 4.5])
        self.assertEqual(len(trace.samples), 8)
        for index, result in enumerate(trace.results):
            self.assertEqual(result["captured_at"], (100.0, 101.5, 103.0, 104.5)[index])
            self.assertEqual(result["matches"][0]["student_id"], "SV001")
            self.assertEqual(result["matches"][0]["person_track_id"], "P1")
            self.assertEqual(result["matches"][0]["association_status"], "matched")
            fresh, cached = trace.samples[index * 2:index * 2 + 2]
            for sample in (fresh, cached):
                self.assertIn("Sleeping: suspected --", sample.labels)
                self.assertIn("Turning Back: unknown --", sample.labels)
                self.assertIn("Eating: not_detected --", sample.labels)
            self.assertEqual([s for s in fresh.labels if "Using Phone:" in s],
                             [s for s in cached.labels if "Using Phone:" in s])
            self.assertNotEqual([s for s in fresh.labels if "Result age:" in s],
                                [s for s in cached.labels if "Result age:" in s])

    def assert_negative_reset(self, state):
        observations = detected_sequence()
        observations[2] = observation(3, 103.0, state=state)
        trace = self.run_scenario(observations)
        self.assert_phone(trace, [0., 1.5, None, 0.], states=["detected", "detected", state, "detected"])

    def test_detected_to_suspected_resets(self):
        self.assert_negative_reset("suspected")

    def test_detected_to_unknown_resets(self):
        self.assert_negative_reset("unknown")

    def test_detected_to_not_detected_resets(self):
        self.assert_negative_reset("not_detected")

    def test_real_iou_tracker_changes_person_id_and_starts_new_episode(self):
        observations = detected_sequence()
        observations[2:] = [observation(3, 103., shift=260), observation(4, 104.5, shift=260)]
        self.assert_phone(self.run_scenario(observations), [0., 1.5, 0., 1.5], persons=["P1", "P1", "P2", "P2"])

    def test_student_change_resets_while_person_id_remains_p1(self):
        observations = detected_sequence()
        observations[2:] = [observation(3, 103., student="SV002"), observation(4, 104.5, student="SV002")]
        self.assert_phone(self.run_scenario(observations), [0., 1.5, 0., 1.5],
                          students=["SV001", "SV001", "SV002", "SV002"])

    def test_geometry_loss_of_association_breaks_duration(self):
        observations = detected_sequence()
        observations[2] = observation(3, 103., associated=False)
        trace = self.run_scenario(observations)
        self.assertEqual(trace.results[2]["matches"], [])
        self.assert_phone(trace, [0., 1.5, None, 0.], students=["SV001", "SV001", "Unknown", "SV001"])

    def test_unknown_student_does_not_inherit_even_when_stable_face_label_is_known(self):
        observations = detected_sequence()[:3] + [observation(4, 104.5, student="Unknown"), observation(5, 106.)]
        trace = self.run_scenario(observations)
        self.assertEqual(trace.results[3]["tracked_faces"][0][2], "SV001")
        self.assert_phone(trace, [0., 1.5, 3., None, 0.],
                          students=["SV001", "SV001", "SV001", "Unknown", "SV001"])

    def test_duplicate_student_claims_suppress_both_panel_pages_and_reset(self):
        observations = detected_sequence()
        observations[2] = observation(3, 103., second_student="SV001")
        trace = self.run_scenario(observations)
        duplicate = trace.results[2]
        self.assertEqual(len(duplicate["matches"]), 2)
        self.assertEqual([p.identity_status for p in duplicate["behavior_frame"].people], ["ambiguous_student"] * 2)
        self.assertIsNone(phone_value(duplicate, 1))
        self.assert_phone(trace, [0., 1.5, None, 0.], students=["SV001", "SV001", "Unknown", "SV001"])
        second_page = next(s for s in trace.samples if s.observation_index == 2 and s.page == 1)
        self.assertIn("Person Track: P2", second_page.labels)
        self.assertIn("Student: Unknown", second_page.labels)
        self.assertIn("Using Phone: detected --", second_page.labels)

    def test_capture_gap_over_five_resets_without_expiring_person_track(self):
        trace = self.run_scenario([observation(i, t) for i, t in enumerate((100., 101.5, 107., 108.5), 1)])
        self.assert_phone(trace, [0., 1.5, 0., 1.5])
        self.assertTrue(all(r["tracked_people"][0]["track_id"] == "P1" for r in trace.results))

    def test_exact_duplicate_frame_is_idempotent_through_snapshot_and_panel(self):
        trace = self.run_scenario([observation(1, 100.), observation(2, 101.5),
                                   observation(2, 101.5), observation(3, 103.)])
        self.assert_phone(trace, [0., 1.5, 1.5, 3.])
        self.assertIs(trace.results[1]["behavior_frame"], trace.results[2]["behavior_frame"])

    def test_duplicate_id_with_different_capture_time_fails_closed(self):
        trace = self.run_scenario([observation(1, 100.), observation(2, 101.5),
                                   observation(2, 103.)], allow_error=True)
        self.assert_phone(trace, [0., 1.5])
        self.assertEqual(len(trace.errors), 1)
        self.assertIn("snapshot and behavior metadata", trace.errors[0])

    def test_non_increasing_capture_time_clears_duration_through_main(self):
        for bad_time in (101.5, 101.0):
            with self.subTest(captured_at=bad_time):
                trace = self.run_scenario([observation(1, 100.), observation(2, 101.5),
                                           observation(3, bad_time), observation(4, 104.5)])
                self.assertFalse(trace.results[2]["behavior_frame"].timestamp_valid)
                self.assert_phone(trace, [0., 1.5, None, 0.])

    def test_briefly_absent_person_keeps_id_but_restarts_duration(self):
        observations = detected_sequence()
        observations[2] = observation(3, 103., present=False)
        trace = self.run_scenario(observations)
        self.assertEqual(trace.results[2]["tracked_people"], [])
        self.assertEqual(trace.results[2]["behavior_frame"].people, ())
        self.assertEqual(trace.results[3]["tracked_people"][0]["track_id"], "P1")
        self.assertEqual(phone_value(trace.results[3]), 0.)
        last = next(s for s in trace.samples if s.observation_index == 3)
        self.assertIn("Using Phone: detected ~0.0s", last.labels)

    def test_exact_five_gap_can_display_over_five_seconds_without_warning_or_actions(self):
        trace = self.run_scenario([observation(1, 100.), observation(2, 101.5), observation(3, 106.5)])
        self.assert_phone(trace, [0., 1.5, 6.5])
        self.assertTrue(all("warnings" not in result for result in trace.results))

    def test_distinct_students_have_independent_durations_on_actual_panel_pages(self):
        observations = [observation(1, 100., second_student="SV002", second_state="not_detected"),
                        observation(2, 101.5, second_student="SV002"),
                        observation(3, 103., second_student="SV002")]
        trace = self.run_scenario(observations)
        self.assert_phone(trace, [0., 1.5, 3.])
        self.assertEqual(phone_value(trace.results[1], 1), 0.)
        self.assertEqual(phone_value(trace.results[2], 1), 1.5)
        last = next(s for s in trace.samples if s.observation_index == 2 and s.page == 1)
        self.assertIn("Student: SV002", last.labels)
        self.assertIn("Person Track: P2", last.labels)
        self.assertIn("Using Phone: detected ~1.5s", last.labels)


if __name__ == "__main__":
    unittest.main()
