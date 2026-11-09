"""CH-20: no camera/GPU required; injected evidence never changes real models/DBs."""

import ast
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np

from ai.behavior_detection.integration import run_ch20, run_video, student_tracking
from ai.behavior_detection import run_httm
from ai.behavior_detection.integration.event_schema import DetectionEvent, FrameMetadata, FrameResult
from ai.behavior_detection.integration.realtime import InferenceSnapshot
from ai.behavior_detection.integration.student_overlay import draw_student_overlays, student_label_lines
from ai.behavior_detection.integration.student_pipeline import (
    StudentFrameResult, StudentInferenceProcessor, StudentObservation,
    StudentOverlay, StudentOverlayController,
)


FRAME = np.zeros((480, 640, 3), dtype=np.uint8)
BOXES = {1: (80, 80, 140, 140), 2: (400, 80, 460, 140)}


def meta(now, source='fixture'):
    return FrameMetadata(source, int(round(now * 30)), int(round(now * 1000)), 640, 480)


def observation(track=1, behavior='using_phone', student=None, confidence=.8, reset=False):
    return StudentObservation(track, BOXES[track], student or f'{track:03d}', behavior,
                              confidence if behavior is not None else 0, reset)


def snapshot(now, observations, source='fixture', status='ok'):
    metadata = meta(now, source)
    detections = tuple(DetectionEvent(o.behavior, 0 if o.behavior == 'sleeping' else 1,
                                     o.confidence, o.bbox, metadata.frame_index,
                                     metadata.timestamp_ms)
                       for o in observations if o.behavior is not None)
    result = StudentFrameResult(FrameResult(status, metadata, detections), tuple(observations))
    return InferenceSnapshot(result, metadata, now, now, now)


def face(bbox, vector):
    return SimpleNamespace(bbox=np.array(bbox), embedding=np.asarray(vector, dtype=np.float32), det_score=.99)


def fake_recognition(vector):
    sid = '001' if vector[0] > vector[1] else '002'
    return {'status': 'MATCH', 'student_id': sid, 'similarity': .99}


class ControllerTests(unittest.TestCase):
    def advance(self, controller, snap, now):
        return controller.advance(snap, meta(now), now)

    def confirm(self, controller, observations):
        for second in range(3):
            self.advance(controller, snapshot(second, observations), second)

    def test_cached_frame_never_counts_as_multiple_confirmations(self):
        controller = StudentOverlayController()
        snap = snapshot(0, [observation()])
        for index in range(30):
            views = self.advance(controller, snap, index / 30)
            self.assertIsNone(views[0].duration)
        self.assertEqual(controller.states.get_state(1).consecutive_hits, 1)
        self.assertEqual(controller.states.get_state(1).status, 'SUSPECTED')
        self.assertEqual(controller.timer.timers, {})

    def test_two_students_have_independent_real_time_durations(self):
        controller = StudentOverlayController()
        for index in range(59):
            now = index / 10
            if index % 10 == 0:
                snap = snapshot(now, [observation(1), observation(2, 'sleeping' if now >= 2 else None)])
            views = self.advance(controller, snap, now)
        self.assertEqual([v.student_id for v in views], ['001', '002'])
        self.assertEqual([v.behavior for v in views], ['using_phone', 'sleeping'])
        self.assertAlmostEqual(views[0].duration, 3.8)
        self.assertAlmostEqual(views[1].duration, 1.8)
        self.assertEqual(controller.timer.warning_seconds, 5)

    def test_display_fps_does_not_determine_duration(self):
        durations = []
        for fps in [10, 30, 60]:
            controller = StudentOverlayController()
            for index in range(fps * 6 + 1):
                now = index / fps
                if index % fps == 0:
                    snap = snapshot(now, [observation()])
                views = self.advance(controller, snap, now)
            durations.append(views[0].duration)
        for duration in durations:
            self.assertAlmostEqual(duration, 4.0)

    def test_explicit_behavior_loss_resets_only_affected_track(self):
        controller = StudentOverlayController()
        self.confirm(controller, [observation(1), observation(2, 'sleeping')])
        self.advance(controller, snapshot(3, [observation(1), observation(2, 'sleeping')]), 3)
        views = self.advance(controller, snapshot(4, [observation(1, None), observation(2, 'sleeping')]), 4)
        self.assertIsNone(views[0].duration)
        self.assertIsNone(controller.timer.get_timer(1))
        self.assertAlmostEqual(views[1].duration, 1.0)  # 0.5 cap on each sparse UI tick
        views = self.advance(controller, snapshot(5, [observation(1), observation(2, 'sleeping')]), 5)
        self.assertIsNone(views[0].duration)
        self.assertEqual(controller.states.get_state(1).status, 'SUSPECTED')

    def test_face_loss_new_track_and_expired_snapshot_never_reuse_elapsed(self):
        controller = StudentOverlayController()
        self.confirm(controller, [observation(1), observation(2, 'sleeping')])
        views = self.advance(controller, snapshot(3, [observation(2, 'sleeping')]), 3)
        self.assertEqual([v.track_id for v in views], [2])
        self.assertIsNone(controller.timer.get_timer(1))
        snap = snapshot(4, [observation(1), observation(2, 'sleeping')])
        self.assertIsNone(self.advance(controller, snap, 4)[0].duration)
        self.assertEqual(self.advance(controller, snap, 6), ())
        self.assertEqual(controller.timer.timers, {})
        self.assertEqual(controller.last_warnings, ())

    def test_identity_change_does_not_inherit_duration(self):
        for explicit_reset in [False, True]:
            controller = StudentOverlayController()
            self.confirm(controller, [observation()])
            views = self.advance(controller, snapshot(3, [observation(student='002', reset=explicit_reset)]), 3)
            self.assertEqual(views[0].student_id, '002')
            self.assertIsNone(views[0].duration)
            self.assertEqual(controller.timer.timers, {})

    def test_source_change_and_inference_error_clear_old_episode(self):
        controller = StudentOverlayController()
        self.confirm(controller, [observation()])
        snap = snapshot(3, [observation()], source='other')
        views = controller.advance(snap, meta(3, 'other'), 3)
        self.assertIsNone(views[0].duration)
        snap = snapshot(4, [observation()], source='other', status='inference_error')
        self.assertIsNone(controller.advance(snap, meta(4, 'other'), 4)[0].duration)

    def test_warning_threshold_stays_five_seconds_and_emits_once(self):
        controller = StudentOverlayController()
        warnings = []
        for index in range(37):
            now = index / 4
            if index % 4 == 0:
                snap = snapshot(now, [observation()])
            views = self.advance(controller, snap, now)
            warnings.extend(controller.last_warnings)
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0]['elapsed'], 5.0)
        self.assertEqual(warnings[0]['timestamp'], 7.0)  # confirmation at 2s, then five timed seconds
        self.assertTrue(views[0].warned)
        self.assertEqual(self.advance(controller, snap, 11), ())
        self.assertEqual(controller.last_warnings, ())

    def test_behavior_switch_needs_new_confirmation_and_new_timer(self):
        controller = StudentOverlayController()
        self.confirm(controller, [observation()])
        for second in [3, 4]:
            views = self.advance(controller, snapshot(second, [observation(behavior='sleeping')]), second)
            self.assertIsNone(views[0].duration)
        views = self.advance(controller, snapshot(5, [observation(behavior='sleeping')]), 5)
        self.assertEqual(views[0].behavior, 'sleeping')
        self.assertEqual(views[0].duration, 0)

    def test_confidence_below_point_four_never_confirms(self):
        controller = StudentOverlayController()
        for second in range(10):
            views = self.advance(controller, snapshot(second, [observation(confidence=.399)]), second)
        self.assertIsNone(views[0].duration)
        self.assertEqual(controller.timer.timers, {})


class ProcessingTests(unittest.TestCase):
    def processor(self, faces, detections):
        app = SimpleNamespace(get=lambda _: faces)
        adapter = SimpleNamespace(process=lambda _, metadata: FrameResult('ok', metadata, tuple(
            DetectionEvent(label, 0 if label == 'sleeping' else 1, .8, bbox,
                           metadata.frame_index, metadata.timestamp_ms) for label, bbox in detections)))
        recognizer = SimpleNamespace(recognize=fake_recognition)
        return StudentInferenceProcessor(adapter, app, recognizer)

    def test_same_frame_geometry_maps_behaviors_to_correct_students(self):
        faces = [face(BOXES[1], [1,0]), face(BOXES[2], [0,1])]
        processor = self.processor(faces, [('using_phone',(40,40,240,430)), ('sleeping',(350,40,550,430))])
        for second in range(4):
            result = processor.process(FRAME, meta(second))
        self.assertEqual([(o.track_id,o.student_id,o.behavior) for o in result.students],
                         [(1,'001','using_phone'),(2,'002','sleeping')])
        self.assertEqual(result.to_dict(), result.behavior_result.to_dict())
        self.assertEqual(set(result.to_dict()), {'status','metadata','detections','error'})
        self.assertEqual(result.unmatched_count, 0)

    def test_ambiguous_box_is_not_arbitrarily_assigned_to_a_student(self):
        faces = [face(BOXES[1], [1,0]), face(BOXES[2], [0,1])]
        processor = self.processor(faces, [('using_phone',(0,0,640,480))])
        for second in range(2): result = processor.process(FRAME, meta(second))
        self.assertEqual([o.behavior for o in result.students], [None,None])
        self.assertEqual(result.unmatched_count, 1)

    def test_duplicate_student_ids_are_suppressed_without_merging_track_ids(self):
        processor = self.processor([face(BOXES[1],[1,0]),face(BOXES[2],[0,1])], [])
        processor.recognizer.recognize=lambda _: {'status':'MATCH','student_id':'001','similarity':.99}
        for second in range(4): result = processor.process(FRAME,meta(second))
        self.assertEqual([o.track_id for o in result.students], [1,2])
        self.assertEqual([o.student_id for o in result.students], [None,None])

    def test_crossing_faces_use_embedding_gate_not_only_nearest_position(self):
        tracker = student_tracking.StableFaceTracker()
        faces = [{'bbox':(100,80,160,140),'embedding':np.array([1.,0.])},
                 {'bbox':(200,80,260,140),'embedding':np.array([0.,1.])}]
        tracker.update(faces, 0)
        tracker.update(faces, 1)
        swapped = [dict(faces[0],bbox=faces[1]['bbox']),dict(faces[1],bbox=faces[0]['bbox'])]
        tracks, assignments = tracker.update(swapped, 2)
        self.assertEqual(tracks[1], (200,80,260,140))
        self.assertEqual(tracks[2], (100,80,160,140))
        np.testing.assert_array_equal(assignments[1]['embedding'], [1.,0.])

    def test_fresh_identity_contradiction_resets_episode_and_old_id_grace(self):
        processor = self.processor([face(BOXES[1],[1,0])], [('using_phone',(40,40,240,430))])
        for second in range(4): result = processor.process(FRAME,meta(second))
        self.assertEqual(result.students[0].student_id, '001')
        processor.recognizer.recognize=lambda _: {'status':'MATCH','student_id':'002','similarity':.99}
        result = processor.process(FRAME,meta(4))
        self.assertTrue(result.students[0].reset_episode)
        self.assertIsNone(result.students[0].student_id)

    def test_missing_local_assets_fail_clearly_before_any_download(self):
        with tempfile.TemporaryDirectory(prefix='httm_ch20_assets_') as folder:
            with self.assertRaisesRegex(FileNotFoundError,'existing InsightFace'):
                StudentInferenceProcessor.from_local(None,face_root=folder)

    def test_reused_helpers_match_original_ast_and_have_no_test_imports(self):
        source = Path(student_tracking.__file__).read_text(encoding='utf-8')
        actual = {n.name: n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
        base=Path(student_tracking.__file__).parents[1]
        groups={
            'test_behavior_tracking_ch20m.py':['cosine_similarity','center','box_iou','StableFaceTracker','StudentIdentityManager','ambiguous_student_ids','safe_student_id','detect_faces'],
            'test_behavior_tracking_ch18.py':['area','intersection','iou','containment','face_overlap','face_center_inside','normalized_distance','remove_duplicates','match_score','associate_behaviors'],
        }
        for filename,names in groups.items():
            original={n.name:n for n in ast.parse((base/filename).read_text(encoding='utf-8-sig')).body
                      if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
            for name in names:
                self.assertEqual(ast.dump(actual[name]),ast.dump(original[name]),name)
        self.assertNotIn('from test_',source)


class OverlayAndEntrypointTests(unittest.TestCase):
    def test_four_fields_and_missing_duration_placeholder(self):
        view=StudentOverlay(1,BOXES[1],'001','using_phone','CONFIRMED',.8,3.8,False)
        self.assertEqual(student_label_lines(view),('Student ID: 001','Track ID: 1',
                                                   'Behavior: Using Phone','Duration: 3.8s'))
        no_duration=replace(view,behavior=None,status='NORMAL',duration=None)
        self.assertEqual(student_label_lines(no_duration)[-1],'Duration: --')
        unknown=replace(no_duration,student_id=None)
        self.assertEqual(student_label_lines(unknown)[0],'Student ID: Unknown')

    def test_overlay_updates_both_tracks_from_their_own_timer_values(self):
        first=StudentOverlay(1,BOXES[1],'001','using_phone','CONFIRMED',.8,3.8,False)
        second=StudentOverlay(2,BOXES[2],'002','sleeping','CONFIRMED',.9,1.2,False)
        with patch.object(cv2,'putText',wraps=cv2.putText) as draw:
            original=draw_student_overlays(FRAME.copy(),[first,second])
        labels=[c.args[1] for c in draw.call_args_list]
        for text in ['Student ID: 001','Track ID: 1','Duration: 3.8s',
                     'Student ID: 002','Track ID: 2','Duration: 1.2s']:
            self.assertIn(text,labels)
        updated=draw_student_overlays(FRAME.copy(),[replace(first,duration=3.9),second])
        self.assertTrue(np.any(updated!=original))

    def test_dedicated_entrypoint_and_existing_three_source_menu_enable_ch20(self):
        with patch.object(run_ch20,'run_video') as run:
            run_ch20.main(['--source','0','--no-display'])
        self.assertIn('--ch20',run.call_args.args[0])
        with patch.object(run_httm,'create_output_paths',return_value=(Path('fixture.mp4'),Path('fixture.jsonl'))), \
                patch.object(run_httm.subprocess,'run') as run,redirect_stdout(io.StringIO()):
            run_httm.run_pipeline('1','droidcam',ch20=True)
        self.assertIn('--ch20',run.call_args.args[0])
        self.assertIn('--warmup',run.call_args.args[0])
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index('--sample-fps')+1],'1')

    def test_ch20_rejects_experimental_pipeline_or_changed_ai_target(self):
        for args in [['--pipeline','unified'],['--sample-fps','2']]:
            with patch('sys.stderr',new_callable=io.StringIO), \
                    patch.object(run_video.DetectorAdapter,'from_default_checkpoint') as load:
                with self.assertRaises(SystemExit): run_ch20.main(['--source','0']+args)
            load.assert_not_called()

    def test_shared_runner_processes_ch20_in_worker_and_updates_overlay_on_main_thread(self):
        with tempfile.TemporaryDirectory(prefix='httm_ch20_playback_') as folder:
            video=Path(folder)/'fixture.avi'
            writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),30,(640,480))
            self.assertTrue(writer.isOpened())
            for _ in range(96): writer.write(FRAME)
            writer.release()
            main_thread=threading.get_ident(); process_threads=[]; draw_threads=[]; durations=[]
            class Processor:
                def process(self,frame,metadata):
                    process_threads.append(threading.get_ident());time.sleep(.1)
                    return StudentFrameResult(FrameResult('ok',metadata),
                                              (observation(1),observation(2,'sleeping')))
                close=Mock()
            real_draw=draw_student_overlays
            def draw(frame,views,*args,**kwargs):
                draw_threads.append(threading.get_ident())
                durations.extend(v.duration for v in views if v.duration is not None)
                return real_draw(frame,views,*args,**kwargs)
            with patch.object(run_video.DetectorAdapter,'from_default_checkpoint'), \
                    patch.object(StudentInferenceProcessor,'from_local',return_value=Processor()), \
                    patch('ai.behavior_detection.integration.student_overlay.draw_student_overlays',side_effect=draw), \
                    redirect_stdout(io.StringIO()) as stdout:
                run_ch20.main(['--source',str(video),'--no-display'])
            stats=json.loads(stdout.getvalue().splitlines()[-1])
            self.assertTrue(stats['ch20'])
            self.assertGreater(stats['video_fps'],24)
            self.assertEqual(stats['ai_target_fps'],1)
            self.assertEqual(set(draw_threads),{main_thread})
            self.assertNotIn(main_thread,process_threads)
            self.assertTrue(any(d>0 for d in durations))
            Processor.close.assert_called_once()
            self.assertFalse(any(t.name=='HTTM-AI' for t in threading.enumerate()))


if __name__=='__main__':
    unittest.main()
