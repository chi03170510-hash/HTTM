# HTTM Round 3: independent video and AI

The current `BehaviorDetector` and legacy adapter use the official Round 3
`ai/behavior_detection/models/best.pt`: 0 `sleeping`, 1 `using_phone`.
The adapter validates these two classes. This launcher uses CPU inference and
does not select experimental V1/V2 weights or add `raising_hand`.

Run from the repository root with the existing Python environment:

```powershell
.\.venv\Scripts\python.exe -m ai.behavior_detection.run_httm
```

The menu retains Windows file selection, laptop webcam 0, and DroidCam 1.
It saves a full-rate annotated video and sampled detection JSONL to `results/`.
Its existing 15-second maximum duration remains configurable through
`MAX_DURATION` in `run_httm.py`. Press Q, Esc, or Ctrl+C to stop early.

## Direct execution

```powershell
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_video --source 0
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_video --source 1 --source-type droidcam
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_video --source "lesson.mp4" --no-display --events "results/lesson_events.jsonl" --output "results/lesson_overlay.mp4"
```

The main thread reads, paces, draws, records, and displays frames. A single AI
worker processes the newest copied frame at a target of **1 FPS**; intermediate
frames replace the pending slot. An inference lasting over one second delays
the next inference, without a catch-up queue. No shared lock is held during
inference or event I/O. JSONL/stdout result output belongs to the AI thread.

`--sample-fps` is retained as the AI-rate option (default 1). File playback and
output recording use the native source FPS, independently of the AI rate.
If FPS metadata is missing/invalid, playback uses a 30 FPS fallback; the real
original rate cannot be recovered from missing metadata. Live sources follow
capture timing. OpenCV GUI calls stay on the main thread for Windows.

The preview window is resizable and opens at **960x720** by default. Set
`--display-width` and `--display-height` to positive pixel dimensions in either
`run_video` or `run_ch20`. OpenCV scales the complete annotated image, including
boxes, Student ID, behavior, and duration. The display size does not resize AI
inputs or recorded frames; a 640x480 camera frame stays 640x480. Window sizing
is skipped with `--no-display`.

The HUD displays measured `VIDEO FPS` and `AI FPS`, plus `AI TARGET`, `MODEL`,
and `SOURCE`. FPS uses recent monotonic completion times, not configured values
or the reciprocal of inference latency. The initial rate is zero until two
operations have completed; AI FPS can be below 1 on slow hardware. In headless
mode VIDEO FPS measures processed video frames; it does not measure a display.
The final JSON summary includes measured whole-run rates and mean AI latency.

Overlay results retain their source frame index/timestamp. `--detection-ttl`
defaults to **1.5 seconds**, measured from input capture and source timeline,
including inference latency. Expired or mismatched results are hidden, even
if they have just completed. Without tracking, a fresh box can still lag a
moving person. Historical detection events keep their original metadata and
existing schema; events are written once per inference, never once per display
frame. These are model detections, not confirmed behavior violations.

Q/EOF stops accepting frames and releases capture, writer, and preview. The
worker is then joined before closing its event stream or inference pipeline.
An in-flight native YOLO call must finish; Python threads cannot cancel it
safely. This can delay process exit, but no pending-frame backlog is drained.

Without `--ch20`, InsightFace, Student ID mapping, tracking, and the five-second
timer remain outside the normal YOLO-only video path. The optional CH-20 path
below connects these components for the student overlay. SQLite violation
logging remains in the separate CH programs; the new overlay path does not
write new SQLite violations or treat sparse detections as proof of continuity.
The `--pipeline unified` option remains a separate prototype and requires its
missing pipeline module/assets; it is not the Round 3 default.

## CH-20: Student ID, Track ID, behavior and duration

Direct laptop webcam demo:

```powershell
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_ch20 --source 0 --warmup --max-duration 30
```

The same mode through the existing Windows file-picker / webcam / DroidCam menu:

```powershell
.\.venv\Scripts\python.exe -m ai.behavior_detection.run_httm --ch20
```

`run_ch20` delegates to `run_video --ch20`; source/output/events/debug options
are shared. Ordinary menu/GUI launches without this flag remain YOLO-only.
The active IDE file `test_yolo_behavior_ch15.py` is a separate legacy program
with different confidence/frame-sampling settings, not this entrypoint.

CH-20 keeps Round 3 `best.pt`, `sleeping`/`using_phone`, confidence 0.40, CPU
inference, and a fixed AI target of 1 FPS. Video playback/recording still run
independently at source FPS. InsightFace detection and recognition run on the
same copied frame as YOLO, in the existing AI worker. It uses only existing
`buffalo_l` model files (`det_10g.onnx`, `w600k_r50.onnx`) under
`~/.insightface/models/buffalo_l`, and the existing `data/students_faces.db`.
Missing resources produce a clear error; no model is auto-downloaded and no
Student ID is invented. Override locations with `--face-model-root` (the
`.insightface` root) and `--student-db` if necessary.

Production `student_tracking.py` contains the existing CH-18 geometry/behavior
association and CH-20M appearance/identity helpers, extracted without changing
their algorithms/constants. Old test scripts stay intact. Importing the
production module does not import test scripts or open their log files.
Unknown identities remain `Unknown`; competing claims to the same enrolled ID
are suppressed. Behavior boxes need an unambiguous face overlap/geometry match;
small phone-only boxes far from a face remain unmatched rather than being
assigned to the nearest student by guesswork.

The main thread consumes immutable student observations and owns the existing
`BehaviorStateManager` and `BehaviorTimer`. It draws each student's **face box**
and a nearby panel:

```text
Student ID: 001
Track ID: 1
Behavior: Using Phone
Duration: 3.8s
```

These are example values, shown only when recognition and behavior confirmation
actually support them. Three distinct AI observations are required for behavior
confirmation; painting the same cached result many times never supplies extra
hits. Student identity retains its separate existing confirmation policy.

Duration is `BehaviorTimer.elapsed`, advanced with monotonic time while the
confirmed state has a fresh evidence lease. It starts when confirmation becomes
active, not at the assumed beginning of the physical action. No duration is
shown for normal/suspected/paused states (`Duration: --`). The existing 0.5s
per-update cap prevents a stalled display from adding a large unsupported jump.
The CH-20 state instance uses `--detection-ttl` (default 1.5s), because the legacy
0.8s behavior timeout cannot span 1 FPS; legacy defaults are unchanged.

New negative evidence, lost faces, expired snapshots, source changes, fresh
identity contradictions, and behavior changes reset the affected episode.
Each Track ID owns its own state/timer; identity changes cannot inherit the
previous student's elapsed time. A new behavior needs its own three observations.
At five elapsed seconds the existing timer raises its once-per-episode flag;
the overlay says **Possible warning**, not a verified real-world violation.
Sparse 1 FPS evidence does not prove continuous action between samples.

CH-20 implements this display/integration stage only. It does not connect
SQLite warning writes or the backend web service, change YOLO accuracy, add
CH-21 functionality, or serialize student identity into the existing detection
JSONL. Legacy CH-19/CH-20M logging remains available in its own scripts.
Slow combined face/YOLO inference can reduce actual AI FPS; stale labels/timers
are hidden/reset instead of displaying fabricated durations. Tracking can still
fail with occlusion, small faces (the inherited face-size minimum is 48 pixels),
or similar appearances. Dense scenes can also leave insufficient space for all
label panels. Full integration details: [CH-20 report](../../../reports/ch20_integration_20261104.md).

## Offline regression tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s ai/behavior_detection/integration/tests -v
```

Tests exercise latest-frame replacement, slow inference, rate limiting,
timestamp/TTL checks, result isolation, native-rate recording, main-thread GUI
dispatch, Q/error cleanup, and the three-source menu/file dialog using mocks
and generated video. Camera hardware and the real Windows preview need a
manual check. CPU inference on generated video checks runtime compatibility,
not recognition accuracy on classroom footage.

## Diagnose empty using_phone results

Keep the official class thresholds at 0.40. Enable debug output to inspect the
YOLO candidates returned at `conf=0.10`, after YOLO NMS and before HTTM's 0.40
filter. Diagnostics go to **stderr**; detection stdout/`--events` keep their
existing schema. Nothing is inferred about candidates below 0.10 or removed by
YOLO NMS. Text debugging alone does not save webcam images or change the default
1 FPS. Image evidence requires the additional `--save-debug-frames` flag below.

```powershell
New-Item -ItemType Directory -Force .\results | Out-Null
$debugStamp = Get-Date -Format "yyyyMMdd_HHmmss"
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_video --source 0 --source-type webcam --debug-detections --warmup --max-duration 30 --events ".\results\phone_events_$debugStamp.jsonl" 2> ".\results\phone_debug_$debugStamp.jsonl"
```

`--warmup` is optional: it runs a dummy BGR uint8 image of 640x480, matching the
recorded laptop webcam dimensions, before capture opens. Startup still takes
time, but predictor setup/internal warm-up no longer consumes the webcam's
capture timeline. Its results are discarded, never logged as real events or
counted toward AI FPS. Other input sizes can still incur additional setup.

Interpret the diagnostic records:

- `yolo_candidates`: original frame metadata, BGR input shape/dtype/range,
  requested `imgsz`, YOLO original image shape, wall time, YOLO's profiled stage
  times, and every returned candidate's class/confidence/XYXY source-pixel box.
  `cold_start` and `unprofiled_ms` help identify predictor setup overhead; the
  latter is not a precise breakdown of setup versus internal warm-up.
- `decision=below_class_threshold`: a candidate exists, but is below 0.40.
  `decision=accepted`: it passed the official threshold.
- `empty_reason=no_yolo_candidates_at_conf_0.10`: YOLO returned no candidate
  at this floor, before HTTM's filter. `all_candidates_filtered` is different.
- `adapter_detections`: counts and reasons for additional class ID/name
  filtering. `adapter_error` reports invalid inputs or inference exceptions.
- `ai_sample_timing`: actual source/start gaps, skipped source frame count,
  processing time, input age, and result output time. This distinguishes slow
  inference from slow stdout/event I/O. Skipped frames are intentional with a
  latest-frame 1 FPS worker; a short phone pose between samples can be missed.

The checked code passes unannotated BGR uint8 to Ultralytics. Ultralytics
converts to RGB, normalizes, and letterboxes internally, then rescales boxes to
the original image. Do not add an extra RGB conversion or manual box scaling.
An empty final event cannot by itself identify low confidence versus absent
candidates, nor prove a camera/model accuracy cause. Hold a pose for several
samples while collecting debug output to distinguish these cases.

## Save original frames and all YOLO candidate boxes

Run a 30-second laptop webcam session without changing Round 3 or its 0.40
thresholds:

```powershell
.\.venv\Scripts\python.exe -m ai.behavior_detection.integration.run_video `
  --source 0 --source-type webcam --sample-fps 1 `
  --debug-detections --save-debug-frames --warmup --max-duration 30
```

`--save-debug-frames` requires `--debug-detections`. Each run creates a unique
directory under `results/debug_frames/` (override the parent with
`--debug-frames-dir`). The console prints the directory and the final stdout
summary includes `debug_report`. The default output directory is Git-ignored.

```text
results/debug_frames/<unique_session>/
  raw/          # exact sampled BGR inputs, stored as lossless PNG, without HUD/boxes
  candidates/   # the same frames, with every returned YOLO candidate at conf=0.10
  frames.jsonl  # frame metadata, paired image paths, candidates, official event result
  summary.json  # totals, per-class counts, box widths, timing, and run status
```

Each pair includes a sequence number, `frame_index`, and timestamp in its
filename. Originals retain every input pixel and their original dimensions.
Candidate images use **green** for accepted candidates and **orange** for
filtered candidates. Numbered boxes remain in original pixel coordinates.
A footer outside the source image lists class, confidence, filter decision,
box width/height and XYXY coordinates, plus the source/frame/timestamp. The
footer can widen the canvas with blank margin; the original image is never
resized or shifted. This avoids covering a small phone region with labels.
Even frames with no candidates, or an inference error, receive an image pair.

PNG encoding, annotation, manifest writes, and per-frame summary updates run
only in the existing AI worker, outside its shared lock. There is no new
thread or image queue. Directory initialization happens before capture; final
report completion happens after the worker joins. Slow storage can lower the
achievable AI rate, but it never makes the display loop wait for image I/O.
Warm-up dummy images are excluded. Normal runs and text-debug-only runs do not
create these artifacts.

`frames.jsonl` and `summary.json` are written directly as UTF-8, independently
of Windows PowerShell's formatting of redirected stderr. No existing run is
overwritten. Each completed image pair is recorded immediately; summary files
are atomically replaced within the new session. Interrupted/incomplete runs
retain their partial evidence and status.

Important report fields:

- `counts.processed_frames`, `successful_frames`, and `inference_error_frames`.
- `frames_without_yolo_candidates`: successful frames with no candidate at 0.10.
- `frames_without_accepted_detections`: successful frames with no official event
  detection at 0.40, including frames whose candidates were all filtered.
- `low_confidence_candidates` counts boxes; `frames_with_low_confidence_candidates`
  counts frames. A frame with both low and accepted boxes contributes to both
  categories, so these counts are not a disjoint partition of frames.
- `accepted_detections` counts boxes that actually survive the adapter, with
  `frames_with_accepted_detections` counting the corresponding frames.
- `per_class.using_phone` (and `sleeping`) gives class-specific counts and
  `bbox_width_px` min/max/mean in original source pixels.
- `mean_image_write_ms` and `runtime` show image I/O cost, measured video/AI FPS,
  and the unchanged AI target.

Review each annotated image against its raw partner before deciding whether
the small phone boxes localize the phone or another region. Confidence and
box size alone do not establish correctness. Images are evidence for review,
not automatic ground-truth labels or authorization to fine-tune the model.

## Hybrid

```powershell
python -m ai.behavior_detection.integration.hybrid_behavior_inference --source "lesson.mp4" --device 0
```

Hybrid uses V2 by default. Pass `--behavior-version v1` to select V1. YOLO11s-Pose, ByteTrack, and the existing temporal/geometry rules remain as implemented. Person/phone and pose auxiliary weights are separate from the V2 artifact; see the integration guide. Hybrid dispositions are not ground-truth events or a validated accuracy result.

Annotation completeness for V2 Validation remains UNVERIFIED. There is no independent Test; the current status is `TEST_NOT_AVAILABLE`.
