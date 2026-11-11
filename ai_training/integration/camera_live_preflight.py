"""Offline Camera Live inventory. Standard library only; no camera/model imports.

PASS checks prerequisites, not model accuracy, class mapping or native DLL/GUI
execution. Hashes are measured fingerprints, not an approved authenticity list.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import zipfile


SCHEMA = "httm_camera_live_preflight_v1"
ROOT = Path(__file__).resolve().parents[2]
FLAGS = ("HTTM_PERSON_SNAPSHOT_OVERLAY", "HTTM_IDENTITY_SHADOW", "HTTM_PERSON_DEBUG",
         "HTTM_FACE_DEBUG", "HTTM_PERSON_DEBUG_SNAPSHOTS")
KNOWN_ENV = set(FLAGS) | {"HTTM_BEHAVIOR_CHECKPOINT", "HTTM_CAMERA_MAX_OBSERVATIONS",
    "HTTM_FACE_TRACKER_TRIAL_MAX_AGE", "HTTM_PERSON_TRIAL_MIN_CONF", "HTTM_IDENTITY_SHADOW_DIR",
    "HTTM_PERSON_DEBUG_DIR", "HTTM_PERSON_DEBUG_MAX_FRAMES", "HTTM_PERSON_DEBUG_MAX_BYTES"}
PACKAGES = {"face": {"numpy": ("numpy",), "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"),
                     "insightface": ("insightface",), "onnxruntime": ("onnxruntime", "onnxruntime-gpu"),
                     "onnx": ("onnx",), "scipy": ("scipy",), "skimage": ("scikit-image",)},
            "hybrid": {"numpy": ("numpy",), "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"),
                       "ultralytics": ("ultralytics",), "torch": ("torch",), "yaml": ("pyyaml",),
                       "mediapipe": ("mediapipe",)}}

# Executed with the exact worker interpreter. find_spec is top-level only: no
# parent package import, YOLO(), FaceAnalysis(), ONNX session or camera call.
PROBE = r'''
import ast,importlib.metadata as md,importlib.util,json,sys,tempfile
from pathlib import Path
versions={}
for d in md.distributions():
    name=d.metadata.get("Name", "").lower().replace("_", "-")
    if name:versions[name]=d.version
modules={}
for name in ("numpy","cv2","insightface","onnxruntime","onnx","scipy","skimage","ultralytics","torch","yaml","mediapipe"):
    try:
        spec=importlib.util.find_spec(name)
        modules[name]=str(spec.origin) if spec is not None else None
    except (ImportError,ValueError,AttributeError):modules[name]=None
headless=None
if modules.get("cv2"):
    version=Path(modules["cv2"]).parent/"version.py"
    if version.is_file():
        for node in ast.parse(version.read_text(encoding="utf-8")).body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="headless" for t in node.targets):
                try:headless=ast.literal_eval(node.value)
                except ValueError:pass
print(json.dumps({"executable":sys.executable,"prefix":sys.prefix,"base_prefix":sys.base_prefix,
 "version":list(sys.version_info[:3]),"home":str(Path.home()),"temp_dir":tempfile.gettempdir(),
 "versions":versions,"modules":modules,"opencv_headless":headless}))
'''


def probe_interpreter(executable):
    result = subprocess.run([str(executable), "-B", "-c", PROBE], capture_output=True,
                            encoding="utf-8", errors="replace", timeout=20, check=False)
    if result.returncode:
        raise ValueError(f"Interpreter probe failed (exit {result.returncode}): {result.stderr.strip()[:400]}")
    data = json.loads(result.stdout)
    if not isinstance(data, dict) or not isinstance(data.get("modules"), dict) or not isinstance(data.get("versions"), dict):
        raise ValueError("Invalid interpreter probe response")
    return data


def assignments(path):
    values = {}
    for node in ast.parse(Path(path).read_text(encoding="utf-8-sig")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    try:
                        values[target.id] = ast.literal_eval(node.value)
                    except (ValueError, TypeError):
                        pass
    return values


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def writable_directory(target):
    """Test only a unique owned temporary marker; never create a session dir."""
    target = Path(target).resolve()
    parent = target
    while not parent.exists():
        if parent == parent.parent:
            raise OSError("no existing parent directory")
        parent = parent.parent
    if not parent.is_dir():
        raise OSError(f"output or an intermediate parent is a file: {parent}")
    with tempfile.NamedTemporaryFile(prefix="httm_preflight_", dir=parent):
        pass
    return {"path": str(target), "tested_parent": str(parent), "target_exists": target.is_dir(),
            "session_created": False, "probe_removed": True}


def run_preflight(root=ROOT, *, environment=None, current_executable=None, probe=probe_interpreter,
                  writable=writable_directory):
    root = Path(root).resolve()
    env = dict(os.environ if environment is None else environment)
    checks, models, interpreters = [], [], {}
    effective = {}

    def add(name, status, message, **details):
        checks.append({"check": name, "status": status, "message": message, **details})

    face_python, hybrid_python = (root / p / "Scripts/python.exe" for p in (".venv", ".venv-hybrid"))
    current = Path(current_executable or sys.executable).resolve()
    add("invocation.python", "PASS" if current == face_python.resolve() else "FAIL",
        "Use repository .venv for coordinator/preflight; Hybrid worker uses .venv-hybrid independently",
        actual=str(current), expected=str(face_python))
    source_names = ("camera_coordinator_live.py", "camera_coordinator.py", "camera_live_runtime.py",
                    "face_id_stream_worker.py", "hybrid_v7_stream_worker.py", "behavior_warning.py",
                    "identity_continuity.py", "identity_continuity_diagnostics.py")
    source = root / "ai_training/integration"
    for name in source_names:
        add("source." + name, "PASS" if (source / name).is_file() else "FAIL", str(source / name))
    try:
        camera = assignments(source / "camera_coordinator.py")["CAMERA_INDEX"]
        if type(camera) is not int or camera < 0:
            raise ValueError("camera index is not a nonnegative integer")
        effective.update(camera_index=camera, camera_backends=["DSHOW", "MSMF"], device="cpu")
        add("camera.configuration", "PASS", f"Current source index={camera}; Live tries DSHOW then MSMF; hardware NOT TESTED")
        live = assignments(source / "camera_coordinator_live.py")
        if live.get("MAX_AI_FRAMES", "missing") is not None:
            raise ValueError("continuous default cannot be confirmed from source")
        warnings = ast.parse((source / "behavior_warning.py").read_text(encoding="utf-8-sig"))
        config = next(n for n in warnings.body if isinstance(n, ast.ClassDef) and n.name == "WarningConfig")
        fields = {n.target.id: ast.literal_eval(n.value) for n in config.body if isinstance(n, ast.AnnAssign)}
        live_tree = ast.parse((source / "camera_coordinator_live.py").read_text(encoding="utf-8-sig"))
        warning_binding = any(isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "WARNING_CONFIG" for t in n.targets)
                              and isinstance(n.value, ast.Name) and n.value.id == "DEFAULT_WARNING_CONFIG" for n in live_tree.body)
        if not warning_binding or any(fields.get(n) is not False for n in ("sleeping", "using_phone", "turning_back", "eating")):
            raise ValueError("disabled WARNING source contract cannot be confirmed")
        resolver_tree = ast.parse((source / "identity_continuity.py").read_text(encoding="utf-8-sig"))
        resolver = next(n for n in resolver_tree.body if isinstance(n, ast.ClassDef) and n.name == "IdentityContinuityResolver")
        constructor = next(n for n in resolver.body if isinstance(n, ast.FunctionDef) and n.name == "__init__")
        hold_defaults = dict(zip([a.arg for a in constructor.args.kwonlyargs], constructor.args.kw_defaults))
        if not isinstance(hold_defaults.get("simulation_hold_seconds"), ast.Constant) or hold_defaults["simulation_hold_seconds"].value is not None:
            raise ValueError("T_hold=None source default cannot be confirmed")
        shadow_tree = ast.parse((source / "identity_continuity_diagnostics.py").read_text(encoding="utf-8-sig"))
        calls = [n for n in ast.walk(shadow_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "IdentityContinuityResolver"]
        if not calls or any(any(k.arg in {None, "simulation_hold_seconds"} for k in call.keywords) for call in calls):
            raise ValueError("Shadow resolver hold-free initialization cannot be confirmed")
        effective.update(warning="disabled", T_hold=None)
        add("safety.source_defaults", "PASS", "Static source: WARNING disabled; Shadow constructor uses T_hold=None")
    except (OSError, ValueError, KeyError, StopIteration, SyntaxError, TypeError) as exc:
        add("safety.source_defaults", "FAIL", str(exc))

    for role, executable in (("face", face_python), ("hybrid", hybrid_python)):
        try:
            if not executable.is_file():
                raise OSError(f"Python interpreter missing: {executable}")
            info = probe(executable)
            version = info["version"]
            if Path(info["executable"]).resolve() != executable.resolve() or Path(info["prefix"]).resolve() != executable.parent.parent.resolve():
                raise ValueError("Probe returned a different interpreter/venv prefix")
            if Path(info["prefix"]).resolve() == Path(info["base_prefix"]).resolve():
                raise ValueError("Worker interpreter is not an isolated venv")
            interpreters[role] = info
            add(f"python.{role}", "FAIL" if tuple(version[:2]) < (3, 10) else "PASS" if version[:2] == [3, 11] else "WARN",
                f"Python {'.'.join(map(str, version))}; source needs >=3.10, audited demo uses 3.11", executable=str(executable))
            for module, alternatives in PACKAGES[role].items():
                versions = {name: info["versions"][name] for name in alternatives if name in info["versions"]}
                present = bool(versions) and bool(info["modules"].get(module))
                optional = role == "hybrid" and module == "mediapipe"
                add(f"package.{role}.{module}", "PASS" if present else "WARN" if optional else "FAIL",
                    "Metadata + top-level module found; native import not executed" if present else
                    "Optional landmark source unavailable" if optional else "Required distribution/module missing",
                    versions=versions, module_origin=info["modules"].get(module))
                if present and not Path(info["modules"][module]).resolve().is_relative_to(Path(info["prefix"]).resolve()):
                    add(f"package.{role}.{module}.external", "WARN", "Module is outside worker venv; launch path/PYTHONPATH may affect reproducibility",
                        module_origin=info["modules"][module])
            cv_names = [n for n in info["versions"] if n.startswith("opencv-")]
            if len(cv_names) > 1:
                add(f"opencv.{role}.collision", "WARN", "Multiple distributions share cv2; installed metadata does not prove GUI/ABI compatibility", distributions=cv_names)
            add(f"opencv.{role}.gui", "FAIL" if role == "face" and info.get("opencv_headless") is True else
                "PASS" if info.get("opencv_headless") is False else "WARN",
                f"Static cv2/version.py headless={info.get('opencv_headless')}; no GUI window opened")
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            add(f"python.{role}", "FAIL", str(exc))

    approved = ()
    # Resolve the one literal Name in the tuple without importing the selector.
    try:
        tree = ast.parse((root / "ai_training/src/behavior_checkpoint.py").read_text(encoding="utf-8-sig"))
        default = assignments(root / "ai_training/src/behavior_checkpoint.py")["DEFAULT_CHECKPOINT"]
        node = next(n.value for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "APPROVED_CHECKPOINTS" for t in n.targets))
        approved = tuple(default if isinstance(n, ast.Name) and n.id == "DEFAULT_CHECKPOINT" else ast.literal_eval(n) for n in node.elts)
        selected = env.get("HTTM_BEHAVIOR_CHECKPOINT", default)
        if selected not in approved:
            raise ValueError(f"HTTM_BEHAVIOR_CHECKPOINT must be one of {approved}")
        effective["behavior_checkpoint"] = selected
        add("model.selector", "PASS", f"Selected {selected}; expected worker classes sleeping/using_phone (checkpoint not loaded)")
    except (OSError, ValueError, KeyError, StopIteration, AttributeError, SyntaxError) as exc:
        selected = None
        add("model.selector", "FAIL", str(exc))

    def inventory(path, role, required):
        path = Path(path)
        try:
            size = path.stat().st_size
            if not path.is_file() or size <= 0:
                raise OSError("missing, empty or not a regular file")
            row = {"role": role, "path": str(path.resolve()), "bytes": size, "sha256": sha256(path), "required": required}
            models.append(row)
            add("model." + role, "PASS", "File readable; measured SHA256, no model load", **row)
        except OSError as exc:
            add("model." + role, "FAIL" if required else "WARN", f"{path}: {exc}")

    if selected:
        inventory(root / "ai_training/models" / selected, "behavior_selected", True)
    for name in approved:
        if name != selected:
            inventory(root / "ai_training/models" / name, "behavior_alternative", False)
    inventory(root / "ai_training/models/pretrained_person/yolo11n_coco.pt", "coco", True)
    inventory(root / "ai_training/models/face_landmarker.task", "face_landmarks_optional", False)
    inventory(root / "ai_training/models/pose_landmarker_full.task", "pose_optional", False)
    if (root / "ai_training/models/disabled_by_v6.task").exists():
        add("model.hand_disabled", "WARN", "disabled_by_v6.task unexpectedly exists; current worker would enable this hand asset")
    else:
        add("model.hand_disabled", "PASS", "Worker intentionally selects absent disabled_by_v6.task; no hand asset required")
    if "face" in interpreters:
        cache = Path(interpreters["face"]["home"]) / ".insightface/models/buffalo_l"
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            inventory(cache / name, "insightface_" + name, True)
        for path in sorted(cache.glob("*.onnx")):
            if path.name not in {"det_10g.onnx", "w600k_r50.onnx"}:
                inventory(path, "insightface_cached_" + path.name, False)

    enrollment = root / "data/face_enrollment"
    archives = sorted(enrollment.glob("SV*/embeddings.npz"))
    if not archives:
        add("enrollment", "FAIL", f"No enrollment archives at {enrollment}; known-SID demo unavailable")
    for path in archives:
        try:
            if not re.fullmatch(r"SV[0-9]{3,8}", path.parent.name):
                raise ValueError("directory SID does not match enrollment convention")
            with zipfile.ZipFile(path) as archive:
                if not {"student_id.npy", "embeddings.npy"} <= set(archive.namelist()):
                    raise ValueError("missing student_id.npy/embeddings.npy")
                if any(archive.getinfo(n).file_size == 0 for n in ("student_id.npy", "embeddings.npy")):
                    raise ValueError("empty enrollment member")
            add("enrollment.archive", "PASS", "Archive and required members present; vectors not loaded/printed", path=str(path), bytes=path.stat().st_size)
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            add("enrollment.archive", "FAIL", f"{path}: {exc}")
    effective["enrollment_archives"] = len(archives)

    for name, allowed, fallback in (("HTTM_FACE_TRACKER_TRIAL_MAX_AGE", {"4.0"}, "2.0s baseline"),
                                    ("HTTM_PERSON_TRIAL_MIN_CONF", {"0.35"}, "baseline; no trial filter")):
        value = env.get(name)
        add("env." + name, "PASS" if value is None or value in allowed else "FAIL",
            f"value={value!r}; unset={fallback}; approved opt-in={sorted(allowed)}")
    value = env.get("HTTM_CAMERA_MAX_OBSERVATIONS")
    try:
        limit = None if value is None else int(value) if value and value.isascii() and value.isdecimal() and int(value) > 0 else None
        if value is not None and limit is None:
            raise ValueError("expected positive ASCII integer; unset for continuous")
        effective["max_observations"] = limit
        add("env.HTTM_CAMERA_MAX_OBSERVATIONS", "PASS", "continuous" if limit is None else f"limited to {limit} observations")
    except ValueError as exc:
        add("env.HTTM_CAMERA_MAX_OBSERVATIONS", "FAIL", str(exc))
    for name in FLAGS:
        value = env.get(name)
        effective[name] = value == "1"
        add("env." + name, "PASS" if value in (None, "0", "1") else "WARN",
            f"effective={'ON' if value == '1' else 'OFF'}; only exact '1' enables; supplied={value!r}")
    if env.get("HTTM_PERSON_DEBUG") != "1" and any(env.get(n) == "1" for n in ("HTTM_FACE_DEBUG", "HTTM_PERSON_DEBUG_SNAPSHOTS")):
        add("env.debug_dependencies", "WARN", "Face debug/snapshots require HTTM_PERSON_DEBUG=1; requested detail will not be recorded")
    for name, default_limit, maximum in (("HTTM_PERSON_DEBUG_MAX_FRAMES", 10, 100), ("HTTM_PERSON_DEBUG_MAX_BYTES", 20 * 1024 * 1024, 100 * 1024 * 1024)):
        supplied = env.get(name)
        try:
            raw = int(supplied) if supplied is not None else default_limit
            value = max(0, min(raw, maximum))
            status = "PASS" if raw == value else "WARN"
        except ValueError:
            value, status = default_limit, "WARN"
        effective[name] = value
        add("env." + name, status, f"diagnostic limit effective={value}; supplied={supplied!r}; runtime clamp/fallback unchanged")
    for name in sorted(k for k in env if k.startswith("HTTM_") and k not in KNOWN_ENV):
        add("env.unknown", "WARN", f"{name} is not used by current Camera Live; does not enable WARNING or retention")
    if env.get("HTTM_IDENTITY_SHADOW") == "1":
        limit = effective.get("max_observations")
        effective["shadow_max_frames"] = min(10 if limit is None else limit, 100)
        add("diagnostic.shadow_budget", "PASS", f"Shadow bounded to {effective['shadow_max_frames']} rows and 1MiB; AI may continue")

    def read_config(path, required):
        duplicates = []
        def pairs(rows):
            output = {}
            for key, value in rows:
                if key in output:
                    duplicates.append(key)
                output[key] = value
            return output
        data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"non-finite JSON {value}")))
        def finite_numbers(value):
            if isinstance(value, dict):
                for item in value.values():
                    finite_numbers(item)
            elif isinstance(value, list):
                for item in value:
                    finite_numbers(item)
            elif isinstance(value, float) and not math.isfinite(value):
                raise ValueError("config contains a non-finite numeric value")
        finite_numbers(data)
        if not isinstance(data, dict) or not set(required) <= data.keys():
            raise ValueError("missing config object/required fields")
        if duplicates:
            add("config.duplicate_keys", "WARN", "JSON has duplicate keys; worker uses last value; preflight does not change config", keys=duplicates)
        add("config." + path.name, "PASS", "JSON parsed using worker last-key-wins semantics", path=str(path), sha256=sha256(path))
        return data
    try:
        config = read_config(root / "ai_training/config/hybrid_v7.json", ("inference_conf", "person_roi_enabled", "person_roi_padding", "dedupe_iou"))
        for key in ("inference_conf", "person_roi_padding", "dedupe_iou"):
            value = config[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"config {key} must be finite in [0,1]")
        if type(config["person_roi_enabled"]) is not bool:
            raise ValueError("person_roi_enabled must be bool")
        spatial = read_config(root / "ai_training/config/hybrid_behavior_v1_4.json", ("spatial",))
        if not isinstance(spatial["spatial"], dict):
            raise ValueError("spatial config must be an object")
    except (OSError, ValueError, TypeError) as exc:
        add("config.validation", "FAIL", str(exc))

    targets = [("runs", root / "ai_training/runs")]
    if "face" in interpreters:
        targets.append(("jpeg_temp", Path(interpreters["face"]["temp_dir"])))
    for flag, key, default_dir in (("HTTM_PERSON_DEBUG", "HTTM_PERSON_DEBUG_DIR", "person_tracker_debug"),
                                   ("HTTM_IDENTITY_SHADOW", "HTTM_IDENTITY_SHADOW_DIR", "identity_continuity_shadow")):
        if env.get(flag) == "1":
            supplied = env.get(key)
            if supplied == "":
                add("output." + key, "WARN", "Empty output directory resolves to current working directory")
            path = Path(supplied) if supplied is not None else root / "ai_training/runs" / default_dir
            if not path.is_absolute():
                path = Path.cwd() / path
                add("output.relative", "WARN", f"{key} resolves against launch working directory; use an absolute path", path=str(path))
            targets.append((key, path))
    for name, target in targets:
        try:
            add("output." + name, "PASS", "Unique temporary write marker created/removed; no session created", **writable(target))
        except OSError as exc:
            add("output." + name, "FAIL", f"{target}: {exc}")

    counts = dict(Counter(c["status"] for c in checks))
    status = "FAIL" if counts.get("FAIL") else "WARN" if counts.get("WARN") else "PASS"
    return {"schema": SCHEMA, "status": status, "counts": counts, "repository": str(root),
            "checks": checks, "effective": effective, "models": models,
            "interpreters": interpreters, "camera_opened": False, "model_loaded": False,
            "limits": ["No camera/backend availability, DLL/native imports, GUI or inference executed.",
                       "Checkpoint class mapping/compatibility is not read from weights; hashes are measured, not authenticated.",
                       "Enrollment vectors, finite values, norm and SID payload binding remain runtime checks.",
                       "PASS/WARN is offline preflight only, never Camera Live/model/identity accuracy PASS."]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT, help="repository root; default resolved from this script")
    parser.add_argument("--output", type=Path, help="optional NEW JSON report; existing file refused")
    parser.add_argument("--json", action="store_true", help="JSON to stdout instead of text")
    args = parser.parse_args(argv)
    try:
        report = run_preflight(args.root)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2)
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"AI Camera Live preflight: {report['status']} {report['counts']}")
            for check in report["checks"]:
                print(f"[{check['status']}] {check['check']}: {check['message']}")
            for model in report["models"]:
                print(f"SHA256 {model['sha256']} | {model['bytes']} bytes | {model['path']}")
            print("Offline only. Camera/GUI/native model execution and actor accuracy are NOT TESTED.")
        return 1 if report["status"] == "FAIL" else 0
    except (OSError, ValueError, TypeError) as exc:
        print(f"Preflight error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
