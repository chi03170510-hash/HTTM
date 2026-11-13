"""Offline Camera Live inventory for AI-Smart core in HTTM. Standard library only.

PASS checks prerequisites, not model accuracy, class mapping or native DLL/GUI
execution. Hashes are measured fingerprints, not an approved authenticity list.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


SCHEMA = "httm_camera_live_preflight_v2_aismart"
ROOT = Path(__file__).resolve().parents[2]
KNOWN_ENV = {
    "HTTM_BEHAVIOR_CHECKPOINT",
    "HTTM_CAMERA_MAX_OBSERVATIONS",
    "HTTM_HYBRID_DEVICE",
}
PACKAGES = {
    "coordinator": {
        "numpy": ("numpy",),
        "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"),
    },
    "ai_worker": {
        "numpy": ("numpy",),
        "cv2": ("opencv-python", "opencv-contrib-python", "opencv-python-headless", "opencv-contrib-python-headless"),
        "ultralytics": ("ultralytics",),
        "torch": ("torch",),
    },
}

PROBE = r'''
import ast,importlib.metadata as md,importlib.util,json,sys,tempfile
from pathlib import Path
versions={}
for d in md.distributions():
    name=d.metadata.get("Name", "").lower().replace("_", "-")
    if name:versions[name]=d.version
modules={}
for name in ("numpy","cv2","ultralytics","torch"):
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
    result = subprocess.run(
        [str(executable), "-B", "-c", PROBE],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=20,
        check=False,
    )
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
    return {
        "path": str(target),
        "tested_parent": str(parent),
        "target_exists": target.is_dir(),
        "session_created": False,
        "probe_removed": True,
    }


def run_preflight(
    root=ROOT,
    *,
    environment=None,
    current_executable=None,
    probe=probe_interpreter,
    writable=writable_directory,
):
    root = Path(root).resolve()
    env = dict(os.environ if environment is None else environment)
    checks, models, interpreters = [], [], {}
    effective = {}

    def add(name, status, message, **details):
        checks.append({"check": name, "status": status, "message": message, **details})

    coordinator_python = root / ".venv/Scripts/python.exe"
    hybrid_python = root / ".venv-hybrid/Scripts/python.exe"
    worker_python = hybrid_python if hybrid_python.is_file() else coordinator_python
    current = Path(current_executable or sys.executable).resolve()
    add(
        "invocation.python",
        "PASS" if current in (coordinator_python.resolve(), worker_python.resolve()) else "WARN",
        "Use repository .venv (and optional .venv-hybrid for AI worker if present)",
        actual=str(current),
        expected=str(coordinator_python),
    )

    source = root / "ai_training/integration"
    source_files = (
        source / "camera_coordinator_live.py",
        source / "camera_coordinator.py",
        source / "camera_live_runtime.py",
        source / "smart_classroom_stream_worker.py",
        source / "live_bbox_overlay.py",
        source / "inference_device.py",
        root / "ai_training/src/smart_classroom_core.py",
        root / "ai_training/src/behavior_checkpoint.py",
    )
    for path in source_files:
        add("source." + path.name, "PASS" if path.is_file() else "FAIL", str(path))

    try:
        camera = assignments(source / "camera_coordinator.py")["CAMERA_INDEX"]
        if type(camera) is not int or camera < 0:
            raise ValueError("camera index is not a nonnegative integer")
        hybrid_device = env.get("HTTM_HYBRID_DEVICE", "auto")
        if hybrid_device not in ("auto", "cpu", "0"):
            raise ValueError("HTTM_HYBRID_DEVICE must be auto, cpu or 0")
        effective.update(
            camera_index=camera,
            camera_backends=["DSHOW", "MSMF"],
            device=hybrid_device,
            device_runtime_verified=False,
        )
        add("camera.configuration", "PASS", f"Current source index={camera}; Live tries DSHOW then MSMF; hardware NOT TESTED")
        live = assignments(source / "camera_coordinator_live.py")
        if live.get("MAX_AI_FRAMES", "missing") is not None:
            raise ValueError("continuous default cannot be confirmed from source")
        add("safety.source_defaults", "PASS", "Static source: continuous AI-Smart pipeline configured")
    except (OSError, ValueError, KeyError, SyntaxError, TypeError) as exc:
        add("safety.source_defaults", "FAIL", str(exc))

    for role, executable in (("coordinator", coordinator_python), ("ai_worker", worker_python)):
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
            add(
                f"python.{role}",
                "FAIL" if tuple(version[:2]) < (3, 10) else "PASS" if version[:2] == [3, 11] else "WARN",
                f"Python {'.'.join(map(str, version))}; source needs >=3.10, audited demo uses 3.11",
                executable=str(executable),
            )
            for module, alternatives in PACKAGES[role].items():
                versions = {name: info["versions"][name] for name in alternatives if name in info["versions"]}
                present = bool(versions) and bool(info["modules"].get(module))
                add(
                    f"package.{role}.{module}",
                    "PASS" if present else "FAIL",
                    "Metadata + top-level module found; native import not executed"
                    if present else "Required distribution/module missing",
                    versions=versions,
                    module_origin=info["modules"].get(module),
                )
            cv_names = [n for n in info["versions"] if n.startswith("opencv-")]
            if len(cv_names) > 1:
                add(
                    f"opencv.{role}.collision",
                    "WARN",
                    "Multiple distributions share cv2; installed metadata does not prove GUI/ABI compatibility",
                    distributions=cv_names,
                )
            add(
                f"opencv.{role}.gui",
                "FAIL" if role == "coordinator" and info.get("opencv_headless") is True else
                "PASS" if info.get("opencv_headless") is False else "WARN",
                f"Static cv2/version.py headless={info.get('opencv_headless')}; no GUI window opened",
            )
        except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
            add(f"python.{role}", "FAIL", str(exc))

    approved = ()
    default = None
    try:
        ckpt_path = root / "ai_training/src/behavior_checkpoint.py"
        tree = ast.parse(ckpt_path.read_text(encoding="utf-8-sig"))
        default = assignments(ckpt_path)["DEFAULT_CHECKPOINT"]
        node = next(
            n.value
            for n in tree.body
            if isinstance(n, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "APPROVED_CHECKPOINTS" for t in n.targets)
        )
        approved = tuple(
            default if isinstance(n, ast.Name) and n.id == "DEFAULT_CHECKPOINT" else ast.literal_eval(n)
            for n in node.elts
        )
        selected = env.get("HTTM_BEHAVIOR_CHECKPOINT", default)
        if selected not in approved:
            raise ValueError(f"HTTM_BEHAVIOR_CHECKPOINT must be one of {approved}")
        effective["behavior_checkpoint"] = selected
        add(
            "model.selector",
            "PASS",
            f"Selected {selected}; AI-Smart 3 behavior classes: Using Phone, Eating, Sleeping (checkpoint not loaded)",
        )
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

    def behavior_model_path(name):
        if name == "best_behavior_v1_4.pt":
            return root / "ai/behavior_detection/models" / name
        local = root / "ai_training/models" / name
        if local.is_file():
            return local
        fallback = root.parent / "models" / name
        return fallback if fallback.is_file() else local

    if selected:
        inventory(behavior_model_path(selected), "behavior_selected", True)
    obj_local = root / "ai_training/models/yolov8n.pt"
    obj_fallback = root.parent / "yolov8n.pt"
    inventory(obj_local if obj_local.is_file() else obj_fallback, "object_detector", True)

    value = env.get("HTTM_CAMERA_MAX_OBSERVATIONS")
    try:
        limit = None if value is None else int(value) if value and value.isascii() and value.isdecimal() and int(value) > 0 else None
        if value is not None and limit is None:
            raise ValueError("expected positive ASCII integer; unset for continuous")
        effective["max_observations"] = limit
        add("env.HTTM_CAMERA_MAX_OBSERVATIONS", "PASS", "continuous" if limit is None else f"limited to {limit} observations")
    except ValueError as exc:
        add("env.HTTM_CAMERA_MAX_OBSERVATIONS", "FAIL", str(exc))

    for name in sorted(k for k in env if k.startswith("HTTM_") and k not in KNOWN_ENV):
        add("env.unknown", "WARN", f"{name} is not used by current AI-Smart Camera Live")

    targets = [("runs", root / "ai_training/runs")]
    if "coordinator" in interpreters:
        targets.append(("jpeg_temp", Path(interpreters["coordinator"]["temp_dir"])))
    for name, target in targets:
        try:
            add("output." + name, "PASS", "Unique temporary write marker created/removed; no session created", **writable(target))
        except OSError as exc:
            add("output." + name, "FAIL", f"{target}: {exc}")

    counts = dict(Counter(c["status"] for c in checks))
    status = "FAIL" if counts.get("FAIL") else "WARN" if counts.get("WARN") else "PASS"
    return {
        "schema": SCHEMA,
        "status": status,
        "counts": counts,
        "repository": str(root),
        "checks": checks,
        "effective": effective,
        "models": models,
        "interpreters": interpreters,
        "camera_opened": False,
        "model_loaded": False,
        "limits": [
            "No camera/backend availability, DLL/native imports, GUI or inference executed.",
            "Checkpoint class mapping/compatibility is checked at worker startup.",
            "PASS/WARN is offline preflight only.",
        ],
    }


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
