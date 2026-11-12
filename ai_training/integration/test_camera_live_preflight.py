"""AI2 stdlib-only preflight fixtures: no camera, model, package install or network."""

import ast
import copy
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
import zipfile

import camera_live_preflight as preflight


class TestCameraPreflight(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "repository with spaces"
        self.root.mkdir()
        source = self.root / "ai_training/integration"; source.mkdir(parents=True)
        for name in ("camera_coordinator_live.py", "camera_coordinator.py", "camera_live_runtime.py",
                     "face_id_stream_worker.py", "hybrid_v7_stream_worker.py", "behavior_warning.py",
                     "identity_continuity.py", "identity_continuity_diagnostics.py"):
            shutil.copy2(preflight.ROOT / "ai_training/integration" / name, source / name)
        selector = self.root / "ai_training/src"; selector.mkdir(parents=True)
        shutil.copy2(preflight.ROOT / "ai_training/src/behavior_checkpoint.py", selector / "behavior_checkpoint.py")
        configs = self.root / "ai_training/config"; configs.mkdir()
        for name in ("hybrid_v7.json", "hybrid_behavior_v1_4.json"):
            data = json.loads((preflight.ROOT / "ai_training/config" / name).read_text())
            (configs / name).write_text(json.dumps(data), encoding="utf-8")
        for name in ("best_behavior_v1_4.pt", "best_behavior_v2.pt", "pretrained_person/yolo11n_coco.pt",
                     "face_landmarker.task", "pose_landmarker_full.task"):
            self.write("ai_training/models/" + name, b"synthetic placeholder; not model weights")
        self.home = Path(self.temp.name) / "synthetic home"
        for name in ("det_10g.onnx", "w600k_r50.onnx"):
            path = self.home / ".insightface/models/buffalo_l" / name
            path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(b"synthetic ONNX placeholder")
        archive = self.root / "data/face_enrollment/SV001/embeddings.npz"
        archive.parent.mkdir(parents=True)
        with zipfile.ZipFile(archive, "w") as stream:
            stream.writestr("student_id.npy", b"synthetic header only")
            stream.writestr("embeddings.npy", b"synthetic header only; no biometric vectors")
        self.python = self.root / ".venv/Scripts/python.exe"
        self.write(".venv/Scripts/python.exe", b"fixture")
        self.write(".venv-hybrid/Scripts/python.exe", b"fixture")
        self.probes = {}
        for role, env in (("face", ".venv"), ("hybrid", ".venv-hybrid")):
            prefix = self.root / env
            versions = {names[0]: "1.0.0" for names in preflight.PACKAGES[role].values()}
            self.probes[role] = {"executable": str(prefix / "Scripts/python.exe"), "prefix": str(prefix),
                "base_prefix": str(Path(self.temp.name) / "base_python"), "version": [3, 11, 9],
                "versions": versions, "modules": {name: str(prefix / "Lib/site-packages" / name / "__init__.py")
                                                   for name in preflight.PACKAGES[role]},
                "home": str(self.home), "temp_dir": self.temp.name, "opencv_headless": False}

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(data)
        return path

    def probe(self, executable):
        role = "hybrid" if ".venv-hybrid" in str(executable) else "face"
        return copy.deepcopy(self.probes[role])

    def run_check(self, env=None, **kwargs):
        return preflight.run_preflight(self.root, environment={} if env is None else env,
            current_executable=kwargs.pop("current_executable", self.python), probe=kwargs.pop("probe", self.probe), **kwargs)

    @staticmethod
    def checks(report, name):
        return [c for c in report["checks"] if c["check"] == name]

    def test_complete_fixture_passes_and_records_hash_without_camera_or_model(self):
        report = self.run_check()
        self.assertEqual(report["status"], "PASS", report["checks"])
        self.assertFalse(report["camera_opened"]); self.assertFalse(report["model_loaded"])
        self.assertEqual(report["effective"]["behavior_checkpoint"], "best_behavior_v1_4.pt")
        self.assertEqual(report["effective"]["camera_index"], 0)
        self.assertEqual(report["effective"]["warning"], "disabled")
        self.assertIsNone(report["effective"]["T_hold"])
        self.assertTrue(all(len(m["sha256"]) == 64 and m["bytes"] > 0 for m in report["models"]))

    def test_device_selection_records_requested_setting_without_runtime_probe(self):
        for device in ("auto", "cpu", "0"):
            with self.subTest(device=device):
                report = self.run_check(env={"HTTM_HYBRID_DEVICE": device})
                self.assertEqual(report["status"], "PASS", report["checks"])
                self.assertEqual(report["effective"]["device"], device)
                self.assertEqual(report["effective"]["face_device"], "auto")
                self.assertFalse(report["effective"]["device_runtime_verified"])

    def test_missing_selected_checkpoint_is_fail(self):
        (self.root / "ai_training/models/best_behavior_v1_4.pt").unlink()
        self.assertEqual(self.checks(self.run_check(), "model.behavior_selected")[0]["status"], "FAIL")

    def test_missing_alternative_checkpoint_is_warn(self):
        (self.root / "ai_training/models/best_behavior_v2.pt").unlink()
        self.assertEqual(self.run_check()["status"], "WARN")

    def test_empty_model_is_fail(self):
        self.write("ai_training/models/pretrained_person/yolo11n_coco.pt", b"")
        self.assertEqual(self.checks(self.run_check(), "model.coco")[0]["status"], "FAIL")

    def test_missing_face_cache_is_fail_without_auto_download(self):
        (self.home / ".insightface/models/buffalo_l/det_10g.onnx").unlink()
        self.assertEqual(self.checks(self.run_check(), "model.insightface_det_10g.onnx")[0]["status"], "FAIL")

    def test_missing_enrollment_is_fail(self):
        (self.root / "data/face_enrollment/SV001/embeddings.npz").unlink()
        self.assertEqual(self.checks(self.run_check(), "enrollment")[0]["status"], "FAIL")

    def test_corrupt_enrollment_is_fail(self):
        self.write("data/face_enrollment/SV001/embeddings.npz", b"not a zip")
        self.assertEqual(self.checks(self.run_check(), "enrollment.archive")[0]["status"], "FAIL")

    def test_missing_venv_interpreter_is_fail_before_probe(self):
        (self.root / ".venv/Scripts/python.exe").unlink()
        probe = Mock(side_effect=self.probe)
        report = self.run_check(probe=probe)
        self.assertEqual(self.checks(report, "python.face")[0]["status"], "FAIL")
        self.assertEqual(probe.call_count, 1)

    def test_wrong_invocation_python_is_fail(self):
        report = self.run_check(current_executable=self.root / ".venv-hybrid/Scripts/python.exe")
        self.assertEqual(self.checks(report, "invocation.python")[0]["status"], "FAIL")

    def test_probe_wrong_prefix_or_non_venv_is_fail(self):
        baseline = copy.deepcopy(self.probes["face"])
        for field in ("prefix", "base_prefix"):
            info = copy.deepcopy(baseline)
            info[field] = info["base_prefix"] if field == "prefix" else info["prefix"]
            self.probes["face"] = info
            self.assertEqual(self.checks(self.run_check(), "python.face")[0]["status"], "FAIL")

    def test_old_python_version_fails(self):
        self.probes["face"]["version"] = [3, 9, 0]
        self.assertEqual(self.checks(self.run_check(), "python.face")[0]["status"], "FAIL")

    def test_missing_required_package_or_module_fails(self):
        self.probes["face"]["modules"]["insightface"] = None
        self.assertEqual(self.checks(self.run_check(), "package.face.insightface")[0]["status"], "FAIL")

    def test_module_outside_worker_venv_warns(self):
        self.probes["face"]["modules"]["numpy"] = str(self.home / "external_numpy/__init__.py")
        self.assertEqual(self.checks(self.run_check(), "package.face.numpy.external")[0]["status"], "WARN")

    def test_optional_mediapipe_or_pose_missing_warns(self):
        self.probes["hybrid"]["modules"]["mediapipe"] = None
        (self.root / "ai_training/models/pose_landmarker_full.task").unlink()
        report = self.run_check()
        self.assertEqual(report["status"], "WARN")
        self.assertEqual(self.checks(report, "package.hybrid.mediapipe")[0]["status"], "WARN")

    def test_headless_coordinator_fails_and_shared_cv2_distributions_warn(self):
        self.probes["face"]["opencv_headless"] = True
        self.probes["hybrid"]["versions"]["opencv-python-headless"] = "1.0.0"
        report = self.run_check()
        self.assertEqual(self.checks(report, "opencv.face.gui")[0]["status"], "FAIL")
        self.assertEqual(self.checks(report, "opencv.hybrid.collision")[0]["status"], "WARN")

    def test_valid_trial_configuration_does_not_mutate_environment_or_source(self):
        env = {"HTTM_BEHAVIOR_CHECKPOINT": "best_behavior_v2.pt", "HTTM_PERSON_TRIAL_MIN_CONF": "0.35",
               "HTTM_FACE_TRACKER_TRIAL_MAX_AGE": "4.0", "HTTM_CAMERA_MAX_OBSERVATIONS": "10"}
        before = copy.deepcopy(env)
        report = self.run_check(env)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["effective"]["max_observations"], 10)
        self.assertEqual(env, before)

    def test_invalid_environment_values_fail(self):
        for key, value in (("HTTM_BEHAVIOR_CHECKPOINT", "../../other.pt"), ("HTTM_PERSON_TRIAL_MIN_CONF", ".35"),
                           ("HTTM_FACE_TRACKER_TRIAL_MAX_AGE", "5.0"), ("HTTM_CAMERA_MAX_OBSERVATIONS", "0"),
                           ("HTTM_CAMERA_MAX_OBSERVATIONS", "nan"), ("HTTM_CAMERA_MAX_OBSERVATIONS", " 10")):
            with self.subTest(key=key, value=value):
                self.assertEqual(self.run_check({key: value})["status"], "FAIL")

    def test_overlay_shadow_switches_and_shadow_budget_are_explicit(self):
        env = {"HTTM_PERSON_SNAPSHOT_OVERLAY": "1", "HTTM_IDENTITY_SHADOW": "1"}
        report = self.run_check(env)
        self.assertTrue(report["effective"]["HTTM_PERSON_SNAPSHOT_OVERLAY"])
        self.assertTrue(report["effective"]["HTTM_IDENTITY_SHADOW"])
        self.assertEqual(report["effective"]["shadow_max_frames"], 10)
        self.assertEqual(self.run_check({**env, "HTTM_CAMERA_MAX_OBSERVATIONS": "20"})["effective"]["shadow_max_frames"], 20)

    def test_nonliteral_flag_warns_that_runtime_effectively_disables_it(self):
        report = self.run_check({"HTTM_PERSON_SNAPSHOT_OVERLAY": "true"})
        self.assertFalse(report["effective"]["HTTM_PERSON_SNAPSHOT_OVERLAY"])
        self.assertEqual(self.checks(report, "env.HTTM_PERSON_SNAPSHOT_OVERLAY")[0]["status"], "WARN")

    def test_face_debug_without_person_debug_warns(self):
        self.assertEqual(self.run_check({"HTTM_FACE_DEBUG": "1"})["status"], "WARN")

    def test_unknown_warning_or_hold_environment_is_not_authorization(self):
        report = self.run_check({"HTTM_WARNING": "1", "HTTM_IDENTITY_HOLD_SECONDS": "2"})
        self.assertEqual(len(self.checks(report, "env.unknown")), 2)
        self.assertEqual(report["effective"]["warning"], "disabled")
        self.assertIsNone(report["effective"]["T_hold"])

    def test_bad_diagnostic_limit_reports_existing_fallback_and_clamp(self):
        report = self.run_check({"HTTM_PERSON_DEBUG_MAX_FRAMES": "101", "HTTM_PERSON_DEBUG_MAX_BYTES": "bad"})
        self.assertEqual(report["effective"]["HTTM_PERSON_DEBUG_MAX_FRAMES"], 100)
        self.assertEqual(report["effective"]["HTTM_PERSON_DEBUG_MAX_BYTES"], 20 * 1024 * 1024)

    def test_malformed_config_and_nonfinite_threshold_fail(self):
        for text in ("{", '{"inference_conf": NaN}', '{"inference_conf":2,"person_roi_enabled":true,"person_roi_padding":0.1,"dedupe_iou":0.5}',
                     '{"inference_conf":0.05,"person_roi_enabled":true,"person_roi_padding":0.1,"dedupe_iou":0.5,"other":1e999}'):
            self.write("ai_training/config/hybrid_v7.json", text.encode())
            self.assertEqual(self.run_check()["status"], "FAIL")

    def test_duplicate_config_keys_warn_and_preserve_last_value_semantics(self):
        path = self.root / "ai_training/config/hybrid_v7.json"
        data = path.read_text()
        path.write_text(data[:-1] + ',"inference_conf":0.05}', encoding="utf-8")
        self.assertEqual(self.checks(self.run_check(), "config.duplicate_keys")[0]["status"], "WARN")

    def test_unwritable_output_fails_without_creating_session(self):
        report = self.run_check(writable=Mock(side_effect=PermissionError("synthetic read-only output")))
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse((self.root / "ai_training/runs").exists())

    def test_paths_with_spaces_write_probe_removes_only_own_marker(self):
        marker = self.root / "keep.txt"; marker.write_text("keep")
        before = set(self.root.iterdir())
        info = preflight.writable_directory(self.root / "new output with spaces" / "session")
        self.assertEqual(set(self.root.iterdir()), before)
        self.assertEqual(marker.read_text(), "keep")
        self.assertFalse(info["target_exists"])

    def test_output_file_instead_of_directory_fails(self):
        path = self.write("output.txt", b"keep")
        with self.assertRaises(OSError):
            preflight.writable_directory(path / "subdir")
        self.assertEqual(path.read_bytes(), b"keep")

    def test_probe_uses_exact_interpreter_argv_no_shell_or_model_import(self):
        payload = self.probes["face"]
        process = Mock(returncode=0, stdout=json.dumps(payload), stderr="")
        with patch.object(preflight.subprocess, "run", return_value=process) as call:
            self.assertEqual(preflight.probe_interpreter(self.python), payload)
        self.assertEqual(call.call_args.args[0][:3], [str(self.python), "-B", "-c"])
        self.assertNotIn("shell", call.call_args.kwargs)
        imported = {alias.name for n in ast.walk(ast.parse(preflight.PROBE)) if isinstance(n, ast.Import) for alias in n.names}
        self.assertFalse(imported & {"cv2", "torch", "ultralytics", "onnxruntime", "insightface", "mediapipe"})

    def test_probe_timeout_or_bad_json_is_reported_as_failure(self):
        for failure in (subprocess.TimeoutExpired("fixture", 20), ValueError("bad probe JSON")):
            self.assertEqual(self.run_check(probe=Mock(side_effect=failure))["status"], "FAIL")

    def test_modified_warning_default_fails_safety_check(self):
        path = self.root / "ai_training/integration/behavior_warning.py"
        path.write_text(path.read_text().replace("sleeping: bool = False", "sleeping: bool = True"), encoding="utf-8")
        self.assertEqual(self.checks(self.run_check(), "safety.source_defaults")[0]["status"], "FAIL")

    def test_modified_shadow_hold_constructor_fails_safety_check(self):
        path = self.root / "ai_training/integration/identity_continuity_diagnostics.py"
        text = path.read_text().replace("max_update_gap=max_update_gap)", "max_update_gap=max_update_gap, simulation_hold_seconds=2.)")
        path.write_text(text, encoding="utf-8")
        self.assertEqual(self.checks(self.run_check(), "safety.source_defaults")[0]["status"], "FAIL")

    def test_cli_report_never_overwrites_existing_file(self):
        path = self.write("existing report.json", b"keep")
        with patch.object(preflight, "run_preflight", return_value=self.run_check()), patch.object(sys, "stderr", io.StringIO()):
            self.assertEqual(preflight.main(["--root", str(self.root), "--output", str(path)]), 2)
        self.assertEqual(path.read_bytes(), b"keep")

    def test_cli_json_serializes_status_and_exit_codes(self):
        for status, expected in (("PASS", 0), ("WARN", 0), ("FAIL", 1)):
            report = self.run_check(); report["status"] = status
            with patch.object(preflight, "run_preflight", return_value=report), patch.object(sys, "stdout", io.StringIO()) as output:
                self.assertEqual(preflight.main(["--json"]), expected)
            self.assertEqual(json.loads(output.getvalue())["status"], status)


if __name__ == "__main__":
    unittest.main()
