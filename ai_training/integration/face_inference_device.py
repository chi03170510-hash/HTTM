"""Camera Live-only InsightFace CUDA setup; enrollment defaults stay unchanged."""
import os
from pathlib import Path
import sys

FACE_DEVICE_ENV = "HTTM_FACE_DEVICE"
_DLL_HANDLES = []


def face_runtime(model):
    """Report session providers, not the requested environment setting."""
    sessions = getattr(model, "models", {})
    rows = []
    if isinstance(sessions, dict):
        for name, component in sessions.items():
            session = getattr(component, "session", None)
            providers = session.get_providers() if session is not None else []
            rows.append({"model": name, "providers": providers})
    backend = ("cuda" if rows and all(r["providers"] and r["providers"][0] == "CUDAExecutionProvider" for r in rows)
               else "cpu" if rows and all(r["providers"] and r["providers"][0] == "CPUExecutionProvider" for r in rows)
               else "unknown")
    return {"backend": backend, "models": rows}


def load_live_face_model(cpu_loader):
    requested = os.environ.get(FACE_DEVICE_ENV, "auto")
    if requested not in ("auto", "cpu", "0"):
        raise ValueError(f"{FACE_DEVICE_ENV} must be auto, cpu or 0")
    if requested == "cpu":
        return cpu_loader()
    try:
        import onnxruntime as ort
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("Installed ONNX Runtime has no CUDA provider")
        if sys.platform == "win32":
            import ctypes
            # Reuse installed CUDA12/cuDNN9 runtimes; do not import torch into Face.
            dll_dir = Path(__file__).resolve().parents[2] / ".venv-hybrid/Lib/site-packages/torch/lib"
            if not dll_dir.is_dir():
                raise RuntimeError("Shared CUDA runtime directory is missing")
            _DLL_HANDLES.append(os.add_dll_directory(str(dll_dir)))
            # cuDNN dynamically loads sibling libraries with native LoadLibrary.
            # add_dll_directory alone does not cover every native search path.
            os.environ["PATH"] = str(dll_dir) + os.pathsep + os.environ.get("PATH", "")
            for name in ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cufft64_11.dll", "cudnn64_9.dll"):
                _DLL_HANDLES.append(ctypes.WinDLL(str(dll_dir / name)))
        from insightface.app import FaceAnalysis
        import numpy as np
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        model = FaceAnalysis(name="buffalo_l", allowed_modules=["detection", "recognition"],
                             providers=[("CUDAExecutionProvider", {"device_id": 0,
                                         "cudnn_conv_algo_search": "HEURISTIC",
                                         "gpu_mem_limit": 768 * 1024 * 1024,
                                         "arena_extend_strategy": "kSameAsRequested"}), "CPUExecutionProvider"],
                             sess_options=options)
        model.prepare(ctx_id=0, det_size=(320, 320))
        if face_runtime(model)["backend"] != "cuda":
            raise RuntimeError("Face sessions fell back instead of activating CUDA")
        # Exercise BOTH ONNX models; CUDA listed as available is not sufficient.
        model.get(np.zeros((320, 320, 3), dtype=np.uint8))
        embedding = model.models["recognition"].get_feat(np.zeros((112, 112, 3), dtype=np.uint8))
        if not np.isfinite(embedding).all():
            raise RuntimeError("CUDA face smoke test returned invalid embedding")
        print("[FACE DEVICE] GPU / CUDA:0 (ONNX Runtime); both models smoke-tested", file=sys.stderr, flush=True)
        return model
    except Exception as exc:
        if requested == "0":
            raise RuntimeError(f"Requested Face CUDA cannot run: {exc}") from exc
        print(f"[FACE DEVICE] CUDA unavailable; falling back to CPU: {exc}", file=sys.stderr, flush=True)
        return cpu_loader()
