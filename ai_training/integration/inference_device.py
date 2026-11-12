"""Select and smoke-test the Hybrid YOLO device; Face/MediaPipe unchanged."""

import os
import sys


DEVICE_ENV = "HTTM_HYBRID_DEVICE"


def configure_hybrid_threads(device):
    """Limit CPU-side YOLO contention only when its inference uses CUDA."""
    if device != "cpu":
        import cv2
        import torch

        torch.set_num_threads(2)
        cv2.setNumThreads(1)


def inference_runtime(models):
    """Read active prediction backends, not the requested device or CPU wrappers."""
    import torch

    rows = []
    for model in models:
        predictor = getattr(model, "predictor", None)
        backend = getattr(predictor, "model", None)
        active = getattr(backend, "backend", backend)
        network = getattr(active, "model", None)
        try:
            parameter = next(network.parameters())
            rows.append({"device": str(parameter.device),
                         "precision": str(parameter.dtype).removeprefix("torch.")})
        except (AttributeError, StopIteration, TypeError):
            rows.append({"device": "unknown", "precision": "unknown"})
    devices = [row["device"] for row in rows]
    backend = ("cuda" if devices and all(d.startswith("cuda:") for d in devices) else
               "cpu" if devices and all(d == "cpu" for d in devices) else
               "unknown" if "unknown" in devices else "mixed")
    return {"backend": backend, "models": rows,
            "device_name": torch.cuda.get_device_name(0) if backend == "cuda" else backend.upper()}


def select_hybrid_device():
    requested = os.environ.get(DEVICE_ENV, "auto")
    if requested not in ("auto", "cpu", "0"):
        raise ValueError(f"Invalid {DEVICE_ENV}={requested!r}; use auto, cpu or 0")
    if requested == "cpu":
        print("[HYBRID DEVICE] CPU explicitly selected", file=sys.stderr, flush=True)
        return "cpu"

    import torch

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the installed PyTorch/driver")
        # Detection alone is insufficient: an unsupported GPU architecture can
        # appear available but fail at its first kernel. Exercise one now.
        sample = torch.ones((8, 8), device="cuda:0")
        value = (sample @ sample).sum().item()
        torch.cuda.synchronize(0)
        if value != 512.0:
            raise RuntimeError("CUDA smoke calculation returned an unexpected value")
        del sample
        name = torch.cuda.get_device_name(0)
        print(f"[HYBRID DEVICE] CUDA:0 {name} | torch={torch.__version__} | CUDA={torch.version.cuda}",
              file=sys.stderr, flush=True)
        return "0"
    except (RuntimeError, AssertionError) as exc:
        if requested == "0":
            raise RuntimeError(f"Requested CUDA device 0 cannot run: {exc}") from exc
        print(f"[HYBRID DEVICE] CUDA not usable; falling back to CPU: {exc}",
              file=sys.stderr, flush=True)
        return "cpu"
