"""Device selection tests without importing CUDA or opening the camera."""

import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from unittest.mock import Mock, patch

from inference_device import DEVICE_ENV, select_hybrid_device, inference_runtime, configure_hybrid_threads


class TestHybridDevice(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(os.environ))
        os.environ.pop(DEVICE_ENV, None)
        self.enterContext(redirect_stderr(io.StringIO()))
        self.torch = Mock()
        self.torch.__version__ = "synthetic"
        self.torch.cuda.is_available.return_value = True
        self.torch.cuda.get_device_name.return_value = "Synthetic GPU"
        self.sample = Mock()
        self.sample.__matmul__ = Mock(return_value=Mock())
        self.sample.__matmul__.return_value.sum.return_value.item.return_value = 512.
        self.torch.ones.return_value = self.sample
        self.enterContext(patch.dict(sys.modules, {"torch": self.torch}))

    def test_auto_selects_cuda_only_after_kernel_smoke_test(self):
        self.assertEqual(select_hybrid_device(), "0")
        self.torch.cuda.synchronize.assert_called_once_with(0)

    def test_cpu_override_never_queries_cuda(self):
        os.environ[DEVICE_ENV] = "cpu"
        self.assertEqual(select_hybrid_device(), "cpu")
        self.torch.cuda.is_available.assert_not_called()

    def test_auto_falls_back_when_cuda_unavailable(self):
        self.torch.cuda.is_available.return_value = False
        self.assertEqual(select_hybrid_device(), "cpu")
        self.torch.ones.assert_not_called()

    def test_auto_falls_back_when_gpu_kernel_is_unsupported(self):
        self.torch.ones.side_effect = RuntimeError("no kernel image is available")
        self.assertEqual(select_hybrid_device(), "cpu")

    def test_explicit_cuda_failure_is_not_silently_hidden(self):
        os.environ[DEVICE_ENV] = "0"
        self.torch.cuda.is_available.return_value = False
        with self.assertRaisesRegex(RuntimeError, "Requested CUDA"):
            select_hybrid_device()

    def test_runtime_reads_active_backend_not_cpu_wrapper(self):
        models = []
        for _ in range(2):
            model = Mock()
            parameter = Mock(device="cuda:0", dtype="torch.float32")
            model.predictor.model.backend.model.parameters.side_effect = lambda: iter([parameter])
            models.append(model)
        runtime = inference_runtime(models)
        self.assertEqual(runtime["backend"], "cuda")
        self.assertEqual(runtime["device_name"], "Synthetic GPU")
        self.assertEqual(runtime["models"][0]["precision"], "float32")

    def test_runtime_missing_predictor_is_not_reported_as_gpu(self):
        self.assertEqual(inference_runtime([object(), object()])["backend"], "unknown")

    def test_cpu_thread_setting_is_unchanged(self):
        configure_hybrid_threads("cpu")
        self.torch.set_num_threads.assert_not_called()

    def test_invalid_setting_rejected(self):
        for value in ("cuda", "1", "", "AUTO"):
            os.environ[DEVICE_ENV] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                select_hybrid_device()


if __name__ == "__main__":
    unittest.main()
