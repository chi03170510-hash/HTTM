"""No hardware/model tests for Camera Live Face device selection."""
import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch
from face_inference_device import FACE_DEVICE_ENV, face_runtime, load_live_face_model


class TestFaceDevice(unittest.TestCase):
    def test_cpu_override_uses_original_loader(self):
        loader = Mock()
        with patch.dict(os.environ, {FACE_DEVICE_ENV: 'cpu'}):
            self.assertIs(load_live_face_model(loader), loader.return_value)
        loader.assert_called_once_with()

    def test_invalid_device_rejected(self):
        with patch.dict(os.environ, {FACE_DEVICE_ENV: 'invalid'}):
            with self.assertRaises(ValueError):
                load_live_face_model(Mock())

    def test_auto_falls_back_without_cuda(self):
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        loader = Mock()
        with patch.dict(os.environ, {FACE_DEVICE_ENV: 'auto'}), patch.dict(sys.modules, {'onnxruntime': ort}):
            self.assertIs(load_live_face_model(loader), loader.return_value)

    def test_explicit_gpu_failure_not_hidden(self):
        ort = SimpleNamespace(get_available_providers=lambda: ['CPUExecutionProvider'])
        with patch.dict(os.environ, {FACE_DEVICE_ENV: '0'}), patch.dict(sys.modules, {'onnxruntime': ort}):
            with self.assertRaisesRegex(RuntimeError, 'Requested Face CUDA'):
                load_live_face_model(Mock())

    def test_actual_session_providers_must_agree(self):
        components = {}
        for name in ('detection', 'recognition'):
            session = Mock()
            session.get_providers.return_value = ['CUDAExecutionProvider', 'CPUExecutionProvider']
            components[name] = SimpleNamespace(session=session)
        model = SimpleNamespace(models=components)
        self.assertEqual(face_runtime(model)['backend'], 'cuda')
        components['recognition'].session.get_providers.return_value = ['CPUExecutionProvider']
        self.assertEqual(face_runtime(model)['backend'], 'unknown')
        self.assertEqual(face_runtime(object())['backend'], 'unknown')


if __name__ == '__main__':
    unittest.main()
