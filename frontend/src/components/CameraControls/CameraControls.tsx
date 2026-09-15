import { useState } from 'react';
import { useStartCamera, useStopCamera, useCameraStatus } from '../../hooks';
import { useAppStore } from '../../store/useAppStore';
import './CameraControls.css';

const CAMERA_OPTIONS = [
  'Cam 01 (Bàn 1-3)',
  'Cam 02 (Bàn 4-6)',
  'Cam 03 (Bàn 7-9)',
];

export const CameraControls = () => {
  const isCameraActive = useAppStore((s) => s.isCameraActive);
  const setCameraActive = useAppStore((s) => s.setCameraActive);
  const selectedCamera = useAppStore((s) => s.selectedCamera);
  const setSelectedCamera = useAppStore((s) => s.setSelectedCamera);

  const { data: cameraStatus } = useCameraStatus();
  const startCamera = useStartCamera();
  const stopCamera = useStopCamera();

  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const handleStart = async () => {
    setErrorMsg(null);
    try {
      await startCamera.mutateAsync();
      setCameraActive(true);
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? 'Không thể bắt đầu camera';
      setErrorMsg(`Lỗi: ${detail}`);
    }
  };

  const handleStop = async () => {
    setErrorMsg(null);
    try {
      await stopCamera.mutateAsync();
      setCameraActive(false);
    } catch (err: any) {
      const detail = err?.response?.data?.detail ?? 'Không thể dừng camera';
      setErrorMsg(`Lỗi: ${detail}`);
    }
  };

  return (
    <div className="cam-controls">
      <div className="cam-controls__buttons">
        <button
          id="btn-start-camera"
          className={`cam-controls__btn cam-controls__btn--start ${isCameraActive ? 'cam-controls__btn--active' : ''}`}
          onClick={handleStart}
          disabled={isCameraActive || startCamera.isPending}
          title="Bắt đầu giám sát"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <path fillRule="evenodd"
              d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z"
              clipRule="evenodd" />
          </svg>
          {startCamera.isPending ? 'Đang kết nối...' : 'BẮT ĐẦU GIÁM SÁT'}
        </button>

        <button
          id="btn-stop-camera"
          className="cam-controls__btn cam-controls__btn--stop"
          onClick={handleStop}
          disabled={!isCameraActive || stopCamera.isPending}
          title="Dừng stream"
        >
          <svg viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
            <rect x="5" y="5" width="10" height="10" rx="1" />
          </svg>
          {stopCamera.isPending ? 'Đang dừng...' : 'Dừng Stream'}
        </button>
      </div>

      <div className="cam-controls__cams">
        {CAMERA_OPTIONS.map((cam) => (
          <button
            key={cam}
            id={`btn-cam-${cam.replace(/\s+/g, '-').toLowerCase()}`}
            className={`cam-controls__cam-btn ${selectedCamera === cam ? 'cam-controls__cam-btn--active' : ''}`}
            onClick={() => setSelectedCamera(cam)}
          >
            {cam}
          </button>
        ))}
      </div>

      {errorMsg && (
        <p className="cam-controls__error">{errorMsg}</p>
      )}

      {cameraStatus && (
        <div className="cam-controls__status-bar">
          <span className={`cam-controls__status-dot ${cameraStatus.is_running ? 'cam-controls__status-dot--on' : ''}`} />
          <span className="cam-controls__status-text">{cameraStatus.message}</span>
        </div>
      )}
    </div>
  );
};
