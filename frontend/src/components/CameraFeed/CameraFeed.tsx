import { useEffect, useRef } from 'react';
import { useAppStore } from '../../store/useAppStore';
import { useWebSocket } from '../../hooks';
import type { WSFaceData } from '../../types';
import './CameraFeed.css';

// Màu bounding-box dựa theo trạng thái mắt
const getBorderColor = (face: WSFaceData): string => {
  if (face.is_warning) return '#ef4444';   // Đỏ - đang vi phạm (nhắm mắt >= 5s)
  if (face.eye_state === 'CLOSED') return '#f59e0b'; // Vàng - nhắm mắt nhưng chưa đến ngưỡng
  return '#22c55e'; // Xanh lá - mắt mở bình thường
};

const getStatusLabel = (face: WSFaceData): string => {
  if (face.is_warning) return `⚠ VI PHẠM  ${face.closed_duration.toFixed(1)}s`;
  if (face.eye_state === 'CLOSED') return `Nhắm mắt ${face.closed_duration.toFixed(1)}s`;
  return 'Bình thường';
};

// Vẽ các thẻ thông tin dạng "card" thay vì bounding box (do backend không gửi bbox tọa độ)
const drawFaceInfo = (
  ctx: CanvasRenderingContext2D,
  face: WSFaceData,
  index: number,
  _canvasWidth: number,
) => {
  const CARD_W = 200;
  const CARD_H = 78;
  const MARGIN = 12;
  const START_X = MARGIN;
  const START_Y = MARGIN + index * (CARD_H + MARGIN);

  const color = getBorderColor(face);
  const label = getStatusLabel(face);
  const isWarning = face.is_warning;

  // Nền mờ
  ctx.save();
  ctx.globalAlpha = 0.78;
  ctx.fillStyle = isWarning ? '#3b0000' : '#0f172a';
  ctx.beginPath();
  ctx.roundRect(START_X, START_Y, CARD_W, CARD_H, 8);
  ctx.fill();
  ctx.globalAlpha = 1;

  // Viền màu
  ctx.strokeStyle = color;
  ctx.lineWidth = isWarning ? 2.5 : 1.5;
  ctx.beginPath();
  ctx.roundRect(START_X, START_Y, CARD_W, CARD_H, 8);
  ctx.stroke();
  ctx.restore();

  // Thanh tiêu đề
  ctx.save();
  ctx.fillStyle = color;
  ctx.globalAlpha = 0.2;
  ctx.beginPath();
  ctx.roundRect(START_X, START_Y, CARD_W, 22, [8, 8, 0, 0]);
  ctx.fill();
  ctx.globalAlpha = 1;
  ctx.restore();

  // Text tiêu đề
  ctx.save();
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.fillStyle = color;
  ctx.fillText(`Người ${face.face_id + 1}`, START_X + 8, START_Y + 14);
  ctx.restore();

  // EAR
  ctx.save();
  ctx.font = '11px "JetBrains Mono", monospace';
  ctx.fillStyle = '#cbd5e1';
  ctx.fillText(`EAR: ${face.avg_ear.toFixed(3)}`, START_X + 8, START_Y + 36);
  ctx.fillStyle = color;
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.fillText(label, START_X + 8, START_Y + 54);
  ctx.restore();

  // Thanh thời gian đóng mắt (chỉ hiện khi đang nhắm)
  if (face.eye_state === 'CLOSED') {
    const WARNING_THRESHOLD = 5.0;
    const progress = Math.min(face.closed_duration / WARNING_THRESHOLD, 1);
    const BAR_Y = START_Y + CARD_H - 6;
    const BAR_W = CARD_W - 16;

    ctx.save();
    ctx.globalAlpha = 0.3;
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.roundRect(START_X + 8, BAR_Y, BAR_W, 4, 2);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(START_X + 8, BAR_Y, BAR_W * progress, 4, 2);
    ctx.fill();
    ctx.restore();
  }
};

export const CameraFeed = () => {
  const imgRef = useRef<HTMLImageElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animFrameRef = useRef<number>(0);
  const isCameraActive = useAppStore((s) => s.isCameraActive);

  const { latestFacesRef, activeAlertsRef } = useWebSocket();

  // RAF loop để vẽ canvas overlay
  useEffect(() => {
    if (!isCameraActive) {
      cancelAnimationFrame(animFrameRef.current);
      const canvas = canvasRef.current;
      if (canvas) {
        const ctx = canvas.getContext('2d');
        ctx?.clearRect(0, 0, canvas.width, canvas.height);
      }
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const ctx = canvas.getContext('2d');
      if (!ctx) return;

      // Đồng bộ kích thước canvas với thẻ img video
      const img = imgRef.current;
      if (img && img.naturalWidth > 0) {
        if (canvas.width !== img.clientWidth || canvas.height !== img.clientHeight) {
          canvas.width = img.clientWidth;
          canvas.height = img.clientHeight;
        }
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const faces = latestFacesRef.current;

      // Vẽ thông tin từng face
      faces.forEach((face, idx) => {
        drawFaceInfo(ctx, face, idx, canvas.width);
      });

      // Hiệu ứng flash đỏ khi có cảnh báo
      if (activeAlertsRef.current.size > 0) {
        const pulse = Math.sin(Date.now() / 200) * 0.5 + 0.5; // 0→1→0
        ctx.save();
        ctx.strokeStyle = `rgba(239,68,68,${pulse * 0.8})`;
        ctx.lineWidth = 6;
        ctx.strokeRect(3, 3, canvas.width - 6, canvas.height - 6);
        ctx.restore();
      }

      animFrameRef.current = requestAnimationFrame(draw);
    };

    animFrameRef.current = requestAnimationFrame(draw);

    return () => {
      cancelAnimationFrame(animFrameRef.current);
    };
  }, [isCameraActive, latestFacesRef, activeAlertsRef]);

  return (
    <div className="camera-feed">
      <div className="camera-feed__wrapper">
        {isCameraActive ? (
          <>
            <img
              ref={imgRef}
              src="/api/camera/video_feed"
              alt="Camera stream"
              className="camera-feed__video"
              onError={() => console.error('MJPEG stream error')}
            />
            <canvas ref={canvasRef} className="camera-feed__canvas" />
          </>
        ) : (
          <div className="camera-feed__placeholder">
            <div className="camera-feed__placeholder-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9a2.25 2.25 0 00-2.25-2.25h-9A2.25 2.25 0 002.25 7.5v9a2.25 2.25 0 002.25 2.25z"
                />
              </svg>
            </div>
            <p className="camera-feed__placeholder-text">Camera chưa được bật</p>
            <p className="camera-feed__placeholder-subtext">Nhấn nút "Bắt đầu giám sát" để khởi động</p>
          </div>
        )}
      </div>

      {/* Badge trạng thái live */}
      {isCameraActive && (
        <div className="camera-feed__badge">
          <span className="camera-feed__badge-dot" />
          LIVE
        </div>
      )}
    </div>
  );
};
