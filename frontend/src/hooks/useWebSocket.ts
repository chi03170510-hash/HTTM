import { useEffect, useRef, useCallback } from 'react';
import { useAppStore } from '../store/useAppStore';
import type { WSFaceData, WSStatusMessage, WSWarningMessage, WSWarningEndMessage } from '../types';

export type WarningAlert = {
  face_id: number;
  duration: number;
  violation_id: number;
  image_url: string;
  message: string;
  timestamp: string;
};

type UseWebSocketReturn = {
  wsRef: React.MutableRefObject<WebSocket | null>;
  latestFacesRef: React.MutableRefObject<WSFaceData[]>;
  activeAlertsRef: React.MutableRefObject<Map<number, WarningAlert>>;
};

export const useWebSocket = (): UseWebSocketReturn => {
  const wsRef = useRef<WebSocket | null>(null);
  // Dùng ref thay vì state để tránh re-render mỗi frame (~10fps)
  const latestFacesRef = useRef<WSFaceData[]>([]);
  const activeAlertsRef = useRef<Map<number, WarningAlert>>(new Map());

  const { isCameraActive } = useAppStore();

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const wsUrl =
      typeof window !== 'undefined' && window.location.host
        ? `${window.location.protocol === 'https:' ? 'wss:' : 'ws:'}//${window.location.host}/ws/monitor`
        : 'ws://127.0.0.1:8000/ws/monitor';
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onopen = () => {
      console.log('[WS] Connected to /ws/monitor');
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data) as WSStatusMessage | WSWarningMessage | WSWarningEndMessage;

        if (data.type === 'status') {
          const msg = data as WSStatusMessage;
          latestFacesRef.current = msg.faces ?? [];
        } else if (data.type === 'warning') {
          const msg = data as WSWarningMessage;
          activeAlertsRef.current.set(msg.face_id, {
            face_id: msg.face_id,
            duration: msg.duration,
            violation_id: msg.violation_id,
            image_url: msg.image_url,
            message: msg.message,
            timestamp: msg.timestamp,
          });
        } else if (data.type === 'warning_end') {
          const msg = data as WSWarningEndMessage;
          activeAlertsRef.current.delete(msg.face_id);
        }
      } catch (error) {
        console.error('[WS] Parse error:', error);
      }
    };

    ws.onclose = () => {
      console.log('[WS] Disconnected');
      latestFacesRef.current = [];
    };

    ws.onerror = (e) => {
      console.error('[WS] Error:', e);
    };
  }, []);

  useEffect(() => {
    if (!isCameraActive) {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
      latestFacesRef.current = [];
      activeAlertsRef.current.clear();
      return;
    }

    connect();

    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [isCameraActive, connect]);

  return { wsRef, latestFacesRef, activeAlertsRef };
};
