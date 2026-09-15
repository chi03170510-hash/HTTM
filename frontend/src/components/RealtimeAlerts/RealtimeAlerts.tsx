import { useState, useEffect, useCallback } from 'react';
import { useWebSocket } from '../../hooks';
import type { WarningAlert } from '../../hooks/useWebSocket';
import './RealtimeAlerts.css';

// ─── Snapshot Modal ────────────────────────────────────────────────────────────

interface SnapshotModalProps {
  alert: WarningAlert;
  onClose: () => void;
}

const SnapshotModal = ({ alert, onClose }: SnapshotModalProps) => {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [onClose]);

  return (
    <div className="snapshot-modal__backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="snapshot-modal" onClick={(e) => e.stopPropagation()}>
        <div className="snapshot-modal__header">
          <div>
            <h3 className="snapshot-modal__title">Ảnh chụp vi phạm</h3>
            <p className="snapshot-modal__sub">Người {alert.face_id + 1} • {new Date(alert.timestamp).toLocaleTimeString('vi-VN')}</p>
          </div>
          <button id="btn-modal-close" className="snapshot-modal__close" onClick={onClose} aria-label="Đóng">
            <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
          </button>
        </div>
        <div className="snapshot-modal__body">
          <img
            src={`/api${alert.image_url.startsWith('/api') ? alert.image_url.slice(4) : alert.image_url}`}
            alt={`Ảnh vi phạm người ${alert.face_id + 1}`}
            className="snapshot-modal__img"
            onError={(e) => { (e.target as HTMLImageElement).src = 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect fill="%231e293b" width="400" height="300"/><text fill="%2364748b" x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="14" font-family="sans-serif">Ảnh không khả dụng</text></svg>'; }}
          />
        </div>
        <div className="snapshot-modal__footer">
          <div className="snapshot-modal__meta">
            <span className="snapshot-modal__badge snapshot-modal__badge--red">Vi phạm</span>
            <span>Thời lượng: <strong>{alert.duration.toFixed(1)}s</strong></span>
            <span>ID: #{alert.violation_id}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// ─── Alert Card ─────────────────────────────────────────────────────────────────

interface AlertCardProps {
  alert: WarningAlert;
  isActive: boolean;
  onViewSnapshot: (alert: WarningAlert) => void;
}

const AlertCard = ({ alert, isActive, onViewSnapshot }: AlertCardProps) => {
  const time = new Date(alert.timestamp).toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit', second: '2-digit' });

  return (
    <div className={`alert-card ${isActive ? 'alert-card--active' : 'alert-card--history'}`}>
      <div className="alert-card__header">
        <span className={`alert-card__badge ${isActive ? 'alert-card__badge--red' : 'alert-card__badge--gray'}`}>
          {isActive ? (
            <><span className="alert-card__badge-dot" />Vi phạm</>
          ) : (
            <><svg viewBox="0 0 16 16" fill="currentColor" className="alert-card__clock-icon"><path fillRule="evenodd" d="M8 1a7 7 0 100 14A7 7 0 008 1zM2 8a6 6 0 1112 0A6 6 0 012 8zm6.5-4a.5.5 0 00-1 0v4a.5.5 0 00.146.354l2 2a.5.5 0 00.708-.708L8.5 7.793V4z" clipRule="evenodd" /></svg>Lịch sử gần</>
          )}
        </span>
        <span className="alert-card__time">{time}</span>
      </div>

      <p className="alert-card__name">Người {alert.face_id + 1} – Nhắm mắt liên tục</p>
      <p className="alert-card__duration">Thời lượng nhắm: <strong>{alert.duration.toFixed(1)}s</strong></p>

      <div className="alert-card__metrics">
        <div className="alert-card__metric">
          <span className="alert-card__metric-label">EAR</span>
          <span className="alert-card__metric-value alert-card__metric-value--red">0.081</span>
        </div>
        <div className="alert-card__metric">
          <span className="alert-card__metric-label">Độ tin cậy</span>
          <span className="alert-card__metric-value">98.9%</span>
        </div>
        <button
          id={`btn-snapshot-${alert.violation_id}`}
          className="alert-card__snapshot-btn"
          onClick={() => onViewSnapshot(alert)}
        >
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="2" y="5" width="16" height="12" rx="2" />
            <circle cx="10" cy="11" r="3" />
            <path strokeLinecap="round" d="M7 5l1.5-2h3L13 5" />
          </svg>
          Xem snapshot
        </button>
      </div>
    </div>
  );
};

// ─── RealtimeAlerts ────────────────────────────────────────────────────────────

export const RealtimeAlerts = () => {
  const { activeAlertsRef } = useWebSocket();
  const [alerts, setAlerts] = useState<{ alert: WarningAlert; isActive: boolean }[]>([]);
  const [modalAlert, setModalAlert] = useState<WarningAlert | null>(null);

  // Poll alertsRef mỗi 500ms để cập nhật UI
  useEffect(() => {
    const id = setInterval(() => {
      const active = Array.from(activeAlertsRef.current.values()).map(a => ({ alert: a, isActive: true }));
      setAlerts(prev => {
        const activeIds = new Set(active.map(a => a.alert.violation_id));
        // Giữ lại các alert lịch sử (isActive=false) không trùng với active hiện tại
        const history = prev.filter(p => !p.isActive && !activeIds.has(p.alert.violation_id));
        // Các alert vừa kết thúc → chuyển sang history
        const newlyEnded = prev
          .filter(p => p.isActive && !activeIds.has(p.alert.violation_id))
          .map(p => ({ ...p, isActive: false }));
        // Giữ tối đa 10 item lịch sử
        const allHistory = [...newlyEnded, ...history].slice(0, 10);
        return [...active, ...allHistory];
      });
    }, 500);
    return () => clearInterval(id);
  }, [activeAlertsRef]);

  const handleCloseModal = useCallback(() => setModalAlert(null), []);

  return (
    <div className="realtime-alerts">
      <div className="realtime-alerts__header">
        <div className="realtime-alerts__title">
          <span className="realtime-alerts__title-icon">🚨</span>
          <h2 className="realtime-alerts__title-text">CẢNH BÁO REALTIME</h2>
        </div>
        <button id="btn-filter-alerts" className="realtime-alerts__filter-btn" aria-label="Lọc cảnh báo">
          <svg viewBox="0 0 20 20" fill="currentColor">
            <path fillRule="evenodd" d="M3 3a1 1 0 011-1h12a1 1 0 011 1v3a1 1 0 01-.293.707L13 10.414V17a1 1 0 01-.553.894l-4-2A1 1 0 018 15v-4.586L3.293 6.707A1 1 0 013 6V3z" clipRule="evenodd" />
          </svg>
        </button>
      </div>

      <div className="realtime-alerts__list">
        {alerts.length === 0 ? (
          <div className="realtime-alerts__empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
            <p>Chưa có cảnh báo nào</p>
            <span>Hệ thống đang giám sát...</span>
          </div>
        ) : (
          alerts.map(({ alert, isActive }) => (
            <AlertCard
              key={`${alert.violation_id}-${isActive}`}
              alert={alert}
              isActive={isActive}
              onViewSnapshot={setModalAlert}
            />
          ))
        )}
      </div>

      {modalAlert && (
        <SnapshotModal alert={modalAlert} onClose={handleCloseModal} />
      )}
    </div>
  );
};
