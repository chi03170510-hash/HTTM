import { CameraFeed } from '../../components/CameraFeed';
import { CameraControls } from '../../components/CameraControls';
import { RealtimeAlerts } from '../../components/RealtimeAlerts';
import './MonitorPage.css';

interface MonitorPageProps {
  onNavigate: (page: 'monitor' | 'history') => void;
}

export const MonitorPage = ({ onNavigate }: MonitorPageProps) => {
  const now = new Date();
  const dateStr = now.toLocaleDateString('vi-VN', { day: '2-digit', month: '2-digit', year: 'numeric' });
  const timeStr = `Ca sáng: 08:30 - 11:00 • ${dateStr}`;

  return (
    <div className="monitor-page">
      {/* ── Topbar ── */}
      <header className="monitor-page__topbar">
        <div className="monitor-page__topbar-left">
          <div className="monitor-page__logo">
            <svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
              <path d="M2 3.993A1 1 0 012.992 3h18.016A1 1 0 0122 3.993v16.014a1 1 0 01-.992.993H2.992A1 1 0 012 20.007V3.993z" />
            </svg>
            <span>CLASSROOM</span>
          </div>
          <nav className="monitor-page__nav">
            <a href="#" id="nav-trangchu" className="monitor-page__nav-link monitor-page__nav-link--active" onClick={(e) => e.preventDefault()}>Trang chủ</a>
            <a href="#" id="nav-violations" className="monitor-page__nav-link" onClick={(e) => { e.preventDefault(); onNavigate('history'); }}>Nhật ký vi phạm</a>
          </nav>
        </div>
        <div className="monitor-page__topbar-right">
          <button id="btn-notification" className="monitor-page__icon-btn" aria-label="Thông báo">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path strokeLinecap="round" strokeLinejoin="round" d="M15 17h5l-1.405-1.405A2.032 2.032 0 0118 14.158V11a6.002 6.002 0 00-4-5.659V5a2 2 0 10-4 0v.341C7.67 6.165 6 8.388 6 11v3.159c0 .538-.214 1.055-.595 1.436L4 17h5m6 0v1a3 3 0 11-6 0v-1m6 0H9" />
            </svg>
          </button>
          <div className="monitor-page__user">
            <div className="monitor-page__user-info">
              <span className="monitor-page__user-name">Giảng viên Giám thị</span>
              <span className="monitor-page__user-room">Phòng A2-302</span>
            </div>
            <div className="monitor-page__avatar">GT</div>
          </div>
        </div>
      </header>

      {/* ── Session Banner ── */}
      <div className="monitor-page__session-banner">
        <div className="monitor-page__session-left">
          <span className="monitor-page__room-badge">
            <span className="monitor-page__room-dot" />
            PHÒNG A2–302
          </span>
          <span className="monitor-page__session-time">
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M5.75 2a.75.75 0 01.75.75V4h3V2.75a.75.75 0 011.5 0V4h.25A2.75 2.75 0 0114 6.75v6.5A2.75 2.75 0 0111.25 16h-6.5A2.75 2.75 0 012 13.25v-6.5A2.75 2.75 0 014.75 4H5V2.75A.75.75 0 015.75 2z" clipRule="evenodd" />
            </svg>
            {timeStr}
          </span>
        </div>
        <div className="monitor-page__session-right">
          <button id="btn-current-session" className="monitor-page__session-btn monitor-page__session-btn--active">
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true"><circle cx="8" cy="8" r="3" /></svg>
            Buổi hiện tại
          </button>
          <button id="btn-weekly-history" className="monitor-page__session-btn">
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M8 1a7 7 0 100 14A7 7 0 008 1zM2 8a6 6 0 1112 0A6 6 0 012 8zm6.5-4a.5.5 0 00-1 0v4a.5.5 0 00.146.354l2 2a.5.5 0 00.708-.708L8.5 7.793V4z" clipRule="evenodd" />
            </svg>
            Lịch sử tuần
          </button>
        </div>
      </div>

      {/* ── Main layout ── */}
      <main className="monitor-page__main">
        {/* Left: Camera panel */}
        <section className="monitor-page__left">
          <div className="monitor-page__camera-panel">
            <div className="monitor-page__camera-header">
              <span className="monitor-page__rec-badge">
                <span className="monitor-page__rec-dot" />
                REC
              </span>
              <span className="monitor-page__live-label">• LIVE WEBCAM</span>
              <span className="monitor-page__clock" id="live-clock" suppressHydrationWarning>
                {now.toLocaleTimeString('vi-VN')} GMT+7
              </span>
            </div>
            <CameraFeed />
            <CameraControls />
          </div>
        </section>

        {/* Right: Alerts panel */}
        <aside className="monitor-page__right">
          <RealtimeAlerts />
        </aside>
      </main>
    </div>
  );
};
