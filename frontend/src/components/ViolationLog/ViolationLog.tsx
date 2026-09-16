import { useState, useCallback } from 'react';
import { useViolations } from '../../hooks';
import type { Violation } from '../../types';
import './ViolationLog.css';

// ─── Snapshot Modal ────────────────────────────────────────────────────────────
interface SnapshotModalProps {
  violation: Violation;
  onClose: () => void;
}

const SnapshotModal = ({ violation, onClose }: SnapshotModalProps) => {
  const imgSrc = `/api/violations/${violation.id}/image`;

  return (
    <div className="vlog-modal__backdrop" onClick={onClose} role="dialog" aria-modal="true">
      <div className="vlog-modal" onClick={(e) => e.stopPropagation()}>
        <div className="vlog-modal__header">
          <div>
            <h3 className="vlog-modal__title">Ảnh chụp bằng chứng vi phạm</h3>
            <p className="vlog-modal__sub">
              Người #{violation.face_id + 1} • {new Date(violation.started_at).toLocaleTimeString('vi-VN')}
            </p>
          </div>
          <button
            id={`btn-close-modal-${violation.id}`}
            className="vlog-modal__close"
            onClick={onClose}
            aria-label="Đóng"
          >
            <svg viewBox="0 0 20 20" fill="currentColor">
              <path fillRule="evenodd" d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z" clipRule="evenodd" />
            </svg>
          </button>
        </div>
        <div className="vlog-modal__body">
          <img
            src={imgSrc}
            alt={`Bằng chứng vi phạm #${violation.id}`}
            className="vlog-modal__img"
            onError={(e) => {
              const t = e.target as HTMLImageElement;
              t.src = 'data:image/svg+xml,' + encodeURIComponent(
                '<svg xmlns="http://www.w3.org/2000/svg" width="400" height="300"><rect fill="#1e293b" width="400" height="300"/><text fill="#64748b" x="50%" y="50%" text-anchor="middle" dy=".3em" font-size="14" font-family="sans-serif">Ảnh không khả dụng</text></svg>'
              );
            }}
          />
        </div>
        <div className="vlog-modal__footer">
          <span className="vlog-modal__badge">Vi phạm</span>
          <span className="vlog-modal__meta">Thời lượng: <strong>{violation.duration.toFixed(1)}s</strong></span>
          <span className="vlog-modal__meta">ID vi phạm: <strong>#{violation.id}</strong></span>
        </div>
      </div>
    </div>
  );
};

// ─── Violation Card ────────────────────────────────────────────────────────────
interface ViolationCardProps {
  violation: Violation;
  onViewSnapshot: (v: Violation) => void;
}

const ViolationCard = ({ violation, onViewSnapshot }: ViolationCardProps) => {
  const startTime = new Date(violation.started_at).toLocaleTimeString('vi-VN', {
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  });

  return (
    <div className="vlog-card">
      {/* Left */}
      <div className="vlog-card__left">
        <div className="vlog-card__top-row">
          <span className="vlog-card__badge">Vi phạm</span>
          <span className="vlog-card__time">
            <svg viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M8 1a7 7 0 100 14A7 7 0 008 1zM2 8a6 6 0 1112 0A6 6 0 012 8zm6.5-4a.5.5 0 00-1 0v4a.5.5 0 00.146.354l2 2a.5.5 0 00.708-.708L8.5 7.793V4z" clipRule="evenodd" />
            </svg>
            {startTime}
          </span>
        </div>

        <p className="vlog-card__student">
          Người #{violation.face_id + 1} • MSSV: SV{String(violation.face_id + 1).padStart(9, '2026160081')} • Bàn {violation.face_id + 1}
        </p>

        <h4 className="vlog-card__type">Người #{violation.face_id + 1} – Nhắm mắt liên tục (&gt; 5s)</h4>

        <div className="vlog-card__metrics">
          <div className="vlog-card__metric-pair">
            <span className="vlog-card__metric-label">Thời lượng:</span>
            <span className="vlog-card__metric-val vlog-card__metric-val--red">
              {violation.duration.toFixed(1)}s
            </span>
          </div>
          
        </div>

       
      </div>

      {/* Right: Snapshot */}
      <button
        id={`btn-snapshot-card-${violation.id}`}
        className="vlog-card__snapshot"
        onClick={() => onViewSnapshot(violation)}
      >
        {violation.image_path ? (
          <img
            src={`/api/violations/${violation.id}/image`}
            alt="Snapshot"
            className="vlog-card__snapshot-img"
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = 'none';
              (e.target as HTMLImageElement).nextElementSibling?.removeAttribute('style');
            }}
          />
        ) : null}
        <div className="vlog-card__snapshot-overlay" style={violation.image_path ? { display: 'none' } : {}}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="2" y="5" width="20" height="15" rx="2" />
            <circle cx="12" cy="12" r="3.5" />
            <path strokeLinecap="round" d="M8.5 5L10 2.5h4L15.5 5" />
          </svg>
        </div>
        <div className="vlog-card__snapshot-label">
          <svg viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5" aria-hidden="true">
            <rect x="2" y="5" width="16" height="12" rx="2" />
            <circle cx="10" cy="11" r="3" />
          </svg>
          Xem snapshot bằng chứng
        </div>
      </button>
    </div>
  );
};

// ─── Pagination ────────────────────────────────────────────────────────────────
interface PaginationProps {
  page: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

const Pagination = ({ page, totalPages, onPageChange }: PaginationProps) => {
  if (totalPages <= 1) return null;

  const pages = Array.from({ length: totalPages }, (_, i) => i + 1);

  return (
    <div className="vlog-pagination">
      <button
        id="btn-prev-page"
        className="vlog-pagination__btn"
        disabled={page === 1}
        onClick={() => onPageChange(page - 1)}
      >
        <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M12.707 5.293a1 1 0 010 1.414L9.414 10l3.293 3.293a1 1 0 01-1.414 1.414l-4-4a1 1 0 010-1.414l4-4a1 1 0 011.414 0z" clipRule="evenodd" /></svg>
        Trang trước
      </button>

      <div className="vlog-pagination__pages">
        {pages.slice(Math.max(0, page - 3), Math.min(totalPages, page + 2)).map((p) => (
          <button
            key={p}
            id={`btn-page-${p}`}
            className={`vlog-pagination__page ${p === page ? 'vlog-pagination__page--active' : ''}`}
            onClick={() => onPageChange(p)}
          >
            {p}
          </button>
        ))}
      </div>

      <button
        id="btn-next-page"
        className="vlog-pagination__btn"
        disabled={page === totalPages}
        onClick={() => onPageChange(page + 1)}
      >
        Trang sau
        <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M7.293 14.707a1 1 0 010-1.414L10.586 10 7.293 6.707a1 1 0 011.414-1.414l4 4a1 1 0 010 1.414l-4 4a1 1 0 01-1.414 0z" clipRule="evenodd" /></svg>
      </button>
    </div>
  );
};

// ─── ViolationLog ──────────────────────────────────────────────────────────────
const PAGE_SIZE = 20;

interface ViolationLogProps {
  dateFrom?: string;
  dateTo?: string;
}

export const ViolationLog = ({ dateFrom, dateTo }: ViolationLogProps) => {
  const [page, setPage] = useState(1);
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);

  const { data, isLoading, isError } = useViolations({
    page,
    page_size: PAGE_SIZE,
    date_from: dateFrom,
    date_to: dateTo,
  });

  const handleClose = useCallback(() => setSelectedViolation(null), []);

  const total = data?.total ?? 0;
  const totalPages = data?.total_pages ?? 1;

  return (
    <div className="vlog">
      <div className="vlog__header">
        <div className="vlog__title-area">
          <span className="vlog__title-icon">📋</span>
          <h2 className="vlog__title">NHẬT KÝ LỖI BUỔI HỌC</h2>
          {total > 0 && (
            <span className="vlog__count-badge">{total} sự kiện</span>
          )}
        </div>
        <button id="btn-filter-log" className="vlog__filter-btn" aria-label="Tuỳ chỉnh bộ lọc">
          <svg viewBox="0 0 20 20" fill="currentColor">
            <path d="M5 4a1 1 0 00-2 0v7.268a2 2 0 000 3.464V16a1 1 0 102 0v-1.268a2 2 0 000-3.464V4zM11 4a1 1 0 10-2 0v1.268a2 2 0 000 3.464V16a1 1 0 102 0V8.732a2 2 0 000-3.464V4zM16 3a1 1 0 011 1v7.268a2 2 0 010 3.464V16a1 1 0 11-2 0v-1.268a2 2 0 010-3.464V4a1 1 0 011-1z" />
          </svg>
        </button>
      </div>

      <div className="vlog__body">
        {isLoading && (
          <div className="vlog__loading">
            <div className="vlog__spinner" />
            <span>Đang tải dữ liệu...</span>
          </div>
        )}

        {isError && (
          <div className="vlog__error">
            <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
            <span>Không thể tải dữ liệu. Kiểm tra kết nối backend.</span>
          </div>
        )}

        {!isLoading && !isError && data?.items.length === 0 && (
          <div className="vlog__empty">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
            </svg>
            <p>Chưa có vi phạm nào được ghi nhận</p>
          </div>
        )}

        {data?.items.map((v) => (
          <ViolationCard key={v.id} violation={v} onViewSnapshot={setSelectedViolation} />
        ))}
      </div>

      <Pagination page={page} totalPages={totalPages} onPageChange={setPage} />

      {selectedViolation && (
        <SnapshotModal violation={selectedViolation} onClose={handleClose} />
      )}
    </div>
  );
};
