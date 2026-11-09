import * as XLSX from 'xlsx';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import { useStats } from '../../hooks';
import { useViolations } from '../../hooks';
import type { Violation } from '../../types';
import './StatsCards.css';

// ─── Export utils ─────────────────────────────────────────────────────────────

const exportExcel = (violations: Violation[]) => {
  const rows = violations.map((v, i) => ({
    STT: i + 1,
    'Người #': v.face_id + 1,
    'Bắt đầu': new Date(v.started_at).toLocaleString('vi-VN'),
    'Kết thúc': v.ended_at ? new Date(v.ended_at).toLocaleString('vi-VN') : '—',
    'Thời lượng (s)': v.duration,
    'Ngày tạo': new Date(v.created_at).toLocaleString('vi-VN'),
  }));
  const ws = XLSX.utils.json_to_sheet(rows);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, ws, 'Vi phạm');
  XLSX.writeFile(wb, `bao-cao-vi-pham-${new Date().toISOString().slice(0, 10)}.xlsx`);
};

const exportPDF = (violations: Violation[], stats: { total_today: number; total_this_week: number }) => {
  const doc = new jsPDF();
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(16);
  doc.text('BAO CAO VI PHAM - HE THONG HTTM', 14, 18);
  doc.setFontSize(10);
  doc.setFont('helvetica', 'normal');
  doc.text(`Ngay xuat: ${new Date().toLocaleString('vi-VN')}`, 14, 26);
  doc.text(`Tong hom nay: ${stats.total_today}  |  Tuan nay: ${stats.total_this_week}`, 14, 32);

  autoTable(doc, {
    startY: 40,
    head: [['STT', 'Nguoi #', 'Bat dau', 'Ket thuc', 'Thoi luong (s)']],
    body: violations.map((v, i) => [
      i + 1,
      `Nguoi ${v.face_id + 1}`,
      new Date(v.started_at).toLocaleString('vi-VN'),
      v.ended_at ? new Date(v.ended_at).toLocaleString('vi-VN') : '—',
      v.duration,
    ]),
    styles: { fontSize: 9 },
    headStyles: { fillColor: [59, 130, 246] },
    alternateRowStyles: { fillColor: [248, 250, 252] },
  });

  doc.save(`bao-cao-vi-pham-${new Date().toISOString().slice(0, 10)}.pdf`);
};

// ─── StatsCards ───────────────────────────────────────────────────────────────

interface StatsCardsProps {
  onCameraFilter: (cam: string) => void;
  activeCam: string;
}

export const StatsCards = ({ onCameraFilter, activeCam }: StatsCardsProps) => {
  const { data: stats } = useStats();
  const { data: allViolations } = useViolations({ page: 1, page_size: 100 });

  const totalViolations = stats?.total_today ?? 0;
  const uniqueFaces = allViolations
    ? new Set(allViolations.items.map((v) => v.face_id)).size
    : 0;
  const avgDuration = stats?.avg_duration ?? 0;
  const minDuration = stats?.min_duration ?? 0;
  const maxDuration = stats?.max_duration ?? 0;

  const cameras = [ 'Cam 01 '];

  const handleExport = (type: 'excel' | 'pdf') => {
    const violations = allViolations?.items ?? [];
    if (type === 'excel') exportExcel(violations);
    else exportPDF(violations, { total_today: totalViolations, total_this_week: stats?.total_this_week ?? 0 });
  };

  return (
    <div className="stats-cards">
      <div className="stats-cards__header">
        <div className="stats-cards__header-icon">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
          </svg>
        </div>
        <div>
          <h2 className="stats-cards__title">TỔNG QUAN VI PHẠM BUỔI HỌC</h2>
          <p className="stats-cards__subtitle">Phân tích AI phát hiện hành vi thời gian thực • Ngưỡng vi phạm &gt; 5.0 giây</p>
        </div>
      </div>

      <div className="stats-cards__grid">
        {/* Card 1 */}
        <div className="stats-card stats-card--red">
          <div className="stats-card__label">
            <span>Tổng vi phạm</span>
            <span className="stats-card__icon stats-card__icon--red">
              <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" /></svg>
            </span>
          </div>
          <div className="stats-card__value stats-card__value--red">
            {String(totalViolations).padStart(2, '0')}
            <span className="stats-card__unit"> lần</span>
          </div>
        </div>

        {/* Card 2 */}
        <div className="stats-card stats-card--blue">
          <div className="stats-card__label">
            <span>Sinh viên mắc lỗi</span>
            <span className="stats-card__icon stats-card__icon--blue">
              <svg viewBox="0 0 20 20" fill="currentColor"><path d="M9 6a3 3 0 11-6 0 3 3 0 016 0zM17 6a3 3 0 11-6 0 3 3 0 016 0zM12.93 17c.046-.327.07-.66.07-1a6.97 6.97 0 00-1.5-4.33A5 5 0 0119 16v1h-6.07zM6 11a5 5 0 015 5v1H1v-1a5 5 0 015-5z" /></svg>
            </span>
          </div>
          <div className="stats-card__value stats-card__value--dark">
            {String(uniqueFaces).padStart(2, '0')}
            <span className="stats-card__unit"> / 42 SV</span>
          </div>
          <p className="stats-card__footnote">{uniqueFaces > 0 ? `${((uniqueFaces / 42) * 100).toFixed(1)}% tổng sĩ số lớp` : 'Chưa có dữ liệu'}</p>
        </div>

        {/* Card 3 */}
        <div className="stats-card stats-card--teal">
          <div className="stats-card__label">
            <span>Thời lượng vi phạm TB</span>
            <span className="stats-card__icon stats-card__icon--teal">
              <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-12a1 1 0 10-2 0v4a1 1 0 00.293.707l2.828 2.829a1 1 0 101.415-1.415L11 9.586V6z" clipRule="evenodd" /></svg>
            </span>
          </div>
          <div className="stats-card__value stats-card__value--dark">
            {avgDuration.toFixed(1)}
            <span className="stats-card__unit"> giây</span>
          </div>
          <p className="stats-card__footnote">
            Khoảng thời gian: {minDuration.toFixed(1)}s - {maxDuration.toFixed(1)}s
          </p>
        </div>
      </div>

      <div className="stats-cards__actions">
        <div className="stats-cards__action-btns">
          <button id="btn-continue-monitor" className="stats-cards__btn stats-cards__btn--primary">
            <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" /></svg>
            Tiếp tục giám sát
          </button>
          <button
            id="btn-export-report"
            className="stats-cards__btn stats-cards__btn--outline"
            onClick={() => handleExport('excel')}
          >
            <svg viewBox="0 0 20 20" fill="currentColor"><path fillRule="evenodd" d="M3 17a1 1 0 011-1h12a1 1 0 110 2H4a1 1 0 01-1-1zm3.293-7.707a1 1 0 011.414 0L9 10.586V3a1 1 0 112 0v7.586l1.293-1.293a1 1 0 111.414 1.414l-3 3a1 1 0 01-1.414 0l-3-3a1 1 0 010-1.414z" clipRule="evenodd" /></svg>
            Xuất báo cáo PDF/Excel
          </button>
        </div>

        <div className="stats-cards__cam-filters">
          {cameras.map((cam) => (
            <button
              key={cam}
              id={`btn-filter-${cam.replace(/\s+/g, '-').toLowerCase()}`}
              className={`stats-cards__cam-btn ${activeCam === cam ? 'stats-cards__cam-btn--active' : ''}`}
              onClick={() => onCameraFilter(cam)}
            >
              {cam}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
};
