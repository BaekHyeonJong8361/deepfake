import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import s from './HistoryPage.module.css';
import AuthBackground from '../components/common/AuthBackground';
import { useAuthStore } from '../stores/authStore';
import { useHistoryStore } from '../stores/historyStore';

interface HistoryRecord {
  id: string;
  thumb: string;
  duration: string;
  title: string;
  date: string;
  size: string;
  result: 'FAKE' | 'REAL';
  percentage: string;
}

const HISTORY_DATA: HistoryRecord[] = [
  {
    id: '1',
    thumb: 'https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?w=300&q=80',
    duration: '0:45',
    title: 'president_speech_dub_v2.mp4',
    date: 'Oct 24, 2023 \u2022 14:23',
    size: '12.4 MB',
    result: 'FAKE',
    percentage: '98.2%',
  },
  {
    id: '2',
    thumb: 'https://images.unsplash.com/photo-1522071820081-009f0129c71c?w=300&q=80',
    duration: '1:12',
    title: 'team_meeting_zoom_rec.mp4',
    date: 'Oct 23, 2023 \u2022 09:15',
    size: '45.1 MB',
    result: 'REAL',
    percentage: '99.8%',
  },
  {
    id: '3',
    thumb: 'https://images.unsplash.com/photo-1492684223066-81342ee5ff30?w=300&q=80',
    duration: '0:15',
    title: 'celebrity_endorsement_tiktok.mp4',
    date: 'Oct 21, 2023 \u2022 18:42',
    size: '4.2 MB',
    result: 'FAKE',
    percentage: '94.5%',
  },
  {
    id: '4',
    thumb: 'https://images.unsplash.com/photo-1506744626753-1fa30d22e3a9?w=300&q=80',
    duration: '2:30',
    title: 'news_broadcast_clip_01.mp4',
    date: 'Oct 20, 2023 \u2022 11:05',
    size: '82.7 MB',
    result: 'REAL',
    percentage: '97.1%',
  },
  {
    id: '5',
    thumb: 'https://images.unsplash.com/photo-1533090161767-e6ffed986c88?w=300&q=80',
    duration: '0:08',
    title: 'funny_cat_meme_generated.mp4',
    date: 'Oct 18, 2023 \u2022 22:14',
    size: '1.8 MB',
    result: 'FAKE',
    percentage: '88.9%',
  },
];

type FilterType = 'All' | 'FAKE' | 'REAL';

export default function HistoryPage() {
  const [activeFilter, setActiveFilter] = useState<FilterType>('All');
  const [searchQuery, setSearchQuery] = useState('');
  const [sortBy, setSortBy] = useState('newest');
  const [currentPage, setCurrentPage] = useState(1);
  const { isAuthenticated } = useAuthStore();
  const { records, isLoading, error, fetchHistory, deleteHistory } = useHistoryStore();

  const filters: FilterType[] = ['All', 'FAKE', 'REAL'];
  const historyData = useMemo<HistoryRecord[]>(() => {
    if (!records.length) return HISTORY_DATA;

    return records.map((record) => ({
      id: record.id,
      thumb: record.thumb ?? 'https://images.unsplash.com/photo-1516259762381-22954d7d3ad2?w=300&q=80',
      duration: record.duration ?? '--:--',
      title: record.title,
      date: record.date || 'Unknown date',
      size: record.size ?? '',
      result: record.result,
      percentage: record.percentage || '-',
    }));
  }, [records]);

  useEffect(() => {
    if (!isAuthenticated) return;
    fetchHistory().catch(() => undefined);
  }, [fetchHistory, isAuthenticated]);

  const filtered = historyData.filter((item) => {
    if (activeFilter !== 'All' && item.result !== activeFilter) return false;
    if (searchQuery && !item.title.toLowerCase().includes(searchQuery.toLowerCase()) && !item.date.toLowerCase().includes(searchQuery.toLowerCase())) return false;
    return true;
  });

  const handleDelete = (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    e.preventDefault();
    if (!isAuthenticated) return;
    deleteHistory(id).catch(() => undefined);
  };

  return (
    <div className={s.page}>
      <AuthBackground />
      {/* Header */}
      <header className={s.header}>
        <div className={s.headerInner}>
          <Link to="/" className={s.logo}>
            <svg viewBox="0 0 24 24" fill="none" className={s.logoIcon}>
              <circle cx="9" cy="12" r="7" stroke="currentColor" strokeWidth="2" />
              <circle cx="15" cy="12" r="7" stroke="currentColor" strokeWidth="2" strokeDasharray="4 4" />
            </svg>
            Solomon AI
          </Link>
          <nav className={s.navLinks}>
            <Link to="/" className={s.navLink}>Scan</Link>
            <Link to="/history" className={`${s.navLink} ${s.navLinkActive}`}>History</Link>
            <Link to="/login" className={s.navLink}>Profile</Link>
          </nav>
        </div>
      </header>

      {/* Content */}
      <div className={s.container}>
        <div className={s.pageHeader}>
          <h1 className={s.pageTitle}>Analysis History</h1>
          <p className={s.pageSubtitle}>
            {isLoading ? 'Loading your analysis records...' : 'Review your past video analysis records and reports.'}
          </p>
          {error && <p className={s.errorText}>{error}</p>}
        </div>

        {/* Summary Cards */}
        <div className={s.summaryGrid}>
          <div className={s.summaryCard}>
            <div className={s.summaryLabel}>Total Analyses</div>
            <div className={s.summaryValue}>{historyData.length}</div>
          </div>
          <div className={s.summaryCard}>
            <div className={s.summaryLabel}>FAKE Detections</div>
            <div className={`${s.summaryValue} ${s.summaryValueRed}`}>
              {historyData.filter((item) => item.result === 'FAKE').length}
            </div>
          </div>
          <div className={s.summaryCard}>
            <div className={s.summaryLabel}>Avg. Detection Prob.</div>
            <div className={s.summaryValue}>{getAveragePercentage(historyData)}</div>
          </div>
        </div>

        {/* Controls */}
        <div className={s.controlsBar}>
          <div className={s.filterTabs}>
            {filters.map((f) => (
              <button
                key={f}
                className={`${s.filterTab} ${activeFilter === f ? s.filterTabActive : ''}`}
                onClick={() => setActiveFilter(f)}
              >
                {f}
              </button>
            ))}
          </div>
          <div className={s.searchSort}>
            <div className={s.searchInput}>
              <svg className={s.searchIcon} width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8" />
                <line x1="21" y1="21" x2="16.65" y2="16.65" />
              </svg>
              <input
                type="text"
                placeholder="Search filename or date..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <select className={s.sortSelect} value={sortBy} onChange={(e) => setSortBy(e.target.value)}>
              <option value="newest">Newest First</option>
              <option value="oldest">Oldest First</option>
              <option value="prob_high">Highest Probability</option>
              <option value="prob_low">Lowest Probability</option>
            </select>
          </div>
        </div>

        {/* History List */}
        <div className={s.historyList}>
          {filtered.map((item) => (
            <Link to="/scan/analysis" key={item.id} className={s.historyItem}>
              <div className={s.itemThumb}>
                <img src={item.thumb} alt="Thumbnail" />
                <span className={s.duration}>{item.duration}</span>
              </div>
              <div className={s.itemDetails}>
                <div className={s.itemTitle}>{item.title}</div>
                <div className={s.itemMeta}>
                  <span>{item.date}</span>
                  <span>{item.size}</span>
                </div>
              </div>
              <div className={s.itemResult}>
                <span className={`${s.badge} ${item.result === 'FAKE' ? s.badgeFake : s.badgeReal}`}>
                  {item.result}
                </span>
                <span className={s.percentage}>{item.percentage}</span>
              </div>
              <div className={s.itemActions}>
                <button className={s.btnIcon} aria-label="Delete record" onClick={(e) => handleDelete(e, item.id)}>
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <polyline points="3 6 5 6 21 6" />
                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                  </svg>
                </button>
              </div>
            </Link>
          ))}
        </div>

        {/* Pagination */}
        <div className={s.pagination}>
          {[1, 2, 3, '...', 12].map((p, i) => (
            <button
              key={i}
              className={`${s.pageBtn} ${p === currentPage ? s.pageBtnActive : ''}`}
              onClick={() => typeof p === 'number' && setCurrentPage(p)}
            >
              {p}
            </button>
          ))}
        </div>
      </div>

      {/* Footer */}
      <div className={s.footer}>
        <span>Solomon AI by RunningSnail</span>
        <a href="#">About</a> &middot;{' '}
        <a href="#">Contact</a> &middot;{' '}
        <a href="#">Blog</a> &middot;{' '}
        <a href="#">FAQ</a>
      </div>
    </div>
  );
}

function getAveragePercentage(records: HistoryRecord[]) {
  const values = records
    .map((record) => Number(record.percentage.replace('%', '')))
    .filter((value) => Number.isFinite(value));

  if (!values.length) return '-';
  return `${(values.reduce((sum, value) => sum + value, 0) / values.length).toFixed(1)}%`;
}
