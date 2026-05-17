import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, Filter, Image, Film, Mic, Layers, ChevronRight } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';

const MEDIA_ICONS = {
  image:     <Image  size={13} style={{ color: 'var(--accent)' }} />,
  video:     <Film   size={13} style={{ color: 'var(--accent)' }} />,
  audio:     <Mic    size={13} style={{ color: 'var(--accent)' }} />,
  multimodal:<Layers size={13} style={{ color: 'var(--accent)' }} />,
};

function RiskBadge({ score }) {
  const pct = score <= 1 ? score * 100 : score;
  const color = pct > 70 ? '#FB7185' : pct > 40 ? '#FBBF24' : '#34D399';
  const bg    = pct > 70 ? 'rgba(251,113,133,0.10)' : pct > 40 ? 'rgba(251,191,36,0.10)' : 'rgba(52,211,153,0.10)';
  return (
    <span
      className="px-2 py-0.5 rounded-full text-[10px] font-medium font-mono"
      style={{ color, background: bg }}
    >
      {pct.toFixed(1)}%
    </span>
  );
}

const fadeUp = {
  hidden:  { opacity: 0, y: 12 },
  visible: { opacity: 1, y: 0, transition: { duration: 0.35, ease: [0.22, 1, 0.36, 1] } },
};

export default function History() {
  const { history, historyTotal, isHistoryLoading, historyError, fetchHistory } = useForensicStore();
  const [filter, setFilter] = useState('all');
  const [limit,  setLimit]  = useState(20);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    fetchHistory(limit, filter === 'all' ? null : filter);
  }, [fetchHistory, filter, limit]);

  const handleViewDetail = async (id) => {
    if (selectedId === id) { setSelectedId(null); setDetail(null); return; }
    setSelectedId(id);
    try {
      const { forensicApi } = await import('../services/api');
      const result = await forensicApi.getAnalysis(id);
      setDetail(result.data);
    } catch { setDetail(null); }
  };

  const formatTime = (ts) => {
    try { return new Date(ts).toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' }); }
    catch { return ts; }
  };

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-5">
      {/* Header */}
      <header className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <div className="flex items-center gap-2.5 mb-1.5">
            <div
              className="w-7 h-7 rounded-lg flex items-center justify-center"
              style={{ background: 'var(--accent-dim)', border: '1px solid rgba(59,130,246,0.15)' }}
            >
              <Clock size={14} style={{ color: 'var(--accent)' }} />
            </div>
            <h1 className="font-display text-xl font-bold tracking-tight" style={{ color: 'var(--text-1)' }}>
              Analysis History
            </h1>
          </div>
          <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
            {historyTotal} total scans recorded
          </p>
        </div>

        {/* Filter */}
        <div className="flex items-center gap-2">
          <Filter size={13} style={{ color: 'var(--text-3)' }} />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="field-input text-[12px] w-auto"
            style={{ minWidth: '140px' }}
          >
            <option value="all">All Types</option>
            <option value="image">Image</option>
            <option value="video">Video</option>
            <option value="audio">Audio</option>
            <option value="multimodal">Multimodal</option>
          </select>
        </div>
      </header>

      {historyError && (
        <div
          className="p-3 rounded-lg text-[12px]"
          style={{ background: 'rgba(251,113,133,0.08)', border: '1px solid rgba(251,113,133,0.20)', color: 'var(--risk-critical)' }}
        >
          {historyError}
        </div>
      )}

      {/* Table */}
      <div className="card overflow-hidden">
        <table className="w-full text-left">
          <thead>
            <tr style={{
              background: 'linear-gradient(135deg, rgba(59,130,246,0.08), rgba(6,182,212,0.04))',
              borderBottom: '1px solid rgba(59,130,246,0.10)',
            }}>
              {['Time', 'Type', 'File', 'Risk', 'Verdict', 'Confidence', ''].map((h) => (
                <th
                  key={h}
                  className="px-4 py-3 text-[9px] uppercase tracking-wide"
                  style={{ color: 'var(--text-3)' }}
                >
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {isHistoryLoading ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center">
                  <div className="space-y-2 max-w-xs mx-auto">
                    {[1,2,3].map(i => (
                      <div key={i} className="skeleton h-5 rounded" />
                    ))}
                  </div>
                </td>
              </tr>
            ) : history.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-12 text-center text-[12px]" style={{ color: 'var(--text-3)' }}>
                  No analyses found. Run an analysis to see results here.
                </td>
              </tr>
            ) : (
              history.map((item) => (
                <React.Fragment key={item.id}>
                  <tr
                    onClick={() => handleViewDetail(item.id)}
                    className="cursor-pointer transition-colors"
                    style={{ borderBottom: '1px solid rgba(255,255,255,0.03)' }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(59,130,246,0.03)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <td className="px-4 py-3 text-[11px] font-mono whitespace-nowrap" style={{ color: 'var(--text-3)' }}>
                      {formatTime(item.timestamp)}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex items-center gap-2">
                        {MEDIA_ICONS[item.media_type]}
                        <span className="text-[11px] capitalize" style={{ color: 'var(--text-2)' }}>{item.media_type}</span>
                      </div>
                    </td>
                    <td className="px-4 py-3 text-[11px] truncate max-w-[160px]" style={{ color: 'var(--text-1)' }}>
                      {item.file_name || '\u2014'}
                    </td>
                    <td className="px-4 py-3"><RiskBadge score={item.risk_score} /></td>
                    <td className="px-4 py-3 text-[11px] max-w-[180px] truncate" style={{ color: 'var(--text-2)' }}>{item.verdict}</td>
                    <td className="px-4 py-3 text-[11px]" style={{ color: 'var(--text-3)' }}>{item.confidence}</td>
                    <td className="px-4 py-3">
                      <ChevronRight
                        size={14}
                        style={{
                          color: 'var(--text-3)',
                          transform: selectedId === item.id ? 'rotate(90deg)' : 'rotate(0)',
                          transition: 'transform 0.2s ease',
                        }}
                      />
                    </td>
                  </tr>

                  {selectedId === item.id && detail && (
                    <tr>
                      <td
                        colSpan={7}
                        style={{ background: 'var(--bg-inset)', borderBottom: '1px solid rgba(255,255,255,0.03)' }}
                      >
                        <div className="px-4 py-4">
                          <pre
                            className="text-[10px] font-mono whitespace-pre-wrap leading-relaxed max-h-[280px] overflow-y-auto no-scrollbar"
                            style={{ color: 'var(--text-2)' }}
                          >
                            {JSON.stringify(detail, null, 2)}
                          </pre>
                        </div>
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))
            )}
          </tbody>
        </table>
      </div>

      {history.length < historyTotal && (
        <div className="flex justify-center">
          <button
            onClick={() => setLimit((p) => p + 20)}
            className="btn-ghost text-[12px]"
          >
            Load more results
          </button>
        </div>
      )}
    </motion.div>
  );
}
