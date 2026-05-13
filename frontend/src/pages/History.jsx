import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { Clock, Filter, Image, Film, Mic, Layers, ChevronRight } from 'lucide-react';
import useForensicStore from '../store/useForensicStore';

const MEDIA_ICONS = {
  image: <Image size={16} className="text-accent" />,
  video: <Film size={16} className="text-indigo-400" />,
  audio: <Mic size={16} className="text-accent-success" />,
  multimodal: <Layers size={16} className="text-accent-warning" />,
};

function RiskBadge({ score }) {
  const pct = score <= 1 ? score * 100 : score;
  let color = 'text-accent-success';
  let bg = 'bg-accent-success/10';
  if (pct > 70) { color = 'text-accent-danger'; bg = 'bg-accent-danger/10'; }
  else if (pct > 40) { color = 'text-accent-warning'; bg = 'bg-accent-warning/10'; }

  return (
    <span className={`px-2.5 py-1 rounded-full text-xs font-bold ${color} ${bg}`}>
      {pct.toFixed(1)}%
    </span>
  );
}

export default function History() {
  const { history, historyTotal, isHistoryLoading, historyError, fetchHistory } = useForensicStore();
  const [filter, setFilter] = useState('all');
  const [limit, setLimit] = useState(20);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);

  useEffect(() => {
    const mediaType = filter === 'all' ? null : filter;
    fetchHistory(limit, mediaType);
  }, [fetchHistory, filter, limit]);

  const handleViewDetail = async (id) => {
    if (selectedId === id) {
      setSelectedId(null);
      setDetail(null);
      return;
    }
    setSelectedId(id);
    try {
      const { forensicApi } = await import('../services/api');
      const result = await forensicApi.getAnalysis(id);
      setDetail(result.data);
    } catch {
      setDetail(null);
    }
  };

  const formatTime = (ts) => {
    try {
      return new Date(ts).toLocaleString();
    } catch {
      return ts;
    }
  };

  return (
    <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">
      <header className="mb-8 flex justify-between items-end flex-wrap gap-4">
        <div>
          <h2 className="text-3xl font-bold tracking-tight text-white flex items-center gap-3">
            <Clock className="text-accent" /> Analysis History
          </h2>
          <p className="text-text-muted mt-2">Browse past analysis results ({historyTotal} total).</p>
        </div>
        <div className="flex items-center gap-3">
          <Filter size={16} className="text-text-muted" />
          <select
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            className="bg-background-surface border border-border-subtle rounded-lg px-3 py-2 text-sm text-text-primary focus:outline-none focus:border-accent"
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
        <div className="p-4 bg-accent-danger/10 border border-accent-danger rounded-xl text-accent-danger">
          {historyError}
        </div>
      )}

      <div className="glass-card overflow-hidden">
        <table className="w-full text-left">
          <thead>
            <tr className="border-b border-border-subtle">
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">Time</th>
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">Type</th>
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">File</th>
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">Risk</th>
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">Verdict</th>
              <th className="px-6 py-4 text-xs font-bold text-text-muted uppercase tracking-wider">Confidence</th>
              <th className="px-6 py-4"></th>
            </tr>
          </thead>
          <tbody>
            {isHistoryLoading ? (
              <tr>
                <td colSpan={7} className="px-6 py-12 text-center text-text-muted">
                  <div className="animate-pulse">Loading history...</div>
                </td>
              </tr>
            ) : history.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-6 py-12 text-center text-text-muted">
                  No analyses found. Run an analysis to see results here.
                </td>
              </tr>
            ) : (
              history.map((item) => (
                <React.Fragment key={item.id}>
                  <tr
                    onClick={() => handleViewDetail(item.id)}
                    className="border-b border-border-subtle hover:bg-background-card-hover cursor-pointer transition-colors"
                  >
                    <td className="px-6 py-4 text-xs text-text-secondary">{formatTime(item.timestamp)}</td>
                    <td className="px-6 py-4">
                      <div className="flex items-center gap-2">
                        {MEDIA_ICONS[item.media_type] || null}
                        <span className="text-xs text-text-primary capitalize">{item.media_type}</span>
                      </div>
                    </td>
                    <td className="px-6 py-4 text-xs text-text-primary truncate max-w-[200px]">{item.file_name || '—'}</td>
                    <td className="px-6 py-4"><RiskBadge score={item.risk_score} /></td>
                    <td className="px-6 py-4 text-xs text-text-primary">{item.verdict}</td>
                    <td className="px-6 py-4 text-xs text-text-secondary">{item.confidence}</td>
                    <td className="px-6 py-4">
                      <ChevronRight size={16} className={`text-text-muted transition-transform ${selectedId === item.id ? 'rotate-90' : ''}`} />
                    </td>
                  </tr>
                  {selectedId === item.id && detail && (
                    <tr>
                      <td colSpan={7} className="px-6 py-4 bg-background-surface">
                        <pre className="text-xs text-text-muted font-mono whitespace-pre-wrap leading-relaxed max-h-[300px] overflow-y-auto">
                          {JSON.stringify(detail, null, 2)}
                        </pre>
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
            onClick={() => setLimit((prev) => prev + 20)}
            className="px-6 py-2 border border-border-subtle rounded-lg text-text-secondary hover:text-text-primary hover:border-border-hover text-sm transition-all"
          >
            Load More
          </button>
        </div>
      )}
    </motion.div>
  );
}
