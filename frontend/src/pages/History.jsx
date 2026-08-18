import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Clock, Filter, Image, Film, Mic, Layers, ChevronRight } from 'lucide-react';
import { getRiskColorRaw, normalizeScore } from '../utils/risk';
import { formatShortDateTime } from '../utils/format';
import useForensicStore from '../store/useForensicStore';
import { forensicApi } from '../services/api';
import PageHeader from '../components/PageHeader';
import ScoreBar from '../components/ScoreBar';
import RiskBadge from '../components/RiskBadge';

const MEDIA_ICONS = {
  image:      <Image  size={14} className="text-accent" />,
  video:      <Film   size={14} className="text-accent" />,
  audio:      <Mic    size={14} className="text-accent" />,
  multimodal: <Layers size={14} className="text-accent" />,
};

function DetailPanel({ detail }) {
  if (!detail) return null;

  const pct = normalizeScore(detail.risk_score);
  const modelScores = detail.model_scores || {};
  const hasModelScores = Object.keys(modelScores).length > 0;

  return (
    <div className="space-y-4">
      {/* Risk score + verdict header */}
      <div className="flex items-center gap-4 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-3">Risk Score</span>
          <span
            className="text-lg font-bold font-mono"
            style={{ color: getRiskColorRaw(pct) }}
          >
            {pct.toFixed(1)}%
          </span>
        </div>
        {detail.verdict && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-3">Verdict</span>
            <span className="text-sm font-medium text-text-1">
              {detail.verdict}
            </span>
          </div>
        )}
        {detail.confidence && (
          <div className="flex items-center gap-2">
            <span className="text-xs text-text-3">Confidence</span>
            <span className="text-sm text-text-2">
              {detail.confidence}
            </span>
          </div>
        )}
      </div>

      {/* Model scores as labeled bars */}
      {hasModelScores && (
        <div>
          <p className="text-xs mb-2 text-text-3">Model Scores</p>
          <div className="grid gap-1 max-w-[420px]">
            {Object.entries(modelScores).map(([name, score]) => (
              <ScoreBar key={name} name={name} score={score} />
            ))}
          </div>
        </div>
      )}

      {/* Explanation */}
      {detail.explanation && (
        <div>
          <p className="text-xs mb-1 text-text-3">Explanation</p>
          <p className="text-sm leading-relaxed text-text-2">
            {detail.explanation}
          </p>
        </div>
      )}

      {/* Processing time */}
      {detail.processing_time_ms > 0 && (
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-3">Processing Time</span>
          <span className="text-sm font-mono text-text-2">
            {(detail.processing_time_ms / 1000).toFixed(2)}s
          </span>
        </div>
      )}
    </div>
  );
}

function DetailSkeleton() {
  return (
    <div className="space-y-4">
      {/* Risk score + verdict header placeholder */}
      <div className="flex items-center gap-6 flex-wrap">
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-3">Risk Score</span>
          <div className="skeleton h-5 w-12 rounded" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-3">Verdict</span>
          <div className="skeleton h-4 w-20 rounded" />
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-text-3">Confidence</span>
          <div className="skeleton h-4 w-16 rounded" />
        </div>
      </div>

      {/* Model scores placeholder */}
      <div>
        <p className="text-xs mb-2 text-text-3">Model Scores</p>
        <div className="grid gap-2 max-w-[420px]">
          <div className="flex items-center justify-between gap-4">
            <div className="skeleton h-3 w-24 rounded" />
            <div className="skeleton h-2 w-32 rounded-full" />
            <div className="skeleton h-3 w-8 rounded" />
          </div>
          <div className="flex items-center justify-between gap-4">
            <div className="skeleton h-3 w-28 rounded" />
            <div className="skeleton h-2 w-32 rounded-full" />
            <div className="skeleton h-3 w-8 rounded" />
          </div>
        </div>
      </div>

      {/* Explanation placeholder */}
      <div>
        <p className="text-xs mb-2 text-text-3">Explanation</p>
        <div className="space-y-2">
          <div className="skeleton h-3 w-full rounded" />
          <div className="skeleton h-3 w-5/6 rounded" />
        </div>
      </div>

      {/* Processing time placeholder */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-text-3">Processing Time</span>
        <div className="skeleton h-4 w-14 rounded" />
      </div>
    </div>
  );
}

function DetailContent({ loading, detail }) {
  return (
    <AnimatePresence mode="wait">
      {loading ? (
        <motion.div
          key="skeleton"
          initial={{ opacity: 0, y: -4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: 4 }}
          transition={{ duration: 0.15 }}
        >
          <DetailSkeleton />
        </motion.div>
      ) : detail ? (
        <motion.div
          key="detail"
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.15 }}
        >
          <DetailPanel detail={detail} />
        </motion.div>
      ) : (
        <motion.div
          key="error"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
        >
          <p className="text-xs text-center py-4 text-text-3">Failed to load details</p>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

export default function History() {
  const { history, historyTotal, isHistoryLoading, historyError, fetchHistory } = useForensicStore();
  const [filter, setFilter] = useState('all');
  const [limit, setLimit] = useState(20);
  const [selectedId, setSelectedId] = useState(null);
  const [detail, setDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const activeIdRef = useRef(null);

  useEffect(() => {
    fetchHistory(limit, filter === 'all' ? null : filter);
  }, [fetchHistory, filter, limit]);

  const handleViewDetail = async (id) => {
    if (selectedId === id) {
      setSelectedId(null);
      activeIdRef.current = null;
      setDetail(null);
      return;
    }
    setSelectedId(id);
    activeIdRef.current = id;
    setDetail(null);
    setDetailLoading(true);
    try {
      const result = await forensicApi.getAnalysis(id);
      if (activeIdRef.current === id) {
        setDetail(result.data);
      }
    } catch {
      if (activeIdRef.current === id) {
        setDetail(null);
      }
    } finally {
      if (activeIdRef.current === id) {
        setDetailLoading(false);
      }
    }
  };

  return (
    <div className="space-y-5">
      <PageHeader
        icon={Clock}
        title="Analysis History"
        subtitle={`${historyTotal} total scans recorded`}
        actions={
          <div className="flex items-center gap-2">
            <Filter size={14} className="text-text-3" />
            <select
              value={filter}
              onChange={(e) => setFilter(e.target.value)}
              className="field-input text-xs w-auto min-w-[140px]"
            >
              <option value="all">All Types</option>
              <option value="image">Image</option>
              <option value="video">Video</option>
              <option value="audio">Audio</option>
              <option value="multimodal">Multimodal</option>
            </select>
          </div>
        }
      />

      {historyError && (
        <div
          role="alert"
          className="p-3 rounded-lg text-sm bg-risk-criticalDim text-risk-critical border border-[rgba(251,113,133,0.20)]"
        >
          {historyError}
        </div>
      )}

      {/* Mobile card list (< md) */}
      <div className="md:hidden space-y-3">
        {isHistoryLoading ? (
          <div className="space-y-2 px-1">
            {[1, 2, 3].map((i) => (
              <div key={i} className="skeleton h-20 rounded-lg" />
            ))}
          </div>
        ) : history.length === 0 ? (
          <p className="text-center text-sm py-12 text-text-3">
            No analyses found. Run an analysis to see results here.
          </p>
        ) : (
          history.map((item) => {
            const isOpen = selectedId === item.id;
            return (
              <div key={item.id}>
                <button
                  type="button"
                  onClick={() => handleViewDetail(item.id)}
                  className="card card-hover w-full text-left cursor-pointer"
                >
                  <div className="flex items-center justify-between mb-2">
                    <div className="flex items-center gap-2">
                      {MEDIA_ICONS[item.media_type]}
                      <span className="text-xs capitalize text-text-2">{item.media_type}</span>
                    </div>
                    <RiskBadge score={item.risk_score} />
                  </div>
                  <p className="text-sm truncate mb-1 text-text-1">
                    {item.file_name || '\u2014'}
                  </p>
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono text-text-3">
                      {formatShortDateTime(item.timestamp)}
                    </span>
                    <ChevronRight
                      size={14}
                      className={`text-text-3 transition-transform duration-200 ${isOpen ? 'rotate-90' : ''}`}
                    />
                  </div>
                </button>
                <AnimatePresence initial={false}>
                  {isOpen && (
                    <motion.div
                      initial={{ opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: 'auto' }}
                      exit={{ opacity: 0, height: 0 }}
                      transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                      className="overflow-hidden"
                    >
                      <div className="card mt-1 bg-bg-inset">
                        <DetailContent loading={detailLoading} detail={detail} />
                      </div>
                    </motion.div>
                  )}
                </AnimatePresence>
              </div>
            );
          })
        )}
      </div>

      {/* Desktop table (>= md) */}
      <div className="hidden md:block card overflow-hidden !p-0">
        <div className="table-scroll scroll-fade-x">
          <table className="w-full text-left min-w-[640px]">
            <thead>
              <tr className="bg-gradient-to-br from-[rgba(59,130,246,0.08)] to-[rgba(56,189,248,0.04)] border-b border-border-glow">
                {['Time', 'Type', 'File', 'Risk', 'Verdict', 'Confidence', ''].map((h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-xs uppercase tracking-wide text-text-3"
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
                      {[1, 2, 3].map((i) => (
                        <div key={i} className="skeleton h-5 rounded" />
                      ))}
                    </div>
                  </td>
                </tr>
              ) : history.length === 0 ? (
                <tr>
                  <td colSpan={7} className="px-4 py-12 text-center text-sm text-text-3">
                    No analyses found. Run an analysis to see results here.
                  </td>
                </tr>
              ) : (
                history.map((item) => (
                  <React.Fragment key={item.id}>
                    <tr
                      onClick={() => handleViewDetail(item.id)}
                      className="cursor-pointer table-row-hover border-b border-border-dim"
                    >
                      <td className="px-4 py-3 text-xs font-mono whitespace-nowrap text-text-3">
                        {formatShortDateTime(item.timestamp)}
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          {MEDIA_ICONS[item.media_type]}
                          <span className="text-xs capitalize text-text-2">{item.media_type}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-sm truncate max-w-[160px] text-text-1">
                        {item.file_name || '\u2014'}
                      </td>
                      <td className="px-4 py-3"><RiskBadge score={item.risk_score} /></td>
                      <td className="px-4 py-3 text-sm max-w-[180px] truncate text-text-2">{item.verdict}</td>
                      <td className="px-4 py-3 text-xs text-text-3">{item.confidence}</td>
                      <td className="px-4 py-3">
                        <ChevronRight
                          size={14}
                          className={`text-text-3 transition-transform duration-200 ${selectedId === item.id ? 'rotate-90' : ''}`}
                        />
                      </td>
                    </tr>

                    <AnimatePresence initial={false}>
                      {selectedId === item.id && (
                        <tr key={`${item.id}-detail`}>
                          <td
                            colSpan={7}
                            className="bg-bg-inset border-b border-border-dim p-0"
                          >
                            <motion.div
                              initial={{ opacity: 0, height: 0 }}
                              animate={{ opacity: 1, height: 'auto' }}
                              exit={{ opacity: 0, height: 0 }}
                              transition={{ duration: 0.25, ease: [0.22, 1, 0.36, 1] }}
                              className="overflow-hidden"
                            >
                              <div className="px-4 py-4">
                                <DetailContent loading={detailLoading} detail={detail} />
                              </div>
                            </motion.div>
                          </td>
                        </tr>
                      )}
                    </AnimatePresence>
                  </React.Fragment>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {history.length < historyTotal && (
        <div className="flex justify-center">
          <button
            onClick={() => setLimit((p) => p + 20)}
            className="btn-ghost text-sm"
          >
            Load more results
          </button>
        </div>
      )}
    </div>
  );
}
