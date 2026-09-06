import React, { useEffect, useState } from 'react';
import { motion } from 'framer-motion';
import { ShieldAlert, Image, Film, Mic, Layers, FileSearch, ChevronRight } from 'lucide-react';
import { fadeUp } from '../utils/animations';
import { formatShortDateTime } from '../utils/format';
import useForensicStore from '../store/useForensicStore';
import { forensicApi } from '../services/api';
import PageHeader from '../components/PageHeader';
import RiskBadge from '../components/RiskBadge';
import ComplaintForm from '../components/ComplaintForm';

const MEDIA_ICONS = {
  image:      <Image      size={14} className="text-accent" />,
  video:      <Film       size={14} className="text-accent" />,
  audio:      <Mic        size={14} className="text-accent" />,
  document:   <FileSearch size={14} className="text-accent" />,
  multimodal: <Layers     size={14} className="text-accent" />,
};

export default function CyberComplaint() {
  const { history, isHistoryLoading, fetchHistory } = useForensicStore();
  const [selectedId, setSelectedId] = useState(null);
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [detailLoading, setDetailLoading] = useState(false);

  useEffect(() => {
    fetchHistory(50, null);
  }, [fetchHistory]);

  const flaggedItems = history.filter((item) => item.verdict === 'AI-GENERATED' || item.verdict === 'MANIPULATED');

  const handleSelect = async (id) => {
    setSelectedId(id);
    setSelectedDetail(null);
    setDetailLoading(true);
    try {
      const result = await forensicApi.getAnalysis(id);
      setSelectedDetail(result.data);
    } catch {
      setSelectedDetail(null);
    } finally {
      setDetailLoading(false);
    }
  };

  return (
    <motion.div initial="hidden" animate="visible" variants={fadeUp} className="space-y-6">
      <PageHeader
        icon={ShieldAlert}
        title="Cyber Crime Complaint"
        subtitle="Generate a complaint document from a past AI-generated or manipulated-document detection, for you to review and file yourself."
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-5">
        {/* Left: pick a flagged analysis */}
        <div className="lg:col-span-1 space-y-2">
          <span className="label-tag block mb-1">Flagged Analyses</span>

          {isHistoryLoading ? (
            <div className="space-y-2">
              {[1, 2, 3].map((i) => <div key={i} className="skeleton h-16 rounded-lg" />)}
            </div>
          ) : flaggedItems.length === 0 ? (
            <div className="card text-center py-8">
              <ShieldAlert size={20} className="mx-auto mb-2 text-text-3 opacity-40" />
              <p className="text-sm text-text-2">No flagged results yet.</p>
              <p className="text-xs mt-1 text-text-3">
                Run an image, video, audio, or document analysis first — anything flagged
                AI-GENERATED or MANIPULATED will show up here.
              </p>
            </div>
          ) : (
            <div className="space-y-2">
              {flaggedItems.map((item) => {
                const selected = selectedId === item.id;
                return (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => handleSelect(item.id)}
                    className={`card card-hover w-full text-left cursor-pointer ${
                      selected ? 'border-accent/40 bg-accent-dim' : ''
                    }`}
                  >
                    <div className="flex items-center justify-between mb-1.5">
                      <div className="flex items-center gap-2">
                        {MEDIA_ICONS[item.media_type] || MEDIA_ICONS.image}
                        <span className="text-xs capitalize text-text-2">{item.media_type}</span>
                      </div>
                      <RiskBadge score={item.risk_score} />
                    </div>
                    <p className="text-sm truncate mb-1 text-text-1">
                      {item.file_name || '—'}
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-mono text-text-3">
                        {formatShortDateTime(item.timestamp)}
                      </span>
                      <ChevronRight
                        size={14}
                        className={`text-text-3 transition-transform duration-200 ${selected ? 'rotate-90' : ''}`}
                      />
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Right: selected summary + complainant form */}
        <div className="lg:col-span-2">
          <div className="card">
            <div className="flex items-center gap-2 mb-3">
              <ShieldAlert size={13} className="text-text-3" />
              <span className="label-tag">Complaint Details</span>
            </div>

            {!selectedId ? (
              <div className="flex items-center justify-center gap-2 py-8 text-text-3">
                <ShieldAlert size={16} className="opacity-30" />
                <p className="text-sm">Select a flagged analysis on the left to begin.</p>
              </div>
            ) : detailLoading ? (
              <div className="space-y-2 py-4">
                <div className="skeleton h-4 w-2/3 rounded" />
                <div className="skeleton h-4 w-1/2 rounded" />
                <div className="skeleton h-24 w-full rounded" />
              </div>
            ) : !selectedDetail ? (
              <p className="text-sm text-center py-8 text-text-3">Failed to load this analysis.</p>
            ) : (
              <div className="space-y-4">
                <div className="p-3 rounded-lg bg-bg-inset border border-border-dim">
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-text-2">
                    <span>File: <strong className="text-text-1">{selectedDetail.file_name || '—'}</strong></span>
                    <span>Verdict: <strong className="text-text-1">{selectedDetail.verdict}</strong></span>
                    <span>Risk: <strong className="text-text-1">{(selectedDetail.risk_percent ?? selectedDetail.risk_score * 100).toFixed(1)}%</strong></span>
                    <span>Confidence: <strong className="text-text-1">{selectedDetail.confidence}</strong></span>
                  </div>
                </div>

                <ComplaintForm
                  analysis={selectedDetail}
                  fileName={selectedDetail.file_name}
                />
              </div>
            )}
          </div>
        </div>
      </div>
    </motion.div>
  );
}
