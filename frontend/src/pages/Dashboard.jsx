import React, { useEffect, useState, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import { Image, Film, Mic, Layers, Clock, Shield, Cpu, Upload, FileSearch, ArrowUpRight, CalendarClock, Timer } from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import useForensicStore from '../store/useForensicStore';
import EmptyDashboard from '../components/EmptyDashboard';
import RiskGauge from '../components/RiskGauge';
import RiskBadge from '../components/RiskBadge';
import VerdictChip from '../components/VerdictChip';
import AnimatedNumber from '../components/AnimatedNumber';
import RiskTrendChart from '../components/RiskTrendChart';
import RiskDistributionChart from '../components/RiskDistributionChart';
import { staggerFadeUp } from '../utils/animations';
import { normalizeScore } from '../utils/risk';
import { formatRelativeTime, detectMediaRoute } from '../utils/format';

const MEDIA_ICONS = { image: Image, video: Film, audio: Mic, multimodal: Layers };
const ANALYSIS_CARDS = [
  { to: '/image', label: 'Image', desc: 'Synthetic face & pixel manipulation', icon: Image },
  { to: '/video', label: 'Video', desc: 'Temporal consistency & frame analysis', icon: Film },
  { to: '/audio', label: 'Audio', desc: 'Voice cloning & spectrogram forensics', icon: Mic },
  { to: '/multimodal', label: 'Multimodal', desc: 'Cross-modal fusion confidence matrix', icon: Layers },
];
// Neutral stat tiles only — Avg Risk gets its own hero gauge card, not a plain tile.
// `animated: false` for Models Active — it's a ratio ("4/13"), not a single
// count-up-able number.
const STAT_DEFS = (totals, todayScans, models, cleared, avgProcessingSec) => [
  { label: 'Total Scans', value: totals, icon: FileSearch, iconClass: 'text-text-2', animated: true },
  { label: "Today's Scans", value: todayScans, icon: CalendarClock, iconClass: 'text-text-2', animated: true },
  { label: 'Models Active', value: models, icon: Cpu, iconClass: 'text-text-2', animated: false },
  { label: 'Cleared', value: cleared, icon: Shield, iconClass: 'text-risk-clear', animated: true },
  { label: 'Avg Process Time', value: avgProcessingSec, icon: Timer, iconClass: 'text-text-2', animated: true, decimals: 1, suffix: 's' },
];

export default function Dashboard() {
  const { history, historyTotal, fetchHistory, systemStatus, fetchStatus, isStatusLoading, setPendingFile } = useForensicStore();
  const navigate = useNavigate();
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef(null);

  const handleQuickFile = useCallback((e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  useEffect(() => { fetchHistory(20); fetchStatus(); }, [fetchHistory, fetchStatus]);

  const modelsOnline = systemStatus.loaded_models?.length || 0;
  const totalModels = modelsOnline + (systemStatus.missing_models?.length || 0);
  const riskCounts = history.reduce((acc, item) => {
    const pct = normalizeScore(item.risk_score);
    if (pct > 70) acc.critical += 1;
    else if (pct > 40) acc.caution += 1;
    else acc.clear += 1;
    return acc;
  }, { clear: 0, caution: 0, critical: 0 });
  const avgRisk = historyTotal > 0
    ? Math.round(history.reduce((s, i) => s + normalizeScore(i.risk_score), 0) / history.length) : 0;
  const todayScans = history.filter(
    (i) => new Date(i.timestamp).toDateString() === new Date().toDateString(),
  ).length;
  const avgProcessingSec = historyTotal > 0
    ? history.reduce((s, i) => s + (i.processing_time_ms || 0), 0) / history.length / 1000
    : 0;

  const handleDrop = useCallback((e) => {
    e.preventDefault(); setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    setPendingFile(file);
    navigate(detectMediaRoute(file));
  }, [navigate, setPendingFile]);

  /* Empty state */
  if (historyTotal === 0 && !isStatusLoading) {
    return (
      <motion.div initial="hidden" animate="visible" className="space-y-5">
        <motion.div variants={staggerFadeUp} custom={0}>
          <h1 className="font-display text-2xl font-bold tracking-tight text-text-1">Forensics Command Center</h1>
          <p className="text-sm mt-1 text-text-2 leading-relaxed">AI-powered deepfake detection & media authentication</p>
        </motion.div>
        <EmptyDashboard />
      </motion.div>
    );
  }

  const recentScans = history.slice(0, 5);
  const stats = STAT_DEFS(historyTotal, todayScans, `${modelsOnline}/${totalModels}`, riskCounts.clear, avgProcessingSec);

  return (
    <div className="space-y-5"
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>

      {/* Header */}
      <div className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-2xl font-bold tracking-tight text-text-1">Forensics Command Center</h1>
          <p className="text-sm mt-1 text-text-2 leading-relaxed">AI-powered deepfake detection & media authentication</p>
        </div>
        <span className="text-xs font-mono text-text-3">{modelsOnline}/{totalModels} models online</span>
      </div>

      {/* Hero: Avg Risk gauge + secondary stat stack */}
      <motion.div variants={staggerFadeUp} custom={1} className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <div className="md:col-span-2 card flex items-center gap-6 flex-wrap sm:flex-nowrap">
          <RiskGauge percentage={avgRisk} size={128} label="Avg Risk" />
          <div>
            <span className="label-tag">Average Risk Score</span>
            <p className="text-xs mt-1 text-text-3">
              Across {historyTotal} recorded scan{historyTotal !== 1 ? 's' : ''}
            </p>
          </div>
        </div>

        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {stats.map(({ label, value, icon: Icon, iconClass, animated, decimals, suffix }) => (
            <div key={label} className="card">
              <div className="flex items-center gap-2 mb-2">
                <Icon size={14} className={iconClass} />
                <span className="label-tag">{label}</span>
              </div>
              <p className="font-display text-lg font-bold text-text-1">
                {animated ? <AnimatedNumber value={value} decimals={decimals || 0} suffix={suffix || ''} /> : value}
              </p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Risk trend + distribution charts */}
      {historyTotal > 0 && (
        <motion.div variants={staggerFadeUp} custom={2} className="grid grid-cols-1 lg:grid-cols-3 gap-3">
          <div className="lg:col-span-2">
            <RiskTrendChart history={history} />
          </div>
          <div className="lg:col-span-1">
            <RiskDistributionChart counts={riskCounts} total={historyTotal} />
          </div>
        </motion.div>
      )}

      {/* Quick upload + analysis cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div
          className={`card flex flex-col items-center justify-center text-center cursor-pointer min-h-[160px] border-dashed ${dragOver ? 'border-accent' : ''}`}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*,audio/*"
            onChange={handleQuickFile}
            className="hidden"
            aria-label="Quick analyze file"
          />
          <div className="w-10 h-10 rounded-xl flex items-center justify-center mb-3 bg-accent-dim">
            <Upload size={18} className="text-accent" />
          </div>
          <p className="text-sm font-semibold text-text-1">Quick Analyze</p>
          <p className="text-xs mt-1 text-text-3">Drop any file or click to browse</p>
        </div>
        <div className="lg:col-span-2 grid grid-cols-2 gap-3">
          {ANALYSIS_CARDS.map(({ to, label, desc, icon }) => {
            const Icon = icon;
            return (
              <Link key={to} to={to} className="group card card-hover flex flex-col no-underline">
                <div className="flex items-center justify-between mb-2">
                  <div className="w-8 h-8 rounded-lg flex items-center justify-center bg-accent-dim">
                    <Icon size={15} className="text-accent" />
                  </div>
                  <ArrowUpRight size={12} className="opacity-0 group-hover:opacity-100 transition-opacity text-accent" />
                </div>
                <span className="text-sm font-semibold text-text-1">{label}</span>
                <span className="text-xs mt-0.5 text-text-3">{desc}</span>
              </Link>
            );
          })}
        </div>
      </div>

      {/* Recent scans table */}
      {recentScans.length > 0 && (
        <div className="card overflow-hidden !p-0">
          <div className="flex items-center justify-between px-4 py-3 border-b border-border-dim">
            <div className="flex items-center gap-2">
              <Clock size={13} className="text-accent" />
              <span className="label-tag">Recent Activity</span>
            </div>
            <Link to="/history" className="text-xs font-medium text-accent no-underline">View all</Link>
          </div>
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-text-3 text-xs">
                <th className="px-4 py-2 font-medium">Time</th>
                <th className="px-4 py-2 font-medium">Type</th>
                <th className="px-4 py-2 font-medium">File</th>
                <th className="px-4 py-2 font-medium">Status</th>
                <th className="px-4 py-2 font-medium text-right">Risk</th>
              </tr>
            </thead>
            <tbody>
              {recentScans.map((item) => {
                const Icon = MEDIA_ICONS[item.media_type] || FileSearch;
                return (
                  <tr key={item.id} className="border-t border-border-dim cursor-pointer table-row-hover"
                    onClick={() => navigate('/history')}>
                    <td className="px-4 py-2 text-xs font-mono text-text-3">{formatRelativeTime(item.created_at || item.timestamp)}</td>
                    <td className="px-4 py-2"><Icon size={14} className="text-text-2" /></td>
                    <td className="px-4 py-2 truncate max-w-[200px] text-text-1">{item.file_name || `${item.media_type} analysis`}</td>
                    <td className="px-4 py-2">
                      <VerdictChip verdict={item.verdict} riskScore={normalizeScore(item.risk_score)} />
                    </td>
                    <td className="px-4 py-2 text-right">
                      <RiskBadge score={item.risk_score} />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
