import React, { useEffect, useState, useCallback, useRef } from 'react';
import { motion } from 'framer-motion';
import { Image, Film, Mic, Layers, Clock, Shield, Cpu, Upload, FileSearch, ArrowUpRight, CalendarClock, Timer, Sparkles } from 'lucide-react';
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
  { to: '/image', label: 'Image AI Forensics', desc: 'Synthetic portrait & pixel manipulation', icon: Image },
  { to: '/audio', label: 'Audio Voice Biometrics', desc: 'Voice clone & spectral vocoder traces', icon: Mic },
  { to: '/video', label: 'Video Deepfake Scan', desc: 'Temporal consistency & frame analysis', icon: Film },
  { to: '/document', label: 'Document Forensics', desc: 'Tampering, ELA & ID checksum verification', icon: FileSearch },
  { to: '/multimodal', label: 'Multimodal Fusion', desc: 'Cross-modal audio-visual confidence matrix', icon: Layers },
];

const STAT_DEFS = (totals, todayScans, models, cleared, avgProcessingSec) => [
  { label: 'Total Scans', value: totals, icon: FileSearch, iconClass: 'text-purple-600', animated: true },
  { label: "Today's Scans", value: todayScans, icon: CalendarClock, iconClass: 'text-indigo-600', animated: true },
  { label: 'Active Pipeline', value: models, icon: Cpu, iconClass: 'text-purple-700', animated: false },
  { label: 'Authentic Cleared', value: cleared, icon: Shield, iconClass: 'text-emerald-600', animated: true },
  { label: 'Avg Latency', value: avgProcessingSec, icon: Timer, iconClass: 'text-cyan-600', animated: true, decimals: 1, suffix: 's' },
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

  const modelsOnline = systemStatus.loaded_models?.length || 7;
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
      <motion.div initial="hidden" animate="visible" className="space-y-6">
        <EmptyDashboard />
      </motion.div>
    );
  }

  const recentScans = history.slice(0, 5);
  const stats = STAT_DEFS(historyTotal, todayScans, `${modelsOnline} Models`, riskCounts.clear, avgProcessingSec);

  return (
    <motion.div initial="hidden" animate="visible" className="space-y-6"
      onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
      onDragLeave={() => setDragOver(false)} onDrop={handleDrop}>

      {/* Header */}
      <motion.div variants={staggerFadeUp} custom={0} className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-2xl sm:text-3xl font-black tracking-tight text-[#1E1238]">Forensics Command Station</h1>
          <p className="text-xs sm:text-sm mt-1 text-[#5B4E75] font-medium leading-relaxed">Multi-spectral AI deepfake detection, speech forensics & provenance verification</p>
        </div>
        <span className="text-xs font-mono font-bold text-purple-700 bg-purple-100 px-3 py-1 rounded-full border border-purple-200">
          {modelsOnline} Models Active
        </span>
      </motion.div>

      {/* Hero: Avg Risk gauge + stats */}
      <motion.div variants={staggerFadeUp} custom={1} className="grid grid-cols-1 md:grid-cols-12 gap-4">
        <div className="md:col-span-4 card flex items-center justify-center p-4">
          <RiskGauge percentage={avgRisk} size={140} label="Avg Risk Score" />
        </div>

        <div className="md:col-span-8 grid grid-cols-2 sm:grid-cols-3 gap-3">
          {stats.map(({ label, value, icon: Icon, iconClass, animated, decimals, suffix }) => (
            <div key={label} className="card p-4 flex flex-col justify-between">
              <div className="flex items-center gap-2 mb-2">
                <div className="w-7 h-7 rounded-lg bg-purple-50 flex items-center justify-center">
                  <Icon size={14} className={iconClass} />
                </div>
                <span className="label-tag text-[10px]">{label}</span>
              </div>
              <p className="font-display text-xl font-black text-[#1E1238]">
                {animated ? <AnimatedNumber value={value} decimals={decimals || 0} suffix={suffix || ''} /> : value}
              </p>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Risk trend + distribution charts */}
      {historyTotal > 0 && (
        <motion.div variants={staggerFadeUp} custom={2} className="grid grid-cols-1 lg:grid-cols-3 gap-4">
          <div className="lg:col-span-2">
            <RiskTrendChart history={history} />
          </div>
          <div className="lg:col-span-1">
            <RiskDistributionChart counts={riskCounts} total={historyTotal} />
          </div>
        </motion.div>
      )}

      {/* Quick upload + analysis cards */}
      <motion.div variants={staggerFadeUp} custom={3} className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        <div
          className={`lg:col-span-4 card flex flex-col items-center justify-center text-center cursor-pointer min-h-[160px] border-2 border-dashed transition-all ${
            dragOver ? 'border-purple-600 bg-purple-100/50' : 'border-purple-200 hover:border-purple-400 bg-white/70'
          }`}
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
          <div className="w-12 h-12 rounded-2xl bg-purple-100 border border-purple-200 flex items-center justify-center mb-3 text-purple-700 shadow-sm">
            <Upload size={20} />
          </div>
          <p className="text-sm font-bold text-[#1E1238]">Instant Forensic Drop</p>
          <p className="text-xs mt-1 text-[#5B4E75]">Drop image, audio, or video</p>
        </div>

        <div className="lg:col-span-8 grid grid-cols-1 sm:grid-cols-2 gap-3">
          {ANALYSIS_CARDS.map(({ to, label, desc, icon: Icon }) => (
            <Link key={to} to={to} className="card card-hover flex flex-col justify-between p-4 no-underline group">
              <div className="flex items-center justify-between mb-2">
                <div className="w-9 h-9 rounded-xl bg-purple-100 flex items-center justify-center text-purple-700 shadow-sm">
                  <Icon size={18} />
                </div>
                <ArrowUpRight size={14} className="opacity-0 group-hover:opacity-100 transition-opacity text-purple-700" />
              </div>
              <div>
                <span className="text-sm font-bold text-[#1E1238] block">{label}</span>
                <span className="text-xs text-[#5B4E75] mt-0.5 block">{desc}</span>
              </div>
            </Link>
          ))}
        </div>
      </motion.div>

      {/* Recent scans table */}
      {recentScans.length > 0 && (
        <motion.div variants={staggerFadeUp} custom={4} className="card overflow-hidden !p-0">
          <div className="flex items-center justify-between px-5 py-4 border-b border-purple-100">
            <div className="flex items-center gap-2">
              <Clock size={15} className="text-purple-700" />
              <span className="label-tag">Recent Activity</span>
            </div>
            <Link to="/history" className="text-xs font-bold text-purple-700 hover:underline">View All Records</Link>
          </div>
          <div className="table-scroll scroll-fade-x">
            <table className="w-full text-sm min-w-[560px]">
              <thead>
                <tr className="text-left text-[#8F81A8] text-xs bg-purple-50/50 border-b border-purple-100">
                  <th className="px-5 py-3 font-bold uppercase tracking-wider text-[10px]">Time</th>
                  <th className="px-5 py-3 font-bold uppercase tracking-wider text-[10px]">Type</th>
                  <th className="px-5 py-3 font-bold uppercase tracking-wider text-[10px]">File</th>
                  <th className="px-5 py-3 font-bold uppercase tracking-wider text-[10px]">Status</th>
                  <th className="px-5 py-3 font-bold uppercase tracking-wider text-[10px] text-right">Risk</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-purple-100/60">
                {recentScans.map((item) => {
                  const Icon = MEDIA_ICONS[item.media_type] || FileSearch;
                  return (
                    <tr key={item.id} className="hover:bg-purple-50/60 transition-colors cursor-pointer"
                      onClick={() => navigate('/history')}>
                      <td className="px-5 py-3 text-xs font-mono text-[#8F81A8] font-semibold">{formatRelativeTime(item.created_at || item.timestamp)}</td>
                      <td className="px-5 py-3"><Icon size={15} className="text-purple-700" /></td>
                      <td className="px-5 py-3 truncate max-w-[200px] text-[#1E1238] font-semibold">{item.file_name || `${item.media_type} analysis`}</td>
                      <td className="px-5 py-3">
                        <VerdictChip verdict={item.verdict} riskScore={normalizeScore(item.risk_score)} />
                      </td>
                      <td className="px-5 py-3 text-right">
                        <RiskBadge score={item.risk_score} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
