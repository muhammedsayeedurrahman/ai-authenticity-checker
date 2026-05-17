import React, { useEffect, useState, useCallback, useMemo } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Image, Film, Mic, Layers, ArrowUpRight, Clock,
  Shield, Activity, Cpu, Upload, CheckCircle, AlertTriangle,
  XCircle, Zap, TrendingUp, FileSearch,
} from 'lucide-react';
import { Link, useNavigate } from 'react-router-dom';
import { PieChart, Pie, Cell, ResponsiveContainer } from 'recharts';
import useForensicStore from '../store/useForensicStore';
import MiniSparkline from '../components/MiniSparkline';
import EmptyDashboard from '../components/EmptyDashboard';

const fadeUp = {
  hidden: { opacity: 0, y: 14 },
  visible: (i = 0) => ({
    opacity: 1,
    y: 0,
    transition: { delay: i * 0.07, duration: 0.4, ease: [0.22, 1, 0.36, 1] },
  }),
};

const ANALYSIS_CARDS = [
  { to: '/image', label: 'Image', desc: 'Synthetic face & pixel manipulation', icon: Image, shortcut: 'I' },
  { to: '/video', label: 'Video', desc: 'Temporal consistency & frame analysis', icon: Film, shortcut: 'V' },
  { to: '/audio', label: 'Audio', desc: 'Voice cloning & spectrogram forensics', icon: Mic, shortcut: 'A' },
  { to: '/multimodal', label: 'Multimodal', desc: 'Cross-modal fusion confidence matrix', icon: Layers, shortcut: 'M' },
];

const PIE_COLORS = ['var(--risk-clear)', 'var(--risk-caution)', 'var(--risk-critical)'];
const PIE_COLORS_RAW = ['#34D399', '#FBBF24', '#FB7185'];

function getRiskColor(score) {
  if (score > 70) return 'var(--risk-critical)';
  if (score > 40) return 'var(--risk-caution)';
  return 'var(--risk-clear)';
}

function getRiskBg(score) {
  if (score > 70) return 'rgba(251,113,133,0.08)';
  if (score > 40) return 'rgba(251,191,36,0.08)';
  return 'rgba(52,211,153,0.08)';
}

function getRiskLabel(score) {
  if (score > 70) return 'Deepfake Detected';
  if (score > 40) return 'Suspicious';
  return 'Authentic';
}

function formatTime(timestamp) {
  if (!timestamp) return '';
  const date = new Date(timestamp);
  const now = new Date();
  const diffMs = now - date;
  const diffMins = Math.floor(diffMs / 60000);
  if (diffMins < 1) return 'just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  const diffHours = Math.floor(diffMins / 60);
  if (diffHours < 24) return `${diffHours}h ago`;
  const diffDays = Math.floor(diffHours / 24);
  return `${diffDays}d ago`;
}

const MEDIA_ICONS = { image: Image, video: Film, audio: Mic, multimodal: Layers };

function AnimatedNumber({ value, suffix = '' }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    if (value === 0) { setDisplay(0); return; }
    const duration = 600;
    const startTime = Date.now();
    const animate = () => {
      const elapsed = Date.now() - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setDisplay(Math.round(eased * value));
      if (progress < 1) requestAnimationFrame(animate);
    };
    requestAnimationFrame(animate);
  }, [value]);
  return <>{display}{suffix}</>;
}

function StatusPulse({ online }) {
  return (
    <span className="relative flex items-center gap-1.5">
      <span
        className="w-2 h-2 rounded-full"
        style={{ background: online ? 'var(--risk-clear)' : 'var(--risk-critical)' }}
      />
      {online && (
        <span
          className="absolute w-2 h-2 rounded-full animate-ping"
          style={{ background: 'var(--risk-clear)', opacity: 0.4 }}
        />
      )}
      <span className="text-[11px] font-medium" style={{ color: online ? 'var(--risk-clear)' : 'var(--risk-critical)' }}>
        {online ? 'Online' : 'Offline'}
      </span>
    </span>
  );
}

function DonutCenter({ viewBox, total }) {
  const { cx, cy } = viewBox;
  return (
    <>
      <text x={cx} y={cy - 4} textAnchor="middle" fill="var(--text-1)" fontSize="18" fontWeight="700" fontFamily="'Space Grotesk', sans-serif">
        {total}
      </text>
      <text x={cx} y={cy + 12} textAnchor="middle" fill="var(--text-3)" fontSize="9" fontWeight="500">
        TOTAL
      </text>
    </>
  );
}

export default function Dashboard() {
  const { history, historyTotal, fetchHistory, systemStatus, fetchStatus, isStatusLoading } = useForensicStore();
  const navigate = useNavigate();
  const [dragOver, setDragOver] = useState(false);

  useEffect(() => {
    fetchHistory(20);
    fetchStatus();
  }, [fetchHistory, fetchStatus]);

  const modelsOnline = systemStatus.loaded_models?.length || 0;
  const modelsMissing = systemStatus.missing_models?.length || 0;
  const totalModels = modelsOnline + modelsMissing;
  const systemOnline = modelsOnline > 0;

  const riskCounts = history.reduce(
    (acc, item) => {
      const pct = item.risk_score <= 1 ? item.risk_score * 100 : item.risk_score;
      if (pct > 70) acc.critical += 1;
      else if (pct > 40) acc.caution += 1;
      else acc.clear += 1;
      return acc;
    },
    { clear: 0, caution: 0, critical: 0 }
  );

  const threatRate = historyTotal > 0
    ? Math.round((riskCounts.critical / historyTotal) * 100)
    : 0;

  // Sparkline data: last N risk scores from history (oldest→newest)
  const sparkData = useMemo(() => {
    const scores = history
      .slice(0, 12)
      .map((item) => (item.risk_score <= 1 ? item.risk_score * 100 : item.risk_score))
      .reverse();
    return scores;
  }, [history]);

  const pieData = useMemo(() => [
    { name: 'Clear', value: riskCounts.clear },
    { name: 'Caution', value: riskCounts.caution },
    { name: 'Critical', value: riskCounts.critical },
  ].filter((d) => d.value > 0), [riskCounts.clear, riskCounts.caution, riskCounts.critical]);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer?.files?.[0];
    if (!file) return;
    const type = file.type;
    if (type.startsWith('image/')) navigate('/image');
    else if (type.startsWith('video/')) navigate('/video');
    else if (type.startsWith('audio/')) navigate('/audio');
    else navigate('/multimodal');
  }, [navigate]);

  // Show empty state when no scans and not loading
  if (historyTotal === 0 && !isStatusLoading) {
    return (
      <motion.div initial="hidden" animate="visible" className="space-y-5">
        {/* Header + Status row stays visible */}
        <motion.div variants={fadeUp} custom={0} className="flex items-start justify-between flex-wrap gap-4">
          <div>
            <h1 className="font-display text-[22px] font-bold tracking-tight gradient-text">
              Forensics Command Center
            </h1>
            <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-3)' }}>
              AI-powered deepfake detection & media authentication
            </p>
          </div>
          <div className="flex items-center gap-4">
            <StatusPulse online={systemOnline} />
            <span className="text-[11px] font-mono" style={{ color: 'var(--text-3)' }}>
              {modelsOnline}/{totalModels} models
            </span>
          </div>
        </motion.div>

        <EmptyDashboard />
      </motion.div>
    );
  }

  return (
    <motion.div initial="hidden" animate="visible" className="space-y-5">
      {/* ── Row 1: Header + System Status ── */}
      <motion.div variants={fadeUp} custom={0} className="flex items-start justify-between flex-wrap gap-4">
        <div>
          <h1 className="font-display text-[22px] font-bold tracking-tight gradient-text">
            Forensics Command Center
          </h1>
          <p className="text-[12px] mt-0.5" style={{ color: 'var(--text-3)' }}>
            AI-powered deepfake detection & media authentication
          </p>
        </div>
        <div className="flex items-center gap-4">
          <StatusPulse online={systemOnline} />
          <span className="text-[11px] font-mono" style={{ color: 'var(--text-3)' }}>
            {modelsOnline}/{totalModels} models
          </span>
        </div>
      </motion.div>

      {/* ── Row 2: Stat Cards ── */}
      <motion.div variants={fadeUp} custom={1} className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {/* Total Scans */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <FileSearch size={14} style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 6px rgba(59,130,246,0.3))' }} />
            <span className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: 'var(--text-3)' }}>
              Total Scans
            </span>
          </div>
          <div className="flex items-end justify-between">
            <p className="font-display text-2xl font-bold" style={{ color: 'var(--text-1)' }}>
              <AnimatedNumber value={historyTotal} />
            </p>
            {sparkData.length >= 2 && <MiniSparkline data={sparkData} color="#3B82F6" />}
          </div>
        </div>

        {/* Threat Rate */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <AlertTriangle size={14} style={{ color: 'var(--risk-critical)', filter: 'drop-shadow(0 0 6px rgba(251,113,133,0.3))' }} />
            <span className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: 'var(--text-3)' }}>
              Threat Rate
            </span>
          </div>
          <div className="flex items-end justify-between">
            <p className="font-display text-2xl font-bold" style={{ color: threatRate > 30 ? 'var(--risk-critical)' : 'var(--text-1)' }}>
              <AnimatedNumber value={threatRate} suffix="%" />
            </p>
            {sparkData.length >= 2 && <MiniSparkline data={sparkData} color="#FB7185" />}
          </div>
        </div>

        {/* Models Active */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Cpu size={14} style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 6px rgba(59,130,246,0.3))' }} />
            <span className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: 'var(--text-3)' }}>
              Models Active
            </span>
          </div>
          <p className="font-display text-2xl font-bold" style={{ color: 'var(--text-1)' }}>
            <AnimatedNumber value={modelsOnline} />
            <span className="text-sm font-normal" style={{ color: 'var(--text-3)' }}>/{totalModels}</span>
          </p>
        </div>

        {/* Detections */}
        <div className="card p-4">
          <div className="flex items-center gap-2 mb-2">
            <Shield size={14} style={{ color: 'var(--risk-clear)', filter: 'drop-shadow(0 0 6px rgba(52,211,153,0.3))' }} />
            <span className="text-[10px] uppercase tracking-wider font-semibold" style={{ color: 'var(--text-3)' }}>
              Cleared
            </span>
          </div>
          <div className="flex items-end justify-between">
            <p className="font-display text-2xl font-bold" style={{ color: 'var(--risk-clear)' }}>
              <AnimatedNumber value={riskCounts.clear} />
            </p>
            {sparkData.length >= 2 && <MiniSparkline data={sparkData} color="#34D399" />}
          </div>
        </div>
      </motion.div>

      {/* ── Row 3: Risk Distribution Donut ── */}
      {historyTotal > 0 && (
        <motion.div variants={fadeUp} custom={2} className="card-elevated p-4">
          <div className="flex items-center justify-between mb-3">
            <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-2)' }}>
              Risk Distribution
            </span>
            <div className="flex items-center gap-4">
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--risk-clear)' }} />
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>{riskCounts.clear} clear</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--risk-caution)' }} />
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>{riskCounts.caution} caution</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: 'var(--risk-critical)' }} />
                <span className="text-[10px] font-mono" style={{ color: 'var(--text-3)' }}>{riskCounts.critical} critical</span>
              </span>
            </div>
          </div>
          <div className="flex items-center justify-center" style={{ height: '140px' }}>
            <ResponsiveContainer width={140} height={140}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={40}
                  outerRadius={55}
                  dataKey="value"
                  strokeWidth={0}
                >
                  {pieData.map((entry, idx) => {
                    const colorIdx = entry.name === 'Clear' ? 0 : entry.name === 'Caution' ? 1 : 2;
                    return <Cell key={entry.name} fill={PIE_COLORS_RAW[colorIdx]} />;
                  })}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
            <div className="flex flex-col items-center ml-[-90px] pointer-events-none">
              <span className="font-display text-lg font-bold" style={{ color: 'var(--text-1)' }}>
                {historyTotal}
              </span>
              <span className="text-[9px] uppercase tracking-wider" style={{ color: 'var(--text-3)' }}>
                total
              </span>
            </div>
          </div>
        </motion.div>
      )}

      {/* ── Row 4: Quick Upload + Analysis Cards ── */}
      <motion.div variants={fadeUp} custom={3} className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        {/* Left: Quick Upload Drop Zone */}
        <div
          className="card-elevated p-5 flex flex-col items-center justify-center text-center cursor-pointer transition-all"
          style={{
            borderStyle: 'dashed',
            borderColor: dragOver ? 'var(--accent)' : 'var(--border-mid)',
            background: dragOver ? 'rgba(59,130,246,0.05)' : undefined,
            boxShadow: dragOver
              ? 'inset 0 0 24px rgba(59,130,246,0.10)'
              : undefined,
            minHeight: '180px',
            animation: dragOver ? 'none' : 'borderGlow 3s ease-in-out infinite',
          }}
          onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => navigate('/image')}
        >
          <div
            className="w-10 h-10 rounded-xl flex items-center justify-center mb-3"
            style={{ background: 'var(--accent-dim)', boxShadow: '0 0 12px rgba(59,130,246,0.15)' }}
          >
            <Upload size={18} style={{ color: 'var(--accent)' }} />
          </div>
          <p className="text-[13px] font-semibold" style={{ color: 'var(--text-1)' }}>
            Quick Analyze
          </p>
          <p className="text-[11px] mt-1 max-w-[200px]" style={{ color: 'var(--text-3)' }}>
            Drop any file here or click to start an analysis
          </p>
          <div className="flex gap-2 mt-3">
            {['JPG', 'PNG', 'MP4', 'WAV'].map((ext) => (
              <span
                key={ext}
                className="text-[9px] px-1.5 py-0.5 rounded font-mono"
                style={{ background: 'var(--bg-inset)', color: 'var(--text-3)' }}
              >
                {ext}
              </span>
            ))}
          </div>
        </div>

        {/* Right: Analysis Cards Grid */}
        <div className="lg:col-span-2 grid grid-cols-2 gap-3">
          {ANALYSIS_CARDS.map((card, i) => {
            const Icon = card.icon;
            return (
              <motion.div key={card.to} variants={fadeUp} custom={i + 4}>
                <Link
                  to={card.to}
                  className="group card card-hover flex flex-col p-4 h-full"
                  style={{ textDecoration: 'none' }}
                >
                  <div className="flex items-center justify-between mb-2">
                    <div
                      className="w-8 h-8 rounded-lg flex items-center justify-center"
                      style={{ background: 'var(--accent-dim)', boxShadow: '0 0 12px rgba(59,130,246,0.15)' }}
                    >
                      <Icon size={15} style={{ color: 'var(--accent)' }} />
                    </div>
                    <ArrowUpRight
                      size={12}
                      className="opacity-0 group-hover:opacity-100 transition-opacity duration-200"
                      style={{ color: 'var(--accent)' }}
                    />
                  </div>
                  <span className="text-[13px] font-semibold" style={{ color: 'var(--text-1)' }}>
                    {card.label}
                  </span>
                  <span className="text-[11px] mt-0.5 leading-relaxed" style={{ color: 'var(--text-3)' }}>
                    {card.desc}
                  </span>
                </Link>
              </motion.div>
            );
          })}
        </div>
      </motion.div>

      {/* ── Row 5: Recent Activity ── */}
      <motion.div variants={fadeUp} custom={8}>
        <div className="card overflow-hidden">
          <div className="flex items-center justify-between px-4 py-3" style={{ borderBottom: '1px solid var(--border-dim)' }}>
            <div className="flex items-center gap-2">
              <Clock size={13} style={{ color: 'var(--accent)' }} />
              <span className="text-[11px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-2)' }}>
                Recent Activity
              </span>
            </div>
            {history.length > 0 && (
              <Link to="/history" className="text-[11px] font-medium" style={{ color: 'var(--accent)', textDecoration: 'none' }}>
                View all
              </Link>
            )}
          </div>

          {history.length === 0 ? (
            <div className="px-4 py-10 text-center">
              <Activity size={24} style={{ color: 'var(--text-3)', margin: '0 auto 8px' }} />
              <p className="text-[12px]" style={{ color: 'var(--text-3)' }}>
                No analyses yet. Upload a file to get started.
              </p>
            </div>
          ) : (
            <div>
              {history.slice(0, 6).map((item, idx) => {
                const pct = item.risk_score <= 1 ? item.risk_score * 100 : item.risk_score;
                const Icon = MEDIA_ICONS[item.media_type] || FileSearch;
                return (
                  <motion.div
                    key={item.id}
                    initial={{ opacity: 0, x: -8 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: idx * 0.05, duration: 0.3 }}
                    className="flex items-center gap-3 px-4 py-2.5 cursor-pointer transition-colors"
                    style={{
                      background: idx % 2 === 0 ? 'rgba(59,130,246,0.03)' : 'transparent',
                      borderBottom: idx < 5 ? '1px solid rgba(255,255,255,0.03)' : 'none',
                    }}
                    onClick={() => navigate(`/${item.media_type || 'image'}`)}
                  >
                    {/* Icon */}
                    <div
                      className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
                      style={{ background: getRiskBg(pct) }}
                    >
                      <Icon size={13} style={{ color: getRiskColor(pct) }} />
                    </div>

                    {/* File info */}
                    <div className="flex-1 min-w-0">
                      <p className="text-[12px] font-medium truncate" style={{ color: 'var(--text-1)' }}>
                        {item.file_name || `${item.media_type} analysis`}
                      </p>
                      <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>
                        {item.verdict || getRiskLabel(pct)}
                      </p>
                    </div>

                    {/* Risk score */}
                    <span
                      className="text-[11px] font-bold font-mono px-2 py-0.5 rounded"
                      style={{ color: getRiskColor(pct), background: getRiskBg(pct) }}
                    >
                      {pct.toFixed(0)}%
                    </span>

                    {/* Time */}
                    <span className="text-[10px] font-mono w-14 text-right flex-shrink-0" style={{ color: 'var(--text-3)' }}>
                      {formatTime(item.created_at || item.timestamp)}
                    </span>
                  </motion.div>
                );
              })}
            </div>
          )}
        </div>
      </motion.div>

      {/* ── Row 6: System Health Footer ── */}
      <motion.div variants={fadeUp} custom={9} className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        <div className="card p-3 flex items-center gap-3">
          <Zap size={14} style={{ color: systemOnline ? 'var(--risk-clear)' : 'var(--risk-critical)', filter: 'drop-shadow(0 0 4px currentColor)' }} />
          <div>
            <p className="text-[11px] font-medium" style={{ color: 'var(--text-1)' }}>Inference Engine</p>
            <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>
              {systemStatus.device === 'auto' ? 'Auto (GPU/CPU)' : systemStatus.device}
            </p>
          </div>
        </div>
        <div className="card p-3 flex items-center gap-3">
          <Shield size={14} style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 4px rgba(59,130,246,0.4))' }} />
          <div>
            <p className="text-[11px] font-medium" style={{ color: 'var(--text-1)' }}>Detection Pipeline</p>
            <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>
              {systemStatus.corefakenet_available ? 'CoreFakeNet active' : 'Ensemble mode'}
            </p>
          </div>
        </div>
        <div className="card p-3 flex items-center gap-3">
          <TrendingUp size={14} style={{ color: 'var(--accent)', filter: 'drop-shadow(0 0 4px rgba(59,130,246,0.4))' }} />
          <div>
            <p className="text-[11px] font-medium" style={{ color: 'var(--text-1)' }}>Accuracy</p>
            <p className="text-[10px]" style={{ color: 'var(--text-3)' }}>
              {systemStatus.vit_available ? 'ViT + CLIP ensemble' : 'Standard models'}
            </p>
          </div>
        </div>
      </motion.div>
    </motion.div>
  );
}
